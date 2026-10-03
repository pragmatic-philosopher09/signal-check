"""Google Trends adapter: best-effort live fetch with a snapshot fallback.

There is no open official Google Trends API (the official API is a gated alpha)
and the community client ``pytrends`` was archived in April 2025 and is often
rate-limited (HTTP 429), especially from cloud IPs. So:

1. **Live (best effort).** A pluggable :class:`TrendsFetcher`. The default,
   :class:`PytrendsFetcher`, imports ``pytrends`` lazily and only if the developer
   installed it (it is deliberately *not* a dependency); otherwise live fetching is
   simply unavailable. Results are cached (6h) like every other external call.
2. **Snapshot fallback.** ``<snapshot_dir>/<slug>.json`` (written by
   ``scripts/refresh_samples.py``) or ``<slug>.csv`` (a CSV downloaded from the
   Trends website, optionally with ``<slug>.meta.json`` = ``{"fetched_at": ...}``).
   The deployed demo works from snapshots alone.
3. CSV upload stays available through :mod:`signalcheck.adapters.csv_upload`.

Values are relative (0-100, scaled per request) and sampled, so both caveats are
always added; ``<1`` cells become ``0.5`` with ``imputed=True``. Granularity is set
by the timeframe: <= 90 days daily, <= 5 years weekly, longer monthly.

References (checked 3 Oct 2026):

* Official Trends API alpha: https://developers.google.com/search/apis/trends
* pytrends (archived): https://github.com/GeneralMills/pytrends
* Trends FAQ on sampling/normalisation: https://support.google.com/trends/answer/4365533
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from signalcheck.adapters.base import (
    AdapterError,
    LiveAdapter,
    points_frame,
    require_query,
    slugify,
)
from signalcheck.adapters.csv_upload import TRENDS_CAVEATS, TRENDS_LT1, parse_csv
from signalcheck.config import Config, resolve_path
from signalcheck.http import user_agent
from signalcheck.models import Series
from signalcheck.snapshots import mark_snapshot, read_snapshot

UNKNOWN_FETCH_TIME_CAVEAT = (
    "Snapshot CSV has no recorded fetch time; its last period is treated as incomplete."
)


@dataclass(frozen=True)
class TrendsRequest:
    """One Trends request: ``timeframe`` is ``"YYYY-MM-DD YYYY-MM-DD"``, ``geo`` ``""`` = world."""

    query: str
    start: date
    end: date
    geo: str

    @property
    def timeframe(self) -> str:
        """Trends timeframe string."""
        return f"{self.start:%Y-%m-%d} {self.end:%Y-%m-%d}"


class TrendsFetcher(Protocol):
    """Live source of interest-over-time rows: ``[(date, value_or_'<1'), ...]``.

    Raise :class:`AdapterError` (or anything else; it is caught) when unavailable.
    """

    def __call__(self, request: TrendsRequest) -> list[tuple[date, float | str]]: ...


class PytrendsFetcher:
    """Live fetcher backed by the optional, archived ``pytrends`` package.

    Not installed by default; ``pip install pytrends`` on a developer machine to
    refresh snapshots. Uses this app's User-Agent, timeouts and retry budget.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def __call__(self, request: TrendsRequest) -> list[tuple[date, float | str]]:
        """Interest over time for one keyword (UTC dates)."""
        try:
            module = importlib.import_module("pytrends.request")
        except ImportError as exc:
            raise AdapterError("live fetch unavailable (optional pytrends not installed)") from exc
        http_cfg = self.cfg["http"]
        client = module.TrendReq(
            hl="en-US",
            tz=0,
            timeout=(http_cfg["connect_timeout_s"], http_cfg["read_timeout_s"]),
            retries=int(http_cfg["max_retries"]),
            backoff_factor=float(http_cfg["backoff_base_s"]),
            requests_args={"headers": {"User-Agent": user_agent()}},
        )
        client.build_payload([request.query], timeframe=request.timeframe, geo=request.geo)
        frame = client.interest_over_time()
        if frame is None or frame.empty or request.query not in frame.columns:
            return []
        return [
            (pd.Timestamp(ts).date(), float(v))
            for ts, v in zip(frame.index, frame[request.query], strict=True)
        ]


def granularity(days: int, cfg: Config) -> str:
    """Bucket size Google returns for a ``days``-long timeframe (D, W or M)."""
    gt = cfg["adapters"]["google_trends"]
    if days <= gt["daily_max_days"]:
        return "D"
    if days <= gt["weekly_max_days"]:
        return "W"
    return "M"


class GoogleTrendsAdapter(LiveAdapter):
    """``relative_0_100`` series from live Trends (if available) or a snapshot.

    Params: ``days`` (timeframe length), ``geo``, ``live`` (``False`` = snapshot only).
    """

    source = "google_trends"

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        fetcher: TrendsFetcher | None = None,
        snapshot_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cfg, **kwargs)
        self.fetcher: TrendsFetcher = fetcher if fetcher is not None else PytrendsFetcher(self.cfg)
        self._snapshot_dir = snapshot_dir

    @property
    def snapshot_dir(self) -> Path:
        """Directory holding ``<slug>.json`` / ``<slug>.csv`` snapshots."""
        if self._snapshot_dir is not None:
            return self._snapshot_dir
        return resolve_path(self.settings["snapshot_dir"])

    def fetch(self, query: str, params: Mapping[str, Any]) -> Series:
        """Live series, else the snapshot with a caveat explaining why."""
        self.ensure_enabled()
        query = require_query(query)
        live_reason = "live fetching is switched off"
        if self.settings["live_enabled"] and params.get("live", True):
            try:
                return self.fetch_live(query, params)
            except AdapterError as exc:
                live_reason = exc.reason
            except Exception as exc:
                live_reason = f"live fetch failed ({type(exc).__name__})"
        try:
            series = self.load_snapshot(query)
        except AdapterError as exc:
            raise AdapterError(f"{live_reason}; {exc.reason}") from exc
        caveats = [*series.caveats, f"Live Google Trends unavailable: {live_reason}."]
        return Series(
            series.source,
            series.query,
            series.freq,
            series.points,
            series.scale,
            {**series.meta, "caveats": caveats},
        )

    def fetch_live(self, query: str, params: Mapping[str, Any]) -> Series:
        """Best-effort live fetch through :attr:`fetcher` (cached)."""
        days = int(params.get("days") or self.settings["timeframe_days"])
        if days < 1:
            raise AdapterError("days must be >= 1")
        geo = str(params.get("geo", self.settings["geo"]))
        end = self.now().date()
        request = TrendsRequest(query, end - timedelta(days=days - 1), end, geo)
        payload = self.cache.get_or_fetch(
            self.source,
            query,
            {"timeframe": request.timeframe, "geo": geo},
            lambda: self._fetch_rows(request),
        )
        lt1 = float(self.cfg["adapters"]["csv"]["trends_lt1_value"])
        rows = [
            {
                "ts": date.fromisoformat(d),
                "value": lt1 if v == TRENDS_LT1 else v,
                "imputed": v == TRENDS_LT1,
            }
            for d, v in payload["rows"]
        ]
        meta = self.base_meta(datetime.fromisoformat(payload["fetched_at"]), query, geo=geo)
        meta["caveats"] = list(TRENDS_CAVEATS)
        n_lt1 = sum(r["imputed"] for r in rows)
        if n_lt1:
            meta["caveats"].append(
                f"{n_lt1} value(s) reported as '<1' set to {lt1:g} and marked imputed."
            )
        return Series(
            source=self.source,
            query=query,
            freq=granularity(days, self.cfg),
            points=points_frame(rows),
            scale="relative_0_100",
            meta=meta,
        )

    def _fetch_rows(self, request: TrendsRequest) -> dict[str, Any]:
        fetched_at = self.now()
        rows = self.fetcher(request)
        if not rows:
            raise AdapterError(f"Google Trends returned no data for {request.query!r}")
        return {
            "fetched_at": fetched_at.isoformat(),
            "rows": [(d.isoformat(), v if v == TRENDS_LT1 else float(v)) for d, v in sorted(rows)],
        }

    def load_snapshot(self, query: str) -> Series:
        """Snapshot for ``query`` (JSON preferred, then CSV download) with age caveats."""
        slug = slugify(query)
        json_path = self.snapshot_dir / f"{slug}.json"
        csv_path = self.snapshot_dir / f"{slug}.csv"
        if json_path.exists():
            series = read_snapshot(json_path)
        elif csv_path.exists():
            series = self._read_csv_snapshot(query, csv_path)
        else:
            raise AdapterError(f"no Google Trends snapshot for {query!r}")
        if series.source != self.source:
            raise AdapterError(f"snapshot {json_path.name} is not a Google Trends snapshot")
        series = mark_snapshot(series, self.now())
        age = int(series.meta["snapshot_age_days"])
        stale = int(self.settings["snapshot_stale_days"])
        if age > stale:
            series.meta["caveats"].append(
                f"Snapshot is {age} days old (older than {stale}); recent changes are not shown."
            )
        for caveat in TRENDS_CAVEATS:
            if caveat not in series.meta["caveats"]:
                series.meta["caveats"].append(caveat)
        return series

    def _read_csv_snapshot(self, query: str, path: Path) -> Series:
        """Parse a Trends-website CSV download; fetch time from ``<slug>.meta.json`` if present."""
        sidecar = path.with_suffix(".meta.json")
        fetched_at: datetime | None = None
        if sidecar.exists():
            try:
                raw = json.loads(sidecar.read_text(encoding="utf-8"))["fetched_at"]
                fetched_at = datetime.fromisoformat(raw)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise AdapterError(f"could not read {sidecar.name}: {exc}") from exc
        return series_from_trends_csv(
            path.read_bytes(),
            query,
            fetched_at,
            self.cfg,
            fallback_time=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        )


def series_from_trends_csv(
    data: bytes | str,
    query: str,
    fetched_at: datetime | None,
    cfg: Config,
    *,
    fallback_time: datetime | None = None,
) -> Series:
    """Google Trends series from a CSV downloaded from the Trends website.

    With a known ``fetched_at`` the partial-period rule is ``fetched_at``; without
    it the last period is treated as incomplete (and a caveat says so), dated at
    ``fallback_time`` (default: now).
    """
    if fetched_at is not None and fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=UTC)
    when = fetched_at or fallback_time or datetime.now(UTC)
    series = parse_csv(
        data,
        query=query,
        upload_date=when.date(),
        last_period_incomplete=fetched_at is None,
        source="google_trends",
        now=when,
        cfg=cfg,
    )
    meta = dict(series.meta)
    if meta.get("csv_format") != "google_trends":
        raise AdapterError("CSV is not a Google Trends download (expected '<query>: (<geo>)')")
    if fetched_at is not None:
        meta["partial_rule"] = "fetched_at"
    else:
        meta["caveats"] = [*meta["caveats"], UNKNOWN_FETCH_TIME_CAVEAT]
    return Series(series.source, series.query, series.freq, series.points, series.scale, meta)
