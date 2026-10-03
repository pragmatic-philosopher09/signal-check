"""Adapter protocol, typed failure results and helpers shared by live adapters.

Every source turns ``fetch(query, params)`` into a :class:`Series`. Adapters raise
:class:`AdapterError` (or :class:`AdapterDisabled` when a source is switched off or
has no credentials) with a short, user-presentable reason; callers use
:func:`fetch_safely`, which never raises and returns a :class:`FetchResult` whose
:attr:`FetchResult.message` is the per-card text ("Couldn't fetch Reddit: ..." or
"X disabled (planned)").

Live adapters derive from :class:`LiveAdapter`, which bundles the shared HTTP
session (identifying User-Agent, timeouts, retry/backoff), the disk cache and an
injectable clock/sleep so tests never hit the network or wait.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import requests

from signalcheck import http
from signalcheck.cache import CacheLike, get_cache
from signalcheck.config import Config, get_config
from signalcheck.models import Series

log = logging.getLogger(__name__)

SOURCE_LABELS: dict[str, str] = {
    "google_trends": "Google Trends",
    "reddit": "Reddit",
    "x": "X",
    "wikipedia": "Wikipedia",
    "hackernews": "Hacker News",
    "csv": "CSV upload",
}


class AdapterError(RuntimeError):
    """A source could not produce a series; ``reason`` is safe to show to users."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class AdapterDisabled(AdapterError):
    """The source is switched off (config/feature flag) or lacks credentials."""


@runtime_checkable
class Adapter(Protocol):
    """A data source. ``source`` is one of :data:`signalcheck.models.SOURCES`."""

    source: str

    def fetch(self, query: str, params: Mapping[str, Any]) -> Series:
        """Return the series for ``query``; raise :class:`AdapterError` on failure."""
        ...


@dataclass(frozen=True)
class FetchResult:
    """Outcome of one adapter call: exactly one of ``series`` / ``error`` is set.

    ``disabled`` distinguishes "switched off / no credentials" from a failed fetch,
    so the UI can grey the source out instead of showing an error.
    """

    source: str
    query: str
    series: Series | None = None
    error: str | None = None
    disabled: bool = False

    @property
    def ok(self) -> bool:
        """True when a series was produced."""
        return self.series is not None

    @property
    def label(self) -> str:
        """Display name of the source."""
        return SOURCE_LABELS.get(self.source, self.source)

    @property
    def message(self) -> str | None:
        """Per-card text for a failure (``None`` on success)."""
        if self.ok:
            return None
        if self.disabled:
            return f"{self.label} disabled ({self.error})"
        return f"Couldn't fetch {self.label}: {self.error}"


def fetch_safely(
    adapter: Adapter, query: str, params: Mapping[str, Any] | None = None
) -> FetchResult:
    """Run ``adapter.fetch`` and turn every failure into a :class:`FetchResult`.

    This is the only place that catches broad exceptions: one source failing must
    never crash the app. Unexpected errors are logged (type only, no payloads) and
    reported generically.
    """
    source = adapter.source
    try:
        series = adapter.fetch(query, params or {})
    except AdapterDisabled as exc:
        return FetchResult(source, query, error=exc.reason, disabled=True)
    except AdapterError as exc:
        return FetchResult(source, query, error=exc.reason)
    except http.HttpError as exc:
        return FetchResult(source, query, error=exc.reason)
    except Exception as exc:
        log.error("adapter %s failed with %s", source, type(exc).__name__)
        return FetchResult(source, query, error=f"unexpected error ({type(exc).__name__})")
    return FetchResult(source, query, series=series)


def slugify(query: str) -> str:
    """File-name slug: ``"Perplexity AI!"`` -> ``"perplexity-ai"``."""
    slug = re.sub(r"[^a-z0-9]+", "-", " ".join(query.split()).casefold()).strip("-")
    if not slug:
        raise AdapterError("query must contain at least one letter or digit")
    return slug


def require_query(query: str) -> str:
    """Stripped query, or :class:`AdapterError` when it is blank."""
    cleaned = " ".join(query.split())
    if not cleaned:
        raise AdapterError("enter a topic to search for")
    return cleaned


def to_date(value: date | datetime | str) -> date:
    """A calendar date from a ``date``, ``datetime`` (UTC) or ISO string."""
    if isinstance(value, datetime):
        return (value.astimezone(UTC) if value.tzinfo else value).date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError as exc:
        raise AdapterError(f"{value!r} is not a date (use YYYY-MM-DD)") from exc


def history_range(params: Mapping[str, Any], default_days: int, today: date) -> tuple[date, date]:
    """Inclusive ``(start, end)`` dates from ``params`` (``start``/``end``/``days``).

    ``end`` defaults to ``today`` (UTC; its bucket is partial and preprocessing
    drops it) and ``start`` to ``end - days + 1``.
    """
    end = to_date(params["end"]) if params.get("end") else today
    end = min(end, today)
    if params.get("start"):
        start = to_date(params["start"])
    else:
        days = int(params.get("days") or default_days)
        if days < 1:
            raise AdapterError("days must be >= 1")
        start = end - timedelta(days=days - 1)
    if start > end:
        raise AdapterError(f"start {start} is after end {end}")
    return start, end


def day_range(start: date, end: date) -> list[date]:
    """Every date from ``start`` to ``end`` inclusive."""
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def day_start_utc(day: date) -> datetime:
    """Midnight UTC at the start of ``day``."""
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def epoch_day(seconds: float) -> date:
    """UTC calendar day of a Unix timestamp."""
    return datetime.fromtimestamp(seconds, tz=UTC).date()


def points_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """``points`` DataFrame from per-bucket dicts with ``ts`` (date) and ``value``.

    Optional columns (``contributors``, ``top_share``, ``raw_count``, ``imputed``)
    are kept when any row has them; missing entries become NaN / False.
    """
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns]"), "value": []})
    frame["ts"] = pd.to_datetime(frame["ts"]).astype("datetime64[ns]")
    frame["value"] = frame["value"].astype(float)
    for column in ("contributors", "top_share", "raw_count"):
        if column in frame.columns:
            frame[column] = frame[column].astype(float)
    if "imputed" in frame.columns:
        frame["imputed"] = frame["imputed"].astype("boolean").fillna(False).astype(bool)
    return frame.sort_values("ts").reset_index(drop=True)


def breadth_aggregates(origins: list[str | None]) -> tuple[int, float]:
    """``(contributors, top_share)`` for one bucket's item origins (authors, subreddits).

    ``contributors`` counts distinct known origins; ``top_share`` is the largest
    share of the bucket's items from a single known origin (unknown/deleted origins
    count as items but never as an origin). Empty buckets give ``(0, nan)``.
    The origins themselves are only held in memory by the caller.
    """
    if not origins:
        return 0, float("nan")
    known = [o for o in origins if o]
    if not known:
        return 0, 0.0
    _, counts = np.unique(np.asarray(known, dtype=object), return_counts=True)
    return len(counts), float(counts.max() / len(origins))


class Pacer:
    """Enforces a minimum interval between requests (client-side rate limiting)."""

    def __init__(
        self,
        min_interval_s: float,
        sleep: Callable[[float], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_interval_s = float(min_interval_s)
        self.sleep = sleep
        self.clock = clock
        self._last: float | None = None

    def wait(self) -> None:
        """Sleep until ``min_interval_s`` has passed since the previous call."""
        if self._last is not None:
            remaining = self.min_interval_s - (self.clock() - self._last)
            if remaining > 0:
                self.sleep(remaining)
        self._last = self.clock()


class LiveAdapter:
    """Base for adapters that call external APIs.

    Dependencies are injectable for tests: ``session`` (defaults to the shared
    :func:`signalcheck.http.get_session`), ``cache`` (defaults to the 6h disk
    cache; pass :class:`~signalcheck.cache.NoCache` for fresh data), ``sleep``,
    ``now`` (a callable returning an aware UTC datetime) and ``rng`` (backoff jitter).
    """

    source = ""

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        session: requests.Session | None = None,
        cache: CacheLike | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else get_config()
        self._session = session
        self._cache = cache
        self.sleep = sleep
        self._now = now
        self.rng = rng

    @property
    def label(self) -> str:
        """Display name of this source."""
        return SOURCE_LABELS[self.source]

    @property
    def settings(self) -> dict[str, Any]:
        """This adapter's ``adapters.<source>`` block from ``config.yaml``."""
        settings: dict[str, Any] = self.cfg["adapters"][self.source]
        return settings

    @property
    def session(self) -> requests.Session:
        """Injected session or the shared one."""
        return self._session if self._session is not None else http.get_session()

    @property
    def cache(self) -> CacheLike:
        """Injected cache or the process-wide disk cache."""
        if self._cache is None:
            self._cache = get_cache()
        return self._cache

    def now(self) -> datetime:
        """Current time (aware, UTC)."""
        return self._now() if self._now is not None else datetime.now(UTC)

    def ensure_enabled(self) -> None:
        """Raise :class:`AdapterDisabled` when ``sources.<source>`` is false."""
        if not self.cfg["sources"].get(self.source, False):
            raise AdapterDisabled("switched off in config.yaml")

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """:func:`signalcheck.http.request` with this adapter's dependencies."""
        return http.request(
            method,
            url,
            session=self.session,
            cfg=self.cfg,
            rng=self.rng,
            sleep=self.sleep,
            **kwargs,
        )

    def get_json(self, url: str, **kwargs: Any) -> Any:
        """GET + decode JSON via the shared HTTP layer (raises :class:`HttpError`)."""
        return http.get_json(
            url, session=self.session, cfg=self.cfg, rng=self.rng, sleep=self.sleep, **kwargs
        )

    def base_meta(self, fetched_at: datetime, resolved_query: str, **extra: Any) -> dict[str, Any]:
        """``Series.meta`` skeleton for a live fetch (``partial_rule = fetched_at``)."""
        meta: dict[str, Any] = {
            "fetched_at": fetched_at.astimezone(UTC).isoformat(),
            "caveats": [],
            "resolved_query": resolved_query,
            "dropped_partial": None,
            "truncated_before": None,
            "partial_rule": "fetched_at",
        }
        meta.update(extra)
        return meta
