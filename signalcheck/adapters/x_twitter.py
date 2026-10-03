"""X (Twitter) adapter: daily post counts from the official X API v2 counts endpoint.

**Off by default.** The source runs only when ``ENABLE_X`` is truthy (env /
``st.secrets``; falls back to ``sources.x`` in ``config.yaml`` when unset). When off,
the card reads "X disabled (planned)". Without ``X_BEARER_TOKEN`` it reads
"X disabled (API credentials not configured)".

**Counts, not posts.** ``GET /2/tweets/counts/all`` (full archive, pay-per-use) or
``/2/tweets/counts/recent`` (last 7 days) with ``granularity=day`` returns per-day
``tweet_count`` buckets without returning any posts, billed per request rather
than per post. ``adapters.x.query_suffix`` (``-is:retweet``) keeps reposts out.
Pages are followed with ``next_token``.

**Breadth (optional, ``breadth_enabled``).** For days in the engine's recent window
that ``/2/tweets/search/recent`` still covers (last 7 days), the newest
``breadth_posts_per_day`` posts of each day are read with ``tweet.fields=author_id``:
``contributors`` = distinct author ids in that sample and ``top_share`` = the
largest single-author share. Ids and text are discarded as soon as a page has been
aggregated. Post reads are billed per post returned.

**Budget.** Every billable request is checked against a persisted spend ledger
(``<cache.dir>/<ledger_file>``, gitignored) and refused if it would push total
spend above ``X_MAX_SPEND_USD`` (default ``default_max_spend_usd``). The projected
cost of the whole fetch is checked before the first request, so a fetch that
cannot finish within budget makes no request at all. Costs come from
``config.yaml``; a request is recorded only after a successful response (failed
requests are not billed). An unreadable ledger or budget fails closed. Results
are cached (6h), so repeating a query spends nothing.

Docs (checked 3 Oct 2026):

* Counts endpoints: https://docs.x.com/x-api/posts/counts/introduction
* counts/all reference: https://docs.x.com/x-api/posts/get-count-of-all-posts
* Recent search: https://docs.x.com/x-api/posts/search/api-reference/get-tweets-search-recent
* Pay-per-use pricing (Counts: Recent $0.005 / Counts: All $0.010 per request;
  Posts: Read $0.005 per post): https://docs.x.com/x-api/getting-started/pricing
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from signalcheck.adapters.base import (
    AdapterDisabled,
    AdapterError,
    LiveAdapter,
    breadth_aggregates,
    day_range,
    day_start_utc,
    history_range,
    points_frame,
    require_query,
)
from signalcheck.config import Config, get_secret, resolve_path, secret_flag
from signalcheck.engine.preprocess import recent_window_length
from signalcheck.http import HttpError
from signalcheck.models import Series

API_BASE = "https://api.x.com"
COUNTS_PATHS = {"all": "/2/tweets/counts/all", "recent": "/2/tweets/counts/recent"}
SEARCH_RECENT_PATH = "/2/tweets/search/recent"
RECENT_DAYS = 7
# X rejects end_time values less than 10 seconds before the request.
END_TIME_MARGIN = timedelta(seconds=30)
MATCH_CAVEAT = (
    "X counts are original posts and replies matching the query (reposts excluded), "
    "as counted by X; they are not deduplicated by author."
)
_LEDGER_LOCK = threading.Lock()


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class SpendLedger:
    """Persisted record of X API spend (JSON, atomic writes).

    Holds only ``total_usd`` and entries ``{at, endpoint, cost_usd, units}``; no query
    text, ids or post content.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        """Ledger contents; ``AdapterError`` if the file exists but is unreadable."""
        if not self.path.exists():
            return {"total_usd": 0.0, "entries": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            total = float(data["total_usd"])
            entries = list(data.get("entries", []))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AdapterError(
                f"X spend ledger {self.path.name} is unreadable; refusing to spend"
            ) from exc
        if not math.isfinite(total) or total < 0:
            raise AdapterError(f"X spend ledger {self.path.name} is invalid; refusing to spend")
        return {"total_usd": total, "entries": entries}

    def total(self) -> float:
        """Total recorded spend in USD."""
        return float(self.load()["total_usd"])

    def record(self, endpoint: str, cost_usd: float, units: int, at: datetime) -> float:
        """Add one billed request and return the new total."""
        with _LEDGER_LOCK:
            data = self.load()
            data["total_usd"] = round(data["total_usd"] + cost_usd, 6)
            data["entries"].append(
                {"at": _iso(at), "endpoint": endpoint, "cost_usd": cost_usd, "units": units}
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
            return float(data["total_usd"])


class XAdapter(LiveAdapter):
    """Daily ``count`` series of X posts (counts endpoint) with a hard spend cap.

    Params: ``days``/``start``/``end``. Extra kwarg ``ledger_path`` overrides the
    ledger location (tests).
    """

    source = "x"

    def __init__(
        self, cfg: Config | None = None, *, ledger_path: Path | None = None, **kwargs: Any
    ) -> None:
        super().__init__(cfg, **kwargs)
        if ledger_path is None:
            ledger_path = resolve_path(self.cfg["cache"]["dir"]) / self.settings["ledger_file"]
        self.ledger = SpendLedger(ledger_path)

    def ensure_enabled(self) -> None:
        """``ENABLE_X`` decides; when unset, ``sources.x`` does. Off reads "planned"."""
        flag = secret_flag("ENABLE_X")
        enabled = flag if flag is not None else bool(self.cfg["sources"].get(self.source, False))
        if not enabled:
            raise AdapterDisabled("planned")

    def token(self) -> str:
        """Bearer token or :class:`AdapterDisabled`."""
        token = get_secret("X_BEARER_TOKEN")
        if not token:
            raise AdapterDisabled("API credentials not configured")
        return token

    def budget(self) -> float:
        """``X_MAX_SPEND_USD`` (or the config default); invalid values fail closed."""
        raw = get_secret("X_MAX_SPEND_USD")
        if raw is None or not raw.strip():
            return float(self.settings["default_max_spend_usd"])
        try:
            value = float(raw)
        except ValueError as exc:
            raise AdapterError("X_MAX_SPEND_USD is not a number; refusing to spend") from exc
        if not math.isfinite(value) or value < 0:
            raise AdapterError("X_MAX_SPEND_USD must be a non-negative number")
        return value

    def remaining_budget(self) -> float:
        """Budget minus recorded spend (may be negative if the cap was lowered)."""
        return self.budget() - self.ledger.total()

    def _ensure_affordable(self, cost: float, what: str) -> None:
        spent, budget = self.ledger.total(), self.budget()
        if spent + cost > budget + 1e-9:
            raise AdapterError(
                f"X budget cap reached: {what} would cost ${cost:.3f}, ${spent:.3f} of "
                f"${budget:.2f} already spent (X_MAX_SPEND_USD)"
            )

    def fetch(self, query: str, params: Mapping[str, Any]) -> Series:
        """Cached daily counts (+ optional breadth) as a ``count`` series."""
        self.ensure_enabled()
        query = require_query(query)
        self.token()
        s = self.settings
        now = self.now()
        start, end = history_range(params, int(s["history_days"]), now.date())
        caveats = [MATCH_CAVEAT]
        if s["counts_endpoint"] == "recent":
            earliest = (now - timedelta(days=RECENT_DAYS)).date() + timedelta(days=1)
            if start < earliest:
                caveats.append(
                    f"counts/recent covers only the last {RECENT_DAYS} days, so the series "
                    f"starts on {earliest.isoformat()} instead of {start.isoformat()}."
                )
                start = earliest
        if start > end:
            raise AdapterError("requested range is empty")
        full_query = f"{query} {s['query_suffix']}".strip()
        if len(full_query) > 512:
            raise AdapterError("X queries are limited to 512 characters")
        cache_params = {
            "start": start,
            "end": end,
            "endpoint": s["counts_endpoint"],
            "suffix": s["query_suffix"],
            "breadth": bool(s["breadth_enabled"]),
            "per_day": s["breadth_posts_per_day"] if s["breadth_enabled"] else 0,
        }
        payload = self.cache.get_or_fetch(
            self.source,
            query,
            cache_params,
            lambda: self._fetch_live(full_query, start, end),
        )
        return self._build_series(query, payload, caveats)

    def breadth_days(self, days: list[date], now: datetime) -> list[date]:
        """Recent-window days (plus a partial today) that search/recent still covers."""
        complete = [d for d in days if day_start_utc(d) + timedelta(days=1) <= now]
        partial = [d for d in days if d not in set(complete)]
        recent = complete[-recent_window_length(len(complete), "D", self.cfg) :] if complete else []
        cutoff = now - timedelta(days=RECENT_DAYS) + timedelta(minutes=1)
        return [d for d in recent + partial if day_start_utc(d) >= cutoff]

    def projected_cost(self, start: date, end: date, n_breadth_days: int) -> float:
        """Worst-case cost of a fetch: counts pages plus the full breadth sample."""
        s = self.settings
        n_days = (end - start).days + 1
        if s["counts_endpoint"] == "all":
            pages = math.ceil(n_days / int(s["counts_all_page_days"]))
            cost = pages * float(s["cost_counts_all_usd"])
        else:
            cost = float(s["cost_counts_recent_usd"])
        if s["breadth_enabled"]:
            posts = n_breadth_days * int(s["breadth_posts_per_day"])
            cost += posts * float(s["cost_post_read_usd"])
        return cost

    def _get(self, path: str, params: Mapping[str, Any]) -> Any:
        headers = {"Authorization": f"Bearer {self.token()}"}
        try:
            response = self.request("GET", API_BASE + path, params=dict(params), headers=headers)
        except HttpError as exc:
            if exc.status in (401, 403):
                raise AdapterError(
                    f"X rejected the bearer token or plan access (HTTP {exc.status})"
                ) from exc
            if exc.status == 402:
                raise AdapterError("X account has no API credits left (HTTP 402)") from exc
            raise
        return response.json() or {}

    def _fetch_live(self, full_query: str, start: date, end: date) -> dict[str, Any]:
        fetched_at = self.now()
        days = day_range(start, end)
        breadth = self.breadth_days(days, fetched_at) if self.settings["breadth_enabled"] else []
        projected = self.projected_cost(start, end, len(breadth))
        self._ensure_affordable(projected, "this fetch")
        counts, n_requests = self._fetch_counts(full_query, start, end, fetched_at)
        breadth_rows = {
            day.isoformat(): self._fetch_breadth(full_query, day, fetched_at) for day in breadth
        }
        return {
            "fetched_at": fetched_at.isoformat(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "counts": counts,
            "breadth": breadth_rows,
            "requests": n_requests,
        }

    def _fetch_counts(
        self, full_query: str, start: date, end: date, now: datetime
    ) -> tuple[dict[str, int], int]:
        s = self.settings
        endpoint = str(s["counts_endpoint"])
        cost = float(s["cost_counts_all_usd" if endpoint == "all" else "cost_counts_recent_usd"])
        upper = min(day_start_utc(end + timedelta(days=1)), now - END_TIME_MARGIN)
        params: dict[str, Any] = {
            "query": full_query,
            "granularity": "day",
            "start_time": _iso(day_start_utc(start)),
            "end_time": _iso(upper),
        }
        counts: dict[str, int] = {}
        n_requests = 0
        max_pages = math.ceil(((end - start).days + 1) / int(s["counts_all_page_days"])) + 2
        while True:
            self._ensure_affordable(cost, f"counts/{endpoint} page")
            body = self._get(COUNTS_PATHS[endpoint], params)
            n_requests += 1
            self.ledger.record(f"counts/{endpoint}", cost, 1, self.now())
            for bucket in body.get("data") or []:
                day = datetime.fromisoformat(str(bucket["start"]).replace("Z", "+00:00"))
                key = day.astimezone(UTC).date().isoformat()
                counts[key] = counts.get(key, 0) + int(bucket.get("tweet_count", 0))
            token = (body.get("meta") or {}).get("next_token")
            if not token:
                break
            if n_requests >= max_pages:
                raise AdapterError("X counts pagination did not terminate")
            params["next_token"] = token
        return counts, n_requests

    def _fetch_breadth(self, full_query: str, day: date, now: datetime) -> dict[str, float]:
        """Sample the newest posts of one day; returns aggregates only."""
        s = self.settings
        per_day = int(s["breadth_posts_per_day"])
        cost_per_post = float(s["cost_post_read_usd"])
        self._ensure_affordable(per_day * cost_per_post, "breadth sample")
        lower = max(day_start_utc(day), now - timedelta(days=RECENT_DAYS) + timedelta(minutes=1))
        upper = min(day_start_utc(day) + timedelta(days=1), now - END_TIME_MARGIN)
        body = self._get(
            SEARCH_RECENT_PATH,
            {
                "query": full_query,
                "start_time": _iso(lower),
                "end_time": _iso(upper),
                "max_results": per_day,
                "tweet.fields": "author_id",
            },
        )
        posts = body.get("data") or []
        self.ledger.record("search/recent", len(posts) * cost_per_post, len(posts), self.now())
        origins = [str(p["author_id"]) if p.get("author_id") else None for p in posts]
        contributors, top_share = breadth_aggregates(origins)
        sampled = len(origins)
        origins.clear()
        posts.clear()
        return {"contributors": float(contributors), "top_share": top_share, "sampled": sampled}

    def _build_series(self, query: str, payload: Mapping[str, Any], caveats: list[str]) -> Series:
        start = date.fromisoformat(payload["start"])
        end = date.fromisoformat(payload["end"])
        counts: Mapping[str, int] = payload["counts"]
        breadth: Mapping[str, Mapping[str, float]] = payload["breadth"]
        rows: list[dict[str, Any]] = []
        missing = 0
        for day in day_range(start, end):
            key = day.isoformat()
            value = counts.get(key)
            if value is None:
                missing += 1
            row: dict[str, Any] = {
                "ts": day,
                "value": float(value or 0),
                "raw_count": float(value or 0),
                "imputed": value is None,
            }
            if key in breadth:
                share = breadth[key]["top_share"]
                row["contributors"] = breadth[key]["contributors"]
                row["top_share"] = None if math.isnan(share) else share
            rows.append(row)
        if missing:
            caveats.append(
                f"{missing} day(s) had no bucket in the X counts response; set to 0 and "
                "marked imputed."
            )
        if breadth:
            caveats.append(
                f"Breadth on X is sampled from the newest {self.settings['breadth_posts_per_day']} "
                f"posts per day for {len(breadth)} recent day(s) (search/recent covers 7 days)."
            )
        meta = self.base_meta(
            datetime.fromisoformat(payload["fetched_at"]),
            query,
            counts_endpoint=self.settings["counts_endpoint"],
            billed_requests=payload["requests"],
        )
        meta["caveats"] = caveats
        return Series(
            source=self.source,
            query=query,
            freq="D",
            points=points_frame(rows),
            scale="count",
            meta=meta,
        )
