"""Reddit adapter: daily post counts from the official Data API search, merged
with history accumulated by ``scripts/collect_reddit.py``.

**Access.** OAuth2 application-only ("client_credentials") token from
``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` (``get_secret``); without them the
source is disabled ("API credentials not configured"). The token is kept in
memory only. The Data API is free for non-commercial use and needs app approval.

**Counts.** ``GET https://oauth.reddit.com/search?q=..&sort=new&t=all&type=link
&limit=100`` paginated with ``after`` until posts are older than the requested
range, the listing ends, or ``result_cap`` (1,000) posts were read; Reddit search
stops at about 1,000 results. Posts are bucketed by UTC day (``created_utc``).
Search returns posts only, not comments.

**Breadth.** Per day, ``contributors`` = distinct known authors and ``top_share``
= max(largest single-subreddit share, largest single-author share). Authors and
subreddits are held in memory only while a page is aggregated; nothing but
per-day aggregates is cached or written.

**Truncation.** If the cap is hit, the oldest returned post's day is only partly
covered, so the earliest fully covered day is the next one: ``meta.truncated_before``
is set, earlier live days are dropped and a caveat explains it. Collected history
(``data/collected/reddit/<slug>.csv``, complete days only) fills days before the
live coverage starts.

**Rate limits.** 100 queries/minute per OAuth client id (averaged over 10 min),
reported in ``X-Ratelimit-Used/Remaining/Reset``; requests are paced at
``min_interval_s`` and pause until the reset when ``Remaining`` runs out.

Docs (checked 3 Oct 2026):

* Search endpoint: https://www.reddit.com/dev/api#GET_search
* OAuth2 / app-only grant: https://github.com/reddit-archive/reddit/wiki/OAuth2
* API rules, rate limits and User-Agent format:
  https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki
* Responsible Builder Policy (approval, non-commercial use):
  https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy
"""

from __future__ import annotations

import csv
import io
import math
import os
from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from signalcheck.adapters.base import (
    AdapterDisabled,
    AdapterError,
    LiveAdapter,
    Pacer,
    breadth_aggregates,
    day_range,
    day_start_utc,
    epoch_day,
    history_range,
    points_frame,
    require_query,
    slugify,
)
from signalcheck.config import Config, get_secret, resolve_path
from signalcheck.http import HttpError, user_agent
from signalcheck.models import Series

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
SEARCH_URL = "https://oauth.reddit.com/search"
DELETED_AUTHORS = frozenset({"[deleted]", "[removed]", ""})
COLLECTED_COLUMNS = ("date", "count", "contributors", "top_share")
POSTS_ONLY_CAVEAT = (
    "Reddit counts are posts whose title or text match the query; comments are not searched."
)


def collected_path(query: str, cfg: Config) -> Path:
    """``<collected_dir>/<slug>.csv`` for a topic."""
    return resolve_path(cfg["adapters"]["reddit"]["collected_dir"]) / f"{slugify(query)}.csv"


def read_collected(path: Path) -> dict[str, dict[str, float]]:
    """Collected day aggregates ``{iso_date: {count, contributors, top_share}}``.

    Raises :class:`AdapterError` if the file exists but is malformed; returns ``{}``
    when it does not exist.
    """
    if not path.exists():
        return {}
    try:
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
        out: dict[str, dict[str, float]] = {}
        for row in rows:
            day = date.fromisoformat(row["date"]).isoformat()
            share = (row.get("top_share") or "").strip()
            out[day] = {
                "count": float(row["count"]),
                "contributors": float(row["contributors"]),
                "top_share": float(share) if share else math.nan,
            }
        return out
    except (KeyError, ValueError, TypeError, OSError) as exc:
        raise AdapterError(f"collected history {path.name} is malformed ({exc})") from exc


def write_collected(path: Path, days: Mapping[str, Mapping[str, float]]) -> None:
    """Write day aggregates as CSV (sorted, atomic). Aggregates only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLLECTED_COLUMNS)
    for day in sorted(days):
        row = days[day]
        share = row["top_share"]
        writer.writerow(
            [
                day,
                int(row["count"]),
                int(row["contributors"]),
                "" if share is None or math.isnan(share) else f"{share:.4f}",
            ]
        )
    tmp = path.with_suffix(".csv.tmp")
    tmp.write_text(buffer.getvalue(), encoding="utf-8")
    os.replace(tmp, path)


class RedditAdapter(LiveAdapter):
    """Daily ``count`` series of Reddit posts with breadth; merges collected history.

    Params: ``days``/``start``/``end`` (live search range).
    """

    source = "reddit"

    def __init__(self, cfg: Config | None = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._token: str | None = None
        self._pacer: Pacer | None = None

    def credentials(self) -> tuple[str, str]:
        """``(client_id, client_secret)`` or :class:`AdapterDisabled`."""
        client_id = get_secret("REDDIT_CLIENT_ID")
        secret = get_secret("REDDIT_CLIENT_SECRET")
        if not client_id or not secret:
            raise AdapterDisabled("API credentials not configured")
        return client_id, secret

    def headers(self) -> dict[str, str]:
        """Reddit asks for a unique UA naming the app and, ideally, the account."""
        username = get_secret("REDDIT_USERNAME")
        ua = user_agent() + (f" (by /u/{username})" if username else "")
        return {"User-Agent": ua}

    def fetch(self, query: str, params: Mapping[str, Any]) -> Series:
        """Live counts (cached) merged with collected history, with truncation caveats."""
        self.ensure_enabled()
        query = require_query(query)
        self.credentials()
        start, end = history_range(params, int(self.settings["history_days"]), self.now().date())
        payload = self.live_days(query, start, end)
        return self._build_series(query, payload)

    def live_days(self, query: str, start: date, end: date) -> dict[str, Any]:
        """Cached aggregate payload for the live search over ``[start, end]``."""
        s = self.settings
        return self.cache.get_or_fetch(
            self.source,
            query,
            {"start": start, "end": end, "cap": s["result_cap"], "limit": s["page_limit"]},
            lambda: self._fetch_live(query, start, end),
        )

    def _token_value(self) -> str:
        if self._token is None:
            client_id, secret = self.credentials()
            try:
                response = self.request(
                    "POST",
                    TOKEN_URL,
                    data={"grant_type": "client_credentials"},
                    auth=(client_id, secret),
                    headers=self.headers(),
                )
            except HttpError as exc:
                if exc.status in (400, 401, 403):
                    raise AdapterError(
                        f"Reddit rejected the API credentials (HTTP {exc.status})"
                    ) from exc
                raise
            token = response.json().get("access_token")
            if not token:
                raise AdapterError("Reddit did not return an access token")
            self._token = str(token)
        return self._token

    def _wait_for_rate_limit(self, headers: Mapping[str, str]) -> None:
        """Pause until the window resets when Reddit says no requests remain."""
        try:
            remaining = float(headers.get("X-Ratelimit-Remaining", "inf"))
            reset = float(headers.get("X-Ratelimit-Reset", "0"))
        except ValueError:
            return
        if remaining < 1 and reset > 0:
            if reset > float(self.cfg["http"]["backoff_max_s"]):
                raise AdapterError(f"Reddit rate limit reached; resets in {reset:.0f}s")
            self.sleep(reset)

    def _search_page(
        self, query: str, after: str | None
    ) -> tuple[list[dict[str, Any]], str | None]:
        """One page of ``(created_utc, author, subreddit)`` dicts and the next cursor."""
        if self._pacer is None:
            self._pacer = Pacer(float(self.settings["min_interval_s"]), self.sleep)
        self._pacer.wait()
        params: dict[str, Any] = {
            "q": query,
            "sort": "new",
            "t": "all",
            "type": "link",
            "limit": int(self.settings["page_limit"]),
            "raw_json": 1,
        }
        if after:
            params["after"] = after
        headers = {**self.headers(), "Authorization": f"bearer {self._token_value()}"}
        response = self.request("GET", SEARCH_URL, params=params, headers=headers)
        self._wait_for_rate_limit(response.headers)
        listing = (response.json() or {}).get("data") or {}
        posts = []
        for child in listing.get("children") or []:
            data = child.get("data") or {}
            if "created_utc" not in data:
                continue
            posts.append(
                {
                    "created_utc": float(data["created_utc"]),
                    "author": data.get("author"),
                    "subreddit": data.get("subreddit"),
                }
            )
        return posts, listing.get("after")

    def _fetch_live(self, query: str, start: date, end: date) -> dict[str, Any]:
        """Paginate search (newest first) and aggregate per day; no names leave this call."""
        if len(query) > 512:
            raise AdapterError("Reddit queries are limited to 512 characters")
        fetched_at = self.now()
        lower = day_start_utc(start).timestamp()
        upper = day_start_utc(end + timedelta(days=1)).timestamp()
        cap = int(self.settings["result_cap"])
        by_day: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
        n_results = 0
        oldest: float | None = None
        after: str | None = None
        reached_start = False
        while n_results < cap:
            posts, after = self._search_page(query, after)
            for post in posts:
                n_results += 1
                ts = post["created_utc"]
                oldest = ts if oldest is None else min(oldest, ts)
                if ts < lower:
                    reached_start = True
                    continue
                if ts >= upper:
                    continue
                author = post["author"]
                author = None if author is None or author in DELETED_AUTHORS else str(author)
                by_day[epoch_day(ts).isoformat()].append((author, post["subreddit"]))
            if reached_start or not after or not posts:
                break
        capped = not reached_start and bool(after) and n_results >= cap
        if capped and oldest is not None:
            coverage_start = max(start, epoch_day(oldest) + timedelta(days=1))
        else:
            coverage_start = start
        days: dict[str, dict[str, float]] = {}
        for day in day_range(coverage_start, end):
            items = by_day.get(day.isoformat(), [])
            contributors, author_share = breadth_aggregates([a for a, _ in items])
            _, sub_share = breadth_aggregates([s for _, s in items])
            top_share = max(author_share, sub_share) if items else math.nan
            days[day.isoformat()] = {
                "count": float(len(items)),
                "contributors": float(contributors),
                "top_share": top_share,
            }
        by_day.clear()
        return {
            "fetched_at": fetched_at.isoformat(),
            "start": start.isoformat(),
            "coverage_start": coverage_start.isoformat() if coverage_start <= end else None,
            "capped": capped,
            "n_results": n_results,
            "days": days,
        }

    def _build_series(self, query: str, payload: Mapping[str, Any]) -> Series:
        caveats = [POSTS_ONLY_CAVEAT]
        try:
            collected = read_collected(collected_path(query, self.cfg))
        except AdapterError as exc:
            collected = {}
            caveats.append(f"Collected history ignored: {exc.reason}.")
        live: dict[str, dict[str, float]] = payload["days"]
        coverage = payload["coverage_start"]
        merged = dict(live)
        n_collected = 0
        for day, row in collected.items():
            if coverage is None or day < coverage:
                merged[day] = row
                n_collected += 1
        if not merged:
            raise AdapterError(
                "Reddit search hit its 1,000-result limit within one day, so no complete day "
                "could be counted; try a narrower query"
            )
        if n_collected:
            caveats.append(
                f"{n_collected} earlier day(s) come from the daily collector "
                "(data/collected/reddit)."
            )
        first_day = min(merged)
        truncated_before = None
        if payload["capped"]:
            truncated_before = first_day
            live_from = coverage or "no complete day"
            caveats.append(
                f"Reddit search stopped at its {self.settings['result_cap']:,}-result limit, so "
                f"live counts start on {live_from}; earlier live days were dropped and the "
                f"series starts on {first_day}."
            )
        rows = [
            {
                "ts": date.fromisoformat(day),
                "value": row["count"],
                "contributors": row["contributors"],
                "top_share": None if math.isnan(row["top_share"]) else row["top_share"],
                "raw_count": row["count"],
            }
            for day, row in sorted(merged.items())
        ]
        meta = self.base_meta(
            datetime.fromisoformat(payload["fetched_at"]),
            query,
            live_coverage_start=coverage,
            collected_days=n_collected,
            result_count=payload["n_results"],
        )
        meta["caveats"] = caveats
        meta["truncated_before"] = truncated_before
        return Series(
            source=self.source,
            query=query,
            freq="D",
            points=points_frame(rows),
            scale="count",
            meta=meta,
        )
