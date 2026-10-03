"""Hacker News adapter: daily mention counts from the Algolia HN Search API.

**Counts.** One ``search_by_date`` request per UTC day with
``numericFilters=created_at_i>=<day start>,created_at_i<<day end>`` and
``hitsPerPage=0``; the bucket value is ``nbHits`` (stories + comments matching the
query words). Algolia may report ``nbHits`` as an estimate
(``exhaustiveNbHits=false``) for broad queries; those days are listed in a caveat.

**Breadth.** Only for the recent window (sized like the engine's recent window),
the day's hits are fetched (``hitsPerPage<=1000``, attributes limited to
``author`` and ``created_at_i``, highlighting/snippets off, so no text is
returned) to compute ``contributors`` (distinct authors) and ``top_share`` (the
largest single-author share). Author names exist only in memory while a day is
aggregated; baseline days leave breadth null. Algolia pagination stops at 1,000
hits per query, so breadth on busier days uses the newest 1,000 items (caveat).

Docs (verified 3 Oct 2026):

* API: https://hn.algolia.com/api
* Parameters (numericFilters, attributesToRetrieve, typoTolerance):
  https://www.algolia.com/doc/api-reference/search-api-parameters/
* ``exhaustiveNbHits``:
  https://www.algolia.com/doc/api-reference/api-methods/search/#response-exhaustivenbhits
* Rate limit: 10,000 requests per hour per IP (stated on https://hn.algolia.com/api).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timedelta
from typing import Any

from signalcheck.adapters.base import (
    LiveAdapter,
    Pacer,
    breadth_aggregates,
    day_range,
    day_start_utc,
    history_range,
    points_frame,
    require_query,
)
from signalcheck.engine.preprocess import recent_window_length
from signalcheck.models import Series

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
MATCH_CAVEAT = (
    "Hacker News counts are stories and comments containing the query words "
    "(full-text match), which can include unrelated uses of the words."
)


class HackerNewsAdapter(LiveAdapter):
    """Daily ``count`` series of HN stories + comments, with recent-window breadth.

    Params: ``days``/``start``/``end`` (history range).
    """

    source = "hackernews"

    def fetch(self, query: str, params: Mapping[str, Any]) -> Series:
        """Count per day; breadth on the recent window; caveats for estimates/samples."""
        self.ensure_enabled()
        query = require_query(query)
        now = self.now()
        start, end = history_range(params, int(self.settings["history_days"]), now.date())
        s = self.settings
        payload = self.cache.get_or_fetch(
            self.source,
            query,
            {
                "start": start,
                "end": end,
                "tags": s["tags"],
                "typo_tolerance": s["typo_tolerance"],
                "max_hits": s["max_hits_per_day"],
            },
            lambda: self._fetch_days(query, start, end),
        )
        return self._build_series(query, payload)

    def breadth_days(self, days: list[date], now: datetime) -> set[date]:
        """Days that get breadth: the engine-sized recent window plus any partial day."""
        complete = [d for d in days if day_start_utc(d) + timedelta(days=1) <= now]
        partial = [d for d in days if d not in set(complete)]
        if not complete:
            return set(partial)
        n_recent = recent_window_length(len(complete), "D", self.cfg)
        return set(complete[-n_recent:]) | set(partial)

    def _search(self, query: str, day: date, now: datetime, hits_per_page: int) -> Any:
        start = day_start_utc(day)
        stop = min(start + timedelta(days=1), now)
        return self.get_json(
            SEARCH_URL,
            params={
                "query": query,
                "tags": self.settings["tags"],
                "numericFilters": (
                    f"created_at_i>={int(start.timestamp())},created_at_i<{int(stop.timestamp())}"
                ),
                "hitsPerPage": hits_per_page,
                "attributesToRetrieve": "author,created_at_i",
                "attributesToHighlight": "[]",
                "attributesToSnippet": "[]",
                "typoTolerance": str(bool(self.settings["typo_tolerance"])).lower(),
            },
        )

    def _fetch_days(self, query: str, start: date, end: date) -> dict[str, Any]:
        """Aggregate payload ``{fetched_at, days: [...]}``; never contains author names."""
        now = self.now()
        days = day_range(start, end)
        breadth = self.breadth_days(days, now)
        pacer = Pacer(float(self.settings["min_interval_s"]), self.sleep)
        max_hits = int(self.settings["max_hits_per_day"])
        rows: list[dict[str, Any]] = []
        for day in days:
            pacer.wait()
            data = self._search(query, day, now, max_hits if day in breadth else 0)
            count = int(data.get("nbHits") or 0)
            approx = data.get("exhaustiveNbHits") is False
            row: dict[str, Any] = {"d": day.isoformat(), "count": count, "approx": approx}
            if day in breadth:
                authors = [hit.get("author") for hit in data.get("hits") or []]
                if approx and len(authors) < max_hits:
                    # Every match fits on the page, so the page length is the exact count.
                    row.update(count=len(authors), approx=False)
                contributors, top_share = breadth_aggregates(authors)
                row.update(
                    contributors=contributors,
                    top_share=top_share,
                    raw_count=len(authors),
                    sampled=len(authors) < row["count"],
                )
                del authors
            rows.append(row)
        return {"fetched_at": now.isoformat(), "days": rows}

    def _build_series(self, query: str, payload: Mapping[str, Any]) -> Series:
        rows: list[dict[str, Any]] = []
        approx_days: list[str] = []
        sampled_days: list[str] = []
        for row in payload["days"]:
            point: dict[str, Any] = {"ts": date.fromisoformat(row["d"]), "value": row["count"]}
            if "contributors" in row:
                point.update(
                    contributors=row["contributors"],
                    top_share=row["top_share"],
                    raw_count=row["raw_count"],
                )
                if row["sampled"]:
                    sampled_days.append(row["d"])
            else:
                point.update(contributors=None, top_share=None, raw_count=None)
            if row["approx"]:
                approx_days.append(row["d"])
            rows.append(point)
        meta = self.base_meta(
            datetime.fromisoformat(payload["fetched_at"]),
            query,
            tags=self.settings["tags"],
            breadth_days=sum(1 for r in payload["days"] if "contributors" in r),
        )
        meta["caveats"] = [MATCH_CAVEAT]
        if approx_days:
            meta["caveats"].append(
                f"Algolia reported estimated (non-exhaustive) counts for {len(approx_days)} "
                f"day(s), e.g. {approx_days[0]}; treat those values as approximate."
            )
        if sampled_days:
            max_hits = self.settings["max_hits_per_day"]
            meta["caveats"].append(
                f"Breadth (contributors, top share) on {len(sampled_days)} day(s) uses only the "
                f"newest {max_hits} items (Algolia's pagination limit)."
            )
        return Series(
            source=self.source,
            query=query,
            freq="D",
            points=points_frame(rows),
            scale="count",
            meta=meta,
        )
