"""Wikipedia pageviews adapter with a resolved, user-overridable article.

The topic is resolved to one article: an exact (redirect-following) title match
that is not a disambiguation page, else the top non-disambiguation search hit.
The resolved title and the other search candidates go into ``meta`` so the UI can
show the article and let the user override it (``params["article"]``).

Daily views come from the Wikimedia REST pageviews API with ``agent=user``, which
excludes traffic Wikimedia identifies as spiders/automated; unflagged automated
traffic can still cause spikes, which is recorded as a caveat. The API omits days
with zero views, so internal missing days are set to 0, marked ``imputed`` and
listed in a caveat; days before the first returned day are not invented.

Docs (verified 3 Oct 2026):

* Pageviews per-article:
  https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html
* Search / query API: https://www.mediawiki.org/wiki/API:Search and
  https://www.mediawiki.org/wiki/API:Query
* User-Agent policy (descriptive UA with contact info is required):
  https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any
from urllib.parse import quote

from signalcheck.adapters.base import (
    AdapterError,
    LiveAdapter,
    day_range,
    history_range,
    points_frame,
    require_query,
)
from signalcheck.http import HttpError
from signalcheck.models import Series

PAGEVIEWS_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
AGENT_CAVEAT = (
    "Wikipedia pageviews use agent=user, which excludes known bots; unflagged automated "
    "traffic can still cause spikes."
)
REDIRECT_CAVEAT = "Views of redirects to this article are not included."


def api_url(project: str) -> str:
    """MediaWiki action API endpoint for a project such as ``en.wikipedia``."""
    return f"https://{project}.org/w/api.php"


def article_path(title: str) -> str:
    """Title as a pageviews URL segment: spaces -> underscores, everything else escaped."""
    return quote(title.replace(" ", "_"), safe="")


def _is_disambiguation(page: Mapping[str, Any]) -> bool:
    return "disambiguation" in (page.get("pageprops") or {})


class WikipediaAdapter(LiveAdapter):
    """Daily ``pageviews`` series for the article a topic resolves to.

    Params: ``article`` (override title), ``days``/``start``/``end`` (history range).
    """

    source = "wikipedia"

    def fetch(self, query: str, params: Mapping[str, Any]) -> Series:
        """Resolve the article, fetch daily user pageviews, record caveats."""
        self.ensure_enabled()
        query = require_query(query)
        override = str(params.get("article") or "").strip()
        title, candidates = self.resolve(query, override or None)
        now = self.now()
        start, end = history_range(params, int(self.settings["history_days"]), now.date())
        payload = self.cache.get_or_fetch(
            self.source,
            title,
            {
                "step": "pageviews",
                "project": self.settings["project"],
                "access": self.settings["access"],
                "agent": self.settings["agent"],
                "start": start,
                "end": end,
                "title": title,
            },
            lambda: self._fetch_views(title, start, end),
        )
        return self._build_series(query, title, candidates, bool(override), payload)

    def resolve(self, query: str, override: str | None = None) -> tuple[str, list[str]]:
        """``(title, candidates)``: the override if it exists, else exact match, else search."""
        project = self.settings["project"]
        candidates: list[str] = self.cache.get_or_fetch(
            self.source,
            query,
            {"step": "search", "project": project, "limit": self.settings["search_limit"]},
            lambda: self._search(query),
        )
        if override:
            title = self.cache.get_or_fetch(
                self.source,
                override,
                {"step": "title", "project": project, "title": override},
                lambda: self._lookup_title(override, allow_disambiguation=True),
            )
            if title is None:
                raise AdapterError(f"Wikipedia article {override!r} does not exist")
            return title, candidates
        exact = self.cache.get_or_fetch(
            self.source,
            query,
            {"step": "title", "project": project, "title": query},
            lambda: self._lookup_title(query, allow_disambiguation=False),
        )
        if exact is not None:
            return exact, candidates
        if not candidates:
            raise AdapterError(f"no Wikipedia article found for {query!r}")
        return candidates[0], candidates

    def _lookup_title(self, title: str, *, allow_disambiguation: bool) -> str | None:
        """Canonical title of an existing article (following redirects), else ``None``."""
        data = self.get_json(
            api_url(self.settings["project"]),
            params={
                "action": "query",
                "titles": title,
                "redirects": 1,
                "prop": "pageprops",
                "ppprop": "disambiguation",
                "format": "json",
                "formatversion": 2,
            },
        )
        pages = (data.get("query") or {}).get("pages") or []
        for page in pages:
            if page.get("missing") or page.get("invalid") or page.get("ns", 0) != 0:
                continue
            if _is_disambiguation(page) and not allow_disambiguation:
                continue
            return str(page["title"])
        return None

    def _search(self, query: str) -> list[str]:
        """Article titles from full-text search, best first, disambiguation pages removed."""
        data = self.get_json(
            api_url(self.settings["project"]),
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 0,
                "gsrlimit": int(self.settings["search_limit"]),
                "prop": "pageprops",
                "ppprop": "disambiguation",
                "redirects": 1,
                "format": "json",
                "formatversion": 2,
            },
        )
        pages = (data.get("query") or {}).get("pages") or []
        ranked = sorted(pages, key=lambda p: p.get("index", 0))
        return [str(p["title"]) for p in ranked if not _is_disambiguation(p)]

    def _fetch_views(self, title: str, start: date, end: date) -> dict[str, Any]:
        """Aggregate payload ``{fetched_at, views: {iso_date: int}}`` for the cache."""
        s = self.settings
        url = (
            f"{PAGEVIEWS_URL}/{s['project']}/{s['access']}/{s['agent']}/"
            f"{article_path(title)}/daily/{start:%Y%m%d}00/{end:%Y%m%d}00"
        )
        fetched_at = self.now()
        try:
            data = self.get_json(url)
        except HttpError as exc:
            if exc.status == 404:
                raise AdapterError(
                    f"no pageview data for {title!r} between {start} and {end}"
                ) from exc
            raise
        views: dict[str, int] = {}
        for item in data.get("items") or []:
            stamp = str(item["timestamp"])
            day = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
            views[day.isoformat()] = int(item["views"])
        if not views:
            raise AdapterError(f"no pageview data for {title!r} between {start} and {end}")
        return {"fetched_at": fetched_at.isoformat(), "views": views}

    def _build_series(
        self,
        query: str,
        title: str,
        candidates: list[str],
        overridden: bool,
        payload: Mapping[str, Any],
    ) -> Series:
        views: dict[str, int] = payload["views"]
        days = sorted(date.fromisoformat(d) for d in views)
        rows: list[dict[str, Any]] = []
        n_zero_filled = 0
        for day in day_range(days[0], days[-1]):
            value = views.get(day.isoformat())
            if value is None:
                n_zero_filled += 1
            rows.append({"ts": day, "value": value or 0, "imputed": value is None})
        meta = self.base_meta(
            datetime.fromisoformat(payload["fetched_at"]),
            title,
            article=title,
            article_overridden=overridden,
            candidates=candidates,
            project=self.settings["project"],
            agent=self.settings["agent"],
        )
        meta["caveats"] = [AGENT_CAVEAT, REDIRECT_CAVEAT]
        if n_zero_filled:
            meta["caveats"].append(
                f"{n_zero_filled} day(s) missing from the Wikimedia response were set to 0 views "
                "(the API omits zero-view days) and marked imputed."
            )
        return Series(
            source=self.source,
            query=query,
            freq="D",
            points=points_frame(rows),
            scale="pageviews",
            meta=meta,
        )
