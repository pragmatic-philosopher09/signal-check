"""The refresh watchlist from ``config.yaml`` (topics refreshed daily and shown as samples).

``config.load_config`` has already validated every entry, so this module only
turns the raw mappings into typed :class:`WatchItem` values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from signalcheck.config import WATCHLIST_SOURCES, Config, get_config


@dataclass(frozen=True)
class WatchItem:
    """One watchlist topic and the sources the daily refresh fetches for it."""

    topic: str
    wikipedia: bool
    wiki_article: str | None
    hackernews: bool
    reddit: bool

    @property
    def sources(self) -> list[str]:
        """Enabled sources in :data:`~signalcheck.config.WATCHLIST_SOURCES` order."""
        return [s for s in WATCHLIST_SOURCES if getattr(self, s)]

    def params(self, source: str) -> dict[str, Any]:
        """Adapter params for ``source`` (the Wikipedia article override, if any)."""
        if source == "wikipedia" and self.wiki_article:
            return {"article": self.wiki_article}
        return {}


def watch_item(entry: Mapping[str, Any]) -> WatchItem:
    """A :class:`WatchItem` from one validated ``watchlist`` mapping."""
    wiki = entry.get("wikipedia", False)
    return WatchItem(
        topic=" ".join(str(entry["topic"]).split()),
        wikipedia=bool(wiki),
        wiki_article=wiki.strip() if isinstance(wiki, str) else None,
        hackernews=bool(entry.get("hackernews", False)),
        reddit=bool(entry.get("reddit", False)),
    )


def watchlist(cfg: Config | None = None) -> list[WatchItem]:
    """Every watchlist topic, in config order."""
    cfg = cfg if cfg is not None else get_config()
    return [watch_item(entry) for entry in cfg["watchlist"]]


def watchlist_topics(cfg: Config | None = None) -> list[str]:
    """Watchlist topic names, in config order (the sample chips)."""
    return [item.topic for item in watchlist(cfg)]


def topics_for(source: str, cfg: Config | None = None) -> list[str]:
    """Watchlist topics with ``source`` enabled."""
    return [item.topic for item in watchlist(cfg) if source in item.sources]
