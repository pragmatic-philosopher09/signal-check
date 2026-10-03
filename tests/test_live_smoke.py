"""Opt-in live smoke tests against the free, keyless APIs (Wikipedia, Hacker News).

Skipped by default (``addopts = -m 'not live'``) and in CI. Run with
``pytest -m live``. They use a throwaway cache so results are fresh.
"""

from __future__ import annotations

import pytest

from signalcheck.adapters.base import fetch_safely
from signalcheck.adapters.hackernews import HackerNewsAdapter
from signalcheck.adapters.wikipedia import WikipediaAdapter
from signalcheck.cache import NoCache
from signalcheck.config import Config
from signalcheck.engine import analyse

pytestmark = pytest.mark.live


def test_wikipedia_live(cfg: Config) -> None:
    result = fetch_safely(
        WikipediaAdapter(cfg, cache=NoCache()), "python programming", {"days": 60}
    )
    assert result.ok, result.message
    assert result.series is not None
    assert result.series.meta["resolved_query"]
    assert len(result.series.points) >= 59
    assert analyse(result.series, cfg).label


def test_hackernews_live(cfg: Config) -> None:
    result = fetch_safely(HackerNewsAdapter(cfg, cache=NoCache()), "python", {"days": 10})
    assert result.ok, result.message
    assert result.series is not None
    assert len(result.series.points) == 10
    assert result.series.points["value"].sum() > 0
