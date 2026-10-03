"""Tests for the Wikipedia pageviews adapter (HTTP mocked with ``responses``)."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import responses

from signalcheck.adapters.base import AdapterError, fetch_safely
from signalcheck.adapters.wikipedia import AGENT_CAVEAT, PAGEVIEWS_URL, WikipediaAdapter
from signalcheck.engine import analyse

API = "https://en.wikipedia.org/w/api.php"
VIEWS = re.compile(re.escape(PAGEVIEWS_URL) + r"/.*")

PAGES: dict[str, dict[str, Any]] = {
    "Perplexity AI": {"title": "Perplexity AI"},
    "Mercury": {"title": "Mercury", "pageprops": {"disambiguation": ""}},
    "Freddie Mercury": {"title": "Freddie Mercury"},
}
SEARCH: dict[str, list[str]] = {
    "perplexity ai": ["Perplexity AI", "Comet (browser)"],
    "mercury": ["Mercury", "Freddie Mercury", "Mercury (planet)"],
    "zzzz": [],
}


def _api_callback(request: Any) -> tuple[int, dict[str, str], str]:
    params = {k: v[0] for k, v in parse_qs(urlparse(request.url).query).items()}
    if "titles" in params:
        title = params["titles"]
        page = PAGES.get(title, {"title": title, "missing": True})
        body = {"query": {"pages": [{"ns": 0, **page}]}}
    else:
        titles = SEARCH.get(params["gsrsearch"].casefold(), [])
        pages = [
            {"ns": 0, "index": i + 1, **PAGES.get(t, {"title": t})} for i, t in enumerate(titles)
        ]
        body = {"query": {"pages": pages}} if pages else {"batchcomplete": True}
    return 200, {}, json.dumps(body)


def _views(days: list[tuple[str, int]], article: str = "Perplexity_AI") -> dict[str, Any]:
    return {
        "items": [
            {
                "article": article,
                "timestamp": d.replace("-", "") + "00",
                "views": v,
                "agent": "user",
            }
            for d, v in days
        ]
    }


def _mock_api() -> None:
    responses.add_callback(responses.GET, API, callback=_api_callback)


@responses.activate
def test_exact_title_series_and_caveats(adapter_deps: dict[str, Any]) -> None:
    _mock_api()
    responses.get(
        VIEWS, json=_views([("2026-09-29", 100), ("2026-10-01", 120), ("2026-10-02", 90)])
    )
    series = WikipediaAdapter(**adapter_deps).fetch("Perplexity AI", {"days": 10})

    assert series.source == "wikipedia" and series.scale == "pageviews" and series.freq == "D"
    assert series.meta["resolved_query"] == "Perplexity AI"
    assert series.meta["candidates"] == ["Perplexity AI", "Comet (browser)"]
    assert series.meta["fetched_at"].startswith("2026-10-03T12:00")
    assert series.meta["partial_rule"] == "fetched_at"
    assert series.points["value"].tolist() == [100, 0, 120, 90]
    assert series.points["imputed"].tolist() == [False, True, False, False]
    caveats = " ".join(series.caveats)
    assert AGENT_CAVEAT in series.caveats
    assert "1 day(s) missing" in caveats and "imputed" in caveats

    views_call = next(c for c in responses.calls if "pageviews" in str(c.request.url))
    url = str(views_call.request.url)
    assert "/all-access/user/Perplexity_AI/daily/2026092400/2026100300" in url
    assert views_call.request.headers["User-Agent"].startswith("SignalCheck/")


@responses.activate
def test_search_fallback_skips_disambiguation(adapter_deps: dict[str, Any]) -> None:
    _mock_api()
    responses.get(VIEWS, json=_views([("2026-10-01", 5), ("2026-10-02", 6)], "Freddie_Mercury"))
    series = WikipediaAdapter(**adapter_deps).fetch("mercury", {"days": 5})
    assert series.meta["resolved_query"] == "Freddie Mercury"
    assert series.meta["candidates"] == ["Freddie Mercury", "Mercury (planet)"]
    assert series.meta["article_overridden"] is False


@responses.activate
def test_override_is_used_and_validated(adapter_deps: dict[str, Any]) -> None:
    _mock_api()
    responses.get(VIEWS, json=_views([("2026-10-01", 5), ("2026-10-02", 6)], "Mercury"))
    adapter = WikipediaAdapter(**adapter_deps)
    series = adapter.fetch("mercury", {"article": "Mercury", "days": 5})
    assert series.meta["resolved_query"] == "Mercury"
    assert series.meta["article_overridden"] is True
    assert "/Mercury/daily/" in str(responses.calls[-1].request.url)

    result = fetch_safely(adapter, "mercury", {"article": "Not A Page"})
    assert not result.ok
    assert (
        result.message == "Couldn't fetch Wikipedia: Wikipedia article 'Not A Page' does not exist"
    )


@responses.activate
def test_no_article_found(adapter_deps: dict[str, Any]) -> None:
    _mock_api()
    result = fetch_safely(WikipediaAdapter(**adapter_deps), "zzzz", {})
    assert result.message == "Couldn't fetch Wikipedia: no Wikipedia article found for 'zzzz'"


@responses.activate
def test_404_and_empty_views_are_graceful(adapter_deps: dict[str, Any]) -> None:
    _mock_api()
    responses.get(VIEWS, status=404, json={"title": "Not found."})
    result = fetch_safely(WikipediaAdapter(**adapter_deps), "Perplexity AI", {"days": 5})
    assert result.error is not None and result.error.startswith("no pageview data")

    responses.replace(responses.GET, VIEWS, json={"items": []})
    result = fetch_safely(WikipediaAdapter(**adapter_deps), "Perplexity AI", {"days": 6})
    assert result.error is not None and result.error.startswith("no pageview data")


@responses.activate
def test_5xx_retry_exhaustion_is_graceful(adapter_deps: dict[str, Any]) -> None:
    _mock_api()
    responses.get(VIEWS, status=503)
    result = fetch_safely(WikipediaAdapter(**adapter_deps), "Perplexity AI", {"days": 5})
    assert result.message == "Couldn't fetch Wikipedia: HTTP 503 after 3 attempt(s)"
    assert sum("pageviews" in str(c.request.url) for c in responses.calls) == 3


@responses.activate
def test_429_retry_exhaustion_is_graceful(adapter_deps: dict[str, Any]) -> None:
    responses.get(API, status=429)
    result = fetch_safely(WikipediaAdapter(**adapter_deps), "Perplexity AI", {})
    assert result.error == "HTTP 429 after 3 attempt(s)"


@responses.activate
def test_results_are_cached(adapter_deps: dict[str, Any]) -> None:
    _mock_api()
    responses.get(VIEWS, json=_views([("2026-10-01", 5), ("2026-10-02", 6)]))
    adapter = WikipediaAdapter(**adapter_deps)
    first = adapter.fetch("Perplexity AI", {"days": 5})
    n_calls = len(responses.calls)
    second = adapter.fetch("Perplexity AI", {"days": 5})
    assert len(responses.calls) == n_calls
    assert second.points.equals(first.points)


def test_disabled_in_config(adapter_deps: dict[str, Any]) -> None:
    adapter_deps["cfg"]["sources"]["wikipedia"] = False
    result = fetch_safely(WikipediaAdapter(**adapter_deps), "Perplexity AI", {})
    assert result.disabled and result.message == "Wikipedia disabled (switched off in config.yaml)"


def test_blank_query_rejected(adapter_deps: dict[str, Any]) -> None:
    with pytest.raises(AdapterError, match="enter a topic"):
        WikipediaAdapter(**adapter_deps).fetch("   ", {})


@responses.activate
def test_series_runs_through_engine(adapter_deps: dict[str, Any]) -> None:
    _mock_api()
    days = [(f"2026-{m:02d}-{d:02d}", 1000 + (d % 7) * 10) for m in (8, 9) for d in range(1, 31)]
    responses.get(VIEWS, json=_views(days))
    series = WikipediaAdapter(**adapter_deps).fetch("Perplexity AI", {"days": 70})
    verdict = analyse(series, adapter_deps["cfg"])
    assert verdict.label in {"NO_CHANGE", "INCONCLUSIVE", "TREND", "FLUKE", "SEASONAL"}
    assert AGENT_CAVEAT in verdict.caveats
