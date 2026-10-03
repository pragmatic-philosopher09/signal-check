"""Tests for the Hacker News (Algolia) adapter (HTTP mocked with ``responses``)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import responses

from signalcheck.adapters.base import fetch_safely
from signalcheck.adapters.hackernews import MATCH_CAVEAT, SEARCH_URL, HackerNewsAdapter


def _params(request: Any) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlparse(request.url).query).items()}


def _day_of(params: dict[str, str]) -> str:
    lower = params["numericFilters"].split(",")[0].split(">=")[1]
    return datetime.fromtimestamp(int(lower), tz=UTC).date().isoformat()


def _callback(
    counts: dict[str, int],
    authors: dict[str, list[str]] | None = None,
    approx: set[str] | None = None,
) -> Any:
    def handler(request: Any) -> tuple[int, dict[str, str], str]:
        params = _params(request)
        day = _day_of(params)
        per_page = int(params["hitsPerPage"])
        names = (authors or {}).get(day, [])[:per_page]
        body = {
            "nbHits": counts.get(day, 0),
            "exhaustiveNbHits": day not in (approx or set()),
            "hits": [
                {"author": a, "created_at_i": 0, "objectID": str(i)} for i, a in enumerate(names)
            ],
        }
        return 200, {}, json.dumps(body)

    return handler


@responses.activate
def test_counts_and_recent_breadth(adapter_deps: dict[str, Any]) -> None:
    counts = {f"2026-09-{d:02d}": 10 + d for d in range(4, 31)} | {
        "2026-10-01": 50,
        "2026-10-02": 4,
        "2026-10-03": 1,
    }
    authors = {d: [f"user{i}" for i in range(c)] for d, c in counts.items()}
    authors["2026-10-02"] = ["ann", "ann", "bob", "cy"]
    responses.add_callback(responses.GET, SEARCH_URL, callback=_callback(counts, authors))
    series = HackerNewsAdapter(**adapter_deps).fetch("OpenAI", {"days": 30})

    assert series.scale == "count" and series.freq == "D" and len(series.points) == 30
    assert series.caveats == [MATCH_CAVEAT]
    assert series.meta["fetched_at"].startswith("2026-10-03T12:00")
    # 29 complete days -> recent window clamp(round(0.2 * 29), 7, 28) = 7, plus today.
    assert series.meta["breadth_days"] == 8
    pts = series.points.set_index("ts")
    assert pts.loc[pd.Timestamp("2026-09-20"), "value"] == 30
    assert pd.isna(pts.loc[pd.Timestamp("2026-09-20"), "contributors"])
    oct2 = pts.loc[pd.Timestamp("2026-10-02")]
    assert (oct2["value"], oct2["contributors"], oct2["top_share"], oct2["raw_count"]) == (
        4,
        3,
        0.5,
        4,
    )

    sent = [_params(c.request) for c in responses.calls]
    assert len(sent) == 30
    baseline = [p for p in sent if p["hitsPerPage"] == "0"]
    assert len(baseline) == 22
    first = sent[0]
    assert first["tags"] == "(story,comment)" and first["typoTolerance"] == "false"
    assert first["attributesToRetrieve"] == "author,created_at_i"
    assert first["numericFilters"] == "created_at_i>=1788480000,created_at_i<1788566400"
    # Today's window stops at fetch time so the bucket is marked partial, not complete.
    assert sent[-1]["numericFilters"].endswith("created_at_i<1791028800")


@responses.activate
def test_approximate_and_sampled_days_are_caveated(adapter_deps: dict[str, Any]) -> None:
    counts = {"2026-09-30": 5000, "2026-10-01": 1500, "2026-10-02": 3}
    authors = {"2026-10-01": [f"u{i % 300}" for i in range(1000)], "2026-10-02": ["a", "b", "c"]}
    approx = {"2026-09-30", "2026-10-01", "2026-10-02"}
    adapter_deps["cfg"]["preprocess"]["recent_min"]["D"] = 2
    responses.add_callback(responses.GET, SEARCH_URL, callback=_callback(counts, authors, approx))
    series = HackerNewsAdapter(**adapter_deps).fetch("the", {"days": 4})

    pts = series.points.set_index("ts")
    # A short page proves the true count, so 2026-10-02 is exact; 09-30 & 10-01 stay estimates.
    assert pts.loc[pd.Timestamp("2026-10-02"), "value"] == 3
    assert pts.loc[pd.Timestamp("2026-10-01"), "raw_count"] == 1000
    text = " ".join(series.caveats)
    assert "estimated (non-exhaustive) counts for 2 day(s)" in text
    assert "on 1 day(s) uses only the newest 1000 items" in text


@responses.activate
def test_empty_results_give_zero_series(adapter_deps: dict[str, Any]) -> None:
    responses.add_callback(responses.GET, SEARCH_URL, callback=_callback({}))
    series = HackerNewsAdapter(**adapter_deps).fetch("nothing-matches", {"days": 5})
    assert series.points["value"].tolist() == [0, 0, 0, 0, 0]
    assert series.points["contributors"].dropna().eq(0).all()


@responses.activate
def test_429_exhaustion_is_graceful(adapter_deps: dict[str, Any]) -> None:
    responses.get(SEARCH_URL, status=429)
    result = fetch_safely(HackerNewsAdapter(**adapter_deps), "openai", {"days": 5})
    assert result.message == "Couldn't fetch Hacker News: HTTP 429 after 3 attempt(s)"
    assert len(responses.calls) == 3


@responses.activate
def test_cache_and_pacing(adapter_deps: dict[str, Any]) -> None:
    sleeps: list[float] = []
    adapter_deps["sleep"] = sleeps.append
    responses.add_callback(responses.GET, SEARCH_URL, callback=_callback({"2026-10-01": 2}))
    adapter = HackerNewsAdapter(**adapter_deps)
    adapter.fetch("openai", {"days": 3})
    adapter.fetch("OpenAI ", {"days": 3})
    assert len(responses.calls) == 3
    assert len(sleeps) == 2 and all(0 < s <= 0.1 for s in sleeps)


def test_disabled_in_config(adapter_deps: dict[str, Any]) -> None:
    adapter_deps["cfg"]["sources"]["hackernews"] = False
    result = fetch_safely(HackerNewsAdapter(**adapter_deps), "openai", {})
    assert result.disabled and not result.ok
