"""Tests for shared adapter helpers, FetchResult, NoCache and snapshots."""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from signalcheck.adapters.base import (
    AdapterDisabled,
    AdapterError,
    FetchResult,
    Pacer,
    breadth_aggregates,
    fetch_safely,
    history_range,
    points_frame,
    slugify,
)
from signalcheck.cache import NoCache
from signalcheck.engine import analyse
from signalcheck.http import HttpError
from signalcheck.models import Series
from signalcheck.snapshots import (
    load_snapshot,
    read_snapshot,
    series_from_dict,
    series_to_dict,
    snapshot_path,
    write_snapshot,
)


class _Raising:
    source = "reddit"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def fetch(self, query: str, params: Any) -> Series:
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "message", "disabled"),
    [
        (
            AdapterDisabled("API credentials not configured"),
            "Reddit disabled (API credentials not configured)",
            True,
        ),
        (AdapterError("search failed"), "Couldn't fetch Reddit: search failed", False),
        (
            HttpError("HTTP 503 after 3 attempt(s)", 503),
            "Couldn't fetch Reddit: HTTP 503 after 3 attempt(s)",
            False,
        ),
        (KeyError("secret-payload"), "Couldn't fetch Reddit: unexpected error (KeyError)", False),
    ],
)
def test_fetch_safely_never_raises(exc: Exception, message: str, disabled: bool) -> None:
    result = fetch_safely(_Raising(exc), "topic")
    assert isinstance(result, FetchResult)
    assert not result.ok and result.disabled is disabled
    assert result.message == message
    assert "secret-payload" not in (result.message or "")


def test_breadth_aggregates() -> None:
    assert breadth_aggregates(["a", "a", "b", None]) == (2, 0.5)
    assert breadth_aggregates([None, None]) == (0, 0.0)
    contributors, share = breadth_aggregates([])
    assert contributors == 0 and math.isnan(share)


def test_history_range() -> None:
    today = date(2026, 10, 3)
    assert history_range({}, 30, today) == (date(2026, 9, 4), today)
    assert history_range({"days": 3, "end": "2026-09-10"}, 30, today) == (
        date(2026, 9, 8),
        date(2026, 9, 10),
    )
    assert history_range({"end": "2027-01-01", "days": 1}, 30, today) == (today, today)
    with pytest.raises(AdapterError, match="after end"):
        history_range({"start": "2026-10-05", "end": "2026-10-01"}, 30, today)
    with pytest.raises(AdapterError, match="not a date"):
        history_range({"start": "yesterday"}, 30, today)


def test_slugify() -> None:
    assert slugify("  Perplexity   AI! ") == "perplexity-ai"
    with pytest.raises(AdapterError):
        slugify("!!!")


def test_pacer_sleeps_only_the_remainder() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    pacer = Pacer(1.0, sleep, clock=lambda: now[0])
    pacer.wait()
    now[0] += 0.25
    pacer.wait()
    now[0] += 2.0
    pacer.wait()
    assert sleeps == [0.75]


def test_no_cache_always_fetches() -> None:
    calls: list[int] = []
    cache = NoCache()
    for _ in range(2):
        cache.get_or_fetch("hackernews", "q", {}, lambda: calls.append(1) or len(calls))
    assert len(calls) == 2


def _series() -> Series:
    points = points_frame(
        [
            {"ts": date(2026, 9, d), "value": float(d), "contributors": d, "top_share": 0.1}
            for d in range(1, 31)
        ]
        + [{"ts": date(2026, 10, 1), "value": 1.0, "contributors": None, "top_share": None}]
    )
    meta = {
        "fetched_at": datetime(2026, 10, 1, 9, tzinfo=UTC).isoformat(),
        "caveats": ["c1"],
        "resolved_query": "Topic",
        "truncated_before": "2026-09-01",
        "dropped_partial": "2026-10-01",
        "partial_rule": "fetched_at",
    }
    return Series("hackernews", "My Topic", "D", points, "count", meta)


def test_snapshot_roundtrip(tmp_path: Path) -> None:
    original = _series()
    path = write_snapshot(original, tmp_path)
    assert (
        path
        == snapshot_path("hackernews", "My Topic", tmp_path)
        == tmp_path / "hackernews" / "my-topic.json"
    )
    loaded = read_snapshot(path)
    assert loaded.meta["dropped_partial"] is None
    assert loaded.meta["truncated_before"] == "2026-09-01"
    pd.testing.assert_frame_equal(loaded.points, original.points)
    # Partial-period rule still applies after a round trip (last day ends after fetched_at).
    verdict = analyse(loaded)
    assert verdict.label


def test_load_snapshot_adds_caveat(tmp_path: Path) -> None:
    write_snapshot(_series(), tmp_path)
    loaded = load_snapshot("hackernews", "my topic", tmp_path)
    assert loaded.meta["snapshot"] is True
    assert any("Cached snapshot fetched on 2026-10-01" in c for c in loaded.caveats)


def test_bad_snapshots_raise_adapter_error(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="no snapshot"):
        read_snapshot(tmp_path / "missing.json")
    data = series_to_dict(_series())
    del data["meta"]["fetched_at"]
    with pytest.raises(AdapterError, match="fetched_at"):
        series_from_dict(data)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"points": "nope"}), encoding="utf-8")
    with pytest.raises(AdapterError, match="malformed"):
        read_snapshot(bad)
