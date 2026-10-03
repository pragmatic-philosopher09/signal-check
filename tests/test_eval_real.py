"""Real-case collection (eval/fetch_real.py) and the pre-registered labels."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from eval.fetch_real import (
    WEEKLY_CAVEAT,
    fetch_case,
    load_entries,
    real_data_path,
    weekly_sums,
)
from signalcheck.adapters.base import AdapterError
from signalcheck.models import Series

LABELS = ("TREND", "FLUKE", "SEASONAL", "NO_CHANGE", "INCONCLUSIVE")


def daily(values: list[float], start: str = "2024-01-01", **meta: Any) -> Series:
    ts = pd.date_range(start, periods=len(values), freq="D")
    points = pd.DataFrame({"ts": ts, "value": values, "imputed": [False] * len(values)})
    full_meta = {"fetched_at": "2026-01-01T00:00:00+00:00", "caveats": ["c"], **meta}
    return Series("wikipedia", "q", "D", points, "pageviews", full_meta)


def test_real_cases_are_pre_registered() -> None:
    entries = load_entries()
    assert len(entries) >= 10
    labels = Counter(e["label"] for e in entries)
    assert set(labels) == set(LABELS)
    for label in LABELS:
        splits = {e["split"] for e in entries if e["label"] == label}
        if labels[label] >= 2:
            assert splits == {"tune", "test"}, label
    for e in entries:
        assert e["labelled_by"] == "agent (pending human review)"
        assert e["rationale"].strip()
        assert (e["label"] == "TREND") == (e["direction"] in ("up", "down"))
        assert real_data_path(e["id"]).exists(), e["id"]


def test_real_snapshots_are_aggregate_only() -> None:
    allowed = {"ts", "value", "imputed", "contributors", "top_share", "raw_count"}
    for e in load_entries():
        data = json.loads(real_data_path(e["id"]).read_text())
        assert data["meta"]["fetched_at"]
        assert data["freq"] == e["freq"]
        for point in data["points"]:
            assert set(point) <= allowed
        first, last = data["points"][0]["ts"], data["points"][-1]["ts"]
        assert first >= e["start"] and last <= e["end"]


def test_load_entries_validates(tmp_path: Path) -> None:
    base = (
        "cases:\n  - {id: a, source: SRC, query: q, freq: FREQ, start: '2024-01-01', "
        "end: '2024-03-01', as_of: ASOF, label: NO_CHANGE, split: tune}\n"
    )
    path = tmp_path / "real.yaml"

    def write(src: str = "wikipedia", freq: str = "D", as_of: str = "'2024-02-01'") -> Path:
        path.write_text(base.replace("SRC", src).replace("FREQ", freq).replace("ASOF", as_of))
        return path

    assert load_entries(write())[0]["as_of"] == "2024-02-01"
    with pytest.raises(ValueError, match="source"):
        load_entries(write(src="reddit"))
    with pytest.raises(ValueError, match="freq"):
        load_entries(write(freq="M"))
    with pytest.raises(ValueError, match="as_of"):
        load_entries(write(as_of="'2024-04-01'"))


def test_weekly_sums_keeps_complete_monday_weeks() -> None:
    # 2024-01-03 is a Wednesday: the first partial week and the last partial week are dropped.
    values = [1.0] * 14
    series = daily(values, start="2024-01-03")
    series.points.loc[6, "imputed"] = True
    weekly = weekly_sums(series)
    assert weekly.freq == "W"
    assert list(weekly.points["ts"].dt.date.astype(str)) == ["2024-01-08"]
    assert weekly.points["value"].tolist() == [7.0]
    assert weekly.points["imputed"].tolist() == [True]
    assert weekly.meta["caveats"] == ["c", WEEKLY_CAVEAT]
    with pytest.raises(ValueError):
        weekly_sums(weekly)


class FakeAdapter:
    def __init__(self, series: Series) -> None:
        self.series = series
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def fetch(self, query: str, params: Mapping[str, Any]) -> Series:
        self.calls.append((query, dict(params)))
        return self.series


def test_fetch_case_passes_dates_and_article_and_aggregates() -> None:
    fake = FakeAdapter(daily([2.0] * 14, article="Coffee"))
    entry = {
        "source": "wikipedia",
        "query": "coffee",
        "article": "Coffee",
        "freq": "W",
        "start": "2024-01-01",
        "end": "2024-01-14",
    }
    series = fetch_case(entry, {}, {"wikipedia": fake})  # type: ignore[arg-type]
    assert fake.calls == [
        ("coffee", {"start": "2024-01-01", "end": "2024-01-14", "article": "Coffee"})
    ]
    assert series.points["value"].tolist() == [14.0, 14.0]
    wrong = FakeAdapter(daily([2.0] * 14, article="Tea"))
    with pytest.raises(AdapterError, match="Tea"):
        fetch_case(entry, {}, {"wikipedia": wrong})  # type: ignore[arg-type]
