"""Tests for the core data contract in signalcheck.models."""

from __future__ import annotations

import pandas as pd
import pytest

from signalcheck.models import Evidence, HistoryCheck, Series, Verdict


def _points() -> pd.DataFrame:
    return pd.DataFrame({"ts": pd.date_range("2026-01-01", periods=3), "value": [1.0, 2.0, 3.0]})


def test_series_valid() -> None:
    s = Series("csv", "q", "D", _points(), "count", {"caveats": ["x"]})
    assert s.caveats == ["x"]
    s.caveats.append("y")  # returns a copy
    assert s.meta["caveats"] == ["x"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("source", "myspace"), ("freq", "H"), ("scale", "furlongs")],
)
def test_series_rejects_unknown_enums(field: str, value: str) -> None:
    kwargs = {"source": "csv", "query": "q", "freq": "D", "points": _points(), "scale": "count"}
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        Series(**kwargs)  # type: ignore[arg-type]


def test_series_requires_columns_and_types() -> None:
    with pytest.raises(ValueError, match="missing required"):
        Series("csv", "q", "D", pd.DataFrame({"ts": []}), "count")
    bad_ts = pd.DataFrame({"ts": ["2026-01-01"], "value": [1.0]})
    with pytest.raises(ValueError, match="datetime64"):
        Series("csv", "q", "D", bad_ts, "count")
    bad_value = pd.DataFrame({"ts": pd.date_range("2026-01-01", periods=1), "value": ["a"]})
    with pytest.raises(ValueError, match="numeric"):
        Series("csv", "q", "D", bad_value, "count")


def test_evidence_skipped_needs_reason() -> None:
    with pytest.raises(ValueError, match="skip_reason"):
        Evidence("outliers", "skipped", "s", {}, None)
    ev = Evidence("outliers", "skipped", "s", {}, None, skip_reason="too short")
    assert ev.skip_reason == "too short"
    with pytest.raises(ValueError, match="stance"):
        Evidence("outliers", "maybe", "s", {}, None)


def test_verdict_validation() -> None:
    Verdict("NO_CHANGE", None, "low", "R5", [], [], [], {})
    with pytest.raises(ValueError, match="direction"):
        Verdict("TREND", None, "high", "R4", [], [], [], {})
    with pytest.raises(ValueError, match="label"):
        Verdict("MAYBE", None, "low", "R6", [], [], [], {})
    with pytest.raises(ValueError, match="confidence"):
        Verdict("FLUKE", None, "certain", "R3", [], [], [], {})


def test_history_points_needed() -> None:
    assert HistoryCheck(20, 28, False, "short").points_needed == 8
    assert HistoryCheck(40, 28, True, None).points_needed == 0
