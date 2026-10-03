"""Cross-source calendar alignment and summary (signalcheck.engine.cross_source)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eval.generators import linear_trend
from signalcheck.config import Config
from signalcheck.engine import analyse, compare_sources
from signalcheck.engine.cross_source import SourceWindow, date_range, display_name, overlap_fraction
from signalcheck.models import Verdict


def verdict(
    label: str,
    start: str | None,
    end_exclusive: str | None,
    direction: str | None = None,
    confidence: str = "medium",
) -> Verdict:
    window = (
        {"recent_start": start, "recent_end_exclusive": end_exclusive}
        if start and end_exclusive
        else {}
    )
    rule = {"TREND": "R4", "FLUKE": "R3", "SEASONAL": "R2", "NO_CHANGE": "R5"}.get(label, "R6")
    return Verdict(label, direction, confidence, rule, [], [], [], window)


SEP = ("2026-09-01", "2026-09-29")  # 28 days, 1-28 Sep inclusive


def test_example_sentence_from_the_brief(cfg: Config) -> None:
    out = compare_sources(
        {
            "google_trends": verdict("TREND", *SEP, direction="up"),
            "wikipedia": verdict("TREND", *SEP, direction="up"),
            "hackernews": verdict("TREND", *SEP, direction="up"),
            "reddit": verdict("FLUKE", *SEP),
        },
        cfg,
    )
    assert (
        out.sentence
        == "3 of 4 sources show an upward trend over 1\u201328 Sep 2026; Reddit looks like a fluke."
    )
    assert (out.window_start, out.window_end, out.window_days) == ("2026-09-01", "2026-09-28", 28)
    assert out.n_compared == 4
    assert out.counts["TREND_up"] == 3 and out.counts["FLUKE"] == 1
    assert out.counts["TREND_down"] == 0 and out.counts["NO_CHANGE"] == 0


def test_window_uses_shortest_recent_window_and_latest_common_date(cfg: Config) -> None:
    out = compare_sources(
        {
            # daily, 28 days ending 3 Oct
            "wikipedia": verdict("TREND", "2026-09-05", "2026-10-03", direction="up"),
            # weekly, 4 weeks ending 28 Sep (week of 21 Sep)
            "google_trends": verdict("TREND", "2026-08-31", "2026-09-28", direction="up"),
            # daily, 7 days ending 2 Oct
            "reddit": verdict("NO_CHANGE", "2026-09-25", "2026-10-02"),
        },
        cfg,
    )
    # Shortest window = 7 days. Anchoring at 2 Oct keeps 2 sources; 28 Sep keeps
    # Wikipedia + Google Trends (Reddit overlaps 3/7 < 50%); 3 Oct keeps 2. The latest
    # anchor with the most comparable sources is 2 Oct, and both cover it.
    assert out.window_days == 7
    assert (out.window_start, out.window_end) == ("2026-09-25", "2026-10-01")
    assert out.comparable == ["wikipedia", "reddit"]
    assert out.not_comparable == ["google_trends"]
    assert out.overlaps["google_trends"] == round(3 / 7, 4)
    assert out.sentence == (
        "1 of 2 sources shows an upward trend over 25 Sep\u20131 Oct 2026; Reddit shows no change; "
        "Google Trends is not comparable (different period)."
    )


def test_stale_source_is_not_comparable(cfg: Config) -> None:
    out = compare_sources(
        {
            "wikipedia": verdict("TREND", *SEP, direction="down"),
            "reddit": verdict("TREND", *SEP, direction="down"),
            "csv": verdict("FLUKE", "2025-09-01", "2025-09-29"),
        },
        cfg,
    )
    assert out.not_comparable == ["csv"]
    assert out.overlaps["csv"] == 0.0
    assert out.sentence == (
        "2 of 2 sources show a downward trend over 1\u201328 Sep 2026; "
        "Uploaded CSV is not comparable (different period)."
    )


def test_exactly_half_overlap_is_comparable(cfg: Config) -> None:
    out = compare_sources(
        {
            "wikipedia": verdict("NO_CHANGE", "2026-09-01", "2026-09-15"),
            "reddit": verdict("NO_CHANGE", "2026-09-08", "2026-09-22"),
        },
        cfg,
    )
    assert out.window_days == 14
    assert sorted(out.comparable) == ["reddit", "wikipedia"]
    assert out.overlaps == {"wikipedia": 1.0, "reddit": 0.5}
    assert (out.window_start, out.window_end) == ("2026-09-01", "2026-09-14")


def test_min_overlap_from_config(cfg: Config) -> None:
    cfg["cross_source"]["min_overlap"] = 0.9
    out = compare_sources(
        {
            "wikipedia": verdict("NO_CHANGE", "2026-09-01", "2026-09-15"),
            "reddit": verdict("NO_CHANGE", "2026-09-08", "2026-09-22"),
        },
        cfg,
    )
    assert len(out.comparable) == 1 and len(out.not_comparable) == 1


def test_failed_and_short_sources_are_listed_separately(cfg: Config) -> None:
    out = compare_sources(
        {
            "wikipedia": verdict("SEASONAL", *SEP),
            "reddit": None,
            "hackernews": verdict("INCONCLUSIVE", None, None),
        },
        cfg,
    )
    assert out.unavailable == ["reddit", "hackernews"]
    assert out.sentence == (
        "Wikipedia shows the usual seasonal pattern over 1\u201328 Sep 2026; "
        "Reddit and Hacker News have no usable result."
    )


def test_nothing_comparable(cfg: Config) -> None:
    out = compare_sources({"reddit": None}, cfg)
    assert out.n_compared == 0 and out.window_start is None
    assert (
        out.sentence
        == "No sources could be compared over a common period; Reddit has no usable result."
    )


def test_ties_prefer_trend_then_fixed_order(cfg: Config) -> None:
    out = compare_sources(
        {
            "reddit": verdict("INCONCLUSIVE", *SEP),
            "wikipedia": verdict("TREND", *SEP, direction="down"),
        },
        cfg,
    )
    assert out.sentence.startswith("1 of 2 sources shows a downward trend")
    assert out.sentence.endswith("Reddit is inconclusive.")


def test_outcomes_never_include_raw_values(cfg: Config) -> None:
    out = compare_sources({"wikipedia": verdict("TREND", *SEP, direction="up")}, cfg)
    assert out.outcomes == {
        "wikipedia": {"label": "TREND", "direction": "up", "confidence": "medium"}
    }


def test_real_verdicts_align(cfg: Config) -> None:
    a = analyse(linear_trend(np.random.default_rng(1), n=120), cfg)
    b = analyse(linear_trend(np.random.default_rng(2), n=120), cfg)
    out = compare_sources({"wikipedia": a, "google_trends": b}, cfg)
    assert out.n_compared == 2
    assert out.window_end == a.window["recent_end"]


def test_helpers() -> None:
    w = SourceWindow("x", pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-11"))
    assert overlap_fraction(w, pd.Timestamp("2026-09-06"), pd.Timestamp("2026-09-16")) == 0.5
    assert overlap_fraction(w, pd.Timestamp("2026-09-16"), pd.Timestamp("2026-09-16")) == 0.0
    assert (
        date_range(pd.Timestamp("2025-12-29"), pd.Timestamp("2026-01-04"))
        == "29 Dec 2025\u20134 Jan 2026"
    )
    assert date_range(pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-01")) == "1 Sep 2026"
    assert display_name("hackernews") == "Hacker News"
    assert display_name("new_source") == "New Source"
