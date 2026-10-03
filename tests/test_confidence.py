"""Confidence scoring (signalcheck.engine.confidence)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from signalcheck.config import Config
from signalcheck.engine.confidence import level_for, penalties, score_confidence, stance_delta
from signalcheck.engine.preprocess import preprocess
from signalcheck.models import Evidence, Preprocessed
from tests.conftest import SeriesFactory


def ev(check: str, stance: str, **numbers: Any) -> Evidence:
    reason = "n/a" if stance == "skipped" else None
    return Evidence(check, stance, "s", dict(numbers), None, reason)


@pytest.fixture
def pre(make_series: SeriesFactory, rng: np.random.Generator, cfg: Config) -> Preprocessed:
    """120 clean daily points: no data-quality penalties apply."""
    return preprocess(make_series(50 + rng.normal(0, 1, 120)), cfg)


@pytest.mark.parametrize(
    ("label", "stance", "expected"),
    [
        ("TREND", "supports_trend", 1),
        ("TREND", "supports_fluke", -1),
        ("TREND", "supports_no_change", -1),
        ("TREND", "supports_seasonal", -1),
        ("TREND", "neutral", 0),
        ("FLUKE", "supports_fluke", 1),
        ("FLUKE", "supports_trend", -1),
        ("FLUKE", "supports_seasonal", -1),
        ("FLUKE", "supports_no_change", 0),
        ("SEASONAL", "supports_seasonal", 1),
        ("SEASONAL", "supports_fluke", -1),
        ("SEASONAL", "supports_trend", 0),
        ("NO_CHANGE", "supports_no_change", 1),
        ("NO_CHANGE", "supports_trend", -1),
        ("NO_CHANGE", "supports_fluke", -1),
        ("NO_CHANGE", "supports_seasonal", -1),
        ("INCONCLUSIVE", "supports_trend", 0),
    ],
)
def test_stance_matrix(label: str, stance: str, expected: int) -> None:
    direction = "up" if label == "TREND" else None
    assert stance_delta(label, direction, ev("x", stance)) == expected


def test_trend_in_the_opposite_direction_contradicts() -> None:
    assert stance_delta("TREND", "up", ev("persistence", "supports_trend", direction="down")) == -1
    assert stance_delta("TREND", "up", ev("persistence", "supports_trend", direction="up")) == 1


def test_skipped_checks_do_not_count() -> None:
    assert stance_delta("TREND", "up", ev("breadth", "skipped")) == 0


def test_levels_follow_config(cfg: Config) -> None:
    assert [level_for(s, cfg) for s in (3, 2, 1, 0, -1)] == [
        "high",
        "medium",
        "medium",
        "low",
        "low",
    ]
    cfg["confidence"]["conf_high"] = 5
    assert level_for(3, cfg) == "medium"


def test_score_adds_agreements_and_contradictions(pre: Preprocessed, cfg: Config) -> None:
    evidence = [
        ev("persistence", "supports_trend", direction="up"),
        ev("level_shift", "supports_trend", direction="up"),
        ev("outliers", "neutral"),
        ev("concentration", "supports_fluke"),
        ev("breadth", "skipped"),
    ]
    score = score_confidence("TREND", "up", evidence, pre, cfg)
    assert score.score == 1
    assert score.level == "medium"
    assert ("concentration contradicts", -1) in score.contributions


def test_high_confidence(pre: Preprocessed, cfg: Config) -> None:
    evidence = [ev(c, "supports_no_change") for c in ("persistence", "outliers", "level_shift")]
    score = score_confidence("NO_CHANGE", None, evidence, pre, cfg)
    assert (score.score, score.level) == (3, "high")


def test_inconclusive_is_always_low(pre: Preprocessed, cfg: Config) -> None:
    evidence = [ev(c, "supports_no_change") for c in ("persistence", "outliers", "level_shift")]
    assert score_confidence("INCONCLUSIVE", None, evidence, pre, cfg).level == "low"


def test_no_penalties_on_clean_long_daily_series(pre: Preprocessed, cfg: Config) -> None:
    seasonality = ev("seasonality", "skipped", skip_code="history")
    assert penalties([seasonality], pre, cfg) == []  # daily: annual seasonality not required


def test_short_history_penalty(make_series: SeriesFactory, cfg: Config) -> None:
    pre = preprocess(make_series(np.full(40, 50.0)), cfg)  # 40 < 2 x 28
    assert [r for r, _ in penalties([], pre, cfg)] == [
        "short history (below the minimum-history multiple)"
    ]


def test_low_count_penalty(pre: Preprocessed, cfg: Config) -> None:
    low = ev("low_count", "supports_trend", active=True)
    assert penalties([low], pre, cfg) == [("low counts", -1)]


@pytest.mark.parametrize(
    ("code", "penalised"), [("history", True), ("gaps", True), ("no_rise", False)]
)
def test_seasonality_penalty_for_weekly(
    make_series: SeriesFactory, cfg: Config, code: str, penalised: bool
) -> None:
    pre = preprocess(make_series(np.full(80, 50.0), freq="W"), cfg)
    out = penalties([ev("seasonality", "skipped", skip_code=code)], pre, cfg)
    assert (("annual seasonality could not be ruled out", -1) in out) is penalised


def test_truncation_penalty(make_series: SeriesFactory, cfg: Config) -> None:
    pre = preprocess(make_series(np.full(120, 50.0), truncated_before="2026-01-01"), cfg)
    assert ("truncated source or imputed points in the recent window", -1) in penalties(
        [], pre, cfg
    )


def test_imputed_recent_point_penalty(make_series: SeriesFactory, cfg: Config) -> None:
    series = make_series(np.full(120, 50.0))
    series.points["imputed"] = False
    series.points.loc[115, "imputed"] = True
    pre = preprocess(series, cfg)
    assert penalties([], pre, cfg) == [
        ("truncated source or imputed points in the recent window", -1)
    ]
    series.points.loc[115, "imputed"] = False
    series.points.loc[5, "imputed"] = True  # baseline only: no penalty
    assert penalties([], preprocess(series, cfg), cfg) == []
