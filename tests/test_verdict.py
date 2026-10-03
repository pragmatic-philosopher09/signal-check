"""Decision table R1-R6 (signalcheck.engine.verdict) on hand-built evidence."""

from __future__ import annotations

from typing import Any

import pytest

from signalcheck.config import Config
from signalcheck.engine.verdict import RULES, decide, extract_signals
from signalcheck.models import RULE_IDS, Evidence, HistoryCheck

OK = HistoryCheck(n_observed=120, min_required=28, sufficient=True, reason=None)
SHORT = HistoryCheck(
    n_observed=10,
    min_required=28,
    sufficient=False,
    reason="not enough history: 10 days observed, need at least 28 days",
)


def ev(check: str, stance: str, **numbers: Any) -> Evidence:
    reason = "not applicable" if stance == "skipped" else None
    return Evidence(check, stance, f"{check} {stance}", dict(numbers), None, reason)


def base(**overrides: Evidence) -> list[Evidence]:
    """A quiet series: every check says 'no change' or is skipped."""
    out = {
        "persistence": ev("persistence", "supports_no_change", direction="up", significant=False),
        "concentration": ev("concentration", "skipped"),
        "seasonality": ev("seasonality", "skipped", skip_code="history"),
        "outliers": ev("outliers", "supports_no_change", n_flagged=0),
        "level_shift": ev("level_shift", "supports_no_change", sustained=False, in_recent=False),
        "low_count": ev("low_count", "skipped"),
        "breadth": ev("breadth", "skipped"),
    }
    out.update(overrides)
    return list(out.values())


TREND_UP = ev("persistence", "supports_trend", direction="up", significant=True)
TREND_DOWN = ev("persistence", "supports_trend", direction="down", significant=True)
SHIFT_UP = ev("level_shift", "supports_trend", sustained=True, in_recent=True, direction="up")
SHIFT_DOWN = ev("level_shift", "supports_trend", sustained=True, in_recent=True, direction="down")
CONC_FLUKE = ev("concentration", "supports_fluke", direction="up")
ISOLATED = ev("outliers", "supports_fluke", n_flagged=1, isolated=True, max_direction="down")


def test_rule_ids_match_models() -> None:
    assert tuple(RULES) == RULE_IDS


def test_r1_not_enough_history(cfg: Config) -> None:
    d = decide(base(persistence=TREND_UP), SHORT, cfg)
    assert (d.label, d.rule, d.direction) == ("INCONCLUSIVE", "R1", None)
    assert d.reason == SHORT.reason


def test_r2_seasonal_wins_over_raw_trend(cfg: Config) -> None:
    seasonal = ev("seasonality", "supports_seasonal", seasonal_share_pct=85.0)
    d = decide(base(seasonality=seasonal, persistence=TREND_UP, level_shift=SHIFT_UP), OK, cfg)
    assert (d.label, d.rule, d.direction) == ("SEASONAL", "R2", None)


def test_r2_falls_through_when_adjusted_series_trends(cfg: Config) -> None:
    # The seasonality check stays neutral when a trend survives seasonal adjustment.
    seasonal = ev("seasonality", "neutral", seasonal_share_pct=85.0, adjusted_trend=True)
    d = decide(base(seasonality=seasonal, persistence=TREND_UP), OK, cfg)
    assert (d.label, d.rule, d.direction) == ("TREND", "R4", "up")


def test_r3a_concentration_fluke(cfg: Config) -> None:
    d = decide(base(concentration=CONC_FLUKE), OK, cfg)
    assert (d.label, d.rule, d.direction) == ("FLUKE", "R3", "up")


def test_r3a_needs_no_trend(cfg: Config) -> None:
    d = decide(base(concentration=CONC_FLUKE, persistence=TREND_UP), OK, cfg)
    # Concentration also blocks R4, so this is a conflict.
    assert (d.label, d.rule) == ("INCONCLUSIVE", "R6")
    assert "concentration: fluke" in d.reason and "persistence: trend" in d.reason


def test_r3b_breadth_fluke_even_with_trend(cfg: Config) -> None:
    breadth = ev("breadth", "supports_fluke", one_origin=True)
    d = decide(base(breadth=breadth, persistence=TREND_UP), OK, cfg)
    assert (d.label, d.rule) == ("FLUKE", "R3")
    assert "single origin" in d.reason


def test_r3c_isolated_outliers_when_concentration_skipped(cfg: Config) -> None:
    d = decide(base(outliers=ISOLATED), OK, cfg)
    assert (d.label, d.rule, d.direction) == ("FLUKE", "R3", "down")
    assert "isolated outliers" in d.reason


def test_r3c_can_be_switched_off(cfg: Config) -> None:
    cfg["verdict"]["fluke_from_isolated_outliers"] = False
    d = decide(base(outliers=ISOLATED), OK, cfg)
    # No outliers-free "no change" either, so it is a conflict.
    assert (d.label, d.rule) == ("INCONCLUSIVE", "R6")


def test_r3c_defers_to_concentration_when_it_ran(cfg: Config) -> None:
    spread = ev("concentration", "neutral", direction="up")
    d = decide(base(outliers=ISOLATED, concentration=spread), OK, cfg)
    assert d.rule == "R6"


@pytest.mark.parametrize("blocker", [TREND_UP, SHIFT_UP])
def test_r3c_blocked_by_trend_or_sustained_shift(cfg: Config, blocker: Evidence) -> None:
    d = decide(base(outliers=ISOLATED, **{blocker.check: blocker}), OK, cfg)
    assert d.label == "TREND"


def test_r4_persistence_trend(cfg: Config) -> None:
    d = decide(base(persistence=TREND_DOWN), OK, cfg)
    assert (d.label, d.rule, d.direction) == ("TREND", "R4", "down")


def test_r4_sustained_level_shift_alone(cfg: Config) -> None:
    d = decide(base(level_shift=SHIFT_UP), OK, cfg)
    assert (d.label, d.rule, d.direction) == ("TREND", "R4", "up")
    assert "level shift" in d.reason


def test_r4_blocked_by_too_few_events(cfg: Config) -> None:
    low = ev("low_count", "neutral", active=True, too_few_to_call=True)
    d = decide(base(persistence=TREND_UP, low_count=low), OK, cfg)
    assert (d.label, d.rule) == ("INCONCLUSIVE", "R6")
    assert "too few events" in d.reason


def test_r4_blocked_by_disagreeing_directions(cfg: Config) -> None:
    d = decide(base(persistence=TREND_UP, level_shift=SHIFT_DOWN), OK, cfg)
    assert (d.label, d.rule) == ("INCONCLUSIVE", "R6")
    assert "disagree on the direction" in d.reason


def test_r4_low_count_trend_must_agree(cfg: Config) -> None:
    low = ev("low_count", "supports_trend", active=True, too_few_to_call=False, direction="down")
    assert decide(base(persistence=TREND_UP, low_count=low), OK, cfg).rule == "R6"
    low_up = ev("low_count", "supports_trend", active=True, too_few_to_call=False, direction="up")
    assert decide(base(persistence=TREND_UP, low_count=low_up), OK, cfg).label == "TREND"


def test_low_count_trend_alone_does_not_trigger_r4(cfg: Config) -> None:
    low = ev("low_count", "supports_trend", active=True, too_few_to_call=False, direction="up")
    d = decide(base(low_count=low), OK, cfg)
    assert d.label == "NO_CHANGE"


def test_r5_no_change(cfg: Config) -> None:
    d = decide(base(), OK, cfg)
    assert (d.label, d.rule, d.direction) == ("NO_CHANGE", "R5", None)


def test_r5_allows_skipped_level_shift(cfg: Config) -> None:
    d = decide(base(level_shift=ev("level_shift", "skipped")), OK, cfg)
    assert d.rule == "R5"


@pytest.mark.parametrize(
    "override",
    [
        ev("persistence", "neutral", direction="up", significant=True),
        ev("outliers", "neutral", n_flagged=5),
        ev("level_shift", "neutral", sustained=False, in_recent=True),
        ev("persistence", "skipped"),
    ],
)
def test_r5_blocked(cfg: Config, override: Evidence) -> None:
    d = decide(base(**{override.check: override}), OK, cfg)
    assert (d.label, d.rule) == ("INCONCLUSIVE", "R6")


def test_r6_lists_conflicting_checks(cfg: Config) -> None:
    noisy = ev("outliers", "neutral", n_flagged=5)
    d = decide(base(outliers=noisy), OK, cfg)
    assert d.reason.startswith("the checks do not line up (")
    assert "outliers: neutral" in d.reason
    assert "persistence: no change" in d.reason
    assert "breadth" not in d.reason  # skipped checks are not listed


def test_extract_signals_ignores_skipped_numbers() -> None:
    skipped_shift = Evidence(
        "level_shift", "skipped", "s", {"sustained": True, "in_recent": True}, None, "gaps"
    )
    s = extract_signals(base(level_shift=skipped_shift))
    assert not s.shift_ran and not s.shift_sustained and not s.shift_in_recent


def test_first_matching_row_wins(cfg: Config) -> None:
    # Seasonal (R2) beats a fluke (R3) beats a trend (R4).
    seasonal = ev("seasonality", "supports_seasonal")
    assert decide(base(seasonality=seasonal, concentration=CONC_FLUKE), OK, cfg).rule == "R2"
    breadth = ev("breadth", "supports_fluke")
    assert decide(base(breadth=breadth, level_shift=SHIFT_UP), OK, cfg).rule == "R3"
