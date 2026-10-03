"""Top-level ``analyse`` entry point (signalcheck.engine.pipeline)."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from eval.generators import GENERATORS, linear_trend, seasonal_wave
from signalcheck.config import Config
from signalcheck.engine import analyse
from signalcheck.engine.checks import CHECKS
from signalcheck.models import RULE_IDS
from tests.conftest import SeriesFactory


@pytest.mark.parametrize(("freq", "n"), [("D", 0), ("D", 5), ("W", 25), ("M", 1)])
def test_short_series_is_inconclusive_via_r1(
    make_series: SeriesFactory, cfg: Config, freq: str, n: int
) -> None:
    v = analyse(make_series(np.full(n, 10.0), freq=freq), cfg)
    assert (v.label, v.rule_fired, v.direction) == ("INCONCLUSIVE", "R1", None)
    assert all(e.stance == "skipped" for e in v.evidence)
    assert v.window == {}
    need = cfg["preprocess"]["min_history"][freq] - n
    assert v.change_my_mind[0].startswith(f"Need {need} more ")


def test_every_check_reported_in_order(cfg: Config) -> None:
    v = analyse(linear_trend(np.random.default_rng(1)), cfg)
    assert [e.check for e in v.evidence] == list(CHECKS)
    assert v.rule_fired in RULE_IDS
    assert {"recent_start", "recent_end", "baseline_start"} <= set(v.window)
    assert v.reason


def test_deterministic(cfg: Config) -> None:
    a = analyse(seasonal_wave(np.random.default_rng(9)), cfg)
    b = analyse(seasonal_wave(np.random.default_rng(9)), cfg)
    assert asdict(a) == asdict(b)


def test_does_not_mutate_input(cfg: Config) -> None:
    series = linear_trend(np.random.default_rng(1))
    before = series.points.copy()
    caveats = list(series.caveats)
    analyse(series, cfg)
    pd.testing.assert_frame_equal(series.points, before)
    assert series.caveats == caveats


def test_short_gaps_filled_for_regular_checks_only(make_series: SeriesFactory, cfg: Config) -> None:
    rng = np.random.default_rng(4)
    n = 140
    stamps = pd.date_range("2026-01-01", periods=n, freq="D")
    keep = np.ones(n, dtype=bool)
    keep[[40, 41, 90]] = False
    values = 50 + rng.normal(0, 1, n)
    series = make_series(values[keep], ts=stamps[keep])
    v = analyse(series, cfg)
    level_shift = next(e for e in v.evidence if e.check == "level_shift")
    assert level_shift.skip_reason is None or "gap" not in level_shift.skip_reason
    assert level_shift.stance != "skipped"
    assert "3 missing days left as gaps (not interpolated)." in v.caveats
    assert any(
        c.startswith(
            "Filled 3 missing days by linear interpolation for the level-shift and "
            "seasonality checks"
        )
        for c in v.caveats
    )
    assert len(v.caveats) == len(set(v.caveats))


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_every_generator_analyses_cleanly(name: str, cfg: Config) -> None:
    v = analyse(GENERATORS[name](np.random.default_rng(0)), cfg)
    assert v.label in {"TREND", "FLUKE", "SEASONAL", "NO_CHANGE", "INCONCLUSIVE"}
    assert (v.direction is not None) == (v.label in {"TREND", "FLUKE"} and v.direction is not None)
    if v.label == "TREND":
        assert v.direction in {"up", "down"}
    assert v.confidence in {"high", "medium", "low"}
    assert v.change_my_mind
