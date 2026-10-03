"""Level shift check: PELT (L2) on the robust-standardised series."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval import generators as gen
from signalcheck.config import Config
from signalcheck.engine.checks import level_shift
from signalcheck.engine.checks.common import iso
from signalcheck.engine.preprocess import preprocess


def test_step_up_is_a_sustained_shift(rng: np.random.Generator, cfg: Config) -> None:
    series = gen.step_change(rng, delta=0.5, at=-20)
    ev = level_shift.run(preprocess(series, cfg), cfg)
    assert ev.stance == "supports_trend"
    assert ev.numbers["direction"] == "up"
    assert ev.numbers["sustained"] is True
    assert ev.numbers["change_date"] == iso(series.points["ts"].iloc[-20])
    assert ev.numbers["after_median"] > ev.numbers["before_median"]
    assert ev.annotation is not None
    assert ev.annotation["vlines"][0]["ts"] == ev.numbers["change_date"]


def test_step_down_is_a_sustained_shift(rng: np.random.Generator, cfg: Config) -> None:
    ev = level_shift.run(preprocess(gen.step_change(rng, delta=-0.5), cfg), cfg)
    assert ev.stance == "supports_trend"
    assert ev.numbers["direction"] == "down"


def test_shift_just_before_recent_window_counts(rng: np.random.Generator, cfg: Config) -> None:
    pre = preprocess(gen.step_change(rng, at=-27), cfg)
    assert pre.windows is not None
    tolerance = cfg["checks"]["level_shift"]["recent_tolerance"]["D"]
    cp = len(pre.work) - 27
    assert pre.windows.recent.start_idx - tolerance <= cp < pre.windows.recent.start_idx
    assert level_shift.run(pre, cfg).stance == "supports_trend"


def test_flat_noise_has_no_shift(rng: np.random.Generator, cfg: Config) -> None:
    ev = level_shift.run(preprocess(gen.flat_noise(rng), cfg), cfg)
    assert ev.stance == "supports_no_change"
    assert ev.numbers["in_recent"] is False
    n = 120
    expected = cfg["checks"]["level_shift"]["pen_beta"] * math.log(n)
    assert ev.numbers["penalty"] == pytest.approx(expected, abs=0.01)


def test_old_shift_is_not_recent(rng: np.random.Generator, cfg: Config) -> None:
    ev = level_shift.run(preprocess(gen.step_change(rng, at=-80), cfg), cfg)
    assert ev.stance == "supports_no_change"


@pytest.mark.parametrize("builder", [gen.spike, gen.dip])
def test_single_spike_or_dip_is_not_a_sustained_shift(
    builder: gen.Builder, rng: np.random.Generator, cfg: Config
) -> None:
    ev = level_shift.run(preprocess(builder(rng), cfg), cfg)
    assert ev.stance != "supports_trend"
    assert ev.numbers["sustained"] is False


def test_hold_fraction_comes_from_config(rng: np.random.Generator, cfg: Config) -> None:
    pre = preprocess(gen.step_change(rng, delta=0.3, noise=0.15), cfg)
    held = level_shift.run(pre, cfg)
    assert held.stance == "supports_trend"
    assert held.numbers["share_new_side_pct"] < 100.0
    cfg["checks"]["level_shift"]["hold_fraction"] = 1.0
    strict = level_shift.run(pre, cfg)
    assert strict.stance == "neutral"
    assert strict.numbers["sustained"] is False
    assert "did not produce a lasting new level" in strict.summary


def test_series_with_gaps_is_skipped(rng: np.random.Generator, cfg: Config) -> None:
    series = gen.step_change(rng)
    series.points.loc[100, "value"] = np.nan
    ev = level_shift.run(preprocess(series, cfg), cfg)
    assert ev.stance == "skipped"
    assert ev.numbers["n_missing"] == 1


def test_short_history_is_skipped(rng: np.random.Generator, cfg: Config) -> None:
    assert level_shift.run(preprocess(gen.step_change(rng, n=10, at=-3), cfg), cfg).stance == (
        "skipped"
    )


def test_pelt_finds_a_clean_step() -> None:
    z = np.r_[np.zeros(30), np.full(20, 10.0)]
    assert level_shift.pelt_change_points(z, min_size=3, penalty=5.0) == [30]
