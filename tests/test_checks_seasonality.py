"""Seasonality check: annual STL, out-of-sample seasonal expectation, adjusted re-runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval import generators as gen
from signalcheck.config import Config
from signalcheck.engine.checks import seasonality
from signalcheck.engine.preprocess import preprocess


def test_annual_peak_supports_seasonal(rng: np.random.Generator, cfg: Config) -> None:
    ev = seasonality.run(preprocess(gen.seasonal_wave(rng), cfg), cfg)
    assert ev.stance == "supports_seasonal"
    assert ev.numbers["seasonal_share_pct"] >= 100 * cfg["verdict"]["seasonal_explained_min"]
    assert ev.numbers["adjusted_trend"] is False
    assert ev.numbers["adjusted_level_shift"] is False
    assert ev.annotation is not None
    assert ev.annotation["lines"][0]["label"] == "baseline + annual pattern"


def test_exactly_two_years_is_enough(rng: np.random.Generator, cfg: Config) -> None:
    assert cfg["checks"]["seasonality"]["annual_min_points"]["W"] == 104
    ev = seasonality.run(preprocess(gen.seasonal_wave(rng, n=104), cfg), cfg)
    assert ev.stance == "supports_seasonal"


def test_daily_annual_peak_supports_seasonal(rng: np.random.Generator, cfg: Config) -> None:
    series = gen.seasonal_wave(rng, n=1095, freq="D", period=365, peak_at=-15)
    ev = seasonality.run(preprocess(series, cfg), cfg)
    assert ev.stance == "supports_seasonal"


def test_season_with_underlying_trend_is_not_just_seasonal(
    rng: np.random.Generator, cfg: Config
) -> None:
    ev = seasonality.run(preprocess(gen.seasonal_wave(rng, trend=0.01), cfg), cfg)
    assert ev.stance == "neutral"
    assert ev.numbers["seasonal_share_pct"] < 100 * cfg["verdict"]["seasonal_explained_min"]
    assert ev.summary.startswith("Not just seasonal")


def test_step_change_is_not_explained_by_season(rng: np.random.Generator, cfg: Config) -> None:
    ev = seasonality.run(preprocess(gen.step_change(rng, n=156, freq="W", at=-10), cfg), cfg)
    assert ev.stance == "neutral"
    assert ev.numbers["seasonal_share_pct"] < 100 * cfg["verdict"]["seasonal_explained_min"]


@pytest.mark.parametrize(
    ("change", "trend", "shift", "expected"),
    [
        ("up", None, None, False),
        ("up", "up", None, True),
        ("up", None, "up", True),
        ("up", "down", None, False),
        ("up", None, "down", False),
        ("down", "down", "up", True),
    ],
)
def test_residual_with_change_needs_the_same_direction(
    change: str, trend: str | None, shift: str | None, expected: bool
) -> None:
    assert seasonality.residual_with_change(change, trend, shift) is expected


def test_residual_against_the_rise_leaves_it_seasonal(cfg: Config) -> None:
    # The annual peak lifts the recent window while the underlying level fell: the
    # adjusted shift runs against the rise, so it cannot be what produced the rise.
    series = gen.seasonal_wave(np.random.default_rng(1), n=208)
    series.points.loc[series.points.index[-30:], "value"] -= 20.0
    ev = seasonality.run(preprocess(series, cfg), cfg)
    assert ev.numbers["adjusted_level_shift"] is True
    assert ev.numbers["adjusted_level_shift_direction"] == "down"
    assert ev.numbers["residual_with_change"] is False
    assert ev.stance == "supports_seasonal"
    assert "runs against the rise" in ev.summary


def test_explained_min_comes_from_config(rng: np.random.Generator, cfg: Config) -> None:
    pre = preprocess(gen.seasonal_wave(rng), cfg)
    cfg["verdict"]["seasonal_explained_min"] = 0.99
    ev = seasonality.run(pre, cfg)
    assert ev.stance == "neutral"
    assert ev.numbers["explained_min_pct"] == 99.0


def test_short_history_is_skipped(rng: np.random.Generator, cfg: Config) -> None:
    ev = seasonality.run(preprocess(gen.spike(rng, n=120), cfg), cfg)
    assert ev.stance == "skipped"
    assert ev.skip_reason is not None
    assert "2 years" in ev.skip_reason


def test_flat_series_has_nothing_to_explain(rng: np.random.Generator, cfg: Config) -> None:
    cfg["checks"]["seasonality"]["excess_min_mads"] = 1.0
    ev = seasonality.run(preprocess(gen.flat_noise(rng, n=156, freq="W"), cfg), cfg)
    assert ev.stance == "skipped"
    assert ev.skip_reason is not None
    assert "no rise to explain" in ev.skip_reason


def test_flat_series_is_never_seasonal(rng: np.random.Generator, cfg: Config) -> None:
    ev = seasonality.run(preprocess(gen.flat_noise(rng, n=156, freq="W"), cfg), cfg)
    assert ev.stance in {"skipped", "neutral"}


def test_gaps_are_skipped(rng: np.random.Generator, cfg: Config) -> None:
    series = gen.seasonal_wave(rng)
    series.points.loc[50, "value"] = np.nan
    ev = seasonality.run(preprocess(series, cfg), cfg)
    assert ev.stance == "skipped"
    assert ev.numbers["n_missing"] == 1


@pytest.mark.parametrize("n", [800, 803, 806])
def test_day_of_week_pattern_never_supports_seasonal(
    n: int, rng: np.random.Generator, cfg: Config
) -> None:
    ev = seasonality.run(preprocess(gen.weekday_pattern(rng, n=n), cfg), cfg)
    assert ev.stance != "supports_seasonal"


def test_deweekly_removes_day_of_week_cycle(rng: np.random.Generator, cfg: Config) -> None:
    series = gen.weekday_pattern(rng, n=140)
    work = series.points.set_index("ts")["value"]
    cleaned = seasonality.deweekly(work, "D", cfg)
    assert cleaned is not None
    weekday = pd.DatetimeIndex(work.index).weekday

    def spread(x: pd.Series) -> float:
        means = x.groupby(weekday).mean()
        return float(means.max() - means.min())

    assert spread(cleaned) < 0.2 * spread(work)


def test_deweekly_only_applies_to_long_daily_series(cfg: Config) -> None:
    weekly = pd.Series(np.ones(100))
    assert seasonality.deweekly(weekly, "W", cfg) is None
    short = pd.Series(np.ones(7 * cfg["checks"]["seasonality"]["weekly_min_weeks"] - 1))
    assert seasonality.deweekly(short, "D", cfg) is None


def test_deweekly_with_recent_start_uses_a_fixed_baseline_profile(
    rng: np.random.Generator, cfg: Config
) -> None:
    series = gen.weekday_pattern(rng, n=140)
    work = series.points.set_index("ts")["value"].copy()
    work.iloc[-5] *= 4
    cleaned = seasonality.deweekly(work, "D", cfg, recent_start=112)
    assert cleaned is not None
    correction = (work - cleaned).to_numpy()
    period = cfg["checks"]["seasonality"]["weekly_period"]
    assert np.allclose(correction[period:], correction[:-period])
    # The spike keeps its full height and does not leak into the same weekday earlier.
    assert cleaned.iloc[-5] > 3 * cleaned.iloc[:112].median()
    assert abs(cleaned.iloc[-12] - cleaned.iloc[:112].median()) < 0.25 * cleaned.iloc[:112].median()


def test_deweekly_short_baseline_keeps_stl_component(rng: np.random.Generator, cfg: Config) -> None:
    work = gen.weekday_pattern(rng, n=140).points.set_index("ts")["value"]
    plain = seasonality.deweekly(work, "D", cfg)
    short = seasonality.deweekly(work, "D", cfg, recent_start=3)
    assert plain is not None and short is not None
    assert np.allclose(plain.to_numpy(), short.to_numpy())


def test_seasonal_expectation_uses_only_earlier_cycles() -> None:
    seasonal = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 100.0, 100.0, 100.0])
    out = seasonality.seasonal_expectation(seasonal, recent_start=6, cycle=3)
    np.testing.assert_allclose(out[:6], seasonal[:6])
    np.testing.assert_allclose(out[6:], [3.0, 4.0, 5.0])


def test_block_ids_align_to_the_end() -> None:
    np.testing.assert_array_equal(seasonality.block_ids(9, 4), [-1, 0, 0, 0, 0, 1, 1, 1, 1])


def test_cycle_length(cfg: Config) -> None:
    assert seasonality.cycle_length("W", cfg) == 52
    assert seasonality.cycle_length("M", cfg) == 12
    assert seasonality.cycle_length("D", cfg) == 364
