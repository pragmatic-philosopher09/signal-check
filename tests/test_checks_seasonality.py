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
    ev = seasonality.run(preprocess(gen.flat_noise(rng, n=156, freq="W"), cfg), cfg)
    assert ev.stance == "skipped"
    assert ev.skip_reason is not None
    assert "no rise to explain" in ev.skip_reason


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
