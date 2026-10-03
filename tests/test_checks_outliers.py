"""Outlier check: robust z of recent points against the baseline median and MAD."""

from __future__ import annotations

import numpy as np
import pytest

from eval import generators as gen
from signalcheck.config import Config
from signalcheck.engine.checks import outliers
from signalcheck.engine.checks.common import iso
from signalcheck.engine.checks.seasonality import deweekly
from signalcheck.engine.preprocess import preprocess


def test_single_spike_is_an_isolated_outlier(rng: np.random.Generator, cfg: Config) -> None:
    series = gen.spike(rng, at=-3)
    ev = outliers.run(preprocess(series, cfg), cfg)
    assert ev.stance == "supports_fluke"
    assert ev.numbers["n_flagged"] == 1
    assert ev.numbers["flagged_dates"] == [iso(series.points["ts"].iloc[-3])]
    assert ev.numbers["max_abs_z"] >= cfg["checks"]["outliers"]["outlier_z"]
    assert ev.annotation is not None
    assert ev.annotation["points"][0]["ts"] == ev.numbers["max_date"]


def test_single_dip_is_an_isolated_outlier(rng: np.random.Generator, cfg: Config) -> None:
    ev = outliers.run(preprocess(gen.dip(rng), cfg), cfg)
    assert ev.stance == "supports_fluke"
    assert ev.numbers["max_value"] < ev.numbers["baseline_median"]


def test_flat_noise_has_no_outliers(rng: np.random.Generator, cfg: Config) -> None:
    ev = outliers.run(preprocess(gen.flat_noise(rng), cfg), cfg)
    assert ev.stance == "supports_no_change"
    assert ev.numbers["n_flagged"] == 0


def test_step_change_flags_many_points_and_is_neutral(
    rng: np.random.Generator, cfg: Config
) -> None:
    ev = outliers.run(preprocess(gen.step_change(rng), cfg), cfg)
    assert ev.stance == "neutral"
    assert ev.numbers["n_flagged"] > cfg["checks"]["outliers"]["isolated_max_points"]


def test_outlier_z_comes_from_config(rng: np.random.Generator, cfg: Config) -> None:
    pre = preprocess(gen.spike(rng), cfg)
    cfg["checks"]["outliers"]["outlier_z"] = 1000.0
    assert outliers.run(pre, cfg).stance == "supports_no_change"


def test_short_history_is_skipped(rng: np.random.Generator, cfg: Config) -> None:
    ev = outliers.run(preprocess(gen.spike(rng, n=10), cfg), cfg)
    assert ev.stance == "skipped"


def test_robust_z() -> None:
    z = outliers.robust_z(np.array([10.0, 13.0, 4.0, np.nan]), median=10.0, mad=2.0)
    np.testing.assert_allclose(z[:3], [0.0, 1.5, -3.0])
    assert np.isnan(z[3])
    assert z[1] == pytest.approx(1.5)


def test_day_of_week_pattern_is_not_an_outlier_once_removed(
    rng: np.random.Generator, cfg: Config
) -> None:
    pre = preprocess(gen.weekday_pattern(rng, amplitude=0.5, noise=0.02), cfg)
    assert pre.windows is not None
    plain = outliers.run(pre, cfg)
    assert plain.numbers["n_flagged"] >= 2
    adjusted = deweekly(pre.work, "D", cfg, recent_start=pre.windows.recent.start_idx)
    assert adjusted is not None
    ev = outliers.run(pre, cfg, values=adjusted.to_numpy())
    assert ev.numbers["deweekly"] is True
    assert ev.numbers["n_flagged"] == 0
    assert "day-of-week pattern is removed" in ev.summary


def test_spike_survives_day_of_week_removal(rng: np.random.Generator, cfg: Config) -> None:
    series = gen.weekday_pattern(rng)
    series.points.loc[series.points.index[-6], "value"] *= 5
    pre = preprocess(series, cfg)
    assert pre.windows is not None
    adjusted = deweekly(pre.work, "D", cfg, recent_start=pre.windows.recent.start_idx)
    assert adjusted is not None
    ev = outliers.run(pre, cfg, values=adjusted.to_numpy())
    assert ev.stance == "supports_fluke"
    assert ev.numbers["n_flagged"] == 1
    assert ev.numbers["max_value"] == pytest.approx(
        float(series.points["value"].iloc[-6]), rel=0.01
    )
