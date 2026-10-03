"""Persistence check: Hamed-Rao Mann-Kendall + Theil-Sen on recent window plus context."""

from __future__ import annotations

import numpy as np
import pytest

from eval import generators as gen
from signalcheck.config import Config
from signalcheck.engine.checks import persistence
from signalcheck.engine.preprocess import preprocess


def test_upward_trend_supports_trend(rng: np.random.Generator, cfg: Config) -> None:
    pre = preprocess(gen.linear_trend(rng, slope=0.03), cfg)
    ev = persistence.run(pre, cfg)
    assert ev.stance == "supports_trend"
    assert ev.numbers["direction"] == "up"
    assert ev.numbers["test"] == "hamed_rao"
    assert ev.numbers["slope_norm_pct"] == pytest.approx(3.0, abs=0.5)
    assert pre.windows is not None
    context = cfg["checks"]["persistence"]["persistence_context"]
    assert ev.numbers["n_tested"] == pre.windows.recent.length + context
    assert ev.numbers["run_length"] > 0


def test_downward_trend_supports_trend(rng: np.random.Generator, cfg: Config) -> None:
    ev = persistence.run(preprocess(gen.linear_trend(rng, slope=-0.025), cfg), cfg)
    assert ev.stance == "supports_trend"
    assert ev.numbers["direction"] == "down"
    assert ev.numbers["slope_norm_pct"] < -2.0
    assert "fell" in ev.summary


def test_flat_noise_supports_no_change(rng: np.random.Generator, cfg: Config) -> None:
    ev = persistence.run(preprocess(gen.flat_noise(rng), cfg), cfg)
    assert ev.stance == "supports_no_change"
    assert ev.numbers["supports_trend"] is False


def test_single_spike_does_not_support_trend(rng: np.random.Generator, cfg: Config) -> None:
    ev = persistence.run(preprocess(gen.spike(rng), cfg), cfg)
    assert ev.stance != "supports_trend"


def test_significant_but_small_slope_is_neutral(rng: np.random.Generator, cfg: Config) -> None:
    ev = persistence.run(preprocess(gen.linear_trend(rng, slope=-0.015), cfg), cfg)
    assert ev.numbers["significant"] is True
    assert ev.stance == "neutral"


def test_min_slope_comes_from_config(rng: np.random.Generator, cfg: Config) -> None:
    pre = preprocess(gen.linear_trend(rng, slope=-0.015), cfg)
    cfg["checks"]["persistence"]["min_slope"] = 0.01
    assert persistence.run(pre, cfg).stance == "supports_trend"


def test_short_history_is_skipped(rng: np.random.Generator, cfg: Config) -> None:
    ev = persistence.run(preprocess(gen.flat_noise(rng, n=10), cfg), cfg)
    assert ev.stance == "skipped"
    assert ev.skip_reason is not None
    assert "not enough history" in ev.skip_reason


def test_too_few_observed_points_is_skipped(rng: np.random.Generator, cfg: Config) -> None:
    pre = preprocess(gen.flat_noise(rng), cfg)
    assert pre.windows is not None
    values = pre.work.to_numpy(dtype=float).copy()
    values[persistence.segment_start(pre.windows, cfg) + 2 :] = np.nan
    ev = persistence.run(pre, cfg, values=values)
    assert ev.stance == "skipped"
    assert ev.numbers["min_points"] == cfg["checks"]["persistence"]["min_points"]


def test_original_mk_below_hamed_rao_min_n(rng: np.random.Generator, cfg: Config) -> None:
    cfg["checks"]["persistence"]["hamed_rao_min_n"] = 1000
    ev = persistence.run(preprocess(gen.linear_trend(rng), cfg), cfg)
    assert ev.numbers["test"] == "original"
    assert ev.stance == "supports_trend"


def test_log_scale_slope_is_percent_change(rng: np.random.Generator, cfg: Config) -> None:
    pre = preprocess(gen.low_count_poisson(rng, lam=2.0, recent_lam=6.0), cfg)
    assert pre.windows is not None and pre.windows.transform == "log1p"
    ev = persistence.run(pre, cfg)
    assert ev.stance == "supports_trend"
    assert "baseline level" not in ev.summary


def test_exceedance_run_counts_most_recent_consecutive() -> None:
    values = np.array([0.0, 5.0, 0.0, 5.0, 5.0, 5.0])
    assert persistence.exceedance_run(values, center=0.0, half_width=1.0, direction="up") == 3
    assert persistence.exceedance_run(-values, center=0.0, half_width=1.0, direction="down") == 3
    assert persistence.exceedance_run(values, center=0.0, half_width=1.0, direction="down") == 0
    gap = np.array([5.0, 5.0, np.nan, 5.0])
    assert persistence.exceedance_run(gap, center=0.0, half_width=1.0, direction="up") == 1


def test_annotation_has_band_and_trend_segment(rng: np.random.Generator, cfg: Config) -> None:
    ev = persistence.run(preprocess(gen.linear_trend(rng), cfg), cfg)
    assert ev.annotation is not None
    assert ev.annotation["bands"][0]["label"] == "baseline band"
    segment = ev.annotation["segments"][0]
    assert segment["end_value"] > segment["start_value"]
