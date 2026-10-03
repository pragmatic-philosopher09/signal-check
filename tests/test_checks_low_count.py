"""Low-count check: exact conditional test on the recent/baseline rate ratio."""

from __future__ import annotations

import numpy as np
import pytest

from eval import generators as gen
from signalcheck.config import Config
from signalcheck.engine.checks import low_count
from signalcheck.engine.preprocess import preprocess


def test_clear_rise_in_low_counts_supports_trend(rng: np.random.Generator, cfg: Config) -> None:
    ev = low_count.run(preprocess(gen.low_count_poisson(rng, lam=2.0, recent_lam=6.0), cfg), cfg)
    assert ev.stance == "supports_trend"
    assert ev.numbers["active"] is True
    assert ev.numbers["too_few_to_call"] is False
    assert ev.numbers["direction"] == "up"
    assert ev.numbers["ratio_lower"] > 1.0


def test_clear_drop_in_low_counts_supports_trend(rng: np.random.Generator, cfg: Config) -> None:
    ev = low_count.run(preprocess(gen.low_count_poisson(rng, lam=4.0, recent_lam=0.5), cfg), cfg)
    assert ev.stance == "supports_trend"
    assert ev.numbers["direction"] == "down"
    assert ev.numbers["ratio_upper"] < 1.0


def test_unchanged_low_counts_are_too_few_to_call(rng: np.random.Generator, cfg: Config) -> None:
    ev = low_count.run(preprocess(gen.low_count_poisson(rng, lam=2.0), cfg), cfg)
    assert ev.stance == "neutral"
    assert ev.numbers["active"] is True
    assert ev.numbers["too_few_to_call"] is True
    assert ev.numbers["ratio_lower"] <= 1.0 <= ev.numbers["ratio_upper"]
    assert "too few events to call" in ev.summary


def test_high_counts_are_skipped(rng: np.random.Generator, cfg: Config) -> None:
    ev = low_count.run(preprocess(gen.low_count_poisson(rng, lam=40.0), cfg), cfg)
    assert ev.stance == "skipped"
    assert ev.numbers["active"] is False


def test_threshold_comes_from_config(rng: np.random.Generator, cfg: Config) -> None:
    pre = preprocess(gen.low_count_poisson(rng, lam=2.0), cfg)
    cfg["checks"]["low_count"]["low_count_threshold"] = 1.0
    assert low_count.run(pre, cfg).stance == "skipped"


def test_non_count_series_is_skipped(rng: np.random.Generator, cfg: Config) -> None:
    ev = low_count.run(preprocess(gen.flat_noise(rng, level=2.0), cfg), cfg)
    assert ev.stance == "skipped"
    assert ev.skip_reason is not None
    assert "count series" in ev.skip_reason


def test_no_events_is_skipped(rng: np.random.Generator, cfg: Config) -> None:
    ev = low_count.run(preprocess(gen.low_count_poisson(rng, lam=0.0), cfg), cfg)
    assert ev.stance == "skipped"


def test_rate_ratio_equal_rates_includes_one() -> None:
    rr = low_count.rate_ratio(k1=10, n1=10, k0=10, n0=10, level=0.95)
    assert rr.ratio == pytest.approx(1.0)
    assert rr.includes_one


def test_rate_ratio_scales_by_window_lengths() -> None:
    rr = low_count.rate_ratio(k1=20, n1=10, k0=20, n0=40, level=0.95)
    assert rr.ratio == pytest.approx(4.0)
    assert rr.lower > 1.0
    assert not rr.includes_one


def test_rate_ratio_with_no_baseline_events_is_unbounded() -> None:
    rr = low_count.rate_ratio(k1=5, n1=10, k0=0, n0=40, level=0.95)
    assert np.isinf(rr.ratio)
    assert np.isinf(rr.upper)
