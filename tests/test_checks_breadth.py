"""Breadth check: volume-weighted top_share and contributors per item."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from eval import generators as gen
from signalcheck.config import Config
from signalcheck.engine.checks import breadth
from signalcheck.engine.preprocess import preprocess


def test_single_origin_spike_supports_fluke(rng: np.random.Generator, cfg: Config) -> None:
    ev = breadth.run(preprocess(gen.single_origin_spike(rng), cfg), cfg)
    assert ev.stance == "supports_fluke"
    assert ev.numbers["one_origin"] is True
    assert ev.numbers["top_share_pct"] >= 50.0
    assert ev.annotation is not None
    assert ev.annotation["points"][0]["label"] == "single-origin bucket"


def test_broad_spike_is_neutral(rng: np.random.Generator, cfg: Config) -> None:
    series = gen.single_origin_spike(rng, spike_top_share=0.05, spike_contributors=500)
    ev = breadth.run(preprocess(series, cfg), cfg)
    assert ev.stance == "neutral"
    assert ev.numbers["one_origin"] is False
    assert ev.numbers["few_contributors"] is False


def test_few_contributors_supports_fluke(rng: np.random.Generator, cfg: Config) -> None:
    series = gen.single_origin_spike(
        rng, base_contrib_ratio=0.1, spike_top_share=0.05, spike_contributors=50
    )
    ev = breadth.run(preprocess(series, cfg), cfg)
    assert ev.stance == "supports_fluke"
    assert ev.numbers["few_contributors"] is True
    assert ev.numbers["one_origin"] is False


def test_top_share_only_is_enough(rng: np.random.Generator, cfg: Config) -> None:
    series = gen.single_origin_spike(rng)
    series = replace(series, points=series.points.drop(columns=["contributors"]))
    ev = breadth.run(preprocess(series, cfg), cfg)
    assert ev.stance == "supports_fluke"
    assert "contributors_per_item" not in ev.numbers


def test_threshold_comes_from_config(rng: np.random.Generator, cfg: Config) -> None:
    pre = preprocess(gen.single_origin_spike(rng), cfg)
    cfg["checks"]["breadth"]["breadth_top_share_max"] = 0.95
    cfg["checks"]["breadth"]["breadth_min_contrib_ratio"] = 0.0
    assert breadth.run(pre, cfg).stance == "neutral"


def test_without_breadth_columns_is_skipped(rng: np.random.Generator, cfg: Config) -> None:
    ev = breadth.run(preprocess(gen.low_count_poisson(rng), cfg), cfg)
    assert ev.stance == "skipped"


def test_weighted_top_share_weights_by_volume() -> None:
    share = breadth.weighted_top_share(np.array([90.0, 10.0]), np.array([1.0, 0.0]))
    assert share == pytest.approx(0.9)
    assert breadth.weighted_top_share(np.array([0.0]), np.array([0.5])) is None


def test_contributors_per_item() -> None:
    assert breadth.contributors_per_item(np.array([10.0, 30.0]), np.array([4.0, 6.0])) == (
        pytest.approx(0.25)
    )
