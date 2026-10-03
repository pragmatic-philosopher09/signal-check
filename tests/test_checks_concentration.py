"""Concentration check: direction-aware share of excess in the top one or two periods."""

from __future__ import annotations

import numpy as np
import pytest

from eval import generators as gen
from signalcheck.config import Config
from signalcheck.engine.checks import concentration
from signalcheck.engine.checks.common import iso
from signalcheck.engine.preprocess import preprocess


def test_single_spike_supports_fluke(rng: np.random.Generator, cfg: Config) -> None:
    series = gen.spike(rng, at=-5)
    ev = concentration.run(preprocess(series, cfg), cfg)
    assert ev.stance == "supports_fluke"
    assert ev.numbers["direction"] == "up"
    assert ev.numbers["top1_date"] == iso(series.points["ts"].iloc[-5])
    assert ev.numbers["top1_share_pct"] >= 60.0


def test_single_dip_supports_fluke(rng: np.random.Generator, cfg: Config) -> None:
    ev = concentration.run(preprocess(gen.dip(rng, noise=0.02), cfg), cfg)
    assert ev.stance == "supports_fluke"
    assert ev.numbers["direction"] == "down"
    assert "dip" in ev.summary


def test_trend_is_spread_out(rng: np.random.Generator, cfg: Config) -> None:
    ev = concentration.run(preprocess(gen.linear_trend(rng), cfg), cfg)
    assert ev.stance == "neutral"
    assert ev.numbers["top2_share_pct"] < 80.0


def test_step_change_is_spread_out(rng: np.random.Generator, cfg: Config) -> None:
    ev = concentration.run(preprocess(gen.step_change(rng, delta=-0.5), cfg), cfg)
    assert ev.stance == "neutral"
    assert ev.numbers["direction"] == "down"


def test_flat_noise_is_skipped_no_meaningful_excess(rng: np.random.Generator, cfg: Config) -> None:
    ev = concentration.run(preprocess(gen.flat_noise(rng), cfg), cfg)
    assert ev.stance == "skipped"
    assert ev.skip_reason is not None
    assert ev.skip_reason.startswith("no meaningful excess")


def test_short_history_is_skipped(rng: np.random.Generator, cfg: Config) -> None:
    ev = concentration.run(preprocess(gen.spike(rng, n=10), cfg), cfg)
    assert ev.stance == "skipped"


def test_thresholds_come_from_config(rng: np.random.Generator, cfg: Config) -> None:
    pre = preprocess(gen.spike(rng), cfg)
    cfg["checks"]["concentration"]["conc_top1_max"] = 0.99
    cfg["checks"]["concentration"]["conc_top2_max"] = 0.99
    assert concentration.run(pre, cfg).stance == "neutral"


def test_directional_shares_ignore_opposite_sign() -> None:
    direction, order, top1, top2 = concentration.directional_shares(
        np.array([6.0, -10.0, 2.0, 2.0, 12.0])
    )
    assert direction == "up"
    assert order[0] == 4
    assert top1 == pytest.approx(12.0 / 22.0)
    assert top2 == pytest.approx(18.0 / 22.0)


def test_directional_shares_for_drops() -> None:
    direction, order, top1, _ = concentration.directional_shares(np.array([-1.0, -8.0, 1.0]))
    assert direction == "down"
    assert order[0] == 1
    assert top1 == pytest.approx(8.0 / 9.0)
