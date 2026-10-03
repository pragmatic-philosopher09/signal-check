"""The pure-numpy PELT must give exactly ruptures' breakpoints."""

from __future__ import annotations

import math

import numpy as np
import pytest
import ruptures as rpt

from signalcheck.engine.pelt import BadSegmentationParameters, pelt


def _reference(signal: np.ndarray, min_size: int, penalty: float) -> list[int]:
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(signal.reshape(-1, 1))
    return [int(b) for b in algo.predict(pen=penalty)]


def _series(rng: np.random.Generator, kind: str, n: int) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, n)
    if kind == "noise":
        return noise
    if kind == "step":
        at = int(rng.integers(n // 4, 3 * n // 4))
        return noise + np.where(np.arange(n) >= at, rng.uniform(1.0, 6.0), 0.0)
    if kind == "multi":
        out = noise.copy()
        for at in sorted(rng.choice(np.arange(3, n - 3), size=3, replace=False)):
            out[at:] += rng.normal(0.0, 3.0)
        return out
    if kind == "spike":
        out = noise.copy()
        out[int(rng.integers(0, n))] += rng.uniform(5.0, 20.0)
        return out
    if kind == "trend":
        return noise + np.linspace(0.0, rng.uniform(-5.0, 5.0), n)
    if kind == "ties":
        return np.round(noise) + np.where(np.arange(n) >= n // 2, 2.0, 0.0)
    raise AssertionError(kind)


CASES = [
    (seed, kind, n, min_size, beta)
    for seed in range(10)
    for kind in ("noise", "step", "multi", "spike", "trend", "ties")
    for n, min_size in ((30, 3), (60, 4), (104, 4), (180, 7))
    for beta in (1.0, 3.0)
]


@pytest.mark.parametrize(("seed", "kind", "n", "min_size", "beta"), CASES)
def test_matches_ruptures(seed: int, kind: str, n: int, min_size: int, beta: float) -> None:
    rng = np.random.default_rng(1000 * seed + n)
    signal = _series(rng, kind, n)
    penalty = beta * math.log(n)
    assert pelt(signal, min_size, penalty) == _reference(signal, min_size, penalty)


def test_constant_and_tiny_series() -> None:
    flat = np.zeros(20)
    assert pelt(flat, 3, 1.0) == _reference(flat, 3, 1.0) == [20]
    short = np.array([1.0, 2.0, 3.0])
    assert pelt(short, 3, 1.0) == _reference(short, 3, 1.0) == [3]


def test_zero_penalty_and_min_size_one() -> None:
    rng = np.random.default_rng(7)
    signal = rng.normal(size=25)
    for pen in (0.0, 0.5):
        assert pelt(signal, 1, pen) == _reference(signal, 1, pen)


def test_too_short_raises() -> None:
    with pytest.raises(BadSegmentationParameters):
        pelt(np.array([1.0, 2.0]), 3, 1.0)
    with pytest.raises(rpt.exceptions.BadSegmentationParameters):
        _reference(np.array([1.0, 2.0]), 3, 1.0)


def test_recovers_obvious_step() -> None:
    signal = np.r_[np.zeros(40), np.full(40, 5.0)] + np.random.default_rng(3).normal(0, 0.2, 80)
    assert pelt(signal, 4, 3 * math.log(80)) == [40, 80]
