"""Seeded synthetic series builders, shared by the unit tests and the Phase 3 eval.

Every builder takes an explicit :class:`numpy.random.Generator` (never global
randomness) and returns a :class:`~signalcheck.models.Series` whose last period
is complete (``fetched_at`` far in the future). Noise is Gaussian with standard
deviation ``noise * level`` unless stated otherwise; values are clipped at 0.
Positions such as ``at`` and ``onset`` accept negative indices counted from the
end, like Python sequences.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from signalcheck.models import Series

FETCHED_AT = "2100-01-01T00:00:00Z"
START = "2024-01-01"
STEPS: dict[str, str] = {"D": "D", "W": "7D", "M": "MS"}


def to_series(
    values: np.ndarray,
    *,
    freq: str = "D",
    scale: str = "relative_0_100",
    start: str = START,
    extra: Mapping[str, np.ndarray] | None = None,
    query: str = "synthetic",
) -> Series:
    """Wrap values (and optional extra columns) as a complete ``csv`` Series."""
    stamps = pd.date_range(start, periods=len(values), freq=STEPS[freq])
    points = pd.DataFrame({"ts": stamps, "value": np.asarray(values, dtype=float)})
    for name, column in (extra or {}).items():
        points[name] = column
    meta: dict[str, Any] = {"caveats": [], "fetched_at": FETCHED_AT}
    return Series(source="csv", query=query, freq=freq, points=points, scale=scale, meta=meta)


def _index(at: int, n: int) -> int:
    """Normalise a possibly negative position."""
    return at % n


def _noise(rng: np.random.Generator, n: int, level: float, noise: float) -> np.ndarray:
    return rng.normal(0.0, noise * level, size=n)


def flat_noise(
    rng: np.random.Generator,
    *,
    n: int = 120,
    freq: str = "D",
    level: float = 50.0,
    noise: float = 0.05,
    scale: str = "relative_0_100",
) -> Series:
    """Constant level plus noise: the null case (no change of any kind)."""
    values = level + _noise(rng, n, level, noise)
    return to_series(np.clip(values, 0, None), freq=freq, scale=scale)


def linear_trend(
    rng: np.random.Generator,
    *,
    n: int = 120,
    freq: str = "D",
    level: float = 50.0,
    slope: float = 0.03,
    onset: int = -30,
    noise: float = 0.05,
    scale: str = "relative_0_100",
) -> Series:
    """Flat until ``onset``, then a straight line changing ``slope * level`` per period.

    Negative ``slope`` gives a downward trend.
    """
    t0 = _index(onset, n)
    ramp = np.clip(np.arange(n) - t0, 0, None) * slope * level
    values = level + ramp + _noise(rng, n, level, noise)
    return to_series(np.clip(values, 0, None), freq=freq, scale=scale)


def spike(
    rng: np.random.Generator,
    *,
    n: int = 120,
    freq: str = "D",
    level: float = 50.0,
    height: float = 3.0,
    at: int = -5,
    noise: float = 0.05,
    scale: str = "relative_0_100",
) -> Series:
    """Flat noise with one period raised by ``height * level``."""
    values = level + _noise(rng, n, level, noise)
    values[_index(at, n)] += height * level
    return to_series(np.clip(values, 0, None), freq=freq, scale=scale)


def dip(
    rng: np.random.Generator,
    *,
    n: int = 120,
    freq: str = "D",
    level: float = 50.0,
    depth: float = 0.8,
    at: int = -5,
    noise: float = 0.05,
    scale: str = "relative_0_100",
) -> Series:
    """Flat noise with one period lowered by ``depth * level``."""
    values = level + _noise(rng, n, level, noise)
    values[_index(at, n)] -= depth * level
    return to_series(np.clip(values, 0, None), freq=freq, scale=scale)


def step_change(
    rng: np.random.Generator,
    *,
    n: int = 120,
    freq: str = "D",
    level: float = 50.0,
    delta: float = 0.5,
    at: int = -20,
    noise: float = 0.05,
    scale: str = "relative_0_100",
) -> Series:
    """Flat noise whose level moves by ``delta * level`` from position ``at`` onwards."""
    values = level + _noise(rng, n, level, noise)
    values[_index(at, n) :] += delta * level
    return to_series(np.clip(values, 0, None), freq=freq, scale=scale)


def seasonal_wave(
    rng: np.random.Generator,
    *,
    n: int = 156,
    freq: str = "W",
    level: float = 50.0,
    amplitude: float = 0.4,
    period: int = 52,
    peak_at: int = -4,
    trend: float = 0.0,
    power: float = 2.0,
    noise: float = 0.05,
    scale: str = "relative_0_100",
) -> Series:
    """Annual wave peaking at ``peak_at`` (and every ``period`` before it).

    The wave is ``amplitude * level * (2 * c ** power - 1)`` with
    ``c = (1 + cos(phase)) / 2``: a cosine for ``power = 1`` and a sharper annual
    peak (a holiday-style season) for larger ``power``. ``trend`` adds an
    underlying straight line of ``trend * level`` per period over the whole
    series, so a real trend can sit on top of the seasonal pattern.
    """
    t = np.arange(n)
    peak = _index(peak_at, n)
    c = (1 + np.cos(2 * np.pi * (t - peak) / period)) / 2
    wave = amplitude * level * (2 * c**power - 1)
    values = level + wave + trend * level * t + _noise(rng, n, level, noise)
    return to_series(np.clip(values, 0, None), freq=freq, scale=scale)


def weekday_pattern(
    rng: np.random.Generator,
    *,
    n: int = 120,
    level: float = 50.0,
    amplitude: float = 0.3,
    noise: float = 0.05,
    scale: str = "relative_0_100",
) -> Series:
    """Daily series with a strong day-of-week cycle (weekends lower) and no other change."""
    weekday = pd.date_range(START, periods=n, freq="D").weekday.to_numpy()
    values = level * (1 - amplitude * (weekday >= 5)) + _noise(rng, n, level, noise)
    return to_series(np.clip(values, 0, None), freq="D", scale=scale)


def low_count_poisson(
    rng: np.random.Generator,
    *,
    n: int = 90,
    freq: str = "D",
    lam: float = 2.0,
    recent_lam: float | None = None,
    recent: int = 18,
) -> Series:
    """Poisson counts with mean ``lam``; the last ``recent`` periods use ``recent_lam`` if given."""
    rates = np.full(n, float(lam))
    if recent_lam is not None:
        rates[n - recent :] = recent_lam
    return to_series(rng.poisson(rates).astype(float), freq=freq, scale="count")


def single_origin_spike(
    rng: np.random.Generator,
    *,
    n: int = 90,
    freq: str = "D",
    lam: float = 20.0,
    spike_mult: float = 30.0,
    at: int = -5,
    base_top_share: float = 0.1,
    base_contrib_ratio: float = 0.9,
    spike_top_share: float = 0.9,
    spike_contributors: int = 5,
) -> Series:
    """Count series with breadth columns and a spike of ``spike_mult * lam`` items at ``at``.

    Ordinary buckets are broad (``top_share`` ~ ``base_top_share``, contributors ~
    ``base_contrib_ratio`` per item). The spike bucket's ``top_share`` and
    ``contributors`` are set explicitly, so a broad spike can be built by passing
    a low ``spike_top_share`` and many ``spike_contributors``.
    """
    counts = rng.poisson(lam, size=n).astype(float)
    i = _index(at, n)
    counts[i] = round(spike_mult * lam)
    top_share = np.clip(rng.normal(base_top_share, base_top_share / 5, size=n), 0.01, 1.0)
    contributors = np.rint(counts * base_contrib_ratio)
    top_share[i] = spike_top_share
    contributors[i] = spike_contributors
    extra = {"top_share": top_share, "contributors": contributors, "raw_count": counts.copy()}
    return to_series(counts, freq=freq, scale="count", extra=extra)


Builder = Callable[..., Series]

GENERATORS: dict[str, Builder] = {
    "flat_noise": flat_noise,
    "linear_trend": linear_trend,
    "spike": spike,
    "dip": dip,
    "step_change": step_change,
    "seasonal_wave": seasonal_wave,
    "weekday_pattern": weekday_pattern,
    "low_count_poisson": low_count_poisson,
    "single_origin_spike": single_origin_spike,
}
