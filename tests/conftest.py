"""Shared fixtures: validated config, seeded RNG and a synthetic Series builder."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from signalcheck.cache import Cache
from signalcheck.config import Config, load_config
from signalcheck.http import build_session
from signalcheck.models import Series

SEED = 20261003

SeriesFactory = Callable[..., Series]


@pytest.fixture
def cfg() -> Config:
    """A fresh deep copy of the repo config (tests may mutate it)."""
    return copy.deepcopy(load_config())


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator: every random test input is reproducible."""
    return np.random.default_rng(SEED)


def build_series(
    values: Sequence[float] | np.ndarray,
    *,
    freq: str = "D",
    start: str = "2026-01-01",
    scale: str = "relative_0_100",
    fetched_at: str | None = "2100-01-01T00:00:00Z",
    ts: Sequence[Any] | pd.Index | None = None,
    **meta: Any,
) -> Series:
    """A Series with consecutive periods from ``start`` (or explicit ``ts``)."""
    if ts is None:
        step = {"D": "D", "W": "7D", "M": "MS"}[freq]
        stamps = pd.date_range(start, periods=len(values), freq=step)
    else:
        stamps = pd.DatetimeIndex(pd.to_datetime(list(ts), format="ISO8601"))
    points = pd.DataFrame({"ts": stamps, "value": np.asarray(values, dtype=float)})
    full_meta: dict[str, Any] = {"caveats": [], **meta}
    if fetched_at is not None:
        full_meta.setdefault("fetched_at", fetched_at)
    return Series(source="csv", query="test", freq=freq, points=points, scale=scale, meta=full_meta)


@pytest.fixture
def make_series() -> SeriesFactory:
    """Factory fixture around :func:`build_series`."""
    return build_series


FIXED_NOW = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)


@pytest.fixture
def adapter_deps(tmp_path: Path, cfg: Config) -> Iterator[dict[str, Any]]:
    """Injectable adapter dependencies: tmp disk cache, no-op sleep, fixed clock.

    Retries are capped at 2 so retry-exhaustion tests stay fast.
    """
    cfg["http"]["max_retries"] = 2
    cfg["cache"]["dir"] = str(tmp_path / "cache")
    cache = Cache.from_config(cfg)
    yield {
        "cfg": cfg,
        "cache": cache,
        "sleep": lambda _s: None,
        "now": lambda: FIXED_NOW,
        "session": build_session(),
        "rng": np.random.default_rng(SEED),
    }
    cache.close()
