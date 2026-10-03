"""Contract every check must honour, run across all synthetic generators.

- ``summary`` (and ``skip_reason``) only contain numbers/dates recorded in
  ``numbers`` with the same rounding (the narration validator relies on this);
- ``numbers`` are JSON-serialisable and finite;
- annotations use ISO dates within the series range, never indices;
- results are deterministic for a given seed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eval import generators as gen
from signalcheck.config import Config, load_config
from signalcheck.engine.checks import CHECKS, run_all
from signalcheck.engine.checks.common import NumberBook, undisplayed_tokens
from signalcheck.engine.preprocess import preprocess
from signalcheck.models import STANCES, Evidence, Series

SEED = 20261003
DATE_KEYS = {"ts", "start", "end"}
ISO_DATE = r"^\d{4}-\d{2}-\d{2}$"


def _with_gap(series: Series) -> Series:
    series.points.loc[len(series.points) // 2, "value"] = np.nan
    return series


CASES: dict[str, Callable[[np.random.Generator], Series]] = {
    **{name: builder for name, builder in gen.GENERATORS.items()},
    "trend_down": lambda r: gen.linear_trend(r, slope=-0.025),
    "step_down": lambda r: gen.step_change(r, delta=-0.5),
    "weekly_step": lambda r: gen.step_change(r, n=156, freq="W", at=-10),
    "seasonal_with_trend": lambda r: gen.seasonal_wave(r, trend=0.01),
    "seasonal_daily": lambda r: gen.seasonal_wave(r, n=1095, freq="D", period=365, peak_at=-15),
    "seasonal_monthly": lambda r: gen.seasonal_wave(r, n=36, freq="M", period=12, peak_at=-2),
    "low_count_up": lambda r: gen.low_count_poisson(r, recent_lam=6.0),
    "low_count_down": lambda r: gen.low_count_poisson(r, lam=4.0, recent_lam=0.5),
    "low_count_zero": lambda r: gen.low_count_poisson(r, lam=0.0),
    "short_history": lambda r: gen.flat_noise(r, n=8),
    "with_gap": lambda r: _with_gap(gen.step_change(r)),
    "flat_weekly": lambda r: gen.flat_noise(r, n=156, freq="W"),
}


def _evidence_cases() -> Iterator[tuple[str, str]]:
    for case in CASES:
        for check in CHECKS:
            yield case, check


def _run(case: str, check: str, cfg: Config) -> tuple[Series, Evidence]:
    series = CASES[case](np.random.default_rng(SEED))
    return series, CHECKS[check](preprocess(series, cfg), cfg)


def _walk_dates(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in DATE_KEYS:
                yield value
            else:
                yield from _walk_dates(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_dates(item)


@pytest.mark.parametrize(("case", "check"), list(_evidence_cases()))
def test_evidence_contract(case: str, check: str, cfg: Config) -> None:
    series, ev = _run(case, check, cfg)
    assert ev.check == check
    assert ev.stance in STANCES
    assert ev.summary
    assert undisplayed_tokens(ev.summary, ev.numbers) == [], ev.summary
    if ev.stance == "skipped":
        assert ev.skip_reason is not None
        assert undisplayed_tokens(ev.skip_reason, ev.numbers) == []
        assert ev.annotation is None
    json.dumps(ev.numbers, allow_nan=False)
    for value in ev.numbers.values():
        if isinstance(value, float):
            assert math.isfinite(value)
    first, last = series.points["ts"].min(), series.points["ts"].max()
    for date in _walk_dates(ev.annotation):
        assert isinstance(date, str)
        assert pd.Series([date]).str.fullmatch(ISO_DATE).all(), date
        assert first <= pd.Timestamp(date) <= last
    if ev.annotation is not None:
        json.dumps(ev.annotation, allow_nan=False)


def test_every_check_supports_skips_and_stays_out_on_some_case() -> None:
    cfg = load_config()
    seen: dict[str, set[str]] = {check: set() for check in CHECKS}
    for case, check in _evidence_cases():
        seen[check].add(_run(case, check, cfg)[1].stance)
    for check, stances in seen.items():
        assert "skipped" in stances, check
        assert stances - {"skipped", "neutral"}, check


def test_same_seed_gives_identical_evidence(cfg: Config) -> None:
    for name in ("spike", "seasonal_wave", "single_origin_spike", "low_count_up"):
        a = run_all(preprocess(CASES[name](np.random.default_rng(1)), cfg), cfg)
        b = run_all(preprocess(CASES[name](np.random.default_rng(1)), cfg), cfg)
        assert a == b


def test_registry_follows_section_6_order(cfg: Config) -> None:
    assert list(CHECKS) == [
        "persistence",
        "concentration",
        "seasonality",
        "outliers",
        "level_shift",
        "low_count",
        "breadth",
    ]
    evidence = run_all(preprocess(gen.spike(np.random.default_rng(SEED)), cfg), cfg)
    assert [e.check for e in evidence] == list(CHECKS)


def test_generators_are_seeded() -> None:
    for builder in gen.GENERATORS.values():
        a = builder(np.random.default_rng(7)).points
        b = builder(np.random.default_rng(7)).points
        c = builder(np.random.default_rng(8)).points
        pd.testing.assert_frame_equal(a, b)
        assert not a["value"].equals(c["value"])


class TestNumberBook:
    def test_rounding_is_recorded_as_displayed(self, cfg: Config) -> None:
        book = NumberBook(cfg)
        assert book.value("v", 12.345) == "12.3"
        assert book.pct("p", 0.4567) == "45.7%"
        assert book.stat("s", 1.0 / 3.0) == "0.33"
        assert book.numbers == {"v": 12.3, "p": 45.7, "s": 0.33}

    def test_negative_zero_is_displayed_as_zero(self, cfg: Config) -> None:
        book = NumberBook(cfg)
        assert book.value("v", -0.01) == "0.0"
        assert book.numbers["v"] == 0.0

    def test_tiny_p_value_shows_bound(self, cfg: Config) -> None:
        book = NumberBook(cfg)
        assert book.p_value("p", 1e-9) == "p < 0.001"
        assert book.numbers == {"p": 0.0, "p_bound": 0.001}
        assert book.p_value("q", 0.0234) == "p = 0.023"

    def test_conflicting_values_raise(self, cfg: Config) -> None:
        book = NumberBook(cfg)
        book.value("v", 1.0)
        book.value("v", 1.0)
        with pytest.raises(ValueError, match="recorded twice"):
            book.value("v", 2.0)

    def test_non_finite_raises(self, cfg: Config) -> None:
        with pytest.raises(ValueError, match="not finite"):
            NumberBook(cfg).stat("s", float("inf"))

    def test_display_decimals_come_from_config(self, cfg: Config) -> None:
        cfg["display"]["value_decimals"] = 2
        assert NumberBook(cfg).value("v", 12.345) == "12.35"


class TestUndisplayedTokens:
    def test_clean_text_passes(self) -> None:
        numbers = {"a": 12.3, "b": 4, "d": "2024-05-01", "p": 45.0}
        text = "Level 12.3 over 4 weeks since 2024-05-01 (45.0% held)."
        assert undisplayed_tokens(text, numbers) == []

    def test_wrong_rounding_is_caught(self) -> None:
        assert undisplayed_tokens("value 12.35", {"a": 12.3}) == ["12.35"]
        assert undisplayed_tokens("value 12", {"a": 12.3}) == ["12"]

    def test_unknown_numbers_dates_and_words_are_caught(self) -> None:
        bad = undisplayed_tokens("seven days to 2024-01-02, up 9", {"n": 3})
        assert set(bad) == {"2024-01-02", "9", "seven"}

    def test_number_word_matching_a_recorded_count_passes(self) -> None:
        assert undisplayed_tokens("three days", {"n": 3}) == []
