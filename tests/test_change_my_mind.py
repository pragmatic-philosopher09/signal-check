""" "What would change my mind" conditions (signalcheck.engine.change_my_mind)."""

from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict

import numpy as np
import pytest

from eval.generators import (
    flat_noise,
    linear_trend,
    low_count_poisson,
    seasonal_wave,
    single_origin_spike,
    spike,
)
from signalcheck.config import Config
from signalcheck.engine import analyse
from signalcheck.engine.change_my_mind import Condition, change_my_mind
from signalcheck.engine.checks.common import chart_value, undisplayed_tokens
from signalcheck.engine.checks.seasonality import cycle_length
from signalcheck.engine.preprocess import preprocess
from signalcheck.models import Evidence, Series, Verdict
from tests.conftest import SeriesFactory

CASES: dict[str, Series] = {
    "TREND": linear_trend(np.random.default_rng(1), n=120, freq="D", slope=0.03),
    "FLUKE": spike(np.random.default_rng(2), n=120, freq="D"),
    "SEASONAL": seasonal_wave(np.random.default_rng(3)),
    "NO_CHANGE": flat_noise(np.random.default_rng(0), n=80, freq="W", noise=0.05),
    "INCONCLUSIVE": flat_noise(np.random.default_rng(5), n=10, freq="M"),
}


@pytest.fixture(scope="module")
def verdicts() -> dict[str, Verdict]:
    return {label: analyse(series) for label, series in CASES.items()}


def conditions(v: Verdict) -> list[Condition]:
    out = [c for c in v.change_my_mind if isinstance(c, Condition)]
    assert len(out) == len(v.change_my_mind)
    return out


@pytest.mark.parametrize("label", list(CASES))
def test_every_verdict_gets_conditions_that_honour_the_numbers_contract(
    verdicts: dict[str, Verdict], label: str, cfg: Config
) -> None:
    v = verdicts[label]
    assert v.label == label
    conds = conditions(v)
    assert 1 <= len(conds) <= cfg["change_my_mind"]["max_conditions"]
    for c in conds:
        assert c.numbers, c
        assert undisplayed_tokens(c, c.numbers) == [], (c, c.numbers)


def test_frequency_wording(verdicts: dict[str, Verdict]) -> None:
    assert "next 7 days" in verdicts["TREND"].change_my_mind[0]
    assert "next week" in verdicts["SEASONAL"].change_my_mind[0]
    assert "weeks" in verdicts["NO_CHANGE"].change_my_mind[1]
    assert "more months" in verdicts["INCONCLUSIVE"].change_my_mind[0]


def test_trend_condition_uses_fallback_band(verdicts: dict[str, Verdict], cfg: Config) -> None:
    v = verdicts["TREND"]
    pre = preprocess(CASES["TREND"], cfg)
    assert pre.windows is not None
    stats = pre.windows.stats
    k = cfg["change_my_mind"]["fallback_mads"]
    expected = chart_value(stats.median + k * stats.mad, pre.windows.transform, cfg)
    first = conditions(v)[0]
    assert first.numbers["threshold"] == expected
    assert first.startswith("If the next 7 days fall back below")
    assert first.endswith("this was a fluke.")


def test_downward_trend_wording(cfg: Config) -> None:
    v = analyse(linear_trend(np.random.default_rng(7), n=150, slope=-0.015), cfg)
    assert (v.label, v.direction) == ("TREND", "down")
    assert "rise back above" in v.change_my_mind[0]


def test_trend_mentions_missing_annual_check(verdicts: dict[str, Verdict]) -> None:
    second = conditions(verdicts["TREND"])[1]
    assert second.numbers["periods_needed"] == 730 - 120
    assert "annual-pattern check" in second


def test_fluke_condition(verdicts: dict[str, Verdict]) -> None:
    first = conditions(verdicts["FLUKE"])[0]
    assert first.startswith("If the next 7 days all stay above")
    assert first.numbers["mads"] == 2
    assert first.endswith("this becomes a trend.")


def test_breadth_fluke_adds_spread_condition(cfg: Config) -> None:
    v = analyse(single_origin_spike(np.random.default_rng(8)), cfg)
    assert v.label == "FLUKE"
    assert any("top origin's share drops below 50.0%" in c for c in v.change_my_mind)


def test_seasonal_condition_uses_last_years_value(
    verdicts: dict[str, Verdict], cfg: Config
) -> None:
    c = conditions(verdicts["SEASONAL"])[0]
    pre = preprocess(CASES["SEASONAL"], cfg)
    idx = len(pre.work) - cycle_length("W", cfg)
    assert c.numbers["last_year_date"] == pre.series.points["ts"].iloc[idx].date().isoformat()
    assert c.numbers["last_year_value"] == round(float(pre.series.points["value"].iloc[idx]), 1)
    assert "last year's same-week value" in c


def test_no_change_bounds(verdicts: dict[str, Verdict], cfg: Config) -> None:
    c = conditions(verdicts["NO_CHANGE"])[0]
    assert c.numbers["lower"] < c.numbers["upper"]
    assert c.numbers["outlier_z"] == cfg["checks"]["outliers"]["outlier_z"]
    assert "|z| ≥ 3.5" in c


def test_no_change_omits_non_positive_lower_bound(make_series: SeriesFactory, cfg: Config) -> None:
    series = low_count_poisson(np.random.default_rng(1), lam=2.0)
    pre = preprocess(series, cfg)
    quiet = [Evidence("persistence", "supports_no_change", "s", {}, None)]
    conds = change_my_mind("NO_CHANGE", None, "R5", pre, quiet, cfg)
    assert "lower" not in conds[0].numbers
    assert all("below 0.0" not in c for c in conds)


def test_r1_counts_missing_periods(verdicts: dict[str, Verdict]) -> None:
    c = conditions(verdicts["INCONCLUSIVE"])[0]
    assert c.numbers == {"periods_needed": 14, "n_observed": 10, "min_history": 24}
    assert c.startswith("Need 14 more months of data to decide")


def test_r6_conditions_bracket_the_decision(cfg: Config) -> None:
    pre = preprocess(flat_noise(np.random.default_rng(3), n=120), cfg)
    conds = change_my_mind("INCONCLUSIVE", None, "R6", pre, [], cfg)
    assert any("it is a trend" in c for c in conds)
    assert any("the change was a fluke" in c for c in conds)


def test_r6_asks_for_contributor_data_on_count_series(cfg: Config) -> None:
    pre = preprocess(low_count_poisson(np.random.default_rng(3), lam=20.0), cfg)
    skipped_breadth = Evidence("breadth", "skipped", "s", {}, None, "no data")
    conds = change_my_mind("INCONCLUSIVE", None, "R6", pre, [skipped_breadth], cfg)
    assert any(c.startswith("Contributor data") for c in conds)


def test_max_conditions_from_config(cfg: Config) -> None:
    cfg["change_my_mind"]["max_conditions"] = 1
    pre = preprocess(low_count_poisson(np.random.default_rng(3), lam=20.0), cfg)
    assert len(change_my_mind("INCONCLUSIVE", None, "R6", pre, [], cfg)) == 1


def test_condition_survives_pickle_copy_and_json(verdicts: dict[str, Verdict]) -> None:
    c = conditions(verdicts["TREND"])[0]
    for clone in (pickle.loads(pickle.dumps(c)), copy.deepcopy(c), copy.copy(c)):
        assert clone == c and isinstance(clone, Condition) and clone.numbers == c.numbers
    data = asdict(verdicts["TREND"])
    assert json.loads(json.dumps(data["change_my_mind"]))[0] == str(c)


def test_condition_is_a_plain_string() -> None:
    c = Condition("If 3 then 4.", {"a": 3, "b": 4})
    assert c == "If 3 then 4." and c.upper() == "IF 3 THEN 4."
    assert Condition("x").numbers == {}
