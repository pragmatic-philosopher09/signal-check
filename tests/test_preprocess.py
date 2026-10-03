"""Tests for signalcheck.engine.preprocess (COPILOT_BRIEF.md section 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalcheck.config import Config
from signalcheck.engine.preprocess import (
    fill_gaps,
    is_partial_last_period,
    preprocess,
    recent_window_length,
    robust_baseline,
    to_original_scale,
)
from tests.conftest import SeriesFactory, build_series

# --- duplicates and ordering -------------------------------------------------


def test_duplicates_keep_last_and_sort(cfg: Config, make_series: SeriesFactory) -> None:
    ts = ["2026-01-03", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-01 15:00"]
    s = make_series([3, 1, 20, 2, 10], ts=ts)
    out = preprocess(s, cfg).series
    assert out.points["ts"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
    ]
    # Same-day rows collapse onto the day; the last row in input order wins.
    assert out.points["value"].tolist() == [10.0, 2.0, 3.0]
    assert "2 duplicate rows for the same day; kept the last value." in out.caveats


def test_no_duplicate_caveat_when_clean(cfg: Config, make_series: SeriesFactory) -> None:
    out = preprocess(make_series(np.ones(30)), cfg).series
    assert not any("duplicate" in c for c in out.caveats)


def test_weekly_timestamps_snap_to_data_anchor(cfg: Config, make_series: SeriesFactory) -> None:
    # Google Trends weeks start on Sunday; a stray Wednesday stamp joins its Sunday week.
    ts = ["2026-01-04", "2026-01-11", "2026-01-14", "2026-01-18"]
    out = preprocess(make_series([1, 2, 3, 4], freq="W", ts=ts), cfg).series
    assert out.points["ts"].dt.weekday.unique().tolist() == [6]
    assert out.points["value"].tolist() == [1.0, 3.0, 4.0]


# --- gaps ----------------------------------------------------------------------


def test_gaps_left_as_nan_with_caveat(cfg: Config, make_series: SeriesFactory) -> None:
    ts = pd.date_range("2026-01-01", periods=40).delete([5, 6, 20])
    pre = preprocess(make_series(np.arange(37), ts=ts), cfg)
    points = pre.series.points
    assert len(points) == 40
    assert points["value"].isna().sum() == 3
    assert not points["imputed"].any()
    assert "3 missing days left as gaps (not interpolated)." in pre.series.caveats
    assert pre.history.n_observed == 37
    assert np.isnan(pre.work.iloc[5])


def test_explicit_nan_counts_as_gap(cfg: Config, make_series: SeriesFactory) -> None:
    values = np.ones(30)
    values[10] = np.nan
    pre = preprocess(make_series(values), cfg)
    assert "1 missing day left as gaps (not interpolated)." in pre.series.caveats


def test_leading_missing_values_are_trimmed(cfg: Config, make_series: SeriesFactory) -> None:
    values = np.ones(32)
    values[:2] = np.nan
    pre = preprocess(make_series(values), cfg)
    assert len(pre.series.points) == 30
    assert pre.series.points["ts"].iloc[0] == pd.Timestamp("2026-01-03")
    assert not any("missing" in c for c in pre.series.caveats)
    assert pre.windows is not None and pre.windows.stats.n == 23


def test_fill_gaps_only_short_runs(cfg: Config, make_series: SeriesFactory) -> None:
    values = np.arange(40, dtype=float)
    values[[3, 10, 11, 20, 21, 22]] = np.nan  # runs of 1, 2 and 3
    pre = preprocess(make_series(values, scale="count"), cfg)
    filled = fill_gaps(pre, cfg)
    pts = filled.series.points
    assert pts.loc[[3, 10, 11], "value"].tolist() == [3.0, 10.0, 11.0]
    assert pts.loc[[3, 10, 11], "imputed"].all()
    assert pts.loc[[20, 21, 22], "value"].isna().all()
    assert not pts.loc[[20, 21, 22], "imputed"].any()
    assert int(pts["imputed"].sum()) == 3
    caveats = filled.series.caveats
    assert any("Filled 3 missing days by linear interpolation" in c for c in caveats)
    assert "3 missing days in longer gaps left unfilled." in caveats
    assert filled.work.iloc[3] == pytest.approx(np.log1p(3.0))
    # Stats are not recomputed from imputed points; the input is untouched.
    assert filled.windows == pre.windows
    assert pre.series.points["value"].isna().sum() == 6


def test_fill_gaps_noop_without_gaps(cfg: Config, make_series: SeriesFactory) -> None:
    pre = preprocess(make_series(np.ones(30)), cfg)
    assert fill_gaps(pre, cfg) is pre


# --- partial last period ---------------------------------------------------------


@pytest.mark.parametrize(
    ("freq", "last", "fetched_at", "dropped"),
    [
        ("D", "2026-10-03", "2026-10-03T14:00:00Z", True),
        ("D", "2026-10-02", "2026-10-03T00:00:00Z", False),  # ended exactly at fetch
        ("D", "2026-10-02", "2026-10-03T01:00:00+05:30", True),  # 19:30 UTC on the 2nd
        ("W", "2026-09-27", "2026-10-03T09:00:00Z", True),  # Sunday week ends 4 Oct
        ("W", "2026-09-20", "2026-10-03T09:00:00Z", False),
        ("M", "2026-10-01", "2026-10-03T09:00:00Z", True),
        ("M", "2026-09-01", "2026-10-03T09:00:00Z", False),
    ],
)
def test_partial_rule_live_fetch(
    cfg: Config, freq: str, last: str, fetched_at: str, dropped: bool
) -> None:
    step = {"D": "D", "W": "7D", "M": "MS"}[freq]
    ts = pd.date_range(end=last, periods=40, freq=step)
    s = build_series(np.arange(40), freq=freq, ts=ts, fetched_at=fetched_at)
    pre = preprocess(s, cfg)
    assert (pre.series.meta["dropped_partial"] is not None) is dropped
    assert len(pre.series.points) == (39 if dropped else 40)
    if dropped:
        assert pre.series.meta["dropped_partial"] == {"ts": last, "value": 39.0}
        assert any("because it had not ended" in c for c in pre.series.caveats)


@pytest.mark.parametrize(
    ("freq", "last", "upload_date", "toggle", "dropped"),
    [
        ("D", "2026-10-03", "2026-10-03", False, True),
        ("D", "2026-10-02", "2026-10-03", False, False),
        ("D", "2026-10-02", "2026-10-03", True, True),
        ("W", "2026-09-27", "2026-10-03", False, True),  # week 27 Sep - 3 Oct
        ("W", "2026-09-20", "2026-10-03", False, False),
        ("M", "2026-09-01", "2026-10-03", False, False),
        ("M", "2026-09-01", "2026-10-03", True, True),
    ],
)
def test_partial_rule_csv_upload(
    cfg: Config, freq: str, last: str, upload_date: str, toggle: bool, dropped: bool
) -> None:
    step = {"D": "D", "W": "7D", "M": "MS"}[freq]
    ts = pd.date_range(end=last, periods=40, freq=step)
    s = build_series(
        np.arange(40),
        freq=freq,
        ts=ts,
        # fetched_at must be ignored by the upload rule.
        fetched_at="2026-10-03T12:00:00Z",
        partial_rule="upload_date",
        upload_date=upload_date,
        last_period_incomplete=toggle,
    )
    pre = preprocess(s, cfg)
    assert (pre.series.meta["dropped_partial"] is not None) is dropped
    if toggle:
        assert any("marked incomplete by the user" in c for c in pre.series.caveats)


def test_partial_rule_requires_metadata() -> None:
    last = pd.Timestamp("2026-10-01")
    with pytest.raises(ValueError, match="fetched_at"):
        is_partial_last_period(last, "D", {})
    with pytest.raises(ValueError, match="upload_date"):
        is_partial_last_period(last, "D", {"partial_rule": "upload_date"})
    with pytest.raises(ValueError, match="partial_rule"):
        is_partial_last_period(last, "D", {"partial_rule": "vibes"})


def test_dropped_partial_nan_value_is_none(cfg: Config, make_series: SeriesFactory) -> None:
    values = np.ones(30)
    values[-1] = np.nan
    s = make_series(values, start="2026-09-04", fetched_at="2026-10-03T08:00:00Z")
    assert preprocess(s, cfg).series.meta["dropped_partial"] == {"ts": "2026-10-03", "value": None}


def test_input_series_not_mutated(cfg: Config, make_series: SeriesFactory) -> None:
    s = make_series([1, 2, 2], ts=["2026-01-01", "2026-01-03", "2026-01-03"])
    before = s.points.copy()
    preprocess(s, cfg)
    pd.testing.assert_frame_equal(s.points, before)
    assert s.meta["caveats"] == []


# --- minimum history ---------------------------------------------------------------


@pytest.mark.parametrize(("freq", "minimum"), [("D", 28), ("W", 26), ("M", 24)])
def test_too_short_series(cfg: Config, freq: str, minimum: int) -> None:
    short = preprocess(build_series(np.ones(minimum - 1), freq=freq), cfg)
    assert not short.history.sufficient
    assert short.windows is None
    assert short.history.points_needed == 1
    assert short.history.reason is not None
    assert short.history.reason.startswith("not enough history")
    assert f"need at least {minimum}" in short.history.reason
    enough = preprocess(build_series(np.ones(minimum), freq=freq), cfg)
    assert enough.history.sufficient and enough.windows is not None


def test_gaps_do_not_count_towards_history(cfg: Config, make_series: SeriesFactory) -> None:
    ts = pd.date_range("2026-01-01", periods=30).delete([3, 4, 5])
    pre = preprocess(make_series(np.ones(27), ts=ts), cfg)
    assert pre.history.n_observed == 27 and not pre.history.sufficient


def test_empty_series(cfg: Config, make_series: SeriesFactory) -> None:
    pre = preprocess(make_series([]), cfg)
    assert pre.history.n_observed == 0 and pre.windows is None


def test_history_threshold_from_config(cfg: Config, make_series: SeriesFactory) -> None:
    cfg["preprocess"]["min_history"]["D"] = 40
    assert not preprocess(make_series(np.ones(35)), cfg).history.sufficient


# --- recent window ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("freq", "n", "expected"),
    [
        ("D", 28, 7),  # 5.6 -> 6, clamped up to 7
        ("D", 50, 10),
        ("D", 143, 28),  # 28.6 -> 29, clamped down to 28
        ("D", 1000, 28),
        ("W", 26, 5),
        ("W", 10, 4),
        ("W", 66, 13),
        ("W", 520, 13),
        ("M", 24, 5),
        ("M", 12, 3),
        ("M", 120, 6),
    ],
)
def test_recent_window_clamping(cfg: Config, freq: str, n: int, expected: int) -> None:
    assert recent_window_length(n, freq, cfg) == expected


def test_recent_window_rounds_half_up(cfg: Config) -> None:
    cfg["preprocess"]["recent_frac"] = 0.25
    cfg["preprocess"]["recent_min"]["D"] = 1
    assert recent_window_length(42, "D", cfg) == 11  # 10.5 -> 11 (banker's would give 10)


def test_window_bounds(cfg: Config, make_series: SeriesFactory) -> None:
    pre = preprocess(make_series(np.arange(60), start="2026-01-01"), cfg)
    w = pre.windows
    assert w is not None
    assert (w.n, w.baseline.length, w.recent.length) == (60, 48, 12)
    assert (w.baseline.start_idx, w.baseline.end_idx) == (0, 48)
    assert (w.recent.start_idx, w.recent.end_idx) == (48, 60)
    assert w.to_dict() == {
        "baseline_start": "2026-01-01",
        "baseline_end": "2026-02-17",
        "recent_start": "2026-02-18",
        "recent_end": "2026-03-01",
        "recent_end_exclusive": "2026-03-02",
        "n_baseline": 48,
        "n_recent": 12,
    }
    recent = pre.series.points["value"].iloc[w.recent.slice]
    assert recent.tolist() == list(range(48, 60))


def test_monthly_window_end_exclusive(cfg: Config) -> None:
    pre = preprocess(build_series(np.ones(30), freq="M", start="2024-01-01"), cfg)
    assert pre.windows is not None
    assert pre.windows.recent.end == pd.Timestamp("2026-06-01")
    assert pre.windows.recent.end_exclusive == pd.Timestamp("2026-07-01")


# --- robust baseline and MAD ------------------------------------------------------------


def test_mad_estimates_sigma_on_noise(cfg: Config, rng: np.random.Generator) -> None:
    values = 50 + rng.normal(0, 4, size=400)
    values[::40] += 200  # spikes barely move robust statistics
    pre = preprocess(build_series(values, start="2025-01-01"), cfg)
    stats = pre.windows.stats if pre.windows else None
    assert stats is not None and not stats.floored
    assert stats.median == pytest.approx(50, abs=1)
    assert stats.mad == pytest.approx(4, rel=0.15)


@pytest.mark.parametrize(("level", "expected_mad"), [(10.0, 0.5), (0.0, 0.05), (0.4, 0.05)])
def test_mad_zero_uses_floor(cfg: Config, level: float, expected_mad: float) -> None:
    values = np.full(40, level)
    values[-3:] = level + 5  # recent change must not affect the baseline floor
    pre = preprocess(build_series(values), cfg)
    assert pre.windows is not None
    stats = pre.windows.stats
    assert stats.floored and stats.mad_unfloored == 0.0
    assert stats.mad == pytest.approx(expected_mad)  # mad_floor * max(median, 1)


def test_mad_floor_on_log_scale(cfg: Config) -> None:
    pre = preprocess(build_series(np.full(40, 10.0), scale="count"), cfg)
    w = pre.windows
    assert w is not None and w.transform == "log1p"
    assert w.stats.median == pytest.approx(np.log1p(10))
    assert w.stats.mad == pytest.approx(0.05 * np.log1p(10))
    assert w.stats_original.median == 10 and w.stats_original.mad == pytest.approx(0.5)


def test_mad_floor_from_config(cfg: Config) -> None:
    cfg["preprocess"]["mad_floor"] = 0.1
    stats = robust_baseline(pd.Series([20.0] * 10), cfg)
    assert stats.mad == pytest.approx(2.0)


def test_baseline_ignores_nan(cfg: Config) -> None:
    stats = robust_baseline(pd.Series([1.0, np.nan, 3.0, 5.0]), cfg)
    assert stats.n == 3 and stats.median == 3.0
    assert stats.mad == pytest.approx(1.4826 * 2)
    with pytest.raises(ValueError, match="no observed"):
        robust_baseline(pd.Series([np.nan]), cfg)


# --- working scale ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scale", "transform"),
    [
        ("count", "log1p"),
        ("pageviews", "log1p"),
        ("relative_0_100", "identity"),
        ("value", "identity"),
    ],
)
def test_working_scale(cfg: Config, rng: np.random.Generator, scale: str, transform: str) -> None:
    values = rng.poisson(20, size=40).astype(float)
    pre = preprocess(build_series(values, scale=scale), cfg)
    assert pre.windows is not None and pre.windows.transform == transform
    expected = np.log1p(values) if transform == "log1p" else values
    np.testing.assert_allclose(pre.work.to_numpy(), expected)
    np.testing.assert_allclose(to_original_scale(pre.work.to_numpy(), transform), values)
    assert isinstance(pre.work.index, pd.DatetimeIndex)
