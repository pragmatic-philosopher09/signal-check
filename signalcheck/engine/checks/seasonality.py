"""Seasonality check (COPILOT_BRIEF.md section 6.3).

Statistical meaning: how much of the recent rise (or drop) is the usual annual
pattern, and is anything left once that pattern is removed?

- **Day-of-week patterns are a nuisance, not a verdict.** :func:`deweekly` uses a
  robust STL with period ``weekly_period`` on daily data with at least
  ``weekly_min_weeks`` weeks and returns the series minus that weekly component,
  for other checks to use. It never produces ``supports_seasonal``.
- **Annual seasonality** needs at least ``annual_min_points[freq]`` points (two
  full years). Weekly/monthly data get a robust STL with period
  ``annual_period[freq]``; daily data are first averaged into consecutive
  ``weekly_period``-day blocks aligned to the last day, decomposed with the
  weekly period, and each day takes its block's seasonal value.
- STL uses ``stl_robust`` and ``stl_seasonal_deg`` from config (degree 0 makes
  each phase a robust average across years, the near-periodic case).
- Within the recent window ``S`` is the *expected* seasonal value: the mean of
  the fitted component at the same phase in every earlier cycle that ends before
  the recent window (:func:`seasonal_expectation`). The usual pattern is thus
  learnt out of sample, so a recent jump or trend cannot explain itself by
  leaking into the end of the STL fit.
- ``rise = mean(recent) - baseline_median`` and
  ``seasonal_rise = mean(S over recent) - mean(S over baseline)`` on the working
  scale. If ``|rise| < excess_min_mads * MAD`` there is nothing to explain
  (skipped); otherwise ``seasonal_share = clip(seasonal_rise / rise, 0, 1)``.
- Persistence and level shift are re-run on the seasonally adjusted series
  ``x - S``; both results are reported in ``numbers``.

Stance: ``seasonal_share >= seasonal_explained_min`` with no significant trend and
no sustained level shift left after adjustment *in the direction of the rise (or
drop)* supports "seasonal". Otherwise it is neutral (a trend on top of seasonality
is left to the other checks). A residual trend or shift that runs *against* the
change means the usual pattern alone would predict an even larger move (this
season is weaker than usual), so it cannot be what makes the change real; it is
reported (``residual_with_change`` is False) but does not block "seasonal".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from signalcheck.config import Config, get_config
from signalcheck.engine.checks.common import (
    NumberBook,
    baseline_of,
    chart_value,
    iso,
    make_annotation,
    make_evidence,
    period_word,
    plural,
    recent_span,
    skip_if_no_windows,
    skipped,
    timestamps,
)
from signalcheck.engine.checks.level_shift import ShiftResult, detect_shift
from signalcheck.engine.checks.persistence import TrendResult, assess_trend
from signalcheck.models import Evidence, Preprocessed

CHECK = "seasonality"


def stl_seasonal(values: np.ndarray, period: int, cfg: Config) -> np.ndarray:
    """Seasonal component of a (robust, per config) STL decomposition."""
    conf = cfg["checks"]["seasonality"]
    stl = STL(
        values,
        period=period,
        robust=bool(conf["stl_robust"]),
        seasonal_deg=int(conf["stl_seasonal_deg"]),
    )
    return np.asarray(stl.fit().seasonal, dtype=float)


def deweekly(
    work: pd.Series, freq: str, cfg: Config, recent_start: int | None = None
) -> pd.Series | None:
    """Daily series with the day-of-week component removed, or ``None`` if not applicable.

    Only daily data with at least ``weekly_min_weeks`` weeks and no missing values
    qualify. This is nuisance removal for other checks; it is never evidence of
    seasonality. With ``recent_start`` the STL weekly component is replaced by a
    fixed day-of-week profile: its mean per weekday over the baseline (positions
    before ``recent_start``), subtracted from every day. The same correction then
    applies to baseline and recent days, so the baseline spread is not shrunk by
    STL fitting its noise, and a recent spike cannot shape its own correction.
    A baseline shorter than one cycle keeps the plain STL component.
    """
    conf = cfg["checks"]["seasonality"]
    period = int(conf["weekly_period"])
    if freq != "D" or len(work) < int(conf["weekly_min_weeks"]) * period or work.isna().any():
        return None
    seasonal = stl_seasonal(work.to_numpy(dtype=float), period, cfg)
    if recent_start is not None and recent_start >= period:
        phase = np.arange(len(seasonal)) % period
        base = phase[:recent_start]
        profile = np.array([seasonal[:recent_start][base == k].mean() for k in range(period)])
        seasonal = profile[phase]
    return work - seasonal


def residual_with_change(
    change_direction: str, trend_direction: str | None, shift_direction: str | None
) -> bool:
    """Whether the seasonally adjusted series still moves the way the recent change did.

    ``trend_direction`` / ``shift_direction`` are the directions of a supporting
    adjusted trend or sustained adjusted level shift (``None`` if there is none). A
    residual that runs *against* the change (e.g. the level fell while the annual peak
    lifted the recent window) cannot have produced it, so it does not count.
    """
    return change_direction in (trend_direction, shift_direction)


def block_ids(n: int, block: int) -> np.ndarray:
    """Block index per position for consecutive ``block``-sized groups ending at ``n - 1``.

    Leading positions that do not fill a whole block get negative ids.
    """
    return (np.arange(n) - n % block) // block


def annual_component(values: np.ndarray, freq: str, cfg: Config) -> np.ndarray:
    """Annual seasonal component on the working scale, one value per period."""
    conf = cfg["checks"]["seasonality"]
    period = int(conf["annual_period"][freq])
    if freq != "D":
        return stl_seasonal(values, period, cfg)
    block = int(conf["weekly_period"])
    ids = block_ids(len(values), block)
    full = ids >= 0
    weekly = values[full].reshape(-1, block).mean(axis=1)
    seasonal_weekly = stl_seasonal(weekly, period, cfg)
    out = np.empty(len(values))
    out[full] = np.repeat(seasonal_weekly, block)
    out[~full] = seasonal_weekly[0]
    return out


def cycle_length(freq: str, cfg: Config) -> int:
    """Periods per annual cycle at the series frequency (daily: weeks x days)."""
    conf = cfg["checks"]["seasonality"]
    days = int(conf["weekly_period"]) if freq == "D" else 1
    return int(conf["annual_period"][freq]) * days


def seasonal_expectation(seasonal: np.ndarray, recent_start: int, cycle: int) -> np.ndarray:
    """``seasonal`` with each recent value replaced by its mean over earlier cycles.

    For ``t >= recent_start`` the result is the mean of ``seasonal[t - k * cycle]``
    over ``k >= 1`` with ``t - k * cycle < recent_start``; positions with no such
    lag keep their fitted value.
    """
    out = np.asarray(seasonal, dtype=float).copy()
    for t in range(recent_start, len(out)):
        lags = np.arange(t - cycle, -1, -cycle)
        lags = lags[lags < recent_start]
        if lags.size:
            out[t] = float(seasonal[lags].mean())
    return out


def years_of(points: int, freq: str, cfg: Config) -> int:
    """Whole annual cycles covered by ``points`` periods."""
    return int(points // cycle_length(freq, cfg))


def run(pre: Preprocessed, cfg: Config | None = None) -> Evidence:
    """Seasonality evidence for a preprocessed series."""
    cfg = cfg if cfg is not None else get_config()
    early = skip_if_no_windows(CHECK, pre, cfg)
    if early is not None:
        return early
    windows = pre.windows
    assert windows is not None
    conf = cfg["checks"]["seasonality"]
    freq = pre.series.freq
    word = period_word(freq)
    book = NumberBook(cfg)
    work = pre.work.to_numpy(dtype=float)
    n, need = len(work), int(conf["annual_min_points"][freq])
    if n < need:
        book.put("skip_code", "history")
        years = book.count("min_years", years_of(need, freq, cfg))
        return skipped(
            CHECK,
            f"annual seasonality needs at least {years} years of history "
            f"({book.count('min_points', need)} {word}s; this series has "
            f"{book.count('n_points', n)})",
            book,
        )
    missing = int((~np.isfinite(work)).sum())
    if missing:
        book.put("skip_code", "gaps")
        return skipped(
            CHECK,
            f"the series has {book.count('n_missing', missing)} missing "
            f"{plural(missing, word)}; fill short gaps first",
            book,
        )

    stats, transform = windows.stats, windows.transform
    recent = work[windows.recent.slice]
    rise = float(recent.mean()) - stats.median
    min_mads = float(conf["excess_min_mads"])
    if abs(rise) < min_mads * stats.mad:
        book.put("skip_code", "no_rise")
        k = book.stat("excess_min_mads", min_mads)
        return skipped(
            CHECK,
            f"no rise to explain (the recent mean is within {k} MAD of the baseline median)",
            book,
        )

    cycle = cycle_length(freq, cfg)
    fitted = annual_component(work, freq, cfg)
    seasonal = seasonal_expectation(fitted, windows.recent.start_idx, cycle)
    seasonal_rise = float(seasonal[windows.recent.slice].mean()) - float(
        seasonal[windows.baseline.slice].mean()
    )
    share = float(np.clip(seasonal_rise / rise, 0.0, 1.0))
    adjusted = work - seasonal
    adj_stats = baseline_of(adjusted, windows, cfg)
    trend = assess_trend(adjusted, windows, adj_stats, transform, cfg)
    shift = detect_shift(adjusted, windows, adj_stats, freq, cfg)
    adj_trend = trend is not None and trend.supports_trend
    adj_shift = shift is not None and shift.sustained
    left_over = residual_with_change(
        "up" if rise > 0 else "down",
        trend.direction if adj_trend and trend is not None else None,
        shift.direction if adj_shift and shift is not None else None,
    )
    explained = share >= float(cfg["verdict"]["seasonal_explained_min"])
    seasonal_call = explained and not left_over

    book.put("adjusted_trend", adj_trend)
    book.put("adjusted_trend_direction", trend.direction if trend is not None else None)
    book.put("adjusted_level_shift", adj_shift)
    book.put("adjusted_level_shift_direction", shift.direction if shift is not None else None)
    book.put("residual_with_change", left_over)
    change = "rise" if rise > 0 else "drop"
    level = book.value("recent_level", chart_value(float(recent.mean()), transform, cfg))
    median = book.value("baseline_median", chart_value(stats.median, transform, cfg))
    head = (
        f"The annual pattern accounts for {book.pct('seasonal_share_pct', share)} of the recent "
        f"{change} (recent level {level} vs baseline median {median})"
    )
    tail = _adjusted_text(trend, shift, book, word)
    if (adj_trend or adj_shift) and not left_over:
        tail += f", which runs against the {change}, so the annual pattern explains all of it"
    if seasonal_call:
        stance = "supports_seasonal"
        summary = f"Seasonal: {head}; after removing it, {tail}."
    else:
        stance = "neutral"
        need_text = book.pct("explained_min_pct", float(cfg["verdict"]["seasonal_explained_min"]))
        why = "is below" if not explained else "reaches"
        summary = (
            f"Not just seasonal: {head}, which {why} the {need_text} bar; "
            f"after removing it, {tail}."
        )

    ts = timestamps(pre)
    expected = [
        {"ts": iso(ts[i]), "value": chart_value(stats.median + seasonal[i], transform, cfg)}
        for i in range(max(0, n - cycle), n)
    ]
    annotation = make_annotation(
        lines=[{"label": "baseline + annual pattern", "points": expected}],
        spans=[recent_span(pre)],
    )
    return make_evidence(CHECK, stance, summary, book, annotation)


def _adjusted_text(
    trend: TrendResult | None, shift: ShiftResult | None, book: NumberBook, word: str
) -> str:
    """Describe persistence and level shift on the seasonally adjusted series."""
    if trend is None:
        trend_text = "the trend test could not run"
    else:
        p_text = book.p_value("adjusted_p_value", trend.p_value)
        slope = book.pct("adjusted_slope_pct_per_period", abs(trend.slope_norm))
        if trend.supports_trend:
            trend_text = f"values still trend {trend.direction} {slope} per {word} ({p_text})"
        elif trend.significant:
            trend_text = f"only a negligible trend remains ({p_text}, slope {slope} per {word})"
        else:
            trend_text = f"there is no significant trend ({p_text}, slope {slope} per {word})"
    if shift is None:
        shift_text = "the level-shift search could not run"
    elif shift.sustained:
        shift_text = f"a sustained level shift {shift.direction} remains"
    else:
        shift_text = "no sustained level shift remains"
    return f"{trend_text} and {shift_text}"
