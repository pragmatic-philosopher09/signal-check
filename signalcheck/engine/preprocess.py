"""Preprocessing (COPILOT_BRIEF.md section 5).

Turns an adapter's raw :class:`~signalcheck.models.Series` into a regular series
plus the baseline/recent windows and robust baseline statistics that every check
uses. Nothing is interpolated silently: gaps stay NaN (with a caveat) unless a
check explicitly asks for :func:`fill_gaps`, which flags what it filled.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from signalcheck.config import Config, get_config
from signalcheck.models import (
    BaselineStats,
    HistoryCheck,
    Preprocessed,
    Series,
    WindowBounds,
    Windows,
)

PERIOD_WORDS: dict[str, str] = {"D": "day", "W": "week", "M": "month"}
PARTIAL_RULES: tuple[str, ...] = ("fetched_at", "upload_date")


def _plural(count: int, word: str) -> str:
    """``"1 day"`` / ``"3 days"``."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def to_utc_naive(value: Any) -> pd.Timestamp:
    """Convert a datetime/date/ISO string to a naive UTC timestamp.

    Naive inputs are assumed to already be UTC; aware inputs are converted.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.as_unit("ns")


def week_anchor(ts: pd.Series) -> int:
    """Most common weekday (Mon=0) among timestamps; ties go to the earliest weekday.

    Weekly sources label weeks by different start days (Google Trends uses
    Sundays), so the anchor is taken from the data rather than assumed.
    """
    counts = ts.dt.weekday.value_counts()
    best = int(counts.max())
    return int(min(int(day) for day, count in counts.items() if count == best))


def period_start(ts: pd.Series, freq: str, anchor: int = 0) -> pd.Series:
    """Snap timestamps to the start of their ``freq`` period.

    D: midnight; W: the latest ``anchor`` weekday on or before the date; M: the
    first day of the month.
    """
    day = ts.dt.normalize()
    if freq == "D":
        return day
    if freq == "W":
        offset = (day.dt.weekday - anchor) % 7
        return day - pd.to_timedelta(offset, unit="D")
    if freq == "M":
        return day - pd.to_timedelta(day.dt.day - 1, unit="D")
    raise ValueError(f"unsupported freq {freq!r}")


def period_end(start: pd.Timestamp, freq: str) -> pd.Timestamp:
    """Exclusive end of the period starting at ``start`` (the next period's start)."""
    if freq == "D":
        return start + pd.Timedelta(days=1)
    if freq == "W":
        return start + pd.Timedelta(days=7)
    if freq == "M":
        return start + pd.offsets.MonthBegin(1)
    raise ValueError(f"unsupported freq {freq!r}")


def regular_index(first: pd.Timestamp, last: pd.Timestamp, freq: str) -> pd.DatetimeIndex:
    """Every period start from ``first`` to ``last`` inclusive."""
    step = {"D": "D", "W": "7D", "M": "MS"}[freq]
    return pd.date_range(first, last, freq=step, name="ts").as_unit("ns")


def sort_and_dedupe(points: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, int]:
    """Snap ``ts`` to period starts, sort, and keep the last row per period.

    Returns the cleaned frame and the number of rows dropped as duplicates.
    """
    df = points.copy()
    df["ts"] = pd.to_datetime(df["ts"]).astype("datetime64[ns]")
    if df.empty:
        return df.reset_index(drop=True), 0
    anchor = week_anchor(df["ts"]) if freq == "W" else 0
    df["ts"] = period_start(df["ts"], freq, anchor)
    df = df.sort_values("ts", kind="stable")
    deduped = df.drop_duplicates(subset="ts", keep="last")
    return deduped.reset_index(drop=True), len(df) - len(deduped)


def is_partial_last_period(
    last_start: pd.Timestamp, freq: str, meta: dict[str, Any]
) -> tuple[bool, str | None]:
    """Decide whether the last bucket is an incomplete period.

    Live rule (``meta["partial_rule"] == "fetched_at"``, the default): incomplete
    when the period has not ended by ``fetched_at`` (UTC), i.e. its last instant
    is at or after ``fetched_at``. Upload rule (``"upload_date"``): incomplete when
    the period contains ``meta["upload_date"]``. Either way,
    ``meta["last_period_incomplete"]`` (the UI toggle) forces a drop.
    """
    end = period_end(last_start, freq)
    if meta.get("last_period_incomplete"):
        return True, "marked incomplete by the user"
    rule = meta.get("partial_rule", "fetched_at")
    if rule == "fetched_at":
        if meta.get("fetched_at") is None:
            raise ValueError("meta.fetched_at is required to detect a partial last period")
        fetched_at = to_utc_naive(meta["fetched_at"])
        if end > fetched_at:
            return (
                True,
                f"it had not ended when the data was fetched ({fetched_at:%Y-%m-%d %H:%M} UTC)",
            )
        return False, None
    if rule == "upload_date":
        if meta.get("upload_date") is None:
            raise ValueError("meta.upload_date is required for the upload_date partial rule")
        uploaded = to_utc_naive(meta["upload_date"]).normalize()
        if last_start <= uploaded < end:
            return True, f"it contains the upload date ({uploaded:%Y-%m-%d})"
        return False, None
    raise ValueError(f"meta.partial_rule must be one of {PARTIAL_RULES}, got {rule!r}")


def regularise(points: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, int]:
    """Reindex onto every period between the first and last observation.

    Missing periods become NaN rows (``imputed=False``); returns the frame and the
    number of missing values in ``value`` (absent periods plus explicit NaNs).
    Leading rows without a value are trimmed: before the first observation there
    is no data yet, not a gap.
    """
    df = points.copy()
    observed = np.flatnonzero(df["value"].notna().to_numpy())
    df = df.iloc[observed[0] :] if observed.size else df.iloc[0:0]
    if "imputed" not in df.columns:
        df["imputed"] = False
    df["imputed"] = df["imputed"].astype("boolean").fillna(False).astype(bool)
    df["value"] = df["value"].astype(float)
    if df.empty:
        return df.reset_index(drop=True), 0
    index = regular_index(df["ts"].iloc[0], df["ts"].iloc[-1], freq)
    out = df.set_index("ts").reindex(index)
    out["imputed"] = out["imputed"].astype("boolean").fillna(False).astype(bool)
    out = out.reset_index()
    return out, int(out["value"].isna().sum())


def transform_for(scale: str, cfg: Config) -> str:
    """Working-scale transform: ``log1p`` for count-like scales, else ``identity``."""
    return "log1p" if scale in cfg["preprocess"]["log_scales"] else "identity"


def to_work_scale(values: pd.Series, transform: str) -> pd.Series:
    """Map original values onto the working scale used by the checks."""
    if transform == "log1p":
        if (values.dropna() < 0).any():
            raise ValueError("log1p working scale needs non-negative values")
        return np.log1p(values)
    if transform == "identity":
        return values.copy()
    raise ValueError(f"unknown transform {transform!r}")


def to_original_scale(values: Any, transform: str) -> Any:
    """Inverse of :func:`to_work_scale` (numbers or arrays)."""
    if transform == "log1p":
        return np.expm1(values)
    if transform == "identity":
        return values
    raise ValueError(f"unknown transform {transform!r}")


def recent_window_length(n: int, freq: str, cfg: Config) -> int:
    """``clamp(round(recent_frac * n), recent_min[freq], recent_max[freq])``.

    Rounding is half-up (not Python's banker's rounding) so lengths grow
    monotonically with ``n``.
    """
    pre = cfg["preprocess"]
    raw = math.floor(pre["recent_frac"] * n + 0.5)
    return int(min(max(raw, pre["recent_min"][freq]), pre["recent_max"][freq]))


def robust_baseline(values: pd.Series, cfg: Config) -> BaselineStats:
    """Baseline median and scaled MAD, ignoring NaNs.

    ``mad = mad_scale * median(|x - median|)``; with ``mad_scale = 1.4826`` this
    estimates the standard deviation of normal noise while ignoring spikes. A
    perfectly flat baseline has MAD 0, which would make every deviation infinitely
    significant, so it is replaced by ``mad_floor * max(median, 1)``.
    """
    pre = cfg["preprocess"]
    observed = values.dropna().to_numpy(dtype=float)
    if observed.size == 0:
        raise ValueError("baseline has no observed values")
    median = float(np.median(observed))
    mad = float(pre["mad_scale"] * np.median(np.abs(observed - median)))
    floored = mad == 0.0
    used = float(pre["mad_floor"] * max(median, 1.0)) if floored else mad
    return BaselineStats(
        median=median, mad=used, mad_unfloored=mad, floored=floored, n=observed.size
    )


def check_history(n_observed: int, freq: str, cfg: Config) -> HistoryCheck:
    """Rule R1 input: at least ``min_history[freq]`` observed (non-missing) periods."""
    required = int(cfg["preprocess"]["min_history"][freq])
    if n_observed >= required:
        return HistoryCheck(n_observed, required, True, None)
    word = PERIOD_WORDS[freq]
    reason = (
        f"not enough history: {_plural(n_observed, word)} observed, "
        f"need at least {_plural(required, word)}"
    )
    return HistoryCheck(n_observed, required, False, reason)


def compute_windows(
    ts: pd.Series, original: pd.Series, work: pd.Series, freq: str, transform: str, cfg: Config
) -> Windows:
    """Split a regular series into baseline (all earlier points) and recent windows."""
    n = len(ts)
    n_recent = recent_window_length(n, freq, cfg)
    if n_recent >= n:
        raise ValueError(f"recent window ({n_recent}) leaves no baseline in {n} periods")
    split = n - n_recent

    def bounds(start: int, end: int) -> WindowBounds:
        first, last = ts.iloc[start], ts.iloc[end - 1]
        return WindowBounds(start, end, first, last, period_end(last, freq))

    baseline, recent = bounds(0, split), bounds(split, n)
    return Windows(
        n=n,
        baseline=baseline,
        recent=recent,
        transform=transform,
        stats=robust_baseline(work.iloc[baseline.slice], cfg),
        stats_original=robust_baseline(original.iloc[baseline.slice], cfg),
    )


def preprocess(series: Series, cfg: Config | None = None) -> Preprocessed:
    """Sort, dedupe, drop the partial last period, regularise and window a series.

    Steps (each recorded in ``meta.caveats`` when it changes the data):
    1. snap timestamps to period starts and keep the last value per period;
    2. drop the last bucket if incomplete (see :func:`is_partial_last_period`) and
       store it in ``meta.dropped_partial``;
    3. reindex to a regular grid, leaving gaps as NaN;
    4. check the minimum history, then compute windows and baseline statistics.
    """
    cfg = cfg if cfg is not None else get_config()
    meta: dict[str, Any] = dict(series.meta)
    caveats: list[str] = list(meta.get("caveats", []))
    word = PERIOD_WORDS[series.freq]

    points, n_dupes = sort_and_dedupe(series.points, series.freq)
    if n_dupes:
        caveats.append(
            f"{_plural(n_dupes, 'duplicate row')} for the same {word}; kept the last value."
        )

    meta["dropped_partial"] = None
    if not points.empty:
        last = points.iloc[-1]
        partial, why = is_partial_last_period(last["ts"], series.freq, meta)
        if partial:
            value = None if pd.isna(last["value"]) else float(last["value"])
            meta["dropped_partial"] = {"ts": last["ts"].date().isoformat(), "value": value}
            points = points.iloc[:-1]
            caveats.append(
                f"Dropped the last {word} ({last['ts']:%Y-%m-%d}) because {why}; "
                "it is shown greyed out."
            )

    regular, n_gaps = regularise(points, series.freq)
    if n_gaps:
        caveats.append(f"{_plural(n_gaps, f'missing {word}')} left as gaps (not interpolated).")

    meta["caveats"] = caveats
    out = replace(series, points=regular, meta=meta)
    transform = transform_for(series.scale, cfg)
    work = to_work_scale(regular["value"], transform)
    work.index = pd.DatetimeIndex(regular["ts"])

    history = check_history(int(regular["value"].notna().sum()), series.freq, cfg)
    windows: Windows | None = None
    if history.sufficient:
        windows = compute_windows(
            regular["ts"],
            regular["value"],
            work.reset_index(drop=True),
            series.freq,
            transform,
            cfg,
        )
    return Preprocessed(series=out, work=work, history=history, windows=windows)


def _fillable_gaps(values: pd.Series, max_run: int) -> pd.Series:
    """Boolean mask of NaNs in interior runs of length <= ``max_run``."""
    missing = values.isna().to_numpy()
    mask = np.zeros(len(values), dtype=bool)
    observed = np.flatnonzero(~missing)
    if observed.size < 2 or max_run <= 0:
        return pd.Series(mask, index=values.index)
    i = int(observed[0])
    last = int(observed[-1])
    while i < last:
        if missing[i]:
            j = i
            while missing[j]:
                j += 1
            if j - i <= max_run:
                mask[i:j] = True
            i = j
        else:
            i += 1
    return pd.Series(mask, index=values.index)


def fill_gaps(
    pre: Preprocessed, cfg: Config | None = None, for_checks: str | None = None
) -> Preprocessed:
    """Linearly fill short interior gaps for checks that need a regular series.

    Only runs of at most ``gap_fill_max_consecutive`` missing periods are filled;
    longer runs stay NaN. Filled points get ``imputed=True`` and a caveat (scoped to
    ``for_checks`` when given). Windows and baseline statistics are unchanged (they
    were computed on observed data).
    """
    cfg = cfg if cfg is not None else get_config()
    max_run = int(cfg["preprocess"]["gap_fill_max_consecutive"])
    series = pre.series
    points = series.points.copy()
    values = points["value"]
    mask = _fillable_gaps(values, max_run)
    n_missing = int(values.isna().sum())
    if n_missing == 0:
        return pre
    word = PERIOD_WORDS[series.freq]
    caveats = series.caveats
    n_filled = int(mask.sum())
    if n_filled:
        interpolated = values.interpolate(method="linear", limit_area="inside")
        points.loc[mask, "value"] = interpolated[mask]
        points.loc[mask, "imputed"] = True
        caveats.append(
            f"Filled {_plural(n_filled, f'missing {word}')} by linear interpolation"
            f"{f' for the {for_checks}' if for_checks else ''} (gaps of at most {max_run}); "
            "these points are marked imputed."
        )
    if n_missing - n_filled:
        caveats.append(
            f"{_plural(n_missing - n_filled, f'missing {word}')} in longer gaps left unfilled."
        )
    meta = {**series.meta, "caveats": caveats}
    new_series = replace(series, points=points, meta=meta)
    work = to_work_scale(points["value"], transform_for(series.scale, cfg))
    work.index = pre.work.index
    return replace(pre, series=new_series, work=work)
