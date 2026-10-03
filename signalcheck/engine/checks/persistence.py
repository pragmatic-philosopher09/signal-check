"""Persistence check (COPILOT_BRIEF.md section 6.1).

Statistical meaning: is there a monotonic trend through the recent window that
is both statistically significant and practically large?

- The test runs on the recent window plus the last ``persistence_context``
  baseline points, so even a short recent window has enough points for power.
- Significance comes from the Mann-Kendall rank test. With ``n >= hamed_rao_min_n``
  the Hamed-Rao modification is used: it inflates the variance of the S statistic
  for serial (auto)correlation, so smooth noise is not mistaken for a trend.
  Smaller samples use the original test.
- Size comes from the Theil-Sen slope (median of pairwise slopes, robust to
  spikes), normalised per period: on the log1p working scale ``expm1(slope)`` is
  the fractional change per period; on an identity scale the slope is divided by
  the baseline level ``max(|baseline median|, slope_level_floor)``.
- It also counts the consecutive most-recent periods beyond
  ``baseline_median +/- band_mads * MAD`` in the trend's direction.

Supports a trend when ``p < mk_alpha`` and ``|normalised slope| >= min_slope``.
A non-significant test supports "no change"; a significant but tiny slope is
neutral.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymannkendall as mk
from scipy import stats as sps

from signalcheck.config import Config, get_config
from signalcheck.engine.checks.common import (
    NumberBook,
    band,
    baseline_of,
    chart_value,
    highlight,
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
from signalcheck.models import BaselineStats, Evidence, Preprocessed, Windows

CHECK = "persistence"


@dataclass(frozen=True)
class TrendResult:
    """Trend test on the recent window plus context.

    ``start_idx`` is the first tested position; ``n`` the observed points tested;
    ``slope``/``intercept`` the Theil-Sen line on the working scale with x counted
    in periods from ``start_idx``; ``slope_norm`` the normalised slope (fraction
    per period); ``run_length`` the consecutive most-recent periods beyond the
    baseline band in ``direction``.
    """

    start_idx: int
    n: int
    test: str
    p_value: float
    slope: float
    intercept: float
    slope_norm: float
    significant: bool
    large_enough: bool
    direction: str
    run_length: int

    @property
    def supports_trend(self) -> bool:
        """Significant and practically large."""
        return self.significant and self.large_enough


def segment_start(windows: Windows, cfg: Config) -> int:
    """First position tested: the recent window minus ``persistence_context`` points."""
    context = min(int(cfg["checks"]["persistence"]["persistence_context"]), windows.baseline.length)
    return windows.recent.start_idx - context


def normalised_slope(slope: float, stats: BaselineStats, transform: str, cfg: Config) -> float:
    """Slope per period as a fraction of the baseline level (see module docstring)."""
    if transform == "log1p":
        return float(np.expm1(slope))
    floor = float(cfg["checks"]["persistence"]["slope_level_floor"])
    return slope / max(abs(stats.median), floor)


def mann_kendall_p(y: np.ndarray, cfg: Config) -> tuple[float, str]:
    """Two-sided Mann-Kendall p-value and the test used.

    Hamed-Rao's autocorrelation-corrected variance is used when
    ``len(y) >= hamed_rao_min_n`` and it is valid; otherwise the original test.
    """
    per = cfg["checks"]["persistence"]
    if np.ptp(y) == 0:
        return 1.0, "constant"
    alpha = float(per["mk_alpha"])
    if len(y) >= int(per["hamed_rao_min_n"]):
        # The autocorrelation correction can make the variance negative (NaN p);
        # fall back to the uncorrected test rather than inventing a p-value.
        with np.errstate(invalid="ignore", divide="ignore"):
            p = float(mk.hamed_rao_modification_test(y, alpha=alpha).p)
        if np.isfinite(p):
            return p, "hamed_rao"
    p = float(mk.original_test(y, alpha=alpha).p)
    return (p if np.isfinite(p) else 1.0), "original"


def exceedance_run(values: np.ndarray, center: float, half_width: float, direction: str) -> int:
    """Consecutive most-recent values beyond ``center +/- half_width`` in ``direction``.

    A missing value ends the run.
    """
    if direction not in ("up", "down"):
        return 0
    run = 0
    for v in values[::-1]:
        beyond = v > center + half_width if direction == "up" else v < center - half_width
        if not (np.isfinite(v) and beyond):
            break
        run += 1
    return run


def assess_trend(
    values: np.ndarray, windows: Windows, stats: BaselineStats, transform: str, cfg: Config
) -> TrendResult | None:
    """Run the trend test on working-scale ``values``; ``None`` if too few observed points.

    Missing values are skipped (never interpolated); Theil-Sen uses the true
    period positions of the observed points.
    """
    per = cfg["checks"]["persistence"]
    start = segment_start(windows, cfg)
    seg = np.asarray(values[start:], dtype=float)
    x = np.arange(len(seg), dtype=float)
    ok = np.isfinite(seg)
    if int(ok.sum()) < int(per["min_points"]):
        return None
    y, x = seg[ok], x[ok]
    p_value, test = mann_kendall_p(y, cfg)
    if np.ptp(y) == 0:
        slope, intercept = 0.0, float(y[0])
    else:
        fit = sps.theilslopes(y, x)
        slope, intercept = float(fit.slope), float(fit.intercept)
    slope_norm = normalised_slope(slope, stats, transform, cfg)
    direction = "up" if slope > 0 else "down" if slope < 0 else "flat"
    run = exceedance_run(seg, stats.median, float(per["band_mads"]) * stats.mad, direction)
    return TrendResult(
        start_idx=start,
        n=int(ok.sum()),
        test=test,
        p_value=p_value,
        slope=slope,
        intercept=intercept,
        slope_norm=slope_norm,
        significant=p_value < float(per["mk_alpha"]),
        large_enough=abs(slope_norm) >= float(per["min_slope"]),
        direction=direction,
        run_length=run,
    )


def stance_for(result: TrendResult) -> str:
    """``supports_trend`` / ``neutral`` (significant but tiny) / ``supports_no_change``."""
    if result.supports_trend:
        return "supports_trend"
    if result.significant:
        return "neutral"
    return "supports_no_change"


def run(pre: Preprocessed, cfg: Config | None = None, values: np.ndarray | None = None) -> Evidence:
    """Persistence evidence for a preprocessed series.

    ``values`` optionally replaces ``pre.work`` (same length, working scale), e.g.
    with a de-weekly-ised series; baseline statistics are then recomputed on it.
    """
    cfg = cfg if cfg is not None else get_config()
    early = skip_if_no_windows(CHECK, pre, cfg)
    if early is not None:
        return early
    windows = pre.windows
    assert windows is not None
    work = pre.work.to_numpy(dtype=float) if values is None else np.asarray(values, dtype=float)
    stats = windows.stats if values is None else baseline_of(work, windows, cfg)
    result = assess_trend(work, windows, stats, windows.transform, cfg)
    if result is None:
        book = NumberBook(cfg)
        need = book.count("min_points", int(cfg["checks"]["persistence"]["min_points"]))
        return skipped(CHECK, f"fewer than {need} observed points to test for a trend", book)
    return _evidence(pre, result, stats, work, cfg)


def _evidence(
    pre: Preprocessed, result: TrendResult, stats: BaselineStats, work: np.ndarray, cfg: Config
) -> Evidence:
    """Summary, numbers and chart marks for a completed trend test."""
    windows = pre.windows
    assert windows is not None
    ts = timestamps(pre)
    word = period_word(pre.series.freq)
    transform = windows.transform
    per = cfg["checks"]["persistence"]
    book = NumberBook(cfg)
    book.put("test", result.test)
    book.put("direction", result.direction)
    book.put("significant", result.significant)
    book.put("supports_trend", result.supports_trend)
    book.fixed("slope_norm_pct", 100.0 * result.slope_norm, int(cfg["display"]["pct_decimals"]))

    first, last = ts[result.start_idx], ts[-1]
    window_text = (
        f"{book.count('n_tested', result.n)} {plural(result.n, word)} "
        f"({book.date('start', first)} to {book.date('end', last)})"
    )
    of_level = "" if transform == "log1p" else " of the baseline level"
    slope_text = f"{book.pct('slope_pct_per_period', abs(result.slope_norm))}{of_level} per {word}"
    p_text = book.p_value("p_value", result.p_value)
    verb = {"up": "rose", "down": "fell", "flat": "stayed flat"}[result.direction]

    if result.supports_trend:
        summary = (
            f"Values {verb} {slope_text} over the last {window_text}; "
            f"the Mann-Kendall trend test is significant ({p_text})"
        )
    elif result.significant:
        min_text = book.pct("min_slope_pct", float(per["min_slope"]))
        summary = (
            f"Values {verb} over the last {window_text} ({p_text}), but only {slope_text}, "
            f"below the {min_text} needed to call a trend"
        )
    elif result.direction == "flat":
        summary = f"No significant trend over the last {window_text} ({p_text})"
    else:
        summary = (
            f"No significant trend over the last {window_text}: values drifted "
            f"{result.direction} {slope_text} ({p_text})"
        )

    half = float(per["band_mads"]) * stats.mad
    lower = chart_value(stats.median - half, transform, cfg)
    upper = chart_value(stats.median + half, transform, cfg)
    if result.run_length > 0:
        side = "above" if result.direction == "up" else "below"
        summary += (
            f"; the last {book.count('run_length', result.run_length)} "
            f"{plural(result.run_length, word)} sat {side} the baseline band "
            f"({book.value('band_lower', lower)} to {book.value('band_upper', upper)})"
        )
    else:
        book.put("run_length", 0)

    x_end = len(work) - 1 - result.start_idx
    annotation = make_annotation(
        bands=[band("baseline band", ts[0], ts[-1], lower, upper)],
        segments=[
            {
                "label": "Theil-Sen trend",
                "start": iso(first),
                "end": iso(last),
                "start_value": chart_value(result.intercept, transform, cfg),
                "end_value": chart_value(result.intercept + result.slope * x_end, transform, cfg),
            }
        ],
        points=[
            highlight("beyond baseline band", ts[i], chart_value(work[i], transform, cfg))
            for i in range(len(work) - result.run_length, len(work))
        ],
        spans=[recent_span(pre)],
    )
    return make_evidence(CHECK, stance_for(result), summary + ".", book, annotation)
