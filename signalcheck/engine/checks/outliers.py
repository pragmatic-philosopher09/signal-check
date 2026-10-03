"""Outlier check (COPILOT_BRIEF.md section 6.4).

Statistical meaning: which recent observations are extreme relative to the
baseline's typical variation?

For each observed recent period the robust z-score is
``z = (x - baseline_median) / MAD`` on the working scale (MAD scaled to estimate
the standard deviation, so ``|z|`` reads like a normal z-score but is not
distorted by spikes inside the baseline). Points with ``|z| >= outlier_z`` are
flagged; the check reports how many and the largest ``|z|``.

For daily data the pipeline passes the de-weekly-ised series
(:func:`~signalcheck.engine.checks.seasonality.deweekly`) as ``values``: z-scores
and the baseline median/MAD are then computed with the day-of-week pattern
removed, so a regular weekend dip is not mistaken for an extreme day. Displayed
values stay on the observed series.

Stance: no flagged points supports "no change". At most ``isolated_max_points``
flagged points are isolated extremes, which supports a fluke. More flagged points
mean much of the window moved away from the baseline (a trend or level shift),
which this check cannot tell apart, so it is neutral.
"""

from __future__ import annotations

import numpy as np

from signalcheck.config import Config, get_config
from signalcheck.engine.checks.common import (
    NumberBook,
    band,
    baseline_of,
    chart_value,
    highlight,
    make_annotation,
    make_evidence,
    period_word,
    plural,
    recent_span,
    recent_work,
    skip_if_no_windows,
    skipped,
    timestamps,
)
from signalcheck.models import Evidence, Preprocessed

CHECK = "outliers"


def robust_z(values: np.ndarray, median: float, mad: float) -> np.ndarray:
    """Robust z-scores ``(x - median) / mad`` (NaN stays NaN)."""
    return (np.asarray(values, dtype=float) - median) / mad


def run(pre: Preprocessed, cfg: Config | None = None, values: np.ndarray | None = None) -> Evidence:
    """Outlier evidence for a preprocessed series.

    ``values`` optionally replaces ``pre.work`` for the z-scores (same length,
    working scale, e.g. the de-weekly-ised series); baseline statistics are then
    recomputed on it.
    """
    cfg = cfg if cfg is not None else get_config()
    early = skip_if_no_windows(CHECK, pre, cfg)
    if early is not None:
        return early
    windows = pre.windows
    assert windows is not None
    conf = cfg["checks"]["outliers"]
    threshold = float(conf["outlier_z"])
    transform = windows.transform
    word = period_word(pre.series.freq)
    book = NumberBook(cfg)

    recent = recent_work(pre)
    if values is None:
        stats, scored = windows.stats, recent
    else:
        adjusted = np.asarray(values, dtype=float)
        stats, scored = baseline_of(adjusted, windows, cfg), adjusted[windows.recent.slice]
        scored = np.where(np.isfinite(recent), scored, np.nan)
    book.put("deweekly", values is not None)
    z = robust_z(scored, stats.median, stats.mad)
    observed = np.isfinite(z)
    if not observed.any():
        return skipped(CHECK, "no observed values in the recent window", book)
    flagged = [int(i) for i in np.flatnonzero(observed & (np.abs(np.nan_to_num(z)) >= threshold))]
    max_abs = float(np.nanmax(np.abs(z)))
    n_flagged = len(flagged)
    n_obs = int(observed.sum())
    ts = timestamps(pre)
    start = windows.recent.start_idx

    book.put("flagged_dates", [ts[start + i].date().isoformat() for i in flagged])
    recent_text = f"{book.count('n_recent', n_obs)} recent {plural(n_obs, word)}"
    count_text = f"{book.count('n_flagged', n_flagged)} of {recent_text}"
    rule = f"|z| >= {book.stat('outlier_z', threshold)}"
    if values is not None:
        rule += " once the day-of-week pattern is removed"
    max_text = f"largest |z| {book.stat('max_abs_z', max_abs)}"
    if n_flagged == 0:
        stance = "supports_no_change"
        summary = f"No outliers: none of the {recent_text} reach {rule} ({max_text})."
    else:
        i_max = int(flagged[np.argmax(np.abs(z[flagged]))])
        book.put("max_direction", "up" if z[i_max] > 0 else "down")
        peak = chart_value(recent[i_max], transform, cfg)
        where = (
            f"{max_text} on {book.date('max_date', ts[start + i_max])} "
            f"(value {book.value('max_value', peak)} vs baseline median "
            f"{book.value('baseline_median', chart_value(stats.median, transform, cfg))})"
        )
        if n_flagged <= int(conf["isolated_max_points"]):
            stance = "supports_fluke"
            summary = f"Isolated outliers: {count_text} reach {rule}; {where}."
        else:
            stance = "neutral"
            summary = f"Many outliers: {count_text} reach {rule}; {where}."
    book.put("isolated", stance == "supports_fluke")

    half = threshold * stats.mad
    annotation = make_annotation(
        bands=[
            band(
                "outlier threshold",
                windows.recent.start,
                windows.recent.end,
                chart_value(stats.median - half, transform, cfg),
                chart_value(stats.median + half, transform, cfg),
            )
        ],
        points=[
            highlight("outlier", ts[start + i], chart_value(recent[i], transform, cfg))
            for i in flagged
        ],
        spans=[recent_span(pre)],
    )
    return make_evidence(CHECK, stance, summary, book, annotation)
