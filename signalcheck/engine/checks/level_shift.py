"""Level shift check (COPILOT_BRIEF.md section 6.5).

Statistical meaning: did the series move to a new, stable level in or just
before the recent window, and has it stayed there?

The whole series is robust-standardised, ``(x - baseline_median) / MAD`` on the
working scale, and segmented by PELT with an L2 (change-in-mean) cost,
``min_size = min_persist[freq]`` and penalty ``pen_beta * log(n)``; the penalty
grows with ``n`` like BIC, so pure noise rarely yields a change point. The last
change point at or after ``recent start - recent_tolerance[freq]`` is examined:
the medians of everything before it and everything after it give the old and new
levels (so a short transient segment just before the change point cannot pose as
the established level), and the new level *holds* when at least ``min_persist`` periods
follow and at least ``hold_fraction`` of them sit on the new side of the midpoint
between the two medians. Medians make a lone spike inside a segment irrelevant.

Stance: a held shift supports a trend (with direction); a change point whose new
level did not hold is neutral; no recent change point supports "no change".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

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
from signalcheck.engine.pelt import pelt
from signalcheck.models import BaselineStats, Evidence, Preprocessed, Windows

CHECK = "level_shift"


@dataclass(frozen=True)
class ShiftResult:
    """PELT segmentation and the recent change point (positions index the series).

    ``cp`` is the first position of the new segment, or ``None`` when no change
    point falls in or just before the recent window. Medians are on the working
    scale; ``share_new_side`` is the fraction of points from ``cp`` onwards on the
    new side of the midpoint.
    """

    penalty: float
    change_points: tuple[int, ...]
    cp: int | None
    before_median: float
    after_median: float
    n_post: int
    share_new_side: float
    held: bool

    @property
    def direction(self) -> str | None:
        """``up``/``down`` for a recent change point, else ``None``."""
        if self.cp is None or self.after_median == self.before_median:
            return None
        return "up" if self.after_median > self.before_median else "down"

    @property
    def sustained(self) -> bool:
        """A recent change point whose new level has held."""
        return self.cp is not None and self.held and self.direction is not None


def pelt_change_points(z: np.ndarray, min_size: int, penalty: float) -> list[int]:
    """Change points (first index of each new segment) from PELT with an L2 cost.

    Uses the pure-numpy :func:`signalcheck.engine.pelt.pelt`, identical to
    ``ruptures.Pelt(model="l2", jump=1)`` but installable in the browser (Pyodide).
    """
    return [int(b) for b in pelt(z, min_size, penalty)[:-1]]


def detect_shift(
    values: np.ndarray, windows: Windows, stats: BaselineStats, freq: str, cfg: Config
) -> ShiftResult | None:
    """Run the level-shift analysis on working-scale ``values``; ``None`` if any are missing."""
    conf = cfg["checks"]["level_shift"]
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return None
    n = len(values)
    min_size = int(conf["min_persist"][freq])
    penalty = float(conf["pen_beta"]) * math.log(n)
    z = (values - stats.median) / stats.mad
    cps = pelt_change_points(z, min_size, penalty)
    cutoff = windows.recent.start_idx - int(conf["recent_tolerance"][freq])
    recent_cps = [c for c in cps if c >= cutoff]
    if not recent_cps:
        return ShiftResult(penalty, tuple(cps), None, math.nan, math.nan, 0, math.nan, False)
    cp = recent_cps[-1]
    # Everything before the change point, so a short transient segment just before
    # it cannot stand in for the established level.
    before = values[:cp]
    after = values[cp:]
    before_med, after_med = float(np.median(before)), float(np.median(after))
    mid = (before_med + after_med) / 2
    new_side = after > mid if after_med >= before_med else after < mid
    share = float(new_side.mean())
    held = len(after) >= min_size and share >= float(conf["hold_fraction"])
    return ShiftResult(penalty, tuple(cps), cp, before_med, after_med, len(after), share, held)


def run(pre: Preprocessed, cfg: Config | None = None, values: np.ndarray | None = None) -> Evidence:
    """Level-shift evidence for a preprocessed series.

    ``values`` optionally replaces ``pre.work`` (same length, working scale), e.g.
    with a seasonally adjusted series; baseline statistics are recomputed on it.
    """
    cfg = cfg if cfg is not None else get_config()
    early = skip_if_no_windows(CHECK, pre, cfg)
    if early is not None:
        return early
    windows = pre.windows
    assert windows is not None
    work = pre.work.to_numpy(dtype=float) if values is None else np.asarray(values, dtype=float)
    stats = windows.stats if values is None else baseline_of(work, windows, cfg)
    freq = pre.series.freq
    result = detect_shift(work, windows, stats, freq, cfg)
    if result is None:
        book = NumberBook(cfg)
        missing = int((~np.isfinite(work)).sum())
        return skipped(
            CHECK,
            f"the series has {book.count('n_missing', missing)} missing "
            f"{plural(missing, period_word(freq))}; fill short gaps first",
            book,
        )
    return _evidence(pre, result, work, cfg)


def _evidence(pre: Preprocessed, result: ShiftResult, work: np.ndarray, cfg: Config) -> Evidence:
    """Summary, numbers and chart marks for a completed level-shift analysis."""
    windows = pre.windows
    assert windows is not None
    ts = timestamps(pre)
    word = period_word(pre.series.freq)
    transform = windows.transform
    conf = cfg["checks"]["level_shift"]
    book = NumberBook(cfg)
    book.put("change_point_dates", [iso(ts[c]) for c in result.change_points])
    book.put("sustained", result.sustained)
    book.put("direction", result.direction)
    book.put("in_recent", result.cp is not None)

    if result.cp is None or result.direction is None:
        summary = (
            "No level shift in or just before the recent window "
            f"(PELT change-point search, penalty {book.stat('penalty', result.penalty)})."
        )
        annotation = make_annotation(spans=[recent_span(pre)])
        return make_evidence(CHECK, "supports_no_change", summary, book, annotation)

    cp = result.cp
    before = chart_value(result.before_median, transform, cfg)
    after = chart_value(result.after_median, transform, cfg)
    date = book.date("change_date", ts[cp])
    medians = (
        f"median {book.value('before_median', before)} before vs "
        f"{book.value('after_median', after)} after"
    )
    n_post = f"{book.count('n_post', result.n_post)} {plural(result.n_post, word)}"
    share = book.pct("share_new_side_pct", result.share_new_side)
    if result.sustained:
        stance = "supports_trend"
        summary = (
            f"Level shifted {result.direction} on {date}: {medians}, and the new level has held "
            f"for {n_post} ({share} of them on the new side of the midpoint)."
        )
    else:
        stance = "neutral"
        need = book.pct("hold_fraction_pct", float(conf["hold_fraction"]))
        summary = (
            f"A change point on {date} ({medians}) did not produce a lasting new level: "
            f"{share} of the {n_post} since sit on the {result.direction} side of the "
            f"midpoint (needs {need})."
        )

    annotation = make_annotation(
        vlines=[{"label": "change point", "ts": iso(ts[cp])}],
        segments=[
            {
                "label": "level before",
                "start": iso(ts[0]),
                "end": iso(ts[cp - 1]),
                "start_value": before,
                "end_value": before,
            },
            {
                "label": "level after",
                "start": iso(ts[cp]),
                "end": iso(ts[len(work) - 1]),
                "start_value": after,
                "end_value": after,
            },
        ],
        spans=[recent_span(pre)],
    )
    return make_evidence(CHECK, stance, summary, book, annotation)
