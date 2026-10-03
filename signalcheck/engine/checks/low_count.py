"""Low-count check (COPILOT_BRIEF.md section 6.6).

Statistical meaning: with only a handful of events per bucket, is the recent
rate genuinely different from the baseline rate, or within Poisson noise?

Applies to ``count`` series only, and only when the baseline mean per bucket is
below ``low_count_threshold``. Treating counts as Poisson with rates ``l0``
(baseline, ``n0`` buckets) and ``l1`` (recent, ``n1`` buckets), the recent total
``k1`` conditional on the overall total ``N = k0 + k1`` is
``Binomial(N, q)`` with ``q = n1*l1 / (n1*l1 + n0*l0)``. An exact (Clopper-Pearson)
interval for ``q`` from :func:`scipy.stats.binomtest` maps to an interval for the
rate ratio ``l1/l0 = q/(1-q) * n0/n1``. If that ``ci_level`` interval includes 1
there are too few events to call a change.

Stance: interval includes 1 -> neutral with ``too_few_to_call``; interval excludes
1 -> supports a trend in the ratio's direction. When active it always lowers
confidence (applied by the confidence score, not here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

from signalcheck.config import Config, get_config
from signalcheck.engine.checks.common import (
    NumberBook,
    make_annotation,
    make_evidence,
    period_word,
    plural,
    recent_span,
    skip_if_no_windows,
    skipped,
    timestamps,
)
from signalcheck.models import Evidence, Preprocessed

CHECK = "low_count"


@dataclass(frozen=True)
class RateRatio:
    """Recent/baseline Poisson rate ratio with an exact conditional interval.

    ``ratio``/``upper`` are ``inf`` when the baseline has no events.
    """

    k_recent: int
    n_recent: int
    k_baseline: int
    n_baseline: int
    ratio: float
    lower: float
    upper: float

    @property
    def includes_one(self) -> bool:
        """The interval is consistent with an unchanged rate."""
        return self.lower <= 1.0 <= self.upper


def _odds_to_ratio(q: float, n_recent: int, n_baseline: int) -> float:
    """Map the binomial share ``q`` to the rate ratio ``q/(1-q) * n0/n1``."""
    if q >= 1.0:
        return math.inf
    return q / (1.0 - q) * n_baseline / n_recent


def rate_ratio(k1: int, n1: int, k0: int, n0: int, level: float) -> RateRatio:
    """Exact conditional rate-ratio interval for ``k1`` events in ``n1`` vs ``k0`` in ``n0``."""
    total = k0 + k1
    result = binomtest(k1, total, n1 / (n0 + n1))
    ci = result.proportion_ci(confidence_level=level, method="exact")
    ratio = math.inf if k0 == 0 else (k1 / n1) / (k0 / n0)
    return RateRatio(
        k_recent=k1,
        n_recent=n1,
        k_baseline=k0,
        n_baseline=n0,
        ratio=ratio,
        lower=_odds_to_ratio(float(ci.low), n1, n0),
        upper=_odds_to_ratio(float(ci.high), n1, n0),
    )


def run(pre: Preprocessed, cfg: Config | None = None) -> Evidence:
    """Low-count evidence for a preprocessed series."""
    cfg = cfg if cfg is not None else get_config()
    early = skip_if_no_windows(CHECK, pre, cfg)
    if early is not None:
        return early
    windows = pre.windows
    assert windows is not None
    conf = cfg["checks"]["low_count"]
    series = pre.series
    word = period_word(series.freq)
    book = NumberBook(cfg)
    book.put("active", False)
    if series.scale != "count":
        return skipped(CHECK, f"only applies to count series (this series is {series.scale})", book)

    values = series.points["value"].to_numpy(dtype=float)
    base = values[windows.baseline.slice]
    recent = values[windows.recent.slice]
    base, recent = base[np.isfinite(base)], recent[np.isfinite(recent)]
    threshold = float(conf["low_count_threshold"])
    mean_base = float(base.mean())
    mean_text = book.value("baseline_mean", mean_base)
    threshold_text = book.value("low_count_threshold", threshold)
    if mean_base >= threshold:
        return skipped(
            CHECK,
            f"counts are high enough (baseline averages {mean_text} per {word}, "
            f"threshold {threshold_text})",
            book,
        )
    k0, k1 = int(np.rint(base).sum()), int(np.rint(recent).sum())
    if k0 + k1 == 0 or len(recent) == 0:
        return skipped(CHECK, "no events in the baseline or recent window", book)

    rr = rate_ratio(k1, len(recent), k0, len(base), float(conf["ci_level"]))
    book.numbers["active"] = True
    book.put("too_few_to_call", rr.includes_one)
    direction = None if rr.includes_one else ("up" if rr.lower > 1.0 else "down")
    book.put("direction", direction)

    counts = (
        f"{book.count('k_recent', k1)} {plural(k1, 'event')} in "
        f"{book.count('n_recent', rr.n_recent)} recent {plural(rr.n_recent, word)} vs "
        f"{book.count('k_baseline', k0)} in {book.count('n_baseline', rr.n_baseline)} "
        f"baseline {plural(rr.n_baseline, word)}"
    )
    level = book.pct("ci_level_pct", float(conf["ci_level"]))
    if math.isinf(rr.ratio):
        ratio_text = (
            f"the baseline has no events, so the rate ratio is unbounded "
            f"({level} CI lower bound {book.stat('ratio_lower', rr.lower)})"
        )
    else:
        upper = "unbounded" if math.isinf(rr.upper) else book.stat("ratio_upper", rr.upper)
        ratio_text = (
            f"the recent rate is {book.stat('rate_ratio', rr.ratio)}x the baseline rate "
            f"({level} CI {book.stat('ratio_lower', rr.lower)} to {upper})"
        )
    head = (
        f"Low counts: the baseline averages {mean_text} events per {word} "
        f"(below {threshold_text}); {ratio_text}, from {counts}"
    )
    one = book.count("null_ratio", 1)
    if rr.includes_one:
        stance = "neutral"
        summary = f"{head}. The interval includes {one}, so there are too few events to call."
    else:
        stance = "supports_trend"
        side = "above" if direction == "up" else "below"
        summary = f"{head}. The interval lies {side} {one}, so the rate really changed."

    ts = timestamps(pre)
    base_rate = round(k0 / len(base), int(cfg["display"]["value_decimals"]))
    recent_rate = round(k1 / len(recent), int(cfg["display"]["value_decimals"]))
    annotation = make_annotation(
        segments=[
            {
                "label": "baseline mean rate",
                "start": ts[0].date().isoformat(),
                "end": windows.baseline.end.date().isoformat(),
                "start_value": base_rate,
                "end_value": base_rate,
            },
            {
                "label": "recent mean rate",
                "start": windows.recent.start.date().isoformat(),
                "end": windows.recent.end.date().isoformat(),
                "start_value": recent_rate,
                "end_value": recent_rate,
            },
        ],
        spans=[recent_span(pre)],
    )
    return make_evidence(CHECK, stance, summary, book, annotation)
