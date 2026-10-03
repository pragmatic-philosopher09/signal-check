""" "What would change my mind" conditions (COPILOT_BRIEF.md section 8).

Deterministic, 1-``max_conditions`` falsifiable statements per verdict, computed
from the series itself. Thresholds are robust baseline bands built on the
working scale (``median +/- k * MAD``, ``k`` from ``config.yaml``) and mapped back
to the original scale, so every displayed value is a real level of the series.

Each condition is a :class:`Condition`: a plain ``str`` (so ``Verdict`` and JSON
see ordinary text) that also carries ``.numbers``, the rounded values it
displays, recorded with :class:`~signalcheck.engine.checks.common.NumberBook`.
That keeps the numbers contract the narration validator relies on:
``undisplayed_tokens(condition, condition.numbers) == []``.

Wording follows the series frequency (days / weeks / months).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from signalcheck.config import Config, get_config
from signalcheck.engine.checks.common import NumberBook, chart_value, period_word, plural
from signalcheck.engine.checks.seasonality import cycle_length
from signalcheck.engine.verdict import by_check, ran
from signalcheck.models import Evidence, Preprocessed


class Condition(str):
    """A condition sentence that also carries the numbers it displays."""

    numbers: dict[str, Any]

    def __new__(cls, text: str, numbers: dict[str, Any] | None = None) -> Condition:
        obj = super().__new__(cls, text)
        obj.numbers = dict(numbers or {})
        return obj

    def __getnewargs__(self) -> tuple[str, dict[str, Any]]:  # type: ignore[override]
        return (str(self), self.numbers)


Builder = Callable[[NumberBook], str | None]


def _decimals_for(value: float, max_decimals: int = 3) -> int:
    """Fewest decimals (up to ``max_decimals``) that show a config multiplier exactly."""
    for d in range(max_decimals + 1):
        if math.isclose(round(value, d), value, rel_tol=0, abs_tol=1e-12):
            return d
    return max_decimals


def multiplier(book: NumberBook, key: str, value: float) -> str:
    """A config multiplier (``2``, ``3.5``) shown without trailing zeros, and recorded."""
    return book.fixed(key, float(value), _decimals_for(float(value)))


def periods(n: int, freq: str) -> str:
    """``days`` / ``week`` ... for ``n`` periods (the number is formatted separately)."""
    return plural(n, period_word(freq))


def min_persist(freq: str, cfg: Config) -> int:
    """Periods a change must hold before it counts as lasting (``level_shift.min_persist``)."""
    return int(cfg["checks"]["level_shift"]["min_persist"][freq])


def band_edge(pre: Preprocessed, mads: float, cfg: Config) -> float:
    """``baseline median + mads * MAD`` on the working scale, mapped to the original scale."""
    assert pre.windows is not None
    stats = pre.windows.stats
    return chart_value(stats.median + mads * stats.mad, pre.windows.transform, cfg)


def recent_direction(pre: Preprocessed) -> str:
    """``up`` if the recent median sits at or above the baseline median, else ``down``."""
    assert pre.windows is not None
    recent = pre.work.to_numpy(dtype=float)[pre.windows.recent.slice]
    recent = recent[np.isfinite(recent)]
    if recent.size == 0:
        return "up"
    return "up" if float(np.median(recent)) >= pre.windows.stats.median else "down"


def lower_is_meaningful(pre: Preprocessed, edge: float) -> bool:
    """A lower band edge is worth stating unless it is at or below zero on a non-negative series."""
    values = pre.series.points["value"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    non_negative = values.size == 0 or float(values.min()) >= 0
    return edge > 0 or not non_negative


def stay_beyond(pre: Preprocessed, direction: str, cfg: Config, outcome: str) -> Builder:
    """ "If the next N periods all stay above X (median + k MADs), <outcome>."""

    def build(book: NumberBook) -> str | None:
        k = float(cfg["change_my_mind"]["exceed_mads"])
        sign = 1.0 if direction == "up" else -1.0
        raw_edge = band_edge(pre, sign * k, cfg)
        if direction == "down" and not lower_is_meaningful(pre, raw_edge):
            return None
        n = min_persist(pre.series.freq, cfg)
        edge = book.value("threshold", raw_edge)
        side = "above" if direction == "up" else "below"
        op = "+" if direction == "up" else "-"
        return (
            f"If the next {book.count('periods', n)} {periods(n, pre.series.freq)} all stay "
            f"{side} {edge} (baseline median {op} {multiplier(book, 'mads', k)} MADs), "
            f"{outcome}."
        )

    return build


def fall_back(pre: Preprocessed, direction: str, cfg: Config, outcome: str) -> Builder:
    """ "If the next N periods fall back below Y (median + k MADs), <outcome>."""

    def build(book: NumberBook) -> str | None:
        k = float(cfg["change_my_mind"]["fallback_mads"])
        sign = 1.0 if direction == "up" else -1.0
        raw_edge = band_edge(pre, sign * k, cfg)
        if direction == "down" and not lower_is_meaningful(pre, raw_edge):
            return None
        n = min_persist(pre.series.freq, cfg)
        edge = book.value("threshold", raw_edge)
        move = "fall back below" if direction == "up" else "rise back above"
        op = "+" if direction == "up" else "-"
        return (
            f"If the next {book.count('periods', n)} {periods(n, pre.series.freq)} {move} "
            f"{edge} (baseline median {op} {multiplier(book, 'mads', k)} MAD), {outcome}."
        )

    return build


def first_of(*builders: Builder) -> Builder:
    """The first candidate condition that applies."""

    def build(book: NumberBook) -> str | None:
        for candidate in builders:
            text = candidate(book)
            if text is not None:
                return text
        return None

    return build


def unusual_values(pre: Preprocessed, cfg: Config) -> Builder:
    """NO_CHANGE: values beyond ``median +/- outlier_z * MAD`` would be unusual."""

    def build(book: NumberBook) -> str:
        z = float(cfg["checks"]["outliers"]["outlier_z"])
        upper = book.value("upper", band_edge(pre, z, cfg))
        lower_edge = band_edge(pre, -z, cfg)
        z_text = multiplier(book, "outlier_z", z)
        if lower_is_meaningful(pre, lower_edge):
            lower = book.value("lower", lower_edge)
            return f"A value above {upper} or below {lower} would be unusual (|z| ≥ {z_text})."
        return f"A value above {upper} would be unusual (|z| ≥ {z_text})."

    return build


def beyond_last_year(pre: Preprocessed, evidence: Sequence[Evidence], cfg: Config) -> Builder:
    """SEASONAL: the next period beyond last year's same-period value by more than Z."""

    def build(book: NumberBook) -> str | None:
        assert pre.windows is not None
        freq = pre.series.freq
        idx = pre.windows.n - cycle_length(freq, cfg)
        values = pre.series.points["value"].to_numpy(dtype=float)
        work = pre.work.to_numpy(dtype=float)
        if idx < 0 or not (math.isfinite(values[idx]) and math.isfinite(work[idx])):
            return None
        seas = by_check(evidence).get("seasonality")
        falling = seas is not None and float(seas.numbers.get("recent_level", 0.0)) < float(
            seas.numbers.get("baseline_median", 0.0)
        )
        k = float(cfg["change_my_mind"]["seasonal_excess_mads"])
        sign = -1.0 if falling else 1.0
        transform = pre.windows.transform
        last_year = chart_value(work[idx], transform, cfg)
        edge = chart_value(work[idx] + sign * k * pre.windows.stats.mad, transform, cfg)
        word = period_word(freq)
        ts = pre.series.points["ts"].iloc[idx]
        last_text = book.value("last_year_value", last_year)
        date_text = book.date("last_year_date", ts)
        gap = book.value("excess", abs(edge - last_year))
        edge_text = book.value("threshold", edge)
        move, side = ("falls short of", "below") if falling else ("exceeds", "above")
        return (
            f"If the next {word} {move} last year's same-{word} value ({last_text}, {word} of "
            f"{date_text}) by more than {gap} ({side} {edge_text}), something beyond "
            "seasonality is happening."
        )

    return build


def annual_history(pre: Preprocessed, evidence: Sequence[Evidence], cfg: Config) -> Builder:
    """TREND: annual seasonality could not be tested; more history would settle it."""

    def build(book: NumberBook) -> str | None:
        seas = by_check(evidence).get("seasonality")
        if seas is None or seas.numbers.get("skip_code") != "history":
            return None
        freq = pre.series.freq
        need = int(cfg["checks"]["seasonality"]["annual_min_points"][freq]) - len(pre.work)
        if need <= 0:
            return None
        return (
            f"With {book.count('periods_needed', need)} more {periods(need, freq)} of history "
            "the annual-pattern check can run; if the usual yearly pattern explains the "
            "change, it is seasonal rather than a trend."
        )

    return build


def broader_origins(cfg: Config) -> Builder:
    """FLUKE via breadth: the volume spreading beyond one origin would change the call."""

    def build(book: NumberBook) -> str:
        share = float(cfg["checks"]["breadth"]["breadth_top_share_max"])
        return (
            "If the recent volume spreads out so that the top origin's share drops below "
            f"{book.pct('top_share', share)}, it would no longer look like a fluke."
        )

    return build


def more_history(pre: Preprocessed) -> Builder:
    """INCONCLUSIVE via R1: N more periods of history are needed."""

    def build(book: NumberBook) -> str:
        freq = pre.series.freq
        need = pre.history.points_needed
        have = pre.history.n_observed
        return (
            f"Need {book.count('periods_needed', need)} more {periods(need, freq)} of data to "
            f"decide ({book.count('n_observed', have)} observed, at least "
            f"{book.count('min_history', pre.history.min_required)} required)."
        )

    return build


def contributor_data(pre: Preprocessed, evidence: Sequence[Evidence]) -> Builder:
    """INCONCLUSIVE: contributor data would let the breadth check run (count series only)."""

    def build(book: NumberBook) -> str | None:
        if pre.series.scale != "count" or ran(by_check(evidence).get("breadth")):
            return None
        return (
            "Contributor data (distinct authors or sites per period) would show whether a "
            "single origin drives the change."
        )

    return build


def builders_for(
    label: str,
    direction: str | None,
    rule: str,
    pre: Preprocessed,
    evidence: Sequence[Evidence],
    cfg: Config,
) -> list[Builder]:
    """The ordered candidate conditions for a verdict."""
    if rule == "R1" or pre.windows is None:
        return [more_history(pre)]
    d = direction or recent_direction(pre)
    if label == "FLUKE":
        out = [stay_beyond(pre, d, cfg, "this becomes a trend")]
        if ran(by_check(evidence).get("breadth")):
            out.append(broader_origins(cfg))
        return out
    if label == "TREND":
        return [
            fall_back(pre, d, cfg, "this was a fluke"),
            annual_history(pre, evidence, cfg),
        ]
    if label == "SEASONAL":
        return [beyond_last_year(pre, evidence, cfg)]
    if label == "NO_CHANGE":
        return [
            unusual_values(pre, cfg),
            first_of(
                stay_beyond(pre, d, cfg, "that would be a trend"),
                stay_beyond(pre, "up", cfg, "that would be a trend"),
            ),
        ]
    return [
        first_of(
            stay_beyond(pre, d, cfg, "it is a trend"),
            stay_beyond(pre, "up", cfg, "it is a trend"),
        ),
        fall_back(pre, d, cfg, "the change was a fluke"),
        contributor_data(pre, evidence),
    ]


def change_my_mind(
    label: str,
    direction: str | None,
    rule: str,
    pre: Preprocessed,
    evidence: Sequence[Evidence],
    cfg: Config | None = None,
) -> list[Condition]:
    """Up to ``max_conditions`` conditions that would overturn the verdict."""
    cfg = cfg if cfg is not None else get_config()
    limit = int(cfg["change_my_mind"]["max_conditions"])
    out: list[Condition] = []
    for build in builders_for(label, direction, rule, pre, evidence, cfg):
        book = NumberBook(cfg)
        text = build(book)
        if text is not None:
            out.append(Condition(text, book.numbers))
        if len(out) >= limit:
            break
    return out


__all__ = ["Condition", "change_my_mind"]
