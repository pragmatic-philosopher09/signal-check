"""Confidence score (COPILOT_BRIEF.md section 7).

The verdict says *what* the series looks like; confidence says how much the
checks agree about it and how much the data can be trusted::

    score = 0
    +1 per non-skipped check whose stance agrees with the verdict
    -1 per non-skipped check whose stance contradicts it
    -1 if history < history_multiple x minimum history
    -1 if the low-count flag is active
    -1 if seasonality was skipped and freq != "D" (annual seasonality not ruled out)
    -1 if the source is truncated or has imputed points in the recent window
    high if score >= conf_high, medium if score >= conf_medium, else low
    INCONCLUSIVE is always low.

Agreement table (stances not listed for a verdict are neutral, i.e. 0):

=========  =====================  ================================================
Verdict    Agrees                 Contradicts
=========  =====================  ================================================
TREND      trend, same direction  fluke, no change, seasonal, trend opposite way
FLUKE      fluke                  trend, seasonal
SEASONAL   seasonal               fluke
NO_CHANGE  no change              trend, fluke, seasonal
=========  =====================  ================================================

"No change" from a single check is neutral for FLUKE (a spike that is not a
trend is exactly a fluke). For SEASONAL a raw-series "trend" is neutral: a
seasonal rise *is* a rise in the raw data, and the seasonality check has already
re-run the trend and level-shift tests on the seasonally adjusted series.

The seasonality penalty applies only when the check could not run (history or
gaps); a skip because there was "no rise to explain" rules seasonality out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from signalcheck.config import Config, get_config
from signalcheck.models import Evidence, Preprocessed

AGREES: dict[str, frozenset[str]] = {
    "TREND": frozenset({"supports_trend"}),
    "FLUKE": frozenset({"supports_fluke"}),
    "SEASONAL": frozenset({"supports_seasonal"}),
    "NO_CHANGE": frozenset({"supports_no_change"}),
}
CONTRADICTS: dict[str, frozenset[str]] = {
    "TREND": frozenset({"supports_fluke", "supports_no_change", "supports_seasonal"}),
    "FLUKE": frozenset({"supports_trend", "supports_seasonal"}),
    "SEASONAL": frozenset({"supports_fluke"}),
    "NO_CHANGE": frozenset({"supports_trend", "supports_fluke", "supports_seasonal"}),
}
# Seasonality skip codes meaning "could not rule out annual seasonality".
SEASONALITY_UNTESTED: frozenset[str] = frozenset({"history", "gaps"})


@dataclass(frozen=True)
class ConfidenceScore:
    """Confidence level, the integer score and each contribution ``(reason, delta)``."""

    level: str
    score: int
    contributions: tuple[tuple[str, int], ...]


def stance_delta(label: str, direction: str | None, ev: Evidence) -> int:
    """+1 if ``ev`` agrees with the verdict, -1 if it contradicts it, else 0."""
    if ev.stance == "skipped" or label not in AGREES:
        return 0
    if label == "TREND" and ev.stance == "supports_trend":
        claimed = ev.numbers.get("direction")
        if claimed in ("up", "down") and direction is not None and claimed != direction:
            return -1
        return 1
    if ev.stance in AGREES[label]:
        return 1
    if ev.stance in CONTRADICTS[label]:
        return -1
    return 0


def recent_imputed(pre: Preprocessed) -> bool:
    """Any imputed point (gap fill, Trends ``<1``) inside the recent window."""
    if pre.windows is None or "imputed" not in pre.series.points.columns:
        return False
    flags = pre.series.points["imputed"].to_numpy(dtype=bool)[pre.windows.recent.slice]
    return bool(np.any(flags))


def penalties(
    evidence: Sequence[Evidence], pre: Preprocessed, cfg: Config
) -> list[tuple[str, int]]:
    """Data-quality penalties, each -1 (see module docstring)."""
    out: list[tuple[str, int]] = []
    multiple = float(cfg["confidence"]["history_multiple"])
    if pre.history.n_observed < multiple * pre.history.min_required:
        out.append(("short history (below the minimum-history multiple)", -1))
    by_name = {ev.check: ev for ev in evidence}
    low = by_name.get("low_count")
    if low is not None and low.stance != "skipped" and low.numbers.get("active"):
        out.append(("low counts", -1))
    seas = by_name.get("seasonality")
    if (
        pre.series.freq != "D"
        and seas is not None
        and seas.stance == "skipped"
        and seas.numbers.get("skip_code") in SEASONALITY_UNTESTED
    ):
        out.append(("annual seasonality could not be ruled out", -1))
    if pre.series.meta.get("truncated_before") is not None or recent_imputed(pre):
        out.append(("truncated source or imputed points in the recent window", -1))
    return out


def level_for(score: int, cfg: Config) -> str:
    """Map a score to ``high`` / ``medium`` / ``low`` using the config cut-offs."""
    conf = cfg["confidence"]
    if score >= int(conf["conf_high"]):
        return "high"
    if score >= int(conf["conf_medium"]):
        return "medium"
    return "low"


def score_confidence(
    label: str,
    direction: str | None,
    evidence: Sequence[Evidence],
    pre: Preprocessed,
    cfg: Config | None = None,
) -> ConfidenceScore:
    """Score the verdict ``label``/``direction`` against the evidence and data quality."""
    cfg = cfg if cfg is not None else get_config()
    contributions: list[tuple[str, int]] = []
    for ev in evidence:
        delta = stance_delta(label, direction, ev)
        if delta:
            word = "agrees" if delta > 0 else "contradicts"
            contributions.append((f"{ev.check} {word}", delta))
    contributions.extend(penalties(evidence, pre, cfg))
    score = sum(delta for _, delta in contributions)
    level = "low" if label == "INCONCLUSIVE" else level_for(score, cfg)
    return ConfidenceScore(level, score, tuple(contributions))
