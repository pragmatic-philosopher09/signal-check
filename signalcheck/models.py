"""Core data contract (COPILOT_BRIEF.md section 3).

The engine never knows which source a series came from: every adapter returns a
:class:`Series`, preprocessing turns it into a :class:`Preprocessed` bundle
(regular series + :class:`Windows`), checks return :class:`Evidence`, and the
decision table returns a :class:`Verdict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from signalcheck.config import FREQS, SCALES

SOURCES: tuple[str, ...] = ("google_trends", "reddit", "x", "wikipedia", "hackernews", "csv")
STANCES: tuple[str, ...] = (
    "supports_trend",
    "supports_fluke",
    "supports_seasonal",
    "supports_no_change",
    "neutral",
    "skipped",
)
LABELS: tuple[str, ...] = ("TREND", "FLUKE", "SEASONAL", "NO_CHANGE", "INCONCLUSIVE")
DIRECTIONS: tuple[str, ...] = ("up", "down")
CONFIDENCES: tuple[str, ...] = ("low", "medium", "high")
RULE_IDS: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "R6")

REQUIRED_COLUMNS: tuple[str, ...] = ("ts", "value")
OPTIONAL_COLUMNS: tuple[str, ...] = ("contributors", "top_share", "raw_count", "imputed")


def _require_member(name: str, value: object, allowed: tuple[Any, ...]) -> None:
    """Raise ``ValueError`` unless ``value`` is one of ``allowed``."""
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


@dataclass(frozen=True)
class Series:
    """One source's attention series for one query.

    ``points`` has columns ``ts`` (naive UTC timestamps, one per period start) and
    ``value`` (float); optional columns are ``contributors`` (int), ``top_share``
    (largest share of a bucket's items from a single origin, 0-1), ``raw_count``
    (int) and ``imputed`` (bool). ``meta`` carries ``fetched_at``, ``caveats``
    (list[str]), ``resolved_query``, ``dropped_partial`` ({ts, value} | None) and
    ``truncated_before`` (date | None), plus adapter-specific keys.
    """

    source: str
    query: str
    freq: str
    points: pd.DataFrame
    scale: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_member("source", self.source, SOURCES)
        _require_member("freq", self.freq, FREQS)
        _require_member("scale", self.scale, SCALES)
        missing = [c for c in REQUIRED_COLUMNS if c not in self.points.columns]
        if missing:
            raise ValueError(f"points is missing required column(s): {missing}")
        if not pd.api.types.is_datetime64_any_dtype(self.points["ts"]):
            raise ValueError("points.ts must be a datetime64 column")
        if not pd.api.types.is_numeric_dtype(self.points["value"]):
            raise ValueError("points.value must be numeric")

    @property
    def caveats(self) -> list[str]:
        """Caveats recorded so far (empty list if none)."""
        return list(self.meta.get("caveats", []))


@dataclass(frozen=True)
class Evidence:
    """Output of one check: a stance, a one-sentence summary and its numbers.

    ``numbers`` holds every number used in ``summary`` with the same rounding, so
    the narration validator can verify any narrated figure.
    """

    check: str
    stance: str
    summary: str
    numbers: dict[str, Any]
    annotation: dict[str, Any] | None
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        _require_member("stance", self.stance, STANCES)
        if self.stance == "skipped" and not self.skip_reason:
            raise ValueError("a skipped Evidence needs a skip_reason")


@dataclass(frozen=True)
class Verdict:
    """Decision-table result for one series; ``rule_fired`` is the matching row id.

    ``reason`` is a plain-language explanation of why that row matched (the R1
    history shortfall, the R6 conflicting checks, ...). ``window`` is
    :meth:`Windows.to_dict` or empty when the history was too short to split.
    ``change_my_mind`` items are strings; the engine produces
    :class:`~signalcheck.engine.change_my_mind.Condition` strings that also carry
    their displayed numbers in ``.numbers``.
    """

    label: str
    direction: str | None
    confidence: str
    rule_fired: str
    evidence: list[Evidence]
    change_my_mind: list[str]
    caveats: list[str]
    window: dict[str, Any]
    reason: str = ""

    def __post_init__(self) -> None:
        _require_member("label", self.label, LABELS)
        _require_member("confidence", self.confidence, CONFIDENCES)
        _require_member("rule_fired", self.rule_fired, RULE_IDS)
        if self.direction is not None:
            _require_member("direction", self.direction, DIRECTIONS)
        if self.label == "TREND" and self.direction is None:
            raise ValueError("a TREND verdict needs a direction")


@dataclass(frozen=True)
class WindowBounds:
    """A contiguous run of periods: positions ``[start_idx, end_idx)`` in the series.

    ``start``/``end`` are the first and last period starts (inclusive);
    ``end_exclusive`` is the instant the last period ends, used for calendar
    alignment across sources.
    """

    start_idx: int
    end_idx: int
    start: pd.Timestamp
    end: pd.Timestamp
    end_exclusive: pd.Timestamp

    @property
    def length(self) -> int:
        """Number of periods in the window."""
        return self.end_idx - self.start_idx

    @property
    def slice(self) -> slice:
        """Positional slice selecting this window from the regular series."""
        return slice(self.start_idx, self.end_idx)


@dataclass(frozen=True)
class BaselineStats:
    """Robust location/scale of the baseline window.

    ``median`` is the baseline median; ``mad`` is the median absolute deviation
    scaled by ``mad_scale`` (1.4826, so it estimates the standard deviation under
    normality), replaced by ``mad_floor * max(median, 1)`` when it is exactly 0.
    ``mad_unfloored`` keeps the scaled MAD before flooring and ``floored`` says
    whether the floor was applied. ``n`` counts the non-missing baseline points.
    """

    median: float
    mad: float
    mad_unfloored: float
    floored: bool
    n: int


@dataclass(frozen=True)
class Windows:
    """Baseline/recent split plus robust baseline statistics.

    ``transform`` is ``"log1p"`` (count/pageview scales) or ``"identity"``.
    ``stats`` are computed on that working scale and are what checks compare
    against; ``stats_original`` are the same statistics on the original scale, for
    plain-language summaries.
    """

    n: int
    baseline: WindowBounds
    recent: WindowBounds
    transform: str
    stats: BaselineStats
    stats_original: BaselineStats

    def to_dict(self) -> dict[str, Any]:
        """Calendar description of the windows, as stored in ``Verdict.window``."""

        def iso(ts: pd.Timestamp) -> str:
            return ts.date().isoformat()

        return {
            "baseline_start": iso(self.baseline.start),
            "baseline_end": iso(self.baseline.end),
            "recent_start": iso(self.recent.start),
            "recent_end": iso(self.recent.end),
            "recent_end_exclusive": iso(self.recent.end_exclusive),
            "n_baseline": self.baseline.length,
            "n_recent": self.recent.length,
        }


@dataclass(frozen=True)
class HistoryCheck:
    """Whether the series has enough observed points for a verdict (rule R1)."""

    n_observed: int
    min_required: int
    sufficient: bool
    reason: str | None

    @property
    def points_needed(self) -> int:
        """Additional observed periods needed to reach the minimum history."""
        return max(self.min_required - self.n_observed, 0)


@dataclass(frozen=True)
class Preprocessed:
    """Everything the checks consume.

    ``series`` is regular at ``series.freq`` (gaps are NaN rows, the partial last
    period removed) with caveats recorded in ``series.meta``. ``work`` is
    ``series.points.value`` on the working scale, indexed by ``ts``. ``windows`` is
    ``None`` when the history is insufficient.
    """

    series: Series
    work: pd.Series
    history: HistoryCheck
    windows: Windows | None
