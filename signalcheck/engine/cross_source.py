"""Cross-source summary (COPILOT_BRIEF.md section 7).

Sources are compared only through their verdicts, never their raw values, and
only over a common calendar window:

1. Each source's recent window is the calendar span ``[recent_start,
   recent_end_exclusive)`` from ``Verdict.window``.
2. The comparison window length is the *shortest* recent window, in days.
3. It is anchored at the latest complete date covered by all compared sources:
   among the sources' window ends, pick the anchor that keeps the most sources
   comparable (ties -> the latest), then move it to the earliest end within that
   group so every compared source covers the whole window. When every source is
   current this is simply the latest common complete date; a stale source cannot
   drag everyone else's window into the past.
4. A source whose recent window overlaps the comparison window by less than
   ``cross_source.min_overlap`` (50%) is "not comparable (different period)".
   Sources that failed or had too little history to window are listed separately.

The result has one plain sentence ("3 of 4 sources show an upward trend over
1-28 Sep 2026; Reddit looks like a fluke.") plus structured counts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from signalcheck.config import Config, get_config
from signalcheck.models import LABELS, Verdict

DISPLAY_NAMES: dict[str, str] = {
    "csv": "Uploaded CSV",
    "google_trends": "Google Trends",
    "reddit": "Reddit",
    "wikipedia": "Wikipedia",
    "hackernews": "Hacker News",
    "x": "X",
}
# Order used to break ties between equally common outcomes.
OUTCOME_ORDER: tuple[tuple[str, str | None], ...] = (
    ("TREND", "up"),
    ("TREND", "down"),
    ("FLUKE", None),
    ("SEASONAL", None),
    ("NO_CHANGE", None),
    ("INCONCLUSIVE", None),
)


@dataclass(frozen=True)
class SourceWindow:
    """A source's recent window as a half-open calendar span ``[start, end)``."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def days(self) -> int:
        """Length in days."""
        return int((self.end - self.start).days)


@dataclass(frozen=True)
class CrossSourceSummary:
    """One-sentence summary plus the structured counts behind it.

    ``window_start``/``window_end`` are inclusive ISO dates (``None`` if nothing
    could be compared). ``counts`` maps ``LABEL`` or ``TREND_up``/``TREND_down`` to
    the number of comparable sources with that outcome. ``overlaps`` holds each
    windowed source's overlap with the comparison window as a fraction.
    """

    sentence: str
    window_start: str | None
    window_end: str | None
    window_days: int
    n_compared: int
    counts: dict[str, int]
    comparable: list[str]
    not_comparable: list[str]
    unavailable: list[str]
    outcomes: dict[str, dict[str, Any]] = field(default_factory=dict)
    overlaps: dict[str, float] = field(default_factory=dict)


def display_name(name: str) -> str:
    """Human-readable source name."""
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def source_window(name: str, verdict: Verdict | None) -> SourceWindow | None:
    """The calendar recent window of a verdict, or ``None`` when it has none."""
    if verdict is None or not verdict.window:
        return None
    start = verdict.window.get("recent_start")
    end = verdict.window.get("recent_end_exclusive")
    if start is None or end is None:
        return None
    return SourceWindow(name, pd.Timestamp(start), pd.Timestamp(end))


def overlap_fraction(w: SourceWindow, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Share of the comparison window ``[start, end)`` covered by ``w``."""
    length = (end - start).days
    if length <= 0:
        return 0.0
    covered = (min(w.end, end) - max(w.start, start)).days
    return max(covered, 0) / length


def comparable_at(
    windows: list[SourceWindow], anchor: pd.Timestamp, days: int, min_overlap: float
) -> list[SourceWindow]:
    """Sources overlapping ``[anchor - days, anchor)`` by at least ``min_overlap``."""
    start = anchor - pd.Timedelta(days=days)
    return [w for w in windows if overlap_fraction(w, start, anchor) >= min_overlap]


def choose_anchor(windows: list[SourceWindow], days: int, min_overlap: float) -> pd.Timestamp:
    """Anchor (exclusive end) of the comparison window; see the module docstring."""
    candidates = sorted({w.end for w in windows})
    best = max(candidates, key=lambda a: (len(comparable_at(windows, a, days, min_overlap)), a))
    group = comparable_at(windows, best, days, min_overlap)
    return min(w.end for w in group) if group else best


def outcome_key(verdict: Verdict) -> tuple[str, str | None]:
    """``(label, direction)`` with direction kept only for TREND."""
    return verdict.label, verdict.direction if verdict.label == "TREND" else None


def count_key(key: tuple[str, str | None]) -> str:
    """``TREND_up`` / ``FLUKE`` ... for the structured counts."""
    label, direction = key
    return f"{label}_{direction}" if direction else label


def phrase(key: tuple[str, str | None], plural: bool) -> str:
    """Verb phrase for an outcome: "show(s) an upward trend", "look(s) like a fluke"."""
    label, direction = key
    if label == "TREND":
        trend = "an upward trend" if direction == "up" else "a downward trend"
        return f"{'show' if plural else 'shows'} {trend}"
    verbs = {
        "FLUKE": ("look like a fluke", "looks like a fluke"),
        "SEASONAL": ("show the usual seasonal pattern", "shows the usual seasonal pattern"),
        "NO_CHANGE": ("show no change", "shows no change"),
        "INCONCLUSIVE": ("are inconclusive", "is inconclusive"),
    }
    many, one = verbs[label]
    return many if plural else one


DASH = "\u2013"


def date_range(start: pd.Timestamp, end_inclusive: pd.Timestamp) -> str:
    """Compact day range joined by an en dash, e.g. 1-28 Sep 2026 or 25 Aug-28 Sep 2026."""
    if start.year != end_inclusive.year:
        return f"{start.day} {start:%b %Y}{DASH}{end_inclusive.day} {end_inclusive:%b %Y}"
    if start.month != end_inclusive.month:
        return f"{start.day} {start:%b}{DASH}{end_inclusive.day} {end_inclusive:%b %Y}"
    if start.day == end_inclusive.day:
        return f"{start.day} {start:%b %Y}"
    return f"{start.day}{DASH}{end_inclusive.day} {end_inclusive:%b %Y}"


def join_names(names: list[str]) -> str:
    """``A``, ``A and B``, ``A, B and C``."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def lead_outcome(keys: list[tuple[str, str | None]]) -> tuple[str, str | None]:
    """Most common outcome; ties broken by :data:`OUTCOME_ORDER`."""
    counts = Counter(keys)
    return min(counts, key=lambda k: (-counts[k], OUTCOME_ORDER.index(k)))


def build_sentence(
    outcomes: dict[str, tuple[str, str | None]],
    not_comparable: list[str],
    unavailable: list[str],
    span_text: str | None,
) -> str:
    """The one-sentence cross-source summary."""
    clauses: list[str] = []
    if outcomes:
        lead = lead_outcome(list(outcomes.values()))
        leaders = [n for n, k in outcomes.items() if k == lead]
        total = len(outcomes)
        if total == 1:
            head = f"{display_name(leaders[0])} {phrase(lead, plural=False)}"
        else:
            head = f"{len(leaders)} of {total} sources {phrase(lead, plural=len(leaders) != 1)}"
        clauses.append(f"{head} over {span_text}")
        others: dict[tuple[str, str | None], list[str]] = {}
        for name, key in outcomes.items():
            if key != lead:
                others.setdefault(key, []).append(display_name(name))
        for key in OUTCOME_ORDER:
            if key in others:
                names = others[key]
                clauses.append(f"{join_names(names)} {phrase(key, plural=len(names) != 1)}")
    else:
        clauses.append("No sources could be compared over a common period")
    if not_comparable:
        names = [display_name(n) for n in not_comparable]
        verb = "are" if len(names) != 1 else "is"
        clauses.append(f"{join_names(names)} {verb} not comparable (different period)")
    if unavailable:
        names = [display_name(n) for n in unavailable]
        verb = "have" if len(names) != 1 else "has"
        clauses.append(f"{join_names(names)} {verb} no usable result")
    return "; ".join(clauses) + "."


def compare_sources(
    verdicts: Mapping[str, Verdict | None], cfg: Config | None = None
) -> CrossSourceSummary:
    """Align the sources' recent windows on a common calendar span and summarise them.

    ``verdicts`` maps a source name to its verdict, or ``None`` when the source
    failed. Raw values are never compared.
    """
    cfg = cfg if cfg is not None else get_config()
    min_overlap = float(cfg["cross_source"]["min_overlap"])
    windows: list[SourceWindow] = []
    unavailable: list[str] = []
    for name, verdict in verdicts.items():
        w = source_window(name, verdict)
        if w is None or w.days <= 0:
            unavailable.append(name)
        else:
            windows.append(w)
    if not windows:
        sentence = build_sentence({}, [], unavailable, None)
        return CrossSourceSummary(sentence, None, None, 0, 0, {}, [], [], unavailable)

    days = min(w.days for w in windows)
    anchor = choose_anchor(windows, days, min_overlap)
    start = anchor - pd.Timedelta(days=days)
    end_inclusive = anchor - pd.Timedelta(days=1)
    overlaps = {w.name: round(overlap_fraction(w, start, anchor), 4) for w in windows}
    comparable = [w.name for w in windows if overlaps[w.name] >= min_overlap]
    not_comparable = [w.name for w in windows if overlaps[w.name] < min_overlap]

    keys: dict[str, tuple[str, str | None]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    for name in comparable:
        verdict = verdicts[name]
        assert verdict is not None
        keys[name] = outcome_key(verdict)
        outcomes[name] = {
            "label": verdict.label,
            "direction": keys[name][1],
            "confidence": verdict.confidence,
        }
    counts = {count_key(k): n for k, n in Counter(keys.values()).items()}
    for label in LABELS:
        if label != "TREND":
            counts.setdefault(label, 0)
    counts.setdefault("TREND_up", 0)
    counts.setdefault("TREND_down", 0)
    sentence = build_sentence(keys, not_comparable, unavailable, date_range(start, end_inclusive))
    return CrossSourceSummary(
        sentence=sentence,
        window_start=start.date().isoformat(),
        window_end=end_inclusive.date().isoformat(),
        window_days=days,
        n_compared=len(comparable),
        counts=counts,
        comparable=comparable,
        not_comparable=not_comparable,
        unavailable=unavailable,
        outcomes=outcomes,
        overlaps=overlaps,
    )
