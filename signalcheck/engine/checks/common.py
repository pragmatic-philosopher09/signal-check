"""Shared helpers for the section 6 checks.

The central piece is :class:`NumberBook`: every number a check writes into
``Evidence.summary`` is produced by a ``NumberBook`` method, which rounds it,
records the rounded value in ``Evidence.numbers`` and returns its display text.
A summary built only from those strings therefore satisfies the contract that
every number in ``summary`` appears in ``numbers`` with the same rounding, which
the narration validator (section 9.1) relies on. :func:`undisplayed_tokens`
verifies that contract after the fact and is used by the tests.

Annotations describe what the chart should draw, always using ISO dates and
values on the original scale:

- ``bands``:    ``{label, start, end, lower, upper}`` horizontal band over a date span
- ``points``:   ``{label, ts, value}`` highlighted observations
- ``vlines``:   ``{label, ts}`` vertical lines (e.g. a change point)
- ``segments``: ``{label, start, end, start_value, end_value}`` straight lines
- ``spans``:    ``{label, start, end}`` shaded date ranges (e.g. the recent window)
- ``lines``:    ``{label, points: [{ts, value}]}`` polylines (e.g. a seasonal expectation)
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from signalcheck.config import Config
from signalcheck.engine.preprocess import PERIOD_WORDS, robust_baseline, to_original_scale
from signalcheck.models import BaselineStats, Evidence, Preprocessed, Windows

DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_WORDS = """zero one two three four five six seven eight nine ten eleven twelve thirteen
fourteen fifteen sixteen seventeen eighteen nineteen twenty"""
NUMBER_WORDS: tuple[str, ...] = tuple(_WORDS.split())
NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)


def iso(ts: pd.Timestamp) -> str:
    """ISO date (``YYYY-MM-DD``) of a period start."""
    return pd.Timestamp(ts).date().isoformat()


def plural(count: int, word: str) -> str:
    """``word`` pluralised for ``count`` (the count itself is not included)."""
    return word if count == 1 else f"{word}s"


def period_word(freq: str) -> str:
    """``day`` / ``week`` / ``month`` for a series frequency."""
    return PERIOD_WORDS[freq]


@dataclass
class NumberBook:
    """Formats numbers for a summary and records them, rounded, for ``Evidence.numbers``.

    Each method stores the rounded value under ``key`` and returns the exact text
    to put in the summary, so displayed and recorded values can never diverge.
    """

    cfg: Config
    numbers: dict[str, Any] = field(default_factory=dict)

    def _decimals(self, kind: str) -> int:
        return int(self.cfg["display"][f"{kind}_decimals"])

    def _store(self, key: str, value: Any) -> None:
        if key in self.numbers and self.numbers[key] != value:
            raise ValueError(f"number {key!r} recorded twice with different values")
        self.numbers[key] = value

    def fixed(self, key: str, value: float, decimals: int) -> str:
        """Round ``value`` to ``decimals`` places, record it and return its text."""
        if not math.isfinite(value):
            raise ValueError(f"number {key!r} is not finite: {value}")
        rounded = round(float(value), decimals)
        if rounded == 0:
            rounded = 0.0  # avoid displaying "-0.0"
        stored: float | int = int(rounded) if decimals == 0 else rounded
        self._store(key, stored)
        return f"{rounded:.{decimals}f}"

    def value(self, key: str, value: float) -> str:
        """A value on the original scale (median, band edge, level)."""
        return self.fixed(key, value, self._decimals("value"))

    def stat(self, key: str, value: float) -> str:
        """A test statistic, z-score, ratio or penalty."""
        return self.fixed(key, value, self._decimals("stat"))

    def pct(self, key: str, fraction: float) -> str:
        """A fraction shown as a percentage; the percentage (not the fraction) is recorded."""
        return self.fixed(key, 100.0 * fraction, self._decimals("pct")) + "%"

    def count(self, key: str, value: int) -> str:
        """An integer count."""
        self._store(key, int(value))
        return str(int(value))

    def p_value(self, key: str, p: float) -> str:
        """``p = 0.012``, or ``p < 0.001`` when it rounds to zero.

        In the latter case ``key`` holds 0.0 and ``key_bound`` the displayed bound.
        """
        decimals = self._decimals("p")
        if round(p, decimals) == 0:
            self.numbers[key] = 0.0
            bound = self.fixed(f"{key}_bound", 10.0**-decimals, decimals)
            return f"p < {bound}"
        return f"p = {self.fixed(key, p, decimals)}"

    def date(self, key: str, ts: pd.Timestamp) -> str:
        """An ISO date, recorded as a string."""
        text = iso(ts)
        self._store(key, text)
        return text

    def put(self, key: str, value: Any) -> None:
        """Record a machine-readable field (flag, label) that is not displayed as a number."""
        self._store(key, value)


def _numeric_values(numbers: dict[str, Any]) -> list[float]:
    return [
        float(v) for v in numbers.values() if isinstance(v, int | float) and not isinstance(v, bool)
    ]


def undisplayed_tokens(text: str, numbers: dict[str, Any]) -> list[str]:
    """Numbers or dates in ``text`` that do not appear in ``numbers`` as displayed.

    A date must equal a string value in ``numbers``. A number token must equal a
    recorded number exactly, i.e. it was displayed with the rounding it was
    recorded with (``12.3`` matches a recorded ``12.3`` but not ``12.34``, and
    ``12`` does not match ``12.3``). Number words
    (zero-twenty), which the narration validator also reads as numbers, must match
    a recorded integer. Returns the offending tokens; an empty list means the text
    honours the numbers contract.
    """
    strings = {v for v in numbers.values() if isinstance(v, str)}
    bad = [d for d in DATE_RE.findall(text) if d not in strings]
    stripped = DATE_RE.sub(" ", text)
    values = _numeric_values(numbers)
    for token in NUMBER_RE.findall(stripped):
        if not any(math.isclose(float(token), v, rel_tol=1e-12, abs_tol=1e-12) for v in values):
            bad.append(token)
    for word in NUMBER_WORD_RE.findall(stripped):
        if float(NUMBER_WORDS.index(word.lower())) not in values:
            bad.append(word)
    return bad


def make_annotation(
    *,
    bands: Sequence[dict[str, Any]] = (),
    points: Sequence[dict[str, Any]] = (),
    vlines: Sequence[dict[str, Any]] = (),
    segments: Sequence[dict[str, Any]] = (),
    spans: Sequence[dict[str, Any]] = (),
    lines: Sequence[dict[str, Any]] = (),
) -> dict[str, Any] | None:
    """Bundle chart marks (see module docstring); empty kinds are omitted."""
    parts = {
        "bands": list(bands),
        "points": list(points),
        "vlines": list(vlines),
        "segments": list(segments),
        "spans": list(spans),
        "lines": list(lines),
    }
    out = {k: v for k, v in parts.items() if v}
    return out or None


def chart_value(work_value: float, transform: str, cfg: Config) -> float:
    """A working-scale value mapped to the original scale and rounded for drawing."""
    original = float(to_original_scale(float(work_value), transform))
    if transform == "log1p":
        original = max(original, 0.0)
    return round(original, int(cfg["display"]["value_decimals"]))


def span(label: str, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    """A shaded date range."""
    return {"label": label, "start": iso(start), "end": iso(end)}


def band(
    label: str, start: pd.Timestamp, end: pd.Timestamp, lower: float, upper: float
) -> dict[str, Any]:
    """A horizontal band (values already on the original scale)."""
    return {"label": label, "start": iso(start), "end": iso(end), "lower": lower, "upper": upper}


def highlight(label: str, ts: pd.Timestamp, value: float) -> dict[str, Any]:
    """A highlighted observation (value on the original scale)."""
    return {"label": label, "ts": iso(ts), "value": value}


def recent_span(pre: Preprocessed) -> dict[str, Any]:
    """The shaded recent window."""
    assert pre.windows is not None
    return span("recent window", pre.windows.recent.start, pre.windows.recent.end)


def make_evidence(
    check: str,
    stance: str,
    summary: str,
    book: NumberBook,
    annotation: dict[str, Any] | None,
) -> Evidence:
    """Build an :class:`Evidence` from a summary and the book that formatted its numbers."""
    return Evidence(
        check=check,
        stance=stance,
        summary=summary,
        numbers=dict(book.numbers),
        annotation=annotation,
    )


def skipped(check: str, reason: str, book: NumberBook) -> Evidence:
    """A ``skipped`` Evidence; ``reason`` may only contain numbers formatted by ``book``."""
    return Evidence(
        check=check,
        stance="skipped",
        summary=f"Skipped: {reason}.",
        numbers=dict(book.numbers),
        annotation=None,
        skip_reason=reason,
    )


def skip_if_no_windows(check: str, pre: Preprocessed, cfg: Config) -> Evidence | None:
    """The R1 short-circuit: a skipped Evidence when history is insufficient, else ``None``."""
    if pre.windows is not None:
        return None
    book = NumberBook(cfg)
    word = period_word(pre.series.freq)
    have = book.count("n_observed", pre.history.n_observed)
    need = book.count("min_history", pre.history.min_required)
    reason = (
        f"not enough history ({have} {plural(pre.history.n_observed, word)} observed, "
        f"need at least {need})"
    )
    return skipped(check, reason, book)


def recent_work(pre: Preprocessed) -> np.ndarray:
    """Working-scale values of the recent window (may contain NaN)."""
    assert pre.windows is not None
    return pre.work.to_numpy(dtype=float)[pre.windows.recent.slice]


def timestamps(pre: Preprocessed) -> pd.DatetimeIndex:
    """Period starts of the regular series, aligned with ``pre.work``."""
    return pd.DatetimeIndex(pre.work.index)


def baseline_of(values: np.ndarray, windows: Windows, cfg: Config) -> BaselineStats:
    """Robust baseline statistics of an alternative working-scale series (same windows)."""
    return robust_baseline(pd.Series(values[windows.baseline.slice]), cfg)


def finite(values: Iterable[float]) -> np.ndarray:
    """The finite entries of ``values`` as a float array."""
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]
