"""Methodology text for the UI, built from the live ``config.yaml`` values.

Each section is Markdown. Thresholds are never written here as literals: they
are read from the config passed in, so the expander always matches what the
engine actually used.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

from signalcheck.config import Config
from signalcheck.engine import confidence
from signalcheck.engine.checks import CHECKS
from signalcheck.engine.verdict import RULES, STANCE_WORDS
from signalcheck.ui.view_models import check_name

# What each check tests, in plain language (statistical meaning; no thresholds here).
CHECK_MEANINGS: dict[str, str] = {
    "persistence": (
        "Mann-Kendall trend test (Hamed-Rao autocorrelation correction for longer windows) "
        "on the recent window plus a few baseline points, with a Theil-Sen slope expressed "
        "as % of the baseline level per period. Supports a trend when the test is "
        "significant *and* the slope is large enough; also counts the consecutive latest "
        "periods beyond the baseline band."
    ),
    "concentration": (
        "How much of the recent excess over the baseline median comes from the top one or "
        "two periods. A rise (or drop) carried by one or two periods supports a fluke. "
        "Skipped when there is no meaningful excess."
    ),
    "seasonality": (
        "Robust STL decomposition. Day-of-week patterns are removed as a nuisance only. "
        "Annual seasonality (needs two full years) estimates the share of the recent rise "
        "explained by last years' pattern, then re-tests trend and level shift on the "
        "seasonally adjusted series."
    ),
    "outliers": (
        "Robust z-score of each recent point against the baseline median and MAD; a few "
        "isolated extreme points suggest a spike rather than a move of the whole window."
    ),
    "level_shift": (
        "PELT change-point detection on the robust-standardised series. Counts when a "
        "change point falls in or just before the recent window and the new level has held."
    ),
    "low_count": (
        "Count series only: when events are rare, an exact conditional binomial test on the "
        "recent/baseline rate ratio. If its confidence interval includes 1 there are too few "
        "events to call a trend."
    ),
    "breadth": (
        "When the source reports who posted (top-origin share, contributors per item): a "
        "recent window dominated by one origin supports a fluke."
    ),
}


@dataclass(frozen=True)
class MethodologySection:
    """One heading plus its Markdown body."""

    title: str
    body: str


def format_value(value: Any) -> str:
    """Config value as compact text: per-frequency dicts as ``D 28 · W 26 · M 24``."""
    if isinstance(value, dict):
        return " \u00b7 ".join(f"{k} {format_value(v)}" for k, v in value.items())
    if isinstance(value, list | tuple):
        return ", ".join(format_value(v) for v in value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def threshold_lines(settings: dict[str, Any]) -> str:
    """Markdown bullets ``key = value`` for every setting in a config block."""
    return "\n".join(f"- `{key}` = {format_value(value)}" for key, value in settings.items())


def preprocessing_section(cfg: Config) -> MethodologySection:
    """How the raw series is cleaned and split before any check runs."""
    p = cfg["preprocess"]
    body = (
        "Each series is sorted, de-duplicated and regularised to its frequency. Gaps stay "
        "missing (and are listed in the caveats); they are linearly filled, at most "
        f"{p['gap_fill_max_consecutive']} in a row and flagged as imputed (hollow points), "
        "only for checks that need a regular series. A period that had not ended when the "
        "data was fetched is dropped (greyed on the chart). The **recent window** is "
        f"`clamp(round({p['recent_frac']} \u00d7 n), recent_min, recent_max)` periods and the "
        "**baseline** is everything before it. Count and pageview series are analysed on "
        "`log1p` scale; numbers are reported on the original scale.\n\n" + threshold_lines(p)
    )
    return MethodologySection("Preprocessing", body)


def checks_section(cfg: Config) -> MethodologySection:
    """Every check: what it tests and its thresholds from config."""
    parts = []
    for check in CHECKS:
        meaning = CHECK_MEANINGS.get(check, "")
        settings = cfg["checks"].get(check, {})
        parts.append(f"**{check_name(check)}.** {meaning}\n\n{threshold_lines(settings)}")
    return MethodologySection("Checks", "\n\n".join(parts))


def decision_table_section(cfg: Config) -> MethodologySection:
    """The R1-R6 decision table (first matching row wins)."""
    rows = "\n".join(f"| {rule} | {text.replace('->', '\u2192')} |" for rule, text in RULES.items())
    v = cfg["verdict"]
    body = (
        "Rows are evaluated in order and the first match wins; the card shows which row "
        "fired.\n\n| Rule | Condition \u2192 verdict |\n|---|---|\n"
        + rows
        + "\n\n"
        + threshold_lines(v)
    )
    return MethodologySection("Decision table", body)


def confidence_formula() -> str:
    """The score formula exactly as documented in :mod:`signalcheck.engine.confidence`."""
    doc = confidence.__doc__ or ""
    block = doc.split("::", 1)[1].split("Agreement table", 1)[0] if "::" in doc else ""
    return textwrap.dedent(block).strip()


def stance_list(stances: frozenset[str]) -> str:
    """``supports_trend`` -> ``trend`` words, sorted."""
    return ", ".join(sorted(STANCE_WORDS.get(s, s) for s in stances))


def confidence_section(cfg: Config) -> MethodologySection:
    """The confidence score formula with the live cut-offs and the agreement table."""
    c = cfg["confidence"]
    rows = "\n".join(
        f"| {label} | {stance_list(confidence.AGREES[label])} | "
        f"{stance_list(confidence.CONTRADICTS[label])} |"
        for label in confidence.AGREES
    )
    body = (
        f"```text\n{confidence_formula()}\n```\n\n"
        + threshold_lines(c)
        + "\n\nA TREND-supporting check that points the other way contradicts a TREND.\n\n"
        "| Verdict | Agrees | Contradicts |\n|---|---|---|\n" + rows
    )
    return MethodologySection("Confidence", body)


def change_my_mind_section(cfg: Config) -> MethodologySection:
    """How the "what would change my mind" conditions are derived."""
    body = (
        "Up to "
        f"{cfg['change_my_mind']['max_conditions']} deterministic conditions computed from "
        "the series itself (baseline median and MAD on the original scale), never invented.\n\n"
        + threshold_lines(cfg["change_my_mind"])
    )
    return MethodologySection("What would change my mind", body)


def cross_source_section(cfg: Config) -> MethodologySection:
    """How sources are compared."""
    overlap = float(cfg["cross_source"]["min_overlap"])
    body = (
        "Sources are compared over the latest calendar span covered by all of them (length = "
        "the shortest recent window). A source whose recent window overlaps that span by less "
        f"than {overlap:.0%} is listed as not comparable. Verdicts are counted; raw values "
        "are never compared across sources.\n\n" + threshold_lines(cfg["cross_source"])
    )
    return MethodologySection("Cross-source summary", body)


def methodology_sections(cfg: Config) -> list[MethodologySection]:
    """All methodology sections, in reading order."""
    return [
        preprocessing_section(cfg),
        checks_section(cfg),
        decision_table_section(cfg),
        confidence_section(cfg),
        change_my_mind_section(cfg),
        cross_source_section(cfg),
    ]
