"""Verdict decision table (COPILOT_BRIEF.md section 7).

A transparent, ordered rule table over the section 6 check outputs: the first
matching row wins and its id (``R1``-``R6``) is stored in ``Verdict.rule_fired``.
Every threshold lives in ``config.yaml``; this module only combines the checks'
stances and the machine-readable flags they record in ``Evidence.numbers``.

- **R1** Not enough history -> ``INCONCLUSIVE`` (reason stated).
- **R2** The seasonality check supports "seasonal" -> ``SEASONAL``. That stance
  already encodes the full R2 condition: ``seasonal_share >=
  seasonal_explained_min`` *and* no significant trend *and* no sustained level
  shift on the seasonally adjusted series. If something is left after removing
  the annual pattern, the row does not match and a trend on top of seasonality
  can be called by R4 (the seasonality evidence is still shown).
- **R3** ``FLUKE`` when any of:

  a. concentration supports a fluke and persistence does not support a trend;
  b. breadth supports a fluke (one origin drives most of the volume);
  c. (``verdict.fluke_from_isolated_outliers``) concentration was *skipped* for
     lack of excess, the outlier check found isolated outliers (at most
     ``isolated_max_points``), and there is neither a trend nor a sustained level
     shift. This closes the gap where a single spike on a log-scale series is too
     small, averaged over the recent window, for the concentration check to run;
     the outlier check then carries the "short-lived extreme" signal instead.
     When concentration *does* run, its judgement wins over the outliers'.

- **R4** (persistence supports a trend *or* a sustained level shift) *and*
  concentration does not support a fluke -> ``TREND`` with direction. Blocked
  (falls through) when the low-count check says there are too few events to
  call, or when the trend-supporting checks disagree on the direction. When the
  recent window departs meaningfully from the baseline (the concentration check
  ran), that departure's direction must agree too: a significant fall while the
  level is still far *above* the baseline is a decay back towards it after a
  spike, not a downward trend (and vice versa).
- **R5** Persistence ran and supports "no change" (not significant, or a
  significant slope below ``negligible_slope``: a minimum effect size), no recent
  outliers and no change point in (or just before) the recent window ->
  ``NO_CHANGE``. A skipped
  level-shift check (gaps left after filling) does not block this row.
- **R6** Otherwise -> ``INCONCLUSIVE``, listing the checks that disagree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from signalcheck.config import Config, get_config
from signalcheck.models import Evidence, HistoryCheck

RULES: dict[str, str] = {
    "R1": "Not enough history -> INCONCLUSIVE",
    "R2": "Annual pattern explains the change and nothing is left after removing it -> SEASONAL",
    "R3": (
        "Change concentrated in one or two periods without a trend, isolated outliers when "
        "concentration could not run, or a single origin drives the volume -> FLUKE"
    ),
    "R4": (
        "Significant trend or sustained level shift that is not concentrated, with enough "
        "events -> TREND"
    ),
    "R5": "No significant (or only a negligible) slope, no outliers and no recent level shift "
    "-> NO_CHANGE",
    "R6": "Anything else -> INCONCLUSIVE (conflicting checks listed)",
}

STANCE_WORDS: dict[str, str] = {
    "supports_trend": "trend",
    "supports_fluke": "fluke",
    "supports_seasonal": "seasonal",
    "supports_no_change": "no change",
    "neutral": "neutral",
}
CHECK_WORDS: dict[str, str] = {
    "persistence": "persistence",
    "concentration": "concentration",
    "seasonality": "seasonality",
    "outliers": "outliers",
    "level_shift": "level shift",
    "low_count": "low count",
    "breadth": "breadth",
}


@dataclass(frozen=True)
class Decision:
    """Outcome of the decision table: label, direction, rule id and why it matched."""

    label: str
    direction: str | None
    rule: str
    reason: str


@dataclass(frozen=True)
class Signals:
    """The facts the decision table reads, extracted from the check evidence.

    ``*_ran`` is False when the check was skipped (or absent). Directions are
    ``"up"``/``"down"`` or ``None``.
    """

    persistence_ran: bool
    persistence_trend: bool
    persistence_significant: bool
    persistence_no_change: bool
    persistence_direction: str | None
    concentration_ran: bool
    concentration_fluke: bool
    concentration_direction: str | None
    seasonal: bool
    outliers_ran: bool
    n_outliers: int
    isolated_outliers: bool
    outlier_direction: str | None
    shift_ran: bool
    shift_sustained: bool
    shift_in_recent: bool
    shift_direction: str | None
    low_count_active: bool
    too_few_to_call: bool
    low_count_trend: bool
    low_count_direction: str | None
    breadth_fluke: bool


def by_check(evidence: Sequence[Evidence]) -> dict[str, Evidence]:
    """Evidence keyed by check name (later entries win)."""
    return {ev.check: ev for ev in evidence}


def ran(ev: Evidence | None) -> bool:
    """The check produced a result (it exists and was not skipped)."""
    return ev is not None and ev.stance != "skipped"


def _stance(ev: Evidence | None, stance: str) -> bool:
    return ran(ev) and ev is not None and ev.stance == stance


def _flag(ev: Evidence | None, key: str) -> bool:
    return ran(ev) and ev is not None and bool(ev.numbers.get(key, False))


def _direction(ev: Evidence | None, key: str) -> str | None:
    value: Any = None if ev is None else ev.numbers.get(key)
    return value if value in ("up", "down") else None


def extract_signals(evidence: Sequence[Evidence]) -> Signals:
    """Read the decision-table inputs from the checks' stances and ``numbers``."""
    ev = by_check(evidence)
    per, conc, seas = ev.get("persistence"), ev.get("concentration"), ev.get("seasonality")
    out, shift = ev.get("outliers"), ev.get("level_shift")
    low, br = ev.get("low_count"), ev.get("breadth")
    n_flagged = int(out.numbers.get("n_flagged", 0)) if out is not None and ran(out) else 0
    return Signals(
        persistence_ran=ran(per),
        persistence_trend=_stance(per, "supports_trend"),
        persistence_significant=_flag(per, "significant"),
        persistence_no_change=_stance(per, "supports_no_change"),
        persistence_direction=_direction(per, "direction"),
        concentration_ran=ran(conc),
        concentration_fluke=_stance(conc, "supports_fluke"),
        concentration_direction=_direction(conc, "direction"),
        seasonal=_stance(seas, "supports_seasonal"),
        outliers_ran=ran(out),
        n_outliers=n_flagged,
        isolated_outliers=_stance(out, "supports_fluke"),
        outlier_direction=_direction(out, "max_direction"),
        shift_ran=ran(shift),
        shift_sustained=_flag(shift, "sustained"),
        shift_in_recent=_flag(shift, "in_recent"),
        shift_direction=_direction(shift, "direction"),
        low_count_active=_flag(low, "active"),
        too_few_to_call=_flag(low, "too_few_to_call"),
        low_count_trend=_stance(low, "supports_trend"),
        low_count_direction=_direction(low, "direction"),
        breadth_fluke=_stance(br, "supports_fluke"),
    )


def trend_directions(s: Signals) -> list[str]:
    """Directions R4 needs to agree: every trend-supporting check's, plus the departure.

    The departure is the direction of the recent window's excess over the
    baseline median, included only when it is meaningful (the concentration
    check ran rather than skipping for lack of excess).
    """
    claims = [
        (s.persistence_trend, s.persistence_direction),
        (s.shift_sustained, s.shift_direction),
        (s.low_count_trend, s.low_count_direction),
        (s.concentration_ran, s.concentration_direction),
    ]
    return [d for supports, d in claims if supports and d is not None]


def fluke_direction(s: Signals) -> str | None:
    """Direction of a fluke: the concentration check's, else the largest outlier's."""
    return s.concentration_direction or s.outlier_direction


def rule_r3(s: Signals, cfg: Config) -> str | None:
    """R3: the reason a FLUKE matches, or ``None``."""
    if s.concentration_fluke and not s.persistence_trend:
        return "a short-lived spike or dip carries most of the change and there is no trend"
    if s.breadth_fluke:
        return "a single origin drives most of the recent volume"
    if (
        bool(cfg["verdict"]["fluke_from_isolated_outliers"])
        and not s.concentration_ran
        and s.isolated_outliers
        and not s.persistence_trend
        and not s.shift_sustained
    ):
        return (
            "isolated outliers with no trend and no lasting level shift "
            "(the change was too small overall for the concentration check to run)"
        )
    return None


def rule_r4(s: Signals) -> tuple[str | None, str | None, str | None]:
    """R4: ``(direction, reason, blocker)``; a blocker explains why R4 fell through."""
    if not (s.persistence_trend or s.shift_sustained) or s.concentration_fluke:
        return None, None, None
    if s.too_few_to_call:
        return None, None, "low count: too few events to call"
    directions = set(trend_directions(s))
    if len(directions) != 1:
        return (
            None,
            None,
            (
                "the trend-supporting checks and the recent departure from the baseline "
                "disagree on the direction"
            ),
        )
    direction = directions.pop()
    parts = []
    if s.persistence_trend:
        parts.append("a significant, practically large trend")
    if s.shift_sustained:
        parts.append("a sustained level shift")
    return direction, " and ".join(parts) + " that is not concentrated in a single spike", None


def rule_r5(s: Signals) -> bool:
    """R5: no practical slope, no outliers and no recent change point."""
    return (
        s.persistence_ran
        and s.persistence_no_change
        and s.outliers_ran
        and s.n_outliers == 0
        and not s.shift_in_recent
    )


def conflict_text(evidence: Sequence[Evidence], blockers: Sequence[str]) -> str:
    """R6 reason: every check that ran with its stance, plus any rule blockers."""
    items = [
        f"{CHECK_WORDS.get(ev.check, ev.check)}: {STANCE_WORDS[ev.stance]}"
        for ev in evidence
        if ev.stance != "skipped"
    ]
    text = "the checks do not line up (" + "; ".join(items) + ")" if items else "no check ran"
    if blockers:
        text += "; " + "; ".join(blockers)
    return text


def decide(
    evidence: Sequence[Evidence], history: HistoryCheck, cfg: Config | None = None
) -> Decision:
    """Apply rules R1-R6 in order; the first match wins."""
    cfg = cfg if cfg is not None else get_config()
    if not history.sufficient:
        return Decision("INCONCLUSIVE", None, "R1", history.reason or "not enough history")
    s = extract_signals(evidence)
    if s.seasonal:
        return Decision(
            "SEASONAL",
            None,
            "R2",
            "the usual annual pattern explains the recent change and nothing is left after "
            "removing it",
        )
    fluke = rule_r3(s, cfg)
    if fluke is not None:
        return Decision("FLUKE", fluke_direction(s), "R3", fluke)
    direction, trend_reason, blocker = rule_r4(s)
    if direction is not None and trend_reason is not None:
        return Decision("TREND", direction, "R4", trend_reason)
    if rule_r5(s):
        return Decision(
            "NO_CHANGE",
            None,
            "R5",
            "no significant slope, no outliers and no level shift in the recent window",
        )
    blockers = [blocker] if blocker else []
    return Decision("INCONCLUSIVE", None, "R6", conflict_text(evidence, blockers))
