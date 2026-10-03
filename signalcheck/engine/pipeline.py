"""Top-level engine entry point: ``analyse(series) -> Verdict``.

:func:`analyse_detailed` returns the same verdict together with the preprocessed
series it was computed from, so the UI can chart exactly what the checks saw.

preprocess -> checks -> decision table -> confidence -> change-my-mind. Short
gaps are linearly filled (and flagged) only for the checks that need a regular
series (level shift, seasonality); the other checks see the observed data with
gaps left as missing. For daily data the outlier check scores the
de-weekly-ised series (day-of-week pattern removed, section 6.3), so a regular
weekend dip is not flagged as an extreme day. A series that is too short never raises:
every check is skipped and rule R1 returns ``INCONCLUSIVE`` with the history shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass

from signalcheck.config import Config, get_config
from signalcheck.engine.change_my_mind import change_my_mind
from signalcheck.engine.checks import CHECKS, outliers
from signalcheck.engine.checks.seasonality import deweekly
from signalcheck.engine.confidence import score_confidence
from signalcheck.engine.preprocess import fill_gaps, preprocess
from signalcheck.engine.verdict import decide
from signalcheck.models import Evidence, Preprocessed, Series, Verdict

NEEDS_REGULAR: frozenset[str] = frozenset({"level_shift", "seasonality"})
REGULAR_SCOPE = "level-shift and seasonality checks"


def run_checks(pre: Preprocessed, filled: Preprocessed, cfg: Config) -> list[Evidence]:
    """Run every check in section 6 order, on the gap-filled series where required.

    The outlier check gets the de-weekly-ised (gap-filled) series when one applies.
    """
    weekly_free = None
    if pre.windows is not None:
        start = pre.windows.recent.start_idx
        weekly_free = deweekly(filled.work, filled.series.freq, cfg, recent_start=start)
    out: list[Evidence] = []
    for name, check in CHECKS.items():
        if name == outliers.CHECK and weekly_free is not None:
            out.append(outliers.run(pre, cfg, values=weekly_free.to_numpy(dtype=float)))
        else:
            out.append(check(filled if name in NEEDS_REGULAR else pre, cfg))
    return out


def dedupe(items: list[str]) -> list[str]:
    """Keep the first occurrence of each caveat, in order."""
    return list(dict.fromkeys(items))


@dataclass(frozen=True)
class Analysis:
    """A verdict plus the preprocessed series behind it.

    ``pre`` is the regular series (gaps as NaN, partial last period dropped and
    recorded in ``meta.dropped_partial``); ``filled`` is ``pre`` with short gaps
    interpolated and flagged ``imputed`` (identical to ``pre`` when nothing was filled).
    """

    verdict: Verdict
    pre: Preprocessed
    filled: Preprocessed


def analyse_detailed(series: Series, cfg: Config | None = None) -> Analysis:
    """Analyse one series and keep the preprocessed data (never raises for short series)."""
    cfg = cfg if cfg is not None else get_config()
    pre = preprocess(series, cfg)
    filled = fill_gaps(pre, cfg, REGULAR_SCOPE) if pre.windows is not None else pre
    evidence = run_checks(pre, filled, cfg)
    decision = decide(evidence, pre.history, cfg)
    confidence = score_confidence(decision.label, decision.direction, evidence, filled, cfg)
    conditions = change_my_mind(
        decision.label, decision.direction, decision.rule, filled, evidence, cfg
    )
    verdict = Verdict(
        label=decision.label,
        direction=decision.direction,
        confidence=confidence.level,
        rule_fired=decision.rule,
        evidence=evidence,
        change_my_mind=list(conditions),
        caveats=dedupe(filled.series.caveats),
        window=pre.windows.to_dict() if pre.windows is not None else {},
        reason=decision.reason,
    )
    return Analysis(verdict=verdict, pre=pre, filled=filled)


def analyse(series: Series, cfg: Config | None = None) -> Verdict:
    """Analyse one series and return its verdict (never raises for short series)."""
    return analyse_detailed(series, cfg).verdict
