"""Individual statistical checks (COPILOT_BRIEF.md section 6), each returning an Evidence.

Every check is a function ``run(pre, cfg=None) -> Evidence`` over a
:class:`~signalcheck.models.Preprocessed` series. Checks never fill gaps
themselves: callers that want short gaps interpolated call
:func:`signalcheck.engine.preprocess.fill_gaps` first, which records the
imputation in ``meta.caveats``. Checks that need a gap-free series (level shift,
seasonality) are skipped when gaps remain; the others ignore missing points.
"""

from __future__ import annotations

from collections.abc import Callable

from signalcheck.config import Config, get_config
from signalcheck.engine.checks import (
    breadth,
    concentration,
    level_shift,
    low_count,
    outliers,
    persistence,
    seasonality,
)
from signalcheck.models import Evidence, Preprocessed

Check = Callable[[Preprocessed, Config], Evidence]

CHECKS: dict[str, Check] = {
    persistence.CHECK: persistence.run,
    concentration.CHECK: concentration.run,
    seasonality.CHECK: seasonality.run,
    outliers.CHECK: outliers.run,
    level_shift.CHECK: level_shift.run,
    low_count.CHECK: low_count.run,
    breadth.CHECK: breadth.run,
}


def run_all(pre: Preprocessed, cfg: Config | None = None) -> list[Evidence]:
    """Run every check in section 6 order."""
    cfg = cfg if cfg is not None else get_config()
    return [check(pre, cfg) for check in CHECKS.values()]
