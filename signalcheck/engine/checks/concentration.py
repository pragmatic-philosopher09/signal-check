"""Concentration check (COPILOT_BRIEF.md section 6.2).

Statistical meaning: is the recent departure from the baseline carried by one or
two periods (a spike or dip) rather than spread across the window?

On the working scale, ``excess_t = value_t - baseline_median`` for each observed
recent period, and the direction is the sign of their sum. If
``|sum(excess)| < excess_min_mads * MAD * n_recent`` there is no meaningful excess
to attribute and the check is skipped. Otherwise only same-signed (directional)
excess counts, and the shares carried by the largest one and two periods are
compared with ``conc_top1_max`` / ``conc_top2_max``; reaching either supports a
fluke. Rises and drops are treated symmetrically. Excess that is spread out is
neutral here: whether it is a trend, a shift or seasonality is for other checks.
"""

from __future__ import annotations

import numpy as np

from signalcheck.config import Config, get_config
from signalcheck.engine.checks.common import (
    NumberBook,
    band,
    chart_value,
    highlight,
    make_annotation,
    make_evidence,
    period_word,
    plural,
    recent_span,
    recent_work,
    skip_if_no_windows,
    skipped,
    timestamps,
)
from signalcheck.models import Evidence, Preprocessed

CHECK = "concentration"


def directional_shares(excess: np.ndarray) -> tuple[str, np.ndarray, float, float]:
    """Direction, order of periods by directional excess, and top-1/top-2 shares.

    ``excess`` holds observed recent excesses (no NaN) whose sum is non-zero.
    Opposite-signed excess is ignored when computing shares.
    """
    direction = "up" if excess.sum() > 0 else "down"
    directional = np.clip(excess if direction == "up" else -excess, 0.0, None)
    order = np.argsort(-directional, kind="stable")
    total = float(directional.sum())
    top1 = float(directional[order[0]]) / total
    top2 = float(directional[order[:2]].sum()) / total
    return direction, order, top1, top2


def run(pre: Preprocessed, cfg: Config | None = None) -> Evidence:
    """Concentration evidence for a preprocessed series."""
    cfg = cfg if cfg is not None else get_config()
    early = skip_if_no_windows(CHECK, pre, cfg)
    if early is not None:
        return early
    windows = pre.windows
    assert windows is not None
    conf = cfg["checks"]["concentration"]
    stats, transform = windows.stats, windows.transform
    word = period_word(pre.series.freq)

    recent = recent_work(pre)
    observed = np.flatnonzero(np.isfinite(recent))
    excess = recent[observed] - stats.median
    n_recent = len(observed)
    book = NumberBook(cfg)
    needed = float(conf["excess_min_mads"]) * stats.mad * n_recent
    if n_recent == 0 or abs(float(excess.sum())) < needed:
        k = book.stat("excess_min_mads", float(conf["excess_min_mads"]))
        return skipped(
            CHECK,
            f"no meaningful excess (recent values average less than {k} MAD "
            "from the baseline median)",
            book,
        )

    direction, order, top1, top2 = directional_shares(excess)
    fluke = top1 >= float(conf["conc_top1_max"]) or top2 >= float(conf["conc_top2_max"])
    ts = timestamps(pre)
    positions = windows.recent.start_idx + observed
    top_pos = [int(positions[i]) for i in order[:2] if i < n_recent]
    work = pre.work.to_numpy(dtype=float)

    book.put("direction", direction)
    book.put("supports_fluke", fluke)
    side = "upward" if direction == "up" else "downward"
    top_text = (
        f"the largest {word} ({book.date('top1_date', ts[top_pos[0]])}, "
        f"{book.value('top1_value', chart_value(work[top_pos[0]], transform, cfg))}) carries "
        f"{book.pct('top1_share_pct', top1)} of the {side} excess over the baseline median "
        f"({book.value('baseline_median', chart_value(stats.median, transform, cfg))}) "
        f"across {book.count('n_recent', n_recent)} recent {plural(n_recent, word)}, "
        f"and the top {book.count('top_k', 2)} carry {book.pct('top2_share_pct', top2)}"
    )
    burst = "spike" if direction == "up" else "dip"
    if fluke:
        summary = f"Concentrated: {top_text}, so a short-lived {burst} drives the change."
        stance = "supports_fluke"
    else:
        summary = f"Spread out: {top_text}, so the change is not a single {burst}."
        stance = "neutral"

    annotation = make_annotation(
        points=[
            highlight(f"top {rank} excess", ts[p], chart_value(work[p], transform, cfg))
            for rank, p in enumerate(top_pos, start=1)
        ],
        bands=[
            band(
                "baseline median",
                ts[0],
                ts[-1],
                chart_value(stats.median, transform, cfg),
                chart_value(stats.median, transform, cfg),
            )
        ],
        spans=[recent_span(pre)],
    )
    return make_evidence(CHECK, stance, summary, book, annotation)
