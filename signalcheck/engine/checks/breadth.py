"""Breadth check (COPILOT_BRIEF.md section 6.7).

Statistical meaning: is recent volume produced by many independent origins, or
does one origin (a subreddit, an author) account for most of it?

Uses the optional ``top_share`` (largest share of a bucket's items from a single
origin) and ``contributors`` (distinct origins) columns over the recent window:

- volume-weighted mean ``top_share = sum(volume_t * top_share_t) / sum(volume_t)``,
  so busy buckets count more than quiet ones;
- contributors per item ``= sum(contributors_t) / sum(items_t)``, with items taken
  from ``raw_count`` when present, else ``value``.

Mean ``top_share >= breadth_top_share_max`` or contributors per item
``< breadth_min_contrib_ratio`` supports a fluke ("one origin drives most of the
volume"); otherwise the base is broad, which is neutral. Skipped when neither
column has recent data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signalcheck.config import Config, get_config
from signalcheck.engine.checks.common import (
    NumberBook,
    highlight,
    make_annotation,
    make_evidence,
    period_word,
    recent_span,
    skip_if_no_windows,
    skipped,
)
from signalcheck.models import Evidence, Preprocessed

CHECK = "breadth"


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """A numeric column as floats, all-NaN when absent."""
    if name not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def weighted_top_share(volume: np.ndarray, top_share: np.ndarray) -> float | None:
    """Volume-weighted mean of ``top_share`` over buckets with both values and volume > 0."""
    ok = np.isfinite(volume) & np.isfinite(top_share) & (volume > 0)
    if not ok.any():
        return None
    return float((volume[ok] * top_share[ok]).sum() / volume[ok].sum())


def contributors_per_item(items: np.ndarray, contributors: np.ndarray) -> float | None:
    """``sum(contributors) / sum(items)`` over buckets with both values and items > 0."""
    ok = np.isfinite(items) & np.isfinite(contributors) & (items > 0)
    if not ok.any():
        return None
    return float(contributors[ok].sum() / items[ok].sum())


def run(pre: Preprocessed, cfg: Config | None = None) -> Evidence:
    """Breadth evidence for a preprocessed series."""
    cfg = cfg if cfg is not None else get_config()
    early = skip_if_no_windows(CHECK, pre, cfg)
    if early is not None:
        return early
    windows = pre.windows
    assert windows is not None
    conf = cfg["checks"]["breadth"]
    word = period_word(pre.series.freq)
    book = NumberBook(cfg)

    recent = pre.series.points.iloc[windows.recent.slice]
    volume = _column(recent, "value")
    raw = _column(recent, "raw_count")
    items = np.where(np.isfinite(raw), raw, volume)
    top_share = _column(recent, "top_share")
    share = weighted_top_share(volume, top_share)
    ratio = contributors_per_item(items, _column(recent, "contributors"))
    if share is None and ratio is None:
        return skipped(CHECK, "no contributor or top-share data for the recent window", book)

    share_max = float(conf["breadth_top_share_max"])
    ratio_min = float(conf["breadth_min_contrib_ratio"])
    one_origin = share is not None and share >= share_max
    few_contributors = ratio is not None and ratio < ratio_min
    fluke = one_origin or few_contributors
    book.put("one_origin", one_origin)
    book.put("few_contributors", few_contributors)

    parts: list[str] = []
    if share is not None:
        parts.append(
            f"the largest single origin supplies {book.pct('top_share_pct', share)} of recent "
            f"volume (volume-weighted mean per {word}; fluke at "
            f"{book.pct('top_share_max_pct', share_max)} or more)"
        )
    if ratio is not None:
        parts.append(
            f"there are {book.stat('contributors_per_item', ratio)} contributors per item "
            f"(fluke below {book.stat('min_contrib_ratio', ratio_min)})"
        )
    detail = " and ".join(parts)
    if fluke:
        stance = "supports_fluke"
        summary = f"Narrow base: {detail}, so a single origin drives most of the volume."
    else:
        stance = "neutral"
        summary = f"Broad base: {detail}."

    ts = pd.DatetimeIndex(recent["ts"])
    decimals = int(cfg["display"]["value_decimals"])
    flagged = [int(i) for i in np.flatnonzero(np.isfinite(top_share) & (top_share >= share_max))]
    annotation = make_annotation(
        points=[
            highlight("single-origin bucket", ts[i], round(float(volume[i]), decimals))
            for i in flagged
        ],
        spans=[recent_span(pre)],
    )
    return make_evidence(CHECK, stance, summary, book, annotation)
