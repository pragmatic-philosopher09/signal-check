"""JSON export of engine results for the static React frontend.

The browser never recomputes a statistic: every verdict, number, sentence and
chart mark comes from the engine and is serialised here as plain JSON (the same
data the Streamlit view models render). The frontend only lays it out. This is
used twice: at build time over the committed sample snapshots
(``scripts/build_web.py``), and at run time inside the Pyodide web worker
(:mod:`signalcheck.web.worker`).

Schema (``SCHEMA_VERSION``) -- a topic or CSV result::

    {schema, kind: "topic"|"csv", query, sample, generated_at,
     summary: {sentence, chips: [{source, title, badge}], unavailable, not_comparable} | null,
     cards: [{source, title, message, disabled, snapshot, badge, label, direction,
              confidence, rule, rule_text, reason, window_text, evidence, skipped,
              change_my_mind, caveats, wiki, narration, chart}]}

``chart`` holds the analysed series and the raw ``Evidence.annotation`` dicts,
which the frontend maps to marks exactly as :mod:`signalcheck.ui.charts` does.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

from signalcheck.adapters.base import SOURCE_LABELS
from signalcheck.config import APP_VERSION, Config
from signalcheck.engine.pipeline import Analysis
from signalcheck.engine.verdict import RULES
from signalcheck.narrate import TEMPLATE_LABEL, Narration
from signalcheck.ui.methodology import methodology_sections
from signalcheck.ui.runner import (
    CsvResult,
    SourceOutcome,
    SourceStatus,
    TopicResult,
    disabled_outcome,
    source_label,
    summarise,
    template_narrations,
)
from signalcheck.ui.view_models import (
    LABEL_STYLES,
    LABEL_WORDS,
    Badge,
    card_view,
    label_key,
    summary_view,
    verdict_badge,
)

SCHEMA_VERSION = 1

# Sources the static site queries live from the browser (CORS-enabled, keyless APIs).
LIVE_SOURCES: tuple[str, ...] = ("wikipedia", "hackernews")

# Why the other topic sources are not queried live on the static demo.
STATIC_UNAVAILABLE: dict[str, str] = {
    "google_trends": "no keyless public API; upload a Google Trends CSV instead",
    "reddit": "needs server-side credentials; not available on the static demo",
    "x": "needs server-side credentials; not available on the static demo",
}


def jsonable(value: Any) -> Any:
    """``value`` as strict JSON: NaN/inf -> ``None``, dates -> ISO, numpy -> Python."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [jsonable(v) for v in value]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return _day(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)


def _day(ts: Any) -> str:
    return pd.Timestamp(ts).date().isoformat()


def _number(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def badge_json(badge: Badge, key: str) -> dict[str, str]:
    """Badge text, colour name and verdict key (``TREND_up`` ... ``INCONCLUSIVE``)."""
    return {"text": badge.text, "color": badge.color, "key": key}


def chart_json(analysis: Analysis) -> dict[str, Any]:
    """Everything the annotated chart draws, straight from the analysis.

    ``observed``: the analysed regular series ``[date, value|null]`` (gaps break the
    line); ``imputed``: filled points drawn hollow; ``dropped_partial``: the
    incomplete last period that was not analysed; ``annotations``: each check's
    ``Evidence.annotation`` in evidence order (the frontend de-duplicates them).
    """
    pre = analysis.pre.series
    observed = [
        [_day(ts), _number(v)] for ts, v in zip(pre.points["ts"], pre.points["value"], strict=True)
    ]
    filled = analysis.filled.series.points
    imputed: list[list[Any]] = []
    if "imputed" in filled.columns:
        mask = filled["imputed"].fillna(False).astype(bool)
        imputed = [
            [_day(ts), _number(v)]
            for ts, v in zip(filled["ts"][mask], filled["value"][mask], strict=True)
        ]
    dropped = pre.meta.get("dropped_partial")
    return {
        "freq": pre.freq,
        "scale": pre.scale,
        "observed": observed,
        "imputed": imputed,
        "dropped_partial": jsonable(dropped) if dropped else None,
        "window": jsonable(analysis.verdict.window) if analysis.verdict.window else None,
        "annotations": [
            jsonable(ev.annotation) for ev in analysis.verdict.evidence if ev.annotation
        ],
    }


def card_json(outcome: SourceOutcome, narration: Narration | None = None) -> dict[str, Any]:
    """One source card: the Streamlit card view model plus narration and chart data."""
    view = card_view(outcome)
    verdict = outcome.verdict
    card: dict[str, Any] = {
        "source": view.source,
        "title": view.title,
        "message": view.message,
        "disabled": view.disabled,
        "snapshot": view.snapshot,
        "badge": None,
        "label": None,
        "direction": view.direction,
        "confidence": view.confidence,
        "rule": view.rule,
        "rule_text": view.rule_text,
        "reason": view.reason,
        "window_text": view.window_text,
        "evidence": [
            {
                "check": e.check,
                "name": e.name,
                "stance": e.stance,
                "stance_text": e.stance_text,
                "summary": e.summary,
            }
            for e in view.evidence
        ],
        "skipped": [{"name": s.name, "reason": s.reason} for s in view.skipped],
        "change_my_mind": list(view.change_my_mind),
        "caveats": list(view.caveats),
        "wiki": (
            {
                "article": view.wiki.article,
                "overridden": view.wiki.overridden,
                "candidates": list(view.wiki.candidates),
            }
            if view.wiki is not None
            else None
        ),
        "narration": (
            {"text": narration.text, "label": narration.label} if narration is not None else None
        ),
        "chart": chart_json(outcome.analysis) if outcome.analysis is not None else None,
    }
    if verdict is not None and view.badge is not None:
        card["badge"] = badge_json(view.badge, label_key(verdict.label, verdict.direction))
        card["label"] = verdict.label
    return card


def summary_json(result: TopicResult) -> dict[str, Any] | None:
    """The cross-source summary card (``None`` when no enabled source was requested)."""
    view = summary_view(result)
    if view is None or result.summary is None:
        return None
    chips = [
        {
            "source": o.source,
            "title": o.label,
            "badge": badge_json(
                verdict_badge(o.verdict), label_key(o.verdict.label, o.verdict.direction)
            ),
        }
        for o in result.outcomes
        if o.verdict is not None and o.source in result.summary.comparable
    ]
    return {
        "sentence": view.sentence,
        "chips": chips,
        "unavailable": list(view.unavailable),
        "not_comparable": list(view.not_comparable),
    }


def result_json(
    result: TopicResult, cfg: Config, *, kind: str = "topic", generated_at: datetime | None = None
) -> dict[str, Any]:
    """A whole topic (or CSV) result as JSON, with template narration on every card."""
    narrations = template_narrations(result.outcomes, cfg)
    when = generated_at if generated_at is not None else datetime.now(UTC)
    return {
        "schema": SCHEMA_VERSION,
        "kind": kind,
        "query": result.query,
        "sample": result.sample,
        "generated_at": when.astimezone(UTC).isoformat(timespec="seconds"),
        "summary": summary_json(result),
        "cards": [card_json(o, narrations.get(o.source)) for o in result.outcomes],
    }


def with_static_cards(result: TopicResult, cfg: Config, present: Sequence[str]) -> TopicResult:
    """Append a disabled card (with the static-demo reason) for every source not queried."""
    outcomes = list(result.outcomes)
    for source, reason in STATIC_UNAVAILABLE.items():
        if source not in present:
            status = SourceStatus(source, source_label(source), False, reason)
            outcomes.append(disabled_outcome(status))
    return TopicResult(result.query, outcomes, summarise(outcomes, cfg), sample=result.sample)


def csv_json(csv: CsvResult, cfg: Config) -> dict[str, Any]:
    """A CSV run: ``{"columns": [...]}`` when a column must be chosen, else a result."""
    if csv.columns is not None:
        return {"schema": SCHEMA_VERSION, "kind": "columns", "columns": list(csv.columns)}
    assert csv.result is not None
    return result_json(csv.result, cfg, kind="csv")


def site_json(cfg: Config, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Static site metadata: methodology from the live config, labels, sources, samples."""
    return {
        "schema": SCHEMA_VERSION,
        "version": APP_VERSION,
        "methodology": [{"title": s.title, "body": s.body} for s in methodology_sections(cfg)],
        "rules": dict(RULES),
        "labels": {
            key: {
                "text": LABEL_WORDS[key.removesuffix("_up").removesuffix("_down")],
                "color": color,
            }
            for key, (color, _icon) in LABEL_STYLES.items()
        },
        "sources": [
            {
                "source": source,
                "title": SOURCE_LABELS[source],
                "live": source in LIVE_SOURCES,
                "reason": STATIC_UNAVAILABLE.get(source),
            }
            for source in (*LIVE_SOURCES, *STATIC_UNAVAILABLE)
        ],
        "ui": {
            "timeframe_days": list(cfg["ui"]["timeframe_days"]),
            "chart_height_px": int(cfg["ui"]["chart_height_px"]),
            "threshold_headroom": float(cfg["ui"]["threshold_headroom"]),
            "history_days": {s: int(cfg["adapters"][s]["history_days"]) for s in LIVE_SOURCES},
            "template_label": TEMPLATE_LABEL,
        },
        "samples": [dict(s) for s in samples],
    }
