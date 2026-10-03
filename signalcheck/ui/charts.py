"""Annotated Plotly chart for one source card.

The chart shows exactly what the checks saw: the preprocessed regular series
(gaps break the line), points filled for the regular-series checks or imputed by
the adapter drawn hollow, the dropped partial period greyed out, and every
``Evidence.annotation`` overlay (baseline band, recent-window shading, highlighted
points, change-point line, fitted segments, seasonal baseline).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from signalcheck.engine.pipeline import Analysis

SERIES_COLOR = "#1f4e79"
BAND_COLOR = "rgba(31, 119, 180, 0.12)"
RECENT_COLOR = "rgba(255, 193, 7, 0.16)"
THRESHOLD_COLOR = "#9e9e9e"
CHANGE_POINT_COLOR = "#6a1b9a"
PARTIAL_COLOR = "#9e9e9e"

# Highlighted-point styles by annotation label: (colour, marker symbol, size).
POINT_STYLES: dict[str, tuple[str, str, int]] = {
    "beyond baseline band": ("#2e7d32", "circle", 8),
    "largest excess": ("#ef6c00", "diamond", 10),
    "outlier": ("#c62828", "x", 10),
    "single-origin bucket": ("#8e24aa", "square", 8),
}
DEFAULT_POINT_STYLE = ("#c62828", "circle", 8)

# Segment styles by annotation label: (colour, dash, visible-by-default).
SEGMENT_STYLES: dict[str, tuple[str, str, bool]] = {
    "Theil-Sen trend": ("#2e7d32", "solid", True),
    "level before": ("#6a1b9a", "dash", True),
    "level after": ("#6a1b9a", "dash", True),
    "baseline mean rate": ("#616161", "dot", False),
    "recent mean rate": ("#ef6c00", "dot", False),
}
DEFAULT_SEGMENT_STYLE = ("#616161", "dot", False)


def annotation_items(analysis: Analysis, kind: str) -> list[dict[str, Any]]:
    """All annotation entries of one kind across the evidence, de-duplicated, in order."""
    seen: set[tuple[Any, ...]] = set()
    items: list[dict[str, Any]] = []
    for ev in analysis.verdict.evidence:
        for item in (ev.annotation or {}).get(kind, []):
            key = tuple(sorted((k, repr(v)) for k, v in item.items()))
            if key not in seen:
                seen.add(key)
                items.append(item)
    return items


def _dates(values: Iterable[Any]) -> list[str]:
    return [pd.Timestamp(v).date().isoformat() for v in values]


def _period_end(ts: str, freq: str) -> str:
    """Inclusive end of the period starting at ``ts`` (for shading whole periods)."""
    start = pd.Timestamp(ts)
    step = {"D": pd.Timedelta(days=1), "W": pd.Timedelta(days=7)}.get(freq)
    end = start + step if step is not None else start + pd.offsets.MonthBegin(1)
    return pd.Timestamp(end).isoformat()


def add_observed(fig: go.Figure, analysis: Analysis) -> None:
    """The analysed series (gaps left as breaks) and hollow markers for imputed points."""
    points = analysis.pre.series.points
    fig.add_trace(
        go.Scatter(
            x=_dates(points["ts"]),
            y=points["value"],
            mode="lines",
            name=analysis.pre.series.scale.replace("_", " "),
            line={"color": SERIES_COLOR, "width": 1.6},
            connectgaps=False,
            hovertemplate="%{x}: %{y:,.4~g}<extra></extra>",
        )
    )
    filled = analysis.filled.series.points
    if "imputed" in filled.columns:
        imputed = filled[filled["imputed"].fillna(False).astype(bool)]
        if not imputed.empty:
            fig.add_trace(
                go.Scatter(
                    x=_dates(imputed["ts"]),
                    y=imputed["value"],
                    mode="markers",
                    name="imputed (not observed)",
                    marker={
                        "symbol": "circle-open",
                        "size": 8,
                        "color": SERIES_COLOR,
                        "line": {"width": 1.5},
                    },
                    hovertemplate="%{x}: %{y:,.4~g} (imputed)<extra></extra>",
                )
            )


def add_dropped_partial(fig: go.Figure, analysis: Analysis) -> None:
    """Grey marker (and dotted link) for the partial last period that was not analysed."""
    dropped = analysis.pre.series.meta.get("dropped_partial")
    if not dropped:
        return
    points = analysis.pre.series.points.dropna(subset=["value"])
    xs: list[str] = []
    ys: list[float] = []
    if not points.empty:
        xs.append(_dates([points["ts"].iloc[-1]])[0])
        ys.append(float(points["value"].iloc[-1]))
    xs.append(_dates([dropped["ts"]])[0])
    ys.append(float(dropped["value"]))
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name="partial period (dropped)",
            line={"color": PARTIAL_COLOR, "dash": "dot", "width": 1.2},
            marker={"color": PARTIAL_COLOR, "size": [0] * (len(xs) - 1) + [8]},
            hovertemplate="%{x}: %{y:,.4~g} (incomplete, not analysed)<extra></extra>",
        )
    )


def add_recent_window(fig: go.Figure, analysis: Analysis) -> None:
    """Shade the recent window (whole periods), falling back to ``Verdict.window``."""
    freq = analysis.pre.series.freq
    spans = annotation_items(analysis, "spans")
    if not spans and analysis.verdict.window:
        w = analysis.verdict.window
        spans = [{"label": "recent window", "start": w["recent_start"], "end": w["recent_end"]}]
    for i, span in enumerate(spans):
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=pd.Timestamp(span["start"]).isoformat(),
            x1=_period_end(span["end"], freq),
            y0=0,
            y1=1,
            fillcolor=RECENT_COLOR,
            line={"width": 0},
            layer="below",
            name=span.get("label", "recent window"),
            showlegend=i == 0,
        )


def data_peak(analysis: Analysis) -> float | None:
    """Highest value the chart plots (observed, filled, highlighted, fitted or baseline band)."""
    values: list[float] = [
        *analysis.pre.series.points["value"].tolist(),
        *analysis.filled.series.points["value"].tolist(),
        *(float(p["value"]) for p in annotation_items(analysis, "points")),
        *(
            float(s[k])
            for s in annotation_items(analysis, "segments")
            for k in ("start_value", "end_value")
        ),
        *(
            float(p["value"])
            for line in annotation_items(analysis, "lines")
            for p in line.get("points", [])
        ),
        *(
            float(b["upper"])
            for b in annotation_items(analysis, "bands")
            if b.get("label") == "baseline band"
        ),
    ]
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else None


def add_bands(fig: go.Figure, analysis: Analysis, ceiling: float | None = None) -> None:
    """Baseline band (filled), outlier thresholds (dotted) and baseline medians (dashed).

    An upper threshold above ``ceiling`` is left off so it doesn't squash the series.
    """
    freq = analysis.pre.series.freq
    shown: set[str] = set()
    for band in annotation_items(analysis, "bands"):
        label = str(band.get("label", "band"))
        x0 = pd.Timestamp(band["start"]).isoformat()
        x1 = _period_end(band["end"], freq)
        lower, upper = float(band["lower"]), float(band["upper"])
        common: dict[str, Any] = {
            "xref": "x",
            "yref": "y",
            "x0": x0,
            "x1": x1,
            "name": label,
            "showlegend": label not in shown,
            "legendgroup": label,
        }
        shown.add(label)
        if label == "baseline band" and upper > lower:
            fig.add_shape(
                type="rect",
                y0=lower,
                y1=upper,
                fillcolor=BAND_COLOR,
                line={"width": 0},
                layer="below",
                **common,
            )
        elif lower == upper:
            fig.add_shape(
                type="line",
                y0=lower,
                y1=lower,
                line={"color": THRESHOLD_COLOR, "dash": "dash", "width": 1},
                **common,
            )
        else:
            draw_upper = ceiling is None or upper <= ceiling
            if draw_upper:
                fig.add_shape(
                    type="line",
                    y0=upper,
                    y1=upper,
                    line={"color": THRESHOLD_COLOR, "dash": "dot", "width": 1},
                    **common,
                )
            if lower > 0:
                fig.add_shape(
                    type="line",
                    y0=lower,
                    y1=lower,
                    line={"color": THRESHOLD_COLOR, "dash": "dot", "width": 1},
                    **{**common, "showlegend": common["showlegend"] and not draw_upper},
                )
            elif not draw_upper:
                shown.discard(label)


def point_group(label: str) -> str:
    """Legend group for a highlighted point: ``top 1 excess``, ``top 2 excess`` -> one entry."""
    return "largest excess" if re.fullmatch(r"top \d+ excess", label) else label


def add_points(fig: go.Figure, analysis: Analysis) -> None:
    """Highlighted points (spikes, outliers, single-origin buckets), one trace per group."""
    by_label: dict[str, list[dict[str, Any]]] = {}
    for point in annotation_items(analysis, "points"):
        by_label.setdefault(point_group(str(point.get("label", "highlighted"))), []).append(point)
    for label, points in by_label.items():
        color, symbol, size = POINT_STYLES.get(label, DEFAULT_POINT_STYLE)
        fig.add_trace(
            go.Scatter(
                x=_dates(p["ts"] for p in points),
                y=[float(p["value"]) for p in points],
                mode="markers",
                name=label,
                text=[str(p.get("label", label)) for p in points],
                marker={"color": color, "symbol": symbol, "size": size},
                hovertemplate="%{x}: %{y:,.4~g} (%{text})<extra></extra>",
            )
        )


def add_segments(fig: go.Figure, analysis: Analysis) -> None:
    """Fitted lines: Theil-Sen trend, levels before/after, mean rates (hidden by default)."""
    for seg in annotation_items(analysis, "segments"):
        label = str(seg.get("label", "segment"))
        color, dash, visible = SEGMENT_STYLES.get(label, DEFAULT_SEGMENT_STYLE)
        fig.add_trace(
            go.Scatter(
                x=_dates([seg["start"], seg["end"]]),
                y=[float(seg["start_value"]), float(seg["end_value"])],
                mode="lines",
                name=label,
                line={"color": color, "dash": dash, "width": 2},
                visible=True if visible else "legendonly",
                hovertemplate=f"{label}: %{{y:,.4~g}}<extra></extra>",
            )
        )


def add_lines(fig: go.Figure, analysis: Analysis) -> None:
    """Multi-point reference lines (the seasonal baseline)."""
    for line in annotation_items(analysis, "lines"):
        pts = line.get("points", [])
        if not pts:
            continue
        fig.add_trace(
            go.Scatter(
                x=_dates(p["ts"] for p in pts),
                y=[float(p["value"]) for p in pts],
                mode="lines",
                name=str(line.get("label", "reference")),
                line={"color": "#8e24aa", "dash": "dot", "width": 1.5},
                hovertemplate="%{x}: %{y:,.4~g}<extra></extra>",
            )
        )


def add_change_points(fig: go.Figure, analysis: Analysis) -> None:
    """Dashed vertical line at each detected change point."""
    for i, vline in enumerate(annotation_items(analysis, "vlines")):
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=pd.Timestamp(vline["ts"]).isoformat(),
            x1=pd.Timestamp(vline["ts"]).isoformat(),
            y0=0,
            y1=1,
            line={"color": CHANGE_POINT_COLOR, "dash": "dash", "width": 1.5},
            name=str(vline.get("label", "change point")),
            showlegend=i == 0,
        )


def build_chart(
    analysis: Analysis, height: int = 320, threshold_headroom: float | None = None
) -> go.Figure:
    """The annotated chart for one analysed source.

    With ``threshold_headroom``, outlier-threshold lines more than that multiple of the
    highest plotted value are omitted (``ui.threshold_headroom`` in config.yaml).
    """
    fig = go.Figure()
    add_recent_window(fig, analysis)
    peak = data_peak(analysis)
    ceiling = peak * threshold_headroom if peak is not None and threshold_headroom else None
    add_bands(fig, analysis, ceiling)
    add_observed(fig, analysis)
    add_lines(fig, analysis)
    add_segments(fig, analysis)
    add_points(fig, analysis)
    add_change_points(fig, analysis)
    add_dropped_partial(fig, analysis)
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        hovermode="closest",
        legend={
            "orientation": "h",
            "yref": "container",
            "yanchor": "bottom",
            "y": 0,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11},
        },
        xaxis={"title": None},
        yaxis={"title": None, "rangemode": "tozero"},
    )
    return fig
