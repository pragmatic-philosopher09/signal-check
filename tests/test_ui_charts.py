"""Annotated Plotly charts built from Evidence.annotation and meta (signalcheck.ui.charts)."""

from __future__ import annotations

import numpy as np

from signalcheck.config import Config
from signalcheck.engine import analyse_detailed
from signalcheck.engine.pipeline import Analysis
from signalcheck.ui import runner
from signalcheck.ui.charts import annotation_items, build_chart
from tests.conftest import SeriesFactory


def sample_analysis(cfg: Config, query: str, source: str) -> Analysis:
    (outcome,) = runner.run_topic(query, [source], cfg, sample=True).outcomes
    assert outcome.analysis is not None
    return outcome.analysis


def names(fig: object) -> list[str]:
    return [t.name for t in fig.data]  # type: ignore[attr-defined]


def shape_names(fig: object) -> list[str]:
    return [s.name for s in fig.layout.shapes]  # type: ignore[attr-defined]


def test_trend_chart_has_band_recent_window_points_and_change_point(cfg: Config) -> None:
    analysis = sample_analysis(cfg, "rust programming", "wikipedia")
    fig = build_chart(analysis, height=300)
    assert fig.layout.height == 300
    assert names(fig)[0] == "pageviews"
    assert "beyond baseline band" in names(fig)
    shapes = shape_names(fig)
    assert {"recent window", "baseline band", "change point"} <= set(shapes)
    recent = fig.layout.shapes[shapes.index("recent window")]
    window = analysis.verdict.window
    assert str(recent.x0).startswith(window["recent_start"])
    cp = fig.layout.shapes[shapes.index("change point")]
    assert str(cp.x0).startswith("2026-09-21")


def test_dropped_partial_period_is_greyed(cfg: Config) -> None:
    analysis = sample_analysis(cfg, "chatgpt", "hackernews")
    fig = build_chart(analysis)
    trace = fig.data[names(fig).index("partial period (dropped)")]
    assert str(trace.x[-1]).startswith(analysis.pre.series.meta["dropped_partial"]["ts"][:10])
    assert trace.line.dash == "dot"


def test_imputed_points_are_hollow(cfg: Config) -> None:
    analysis = sample_analysis(cfg, "perplexity ai", "wikipedia")
    fig = build_chart(analysis)
    trace = fig.data[names(fig).index("imputed (not observed)")]
    assert trace.marker.symbol == "circle-open"
    assert len(trace.x) == int(
        analysis.filled.series.points["imputed"].fillna(False).astype(bool).sum()
    )


def test_spike_points_are_highlighted(cfg: Config) -> None:
    analysis = sample_analysis(cfg, "taylor swift", "hackernews")
    fig = build_chart(analysis)
    assert "single-origin bucket" in names(fig)
    assert fig.data[names(fig).index("recent mean rate")].visible == "legendonly"


def test_annotation_items_are_deduplicated(cfg: Config) -> None:
    analysis = sample_analysis(cfg, "rust programming", "wikipedia")
    spans = annotation_items(analysis, "spans")
    assert len(spans) == len({(s["start"], s["end"]) for s in spans})


def test_short_series_still_draws(make_series: SeriesFactory, cfg: Config) -> None:
    analysis = analyse_detailed(make_series(np.full(5, 10.0)), cfg)
    assert analysis.verdict.rule_fired == "R1"
    fig = build_chart(analysis)
    assert names(fig) == ["relative 0 100"]
    assert list(fig.layout.shapes) == []


def test_gaps_are_not_bridged(make_series: SeriesFactory, cfg: Config) -> None:
    rng = np.random.default_rng(3)
    values = 50 + rng.normal(0, 1, 140)
    values[60:70] = np.nan
    analysis = analyse_detailed(make_series(values), cfg)
    fig = build_chart(analysis)
    observed = fig.data[0]
    assert observed.connectgaps is False


def test_ranked_excess_points_share_one_legend_entry(cfg: Config) -> None:
    analysis = sample_analysis(cfg, "perplexity ai", "wikipedia")
    fig = build_chart(analysis)
    assert "largest excess" in names(fig)
    assert not any(n.startswith("top ") for n in names(fig))
    trace = fig.data[names(fig).index("largest excess")]
    assert list(trace.text) == ["top 1 excess", "top 2 excess"]


def test_far_threshold_is_left_off_so_the_series_stays_readable(cfg: Config) -> None:
    analysis = sample_analysis(cfg, "taylor swift", "hackernews")
    full = build_chart(analysis)
    upper = max(s.y0 for s in full.layout.shapes if s.name == "outlier threshold")
    peak = float(analysis.pre.series.points["value"].max())
    assert upper > 1.5 * peak
    capped = build_chart(analysis, threshold_headroom=float(cfg["ui"]["threshold_headroom"]))
    assert all(s.y0 <= 1.5 * peak for s in capped.layout.shapes if s.name == "outlier threshold")
    assert "baseline band" in shape_names(capped)


def test_near_threshold_is_kept(cfg: Config) -> None:
    analysis = sample_analysis(cfg, "rust programming", "wikipedia")
    capped = build_chart(analysis, threshold_headroom=1.5)
    assert shape_names(capped).count("outlier threshold") == 2
