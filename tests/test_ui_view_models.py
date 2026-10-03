"""Card and summary view models (signalcheck.ui.view_models)."""

from __future__ import annotations

import pytest

from signalcheck.adapters.base import AdapterError
from signalcheck.config import Config
from signalcheck.engine.verdict import RULES
from signalcheck.models import Evidence, Verdict
from signalcheck.ui import runner
from signalcheck.ui.view_models import (
    LABEL_STYLES,
    card_view,
    evidence_item,
    period_phrase,
    summary_view,
    verdict_badge,
    window_text,
)
from tests.ui_fakes import FakeAdapter, no_network_factories, snapshot_as_live


def make_verdict(label: str, direction: str | None = None, **kw: object) -> Verdict:
    fields: dict[str, object] = {
        "confidence": "low",
        "rule_fired": "R6",
        "evidence": [],
        "change_my_mind": [],
        "caveats": [],
        "window": {},
    }
    fields.update(kw)
    return Verdict(label=label, direction=direction, **fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("label", "direction", "text", "color"),
    [
        ("TREND", "up", "Trend \u2191 up", "green"),
        ("TREND", "down", "Trend \u2193 down", "red"),
        ("FLUKE", None, "Fluke", "orange"),
        ("SEASONAL", None, "Seasonal", "violet"),
        ("NO_CHANGE", None, "No change", "blue"),
        ("INCONCLUSIVE", None, "Inconclusive", "gray"),
    ],
)
def test_verdict_badge(label: str, direction: str | None, text: str, color: str) -> None:
    badge = verdict_badge(make_verdict(label, direction))
    assert (badge.text, badge.color) == (text, color)
    assert badge.icon.startswith(":material/")


def test_every_label_has_a_style() -> None:
    assert set(LABEL_STYLES) == {
        "TREND_up",
        "TREND_down",
        "FLUKE",
        "SEASONAL",
        "NO_CHANGE",
        "INCONCLUSIVE",
    }


def test_evidence_item_words() -> None:
    ev = Evidence("level_shift", "supports_trend", "A step up.", {}, None)
    item = evidence_item(ev)
    assert (item.name, item.stance_text, item.summary) == (
        "Level shift",
        "supports trend",
        "A step up.",
    )
    neutral = evidence_item(Evidence("low_count", "neutral", "Fine.", {}, None))
    assert neutral.stance_text == "neutral"


def test_window_text_uses_the_series_frequency() -> None:
    verdict = make_verdict(
        "NO_CHANGE",
        window={
            "baseline_start": "2025-01-06",
            "baseline_end": "2025-06-30",
            "recent_start": "2025-07-07",
            "recent_end": "2025-09-29",
            "recent_end_exclusive": "2025-10-06",
            "n_baseline": 26,
            "n_recent": 1,
        },
    )
    text = window_text(verdict, "W")
    assert text == (
        "Recent 2025-07-07 \u2013 2025-09-29 (1 week) vs baseline "
        "2025-01-06 \u2013 2025-06-30 (26 weeks)"
    )
    assert window_text(make_verdict("INCONCLUSIVE"), "D") is None
    assert period_phrase(3, "M") == "3 months"


def test_card_view_for_a_sample(cfg: Config) -> None:
    result = runner.run_topic("rust programming", ["wikipedia"], cfg, sample=True)
    (outcome,) = result.outcomes
    view = card_view(outcome)
    assert view.ok and view.message is None and view.snapshot
    assert view.badge is not None and view.badge.text == "Trend \u2191 up"
    assert view.rule == "R4" and view.rule_text == RULES["R4"]
    assert view.reason is not None and view.reason[0].isupper()
    assert view.confidence == "medium" and view.direction == "up"
    assert view.window_text is not None and view.window_text.startswith("Recent 2026-09-05")
    ran = {e.check for e in view.evidence}
    skipped = {s.name for s in view.skipped}
    assert "persistence" in ran and "Seasonality" in skipped
    assert len(ran) + len(skipped) == 7
    assert view.change_my_mind and all(isinstance(c, str) for c in view.change_my_mind)
    assert any("snapshot" in c for c in view.caveats)
    assert view.wiki is not None
    assert view.wiki.article == "Rust (programming language)" and not view.wiki.overridden
    assert view.wiki.article not in view.wiki.candidates


def test_card_view_for_a_failure(cfg: Config) -> None:
    factories = no_network_factories(
        wikipedia=FakeAdapter("wikipedia", error=AdapterError("no Wikipedia article found for 'x'"))
    )
    (outcome,) = runner.run_topic("x", ["wikipedia"], cfg, factories=factories).outcomes
    view = card_view(outcome)
    assert not view.ok and view.badge is None and view.wiki is None
    assert view.message == "Couldn't fetch Wikipedia: no Wikipedia article found for 'x'"


def test_summary_view_lists_unavailable_sources(cfg: Config) -> None:
    factories = no_network_factories(
        wikipedia=FakeAdapter("wikipedia", error=AdapterError("HTTP 503")),
        hackernews=FakeAdapter("hackernews", series=snapshot_as_live("hackernews", "chatgpt")),
    )
    result = runner.run_topic("chatgpt", ["wikipedia", "hackernews", "x"], cfg, factories=factories)
    summary = summary_view(result)
    assert summary is not None
    assert summary.unavailable == ["Wikipedia"]
    assert [label for label, _ in summary.chips] == ["Hacker News"]
    assert "Hacker News" in summary.sentence or "1 of 1" in summary.sentence


def test_summary_view_none_without_sources(cfg: Config) -> None:
    result = runner.run_topic("chatgpt", [], cfg, factories=no_network_factories())
    assert summary_view(result) is None
