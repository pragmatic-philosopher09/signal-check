"""Template narration placeholder (Phase 6 adds the LLM path)."""

from __future__ import annotations

from signalcheck.config import Config
from signalcheck.narrate import AI_LABEL, TEMPLATE_LABEL, template_summary
from signalcheck.ui import runner


def test_template_restates_the_verdict_only(cfg: Config) -> None:
    result = runner.run_topic("rust programming", ["wikipedia"], cfg, sample=True)
    verdict = result.outcomes[0].verdict
    assert verdict is not None
    text = template_summary(verdict, "Wikipedia")
    assert text.startswith("Wikipedia shows a sustained upward trend (TREND, medium confidence")
    assert "rule R4" in text
    assert verdict.change_my_mind[0] in text


def test_labels_are_distinct() -> None:
    assert AI_LABEL == "AI-written summary of the findings above"
    assert "template" in TEMPLATE_LABEL
