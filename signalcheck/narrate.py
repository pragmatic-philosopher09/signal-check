"""LLM narration of validated findings with template fallback (Phase 6).

Phase 5 ships only the template path: :func:`template_summary` restates the
verdict's own fields (label, confidence, rule, reason and the first
change-my-mind condition) without adding any number of its own. The LLM call and
the numbers/dates/label validator from COPILOT_BRIEF.md section 9.1 arrive in
Phase 6 and will fall back to this text.
"""

from __future__ import annotations

from signalcheck.models import Verdict

AI_LABEL = "AI-written summary of the findings above"
TEMPLATE_LABEL = "Summary of the findings above (template text; AI narration is not enabled yet)"

LABEL_PHRASES: dict[str, str] = {
    "TREND": "shows a sustained {direction}ward trend",
    "FLUKE": "looks like a fluke",
    "SEASONAL": "looks seasonal",
    "NO_CHANGE": "shows no meaningful change",
    "INCONCLUSIVE": "is inconclusive",
}


def template_summary(verdict: Verdict, source: str) -> str:
    """Deterministic one-paragraph summary built only from ``verdict``'s fields."""
    phrase = LABEL_PHRASES[verdict.label].format(direction=verdict.direction or "")
    text = (
        f"{source} {phrase} ({verdict.label}, {verdict.confidence} confidence, "
        f"rule {verdict.rule_fired})."
    )
    if verdict.reason:
        reason = verdict.reason.rstrip(".")
        text += f" {reason[:1].upper()}{reason[1:]}."
    if verdict.change_my_mind:
        text += f" {verdict.change_my_mind[0]}"
    return text
