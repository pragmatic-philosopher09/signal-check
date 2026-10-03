"""Card and summary view models: plain data the Streamlit layer renders as-is.

Everything shown on a card is derived here from the engine's validated output
(``Verdict``, ``Evidence``, ``Series.meta``), so the rendering in ``app.py`` stays
a thin loop and the wording can be unit-tested without Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from signalcheck.engine.cross_source import CrossSourceSummary, display_name
from signalcheck.engine.preprocess import PERIOD_WORDS
from signalcheck.engine.verdict import CHECK_WORDS, RULES, STANCE_WORDS
from signalcheck.models import Evidence, Verdict
from signalcheck.ui.runner import SourceOutcome, TopicResult

BadgeColor = Literal["red", "orange", "yellow", "blue", "green", "violet", "gray", "primary"]

LABEL_STYLES: dict[str, tuple[BadgeColor, str]] = {
    "TREND_up": ("green", ":material/trending_up:"),
    "TREND_down": ("red", ":material/trending_down:"),
    "FLUKE": ("orange", ":material/bolt:"),
    "SEASONAL": ("violet", ":material/event_repeat:"),
    "NO_CHANGE": ("blue", ":material/trending_flat:"),
    "INCONCLUSIVE": ("gray", ":material/help:"),
}
LABEL_WORDS: dict[str, str] = {
    "TREND": "Trend",
    "FLUKE": "Fluke",
    "SEASONAL": "Seasonal",
    "NO_CHANGE": "No change",
    "INCONCLUSIVE": "Inconclusive",
}
DIRECTION_ARROWS: dict[str, str] = {"up": "\u2191", "down": "\u2193"}
STANCE_ICONS: dict[str, str] = {
    "supports_trend": ":material/trending_up:",
    "supports_fluke": ":material/bolt:",
    "supports_seasonal": ":material/event_repeat:",
    "supports_no_change": ":material/trending_flat:",
    "neutral": ":material/remove:",
}
DASH = "\u2013"


@dataclass(frozen=True)
class Badge:
    """Text, ``st.badge`` colour and Material icon of a verdict badge."""

    text: str
    color: BadgeColor
    icon: str


@dataclass(frozen=True)
class EvidenceItem:
    """One check's bullet: name, stance words and its one-sentence summary."""

    check: str
    name: str
    stance: str
    stance_text: str
    icon: str
    summary: str


@dataclass(frozen=True)
class SkippedCheck:
    """A check that did not run, with the reason."""

    name: str
    reason: str


@dataclass(frozen=True)
class WikiArticle:
    """Resolved Wikipedia article for the override box."""

    article: str
    overridden: bool
    candidates: list[str]


@dataclass(frozen=True)
class CardView:
    """Everything one source card shows. ``message`` is set instead for failures."""

    source: str
    title: str
    message: str | None = None
    disabled: bool = False
    badge: Badge | None = None
    direction: str | None = None
    confidence: str | None = None
    rule: str | None = None
    rule_text: str | None = None
    reason: str | None = None
    window_text: str | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    skipped: list[SkippedCheck] = field(default_factory=list)
    change_my_mind: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    wiki: WikiArticle | None = None
    snapshot: bool = False

    @property
    def ok(self) -> bool:
        """True when the card has a verdict to show."""
        return self.badge is not None


@dataclass(frozen=True)
class SummaryView:
    """Top cross-source card: the engine's sentence plus per-source chips."""

    sentence: str
    chips: list[tuple[str, Badge]]
    unavailable: list[str]
    not_comparable: list[str]


def label_key(label: str, direction: str | None) -> str:
    """``TREND_up``/``TREND_down`` for trends, else the label."""
    return f"{label}_{direction}" if label == "TREND" and direction else label


def verdict_badge(verdict: Verdict) -> Badge:
    """Badge for a verdict: label words, arrow for trends, colour and icon."""
    color, icon = LABEL_STYLES[label_key(verdict.label, verdict.direction)]
    text = LABEL_WORDS[verdict.label]
    if verdict.label == "TREND" and verdict.direction:
        text = f"{text} {DIRECTION_ARROWS[verdict.direction]} {verdict.direction}"
    return Badge(text, color, icon)


def sentence_case(text: str) -> str:
    """First letter upper-cased, the rest untouched (keeps "MAD", "R3", ...)."""
    return text[:1].upper() + text[1:]


def check_name(check: str) -> str:
    """Capitalised display name of a check."""
    return CHECK_WORDS.get(check, check.replace("_", " ")).capitalize()


def evidence_item(ev: Evidence) -> EvidenceItem:
    """Bullet for a check that ran."""
    words = STANCE_WORDS.get(ev.stance, ev.stance)
    stance_text = "neutral" if ev.stance == "neutral" else f"supports {words}"
    return EvidenceItem(
        check=ev.check,
        name=check_name(ev.check),
        stance=ev.stance,
        stance_text=stance_text,
        icon=STANCE_ICONS.get(ev.stance, ":material/remove:"),
        summary=ev.summary,
    )


def period_phrase(n: int, freq: str) -> str:
    """``"28 days"`` / ``"1 week"``."""
    word = PERIOD_WORDS.get(freq, "period")
    return f"{n} {word}{'' if n == 1 else 's'}"


def window_text(verdict: Verdict, freq: str) -> str | None:
    """Plain description of the recent and baseline windows the verdict compared."""
    w = verdict.window
    if not w:
        return None
    return (
        f"Recent {w['recent_start']} {DASH} {w['recent_end']} "
        f"({period_phrase(int(w['n_recent']), freq)}) vs baseline "
        f"{w['baseline_start']} {DASH} {w['baseline_end']} "
        f"({period_phrase(int(w['n_baseline']), freq)})"
    )


def wiki_article(outcome: SourceOutcome) -> WikiArticle | None:
    """Resolved article metadata from a Wikipedia series."""
    if outcome.source != "wikipedia" or outcome.series is None:
        return None
    meta = outcome.series.meta
    article = meta.get("article") or meta.get("resolved_query")
    if not article:
        return None
    candidates = [str(c) for c in meta.get("candidates") or [] if c != article]
    return WikiArticle(str(article), bool(meta.get("article_overridden")), candidates)


def card_view(outcome: SourceOutcome) -> CardView:
    """View model for one source card."""
    snapshot = bool(outcome.series is not None and outcome.series.meta.get("snapshot"))
    wiki = wiki_article(outcome)
    if outcome.analysis is None:
        return CardView(
            source=outcome.source,
            title=outcome.label,
            message=outcome.message or f"Couldn't fetch {outcome.label}: no data",
            disabled=outcome.disabled,
            wiki=wiki,
            snapshot=snapshot,
        )
    verdict = outcome.analysis.verdict
    freq = outcome.analysis.pre.series.freq
    ran = [ev for ev in verdict.evidence if ev.stance != "skipped"]
    skipped = [
        SkippedCheck(check_name(ev.check), ev.skip_reason or "skipped")
        for ev in verdict.evidence
        if ev.stance == "skipped"
    ]
    return CardView(
        source=outcome.source,
        title=outcome.label,
        badge=verdict_badge(verdict),
        direction=verdict.direction,
        confidence=verdict.confidence,
        rule=verdict.rule_fired,
        rule_text=RULES.get(verdict.rule_fired),
        reason=sentence_case(verdict.reason),
        window_text=window_text(verdict, freq),
        evidence=[evidence_item(ev) for ev in ran],
        skipped=skipped,
        change_my_mind=[str(c) for c in verdict.change_my_mind],
        caveats=list(verdict.caveats),
        wiki=wiki,
        snapshot=snapshot,
    )


def summary_view(result: TopicResult) -> SummaryView | None:
    """Top card for the cross-source summary (``None`` when nothing was requested)."""
    summary: CrossSourceSummary | None = result.summary
    if summary is None:
        return None
    chips = [
        (o.label, verdict_badge(o.verdict))
        for o in result.outcomes
        if o.verdict is not None and o.source in summary.comparable
    ]
    return SummaryView(
        sentence=summary.sentence,
        chips=chips,
        unavailable=[display_name(s) for s in summary.unavailable],
        not_comparable=[display_name(s) for s in summary.not_comparable],
    )
