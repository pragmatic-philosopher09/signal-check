"""Signal Check: Streamlit UI (COPILOT_BRIEF.md section 9).

A thin rendering layer. Fetching/analysis lives in ``signalcheck.ui.runner``,
card wording in ``signalcheck.ui.view_models``, charts in ``signalcheck.ui.charts``
and the Methodology text in ``signalcheck.ui.methodology``. Deterministic
statistics decide every verdict; nothing here changes a verdict.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import streamlit as st

from signalcheck.adapters.base import slugify
from signalcheck.config import APP_NAME, APP_VERSION, get_config
from signalcheck.narrate import Narration, narration_enabled
from signalcheck.ui import runner
from signalcheck.ui.charts import build_chart
from signalcheck.ui.methodology import methodology_sections
from signalcheck.ui.runner import CsvResult, SourceOutcome, SourceStatus, TopicResult
from signalcheck.ui.view_models import CardView, card_view, summary_view

cfg = get_config()
TTL_SECONDS = int(float(cfg["cache"]["ttl_hours"]) * 3600)
REQUEST = "request"
WIKI_INPUT = "wiki_article_input"


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False, max_entries=64)
def cached_topic(
    query: str,
    sources: tuple[str, ...],
    days: int | None,
    sample: bool,
    wiki_article: str | None,
) -> TopicResult:
    """Engine results per request (adapters cache the HTTP calls themselves)."""
    return runner.run_topic(
        query, sources, get_config(), days=days, sample=sample, wiki_article=wiki_article
    )


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False, max_entries=16)
def cached_csv(
    data: bytes, name: str, value_column: str | None, last_period_incomplete: bool
) -> CsvResult:
    """Engine result for an uploaded CSV."""
    return runner.run_csv(
        data,
        name,
        get_config(),
        value_column=value_column,
        last_period_incomplete=last_period_incomplete,
    )


# --- callbacks (run before the next script pass) ---------------------------------


def load_sample(query: str) -> None:
    """Sample chip: committed snapshots only, no network."""
    st.session_state["topic"] = query
    st.session_state[REQUEST] = {
        "query": query,
        "sources": tuple(runner.sample_sources(query, get_config())),
        "days": None,
        "sample": True,
        "wiki_article": None,
    }
    st.session_state.pop("form_error", None)
    st.session_state.pop(WIKI_INPUT, None)


def submit_topic(statuses: list[SourceStatus]) -> None:
    """Form submit: live fetch for the checked, available sources."""
    query = str(st.session_state.get("topic", "")).strip()
    sources = tuple(
        s.source for s in statuses if s.available and st.session_state.get(f"src_{s.source}")
    )
    if not query:
        st.session_state["form_error"] = "Enter a topic to check."
        return
    if not sources:
        st.session_state["form_error"] = "Select at least one source."
        return
    st.session_state.pop("form_error", None)
    st.session_state.pop(WIKI_INPUT, None)
    st.session_state[REQUEST] = {
        "query": query,
        "sources": sources,
        "days": st.session_state.get("timeframe"),
        "sample": False,
        "wiki_article": None,
    }


def set_wiki_article(article: str | None) -> None:
    """Re-fetch Wikipedia with an article override (``None`` = automatic match)."""
    request = st.session_state.get(REQUEST)
    if not request:
        return
    article = (article or "").strip() or None
    st.session_state[REQUEST] = {**request, "wiki_article": article}
    if article is None:
        st.session_state.pop(WIKI_INPUT, None)


def refetch_wiki() -> None:
    """Override button: use whatever is in the article box."""
    set_wiki_article(st.session_state.get(WIKI_INPUT))


# --- rendering ---------------------------------------------------------------------


def render_topic_inputs(statuses: list[SourceStatus]) -> None:
    """Sample chips, then the topic form."""
    topics: list[str] = list(cfg["samples"]["topics"])
    st.caption("Try a sample topic (instant, from committed snapshots):")
    with st.container(horizontal=True, gap="small"):
        for topic in topics:
            st.button(
                topic,
                key=f"sample_{slugify(topic)}",
                on_click=load_sample,
                args=(topic,),
                icon=":material/bolt:",
            )
    timeframes: list[int | None] = [None, *cfg["ui"]["timeframe_days"]]
    with st.form("topic_form", border=True):
        st.text_input("Topic", key="topic", placeholder="e.g. rust programming")
        st.caption("Sources")
        with st.container(horizontal=True, gap="medium"):
            for status in statuses:
                st.checkbox(
                    status.label,
                    key=f"src_{status.source}",
                    value=status.available,
                    disabled=not status.available,
                    help=None if status.available else f"Unavailable: {status.reason}",
                )
        unavailable = [s for s in statuses if not s.available]
        if unavailable:
            st.caption(" \u00b7 ".join(f"{s.label} disabled ({s.reason})" for s in unavailable))
        st.selectbox(
            "Timeframe",
            timeframes,
            key="timeframe",
            format_func=lambda d: "Source defaults" if d is None else f"Last {d} days",
            help="Hacker News makes one request per day, so long timeframes are slower.",
        )
        st.form_submit_button(
            "Check signal", type="primary", on_click=submit_topic, args=(statuses,)
        )
    if st.session_state.get("form_error"):
        st.warning(st.session_state["form_error"])


def wiki_url(outcome: SourceOutcome, article: str) -> str:
    """Link to the resolved article on its Wikipedia project."""
    project = str(cfg["adapters"]["wikipedia"]["project"])
    if outcome.series is not None:
        project = str(outcome.series.meta.get("project") or project)
    host = project if project.endswith(".org") else f"{project}.org"
    return f"https://{host}/wiki/{quote(article.replace(' ', '_'))}"


def render_wiki_box(view: CardView, outcome: SourceOutcome) -> None:
    """Resolved article plus an override box that re-fetches Wikipedia."""
    request = st.session_state.get(REQUEST) or {}
    if view.wiki is not None:
        how = "your override" if view.wiki.overridden else "matched automatically"
        st.markdown(
            f":material/article: Article: [{view.wiki.article}]"
            f"({wiki_url(outcome, view.wiki.article)}) ({how})"
        )
        if view.wiki.candidates:
            st.caption("Other matches: " + ", ".join(view.wiki.candidates[:4]))
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        st.text_input(
            "Wrong article? Enter the exact title",
            key=WIKI_INPUT,
            placeholder=view.wiki.article if view.wiki else "Article title",
        )
        st.button("Re-fetch", key="wiki_refetch", icon=":material/refresh:", on_click=refetch_wiki)
    if request.get("wiki_article"):
        st.button(
            "Use automatic match",
            key="wiki_reset",
            type="tertiary",
            on_click=set_wiki_article,
            args=(None,),
        )


def render_bullets(items: list[str]) -> None:
    """Markdown bullet list."""
    st.markdown("\n".join(f"- {item}" for item in items))


def render_failure(view: CardView, outcome: SourceOutcome) -> None:
    """Card body for a source that failed or is disabled."""
    st.markdown(f"**{view.title}**")
    if view.disabled:
        st.caption(view.message)
    else:
        st.warning(view.message, icon=":material/cloud_off:")
    if outcome.source == "wikipedia":
        render_wiki_box(view, outcome)


NARRATION_ICONS = {"llm": ":material/auto_awesome:", "template": ":material/notes:"}


def fill_narration(slot: Any, narration: Narration) -> None:
    """Show ``narration`` in a card's narration placeholder, labelled by how it was made."""
    with slot.container():
        st.caption(f"{NARRATION_ICONS[narration.path]} {narration.label}")
        st.markdown(narration.text)


def render_card(outcome: SourceOutcome, narration: Narration | None = None) -> Any | None:
    """One source card: verdict, chart, evidence, change-my-mind, caveats, narration.

    Returns the narration placeholder (showing ``narration`` for now) so the caller
    can swap in the AI summary once it arrives; ``None`` for failure cards.
    """
    view = card_view(outcome)
    with st.container(border=True, key=f"card_{outcome.source}"):
        if view.badge is None or outcome.analysis is None:
            render_failure(view, outcome)
            return None
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(f"**{view.title}**")
            st.badge(view.badge.text, icon=view.badge.icon, color=view.badge.color)
            st.badge(f"{view.confidence} confidence", color="gray")
            if view.snapshot:
                st.badge("snapshot", icon=":material/inventory_2:", color="gray")
        st.markdown(f"**Rule {view.rule} fired:** {view.reason}")
        st.caption(f"{view.rule}: {view.rule_text}")
        if view.window_text:
            st.caption(view.window_text)
        if outcome.source == "wikipedia":
            render_wiki_box(view, outcome)
        fig = build_chart(
            outcome.analysis,
            height=int(cfg["ui"]["chart_height_px"]),
            threshold_headroom=float(cfg["ui"]["threshold_headroom"]),
        )
        st.plotly_chart(
            fig,
            key=f"chart_{outcome.source}",
            config={"displayModeBar": False, "responsive": True},
        )
        st.markdown("**Evidence**")
        render_bullets(
            [f"{e.icon} **{e.name}** ({e.stance_text}): {e.summary}" for e in view.evidence]
        )
        if view.skipped:
            with st.expander(f"Checks that did not run ({len(view.skipped)})"):
                render_bullets([f"**{s.name}**: {s.reason}" for s in view.skipped])
        if view.change_my_mind:
            with st.container(border=True):
                st.markdown(":material/psychology_alt: **What would change my mind**")
                render_bullets(view.change_my_mind)
        if view.caveats:
            st.markdown("**Caveats**")
            st.caption("\n".join(f"- {c}" for c in view.caveats))
        slot = st.empty()
        if narration is not None:
            fill_narration(slot, narration)
        return slot


def render_summary(result: TopicResult) -> None:
    """Top cross-source card."""
    summary = summary_view(result)
    if summary is None:
        return
    with st.container(border=True, key="summary_card"):
        st.caption(f"Across sources \u00b7 \u201c{result.query}\u201d")
        st.markdown(f"#### {summary.sentence}")
        if summary.chips:
            with st.container(horizontal=True, gap="small"):
                for label, badge in summary.chips:
                    st.badge(f"{label}: {badge.text}", icon=badge.icon, color=badge.color)
        if summary.not_comparable:
            st.caption("Not comparable (different period): " + ", ".join(summary.not_comparable))
        if summary.unavailable:
            st.caption("Unavailable: " + ", ".join(summary.unavailable))
        if result.sample:
            st.caption("Sample topic: committed snapshots, not live data.")


def render_result(result: TopicResult) -> None:
    """Summary card, then one card per source, then the AI summaries (if enabled).

    Cards render with their template narration first, so a slow or failing LLM
    never delays or breaks a card; validated AI text replaces it when ready.
    """
    render_summary(result)
    templates = runner.template_narrations(result.outcomes, cfg)
    slots = {}
    for outcome in result.outcomes:
        slot = render_card(outcome, templates.get(outcome.source))
        if slot is not None:
            slots[outcome.source] = slot
    if not slots or not narration_enabled():
        return
    with st.spinner("Writing AI summaries\u2026"):
        narrations = runner.narrate_outcomes(result.outcomes, cfg)
    for source, narration in narrations.items():
        if source in slots and narration.path == "llm":
            fill_narration(slots[source], narration)


def run_topic_request(request: dict[str, Any]) -> None:
    """Run (cached) and render the current topic request."""
    with st.spinner("Fetching sources and running the checks\u2026"):
        result = cached_topic(
            request["query"],
            tuple(request["sources"]),
            request["days"],
            bool(request["sample"]),
            request["wiki_article"],
        )
    render_result(result)


def render_csv() -> None:
    """CSV upload with the incomplete-period toggle and a column picker when needed."""
    upload = st.file_uploader("CSV file (date + value, or a Google Trends export)", type="csv")
    st.text_input("What does this series measure? (optional)", key="csv_name")
    st.toggle(
        "Last period is incomplete",
        key="csv_incomplete",
        help="Drop the last row if its period had not ended when the data was exported.",
    )
    if upload is None:
        return
    data = upload.getvalue()
    name = str(st.session_state.get("csv_name") or "").strip() or upload.name
    column = st.session_state.get("csv_column")
    incomplete = bool(st.session_state.get("csv_incomplete"))
    outcome = cached_csv(data, name, column, incomplete)
    if outcome.columns is not None:
        st.selectbox(
            "This file has several value columns. Which one should be analysed?",
            outcome.columns,
            index=None,
            key="csv_column",
            placeholder="Choose a column",
        )
        return
    if outcome.result is not None:
        render_result(outcome.result)


def render_methodology() -> None:
    """Every check, live thresholds, the decision table and the confidence formula."""
    with st.expander("Methodology", icon=":material/menu_book:"):
        st.caption("All thresholds below are read live from config.yaml.")
        for section in methodology_sections(cfg):
            st.markdown(f"##### {section.title}")
            st.markdown(section.body)


def main() -> None:
    """Page entry point."""
    st.set_page_config(page_title=APP_NAME, page_icon=":material/insights:", layout="centered")
    st.title(APP_NAME)
    st.caption(
        "Is a topic really trending, a one-off fluke, seasonal, or flat? Transparent "
        "statistical checks decide; every verdict shows the rule that fired."
    )
    mode = st.radio(
        "Input",
        ["Search a topic", "Upload a CSV"],
        horizontal=True,
        label_visibility="collapsed",
        key="mode",
    )
    if mode == "Upload a CSV":
        render_csv()
    else:
        statuses = runner.source_statuses(cfg)
        render_topic_inputs(statuses)
        request = st.session_state.get(REQUEST)
        if request:
            run_topic_request(request)
    render_methodology()
    st.caption(f"{APP_NAME} {APP_VERSION} \u00b7 official APIs only \u00b7 aggregates only")


main()
