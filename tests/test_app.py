"""Streamlit AppTest smoke tests for app.py (no network: adapters are faked)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from signalcheck.adapters.base import AdapterError
from signalcheck.ui import runner
from tests.ui_fakes import FakeAdapter, no_network_factories, snapshot_as_live

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fresh engine cache and fake adapters that fail loudly on any live fetch."""
    st.cache_data.clear()
    monkeypatch.setattr(runner, "ADAPTER_FACTORIES", no_network_factories())
    yield
    st.cache_data.clear()


def start() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def texts(elements: object) -> list[str]:
    return [str(e.value) for e in elements]  # type: ignore[attr-defined]


def check_signal(at: AppTest) -> None:
    next(b for b in at.button if b.label == "Check signal").click().run()


def test_initial_render_explains_disabled_sources() -> None:
    at = start()
    assert at.title[0].value == "Signal Check"
    assert any("X disabled (planned)" in c for c in texts(at.caption))
    assert any("Reddit disabled (API credentials not configured)" in c for c in texts(at.caption))
    assert at.checkbox(key="src_x").disabled and at.checkbox(key="src_wikipedia").value
    assert not at.get("plotly_chart")
    assert at.expander[-1].label == "Methodology"


def test_sample_chip_renders_cards_offline() -> None:
    at = start()
    at.button(key="sample_rust-programming").click().run()
    assert not at.exception
    assert not at.warning
    assert len(at.get("plotly_chart")) == 2
    markdown = "\n".join(texts(at.markdown))
    assert "Rule R4 fired:" in markdown
    assert "What would change my mind" in markdown
    assert ":green-badge[" in markdown
    captions = "\n".join(texts(at.caption))
    assert "Sample topic: committed snapshots" in captions
    assert "template text" in captions
    assert at.text_input(key="topic").value == "rust programming"


def test_failed_adapter_shows_its_message_and_other_cards_still_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "ADAPTER_FACTORIES",
        no_network_factories(
            wikipedia=FakeAdapter("wikipedia", error=AdapterError("HTTP 503 from Wikimedia")),
            google_trends=FakeAdapter("google_trends", error=AdapterError("rate limited (429)")),
            hackernews=FakeAdapter("hackernews", series=snapshot_as_live("hackernews", "chatgpt")),
        ),
    )
    at = start()
    at.text_input(key="topic").set_value("chatgpt")
    check_signal(at)
    assert not at.exception
    warnings = texts(at.warning)
    assert "Couldn't fetch Wikipedia: HTTP 503 from Wikimedia" in warnings
    assert "Couldn't fetch Google Trends: rate limited (429)" in warnings
    assert len(at.get("plotly_chart")) == 1
    assert any("Unavailable: Wikipedia, Google Trends" in c for c in texts(at.caption))


def test_wiki_override_refetches_with_the_article(monkeypatch: pytest.MonkeyPatch) -> None:
    wiki = FakeAdapter(
        "wikipedia",
        series=snapshot_as_live("wikipedia", "chatgpt", article="ChatGPT", article_overridden=True),
    )
    monkeypatch.setattr(runner, "ADAPTER_FACTORIES", no_network_factories(wikipedia=wiki))
    at = start()
    at.button(key="sample_chatgpt").click().run()
    assert wiki.calls == []
    at.text_input(key="wiki_article_input").set_value("ChatGPT")
    at.button(key="wiki_refetch").click().run()
    assert not at.exception
    assert wiki.calls == [("chatgpt", {"article": "ChatGPT"})]
    assert at.button(key="wiki_reset")


def test_empty_topic_is_rejected() -> None:
    at = start()
    check_signal(at)
    assert texts(at.warning) == ["Enter a topic to check."]


def test_methodology_lists_live_thresholds() -> None:
    at = start()
    markdown = "\n".join(texts(at.markdown))
    assert "`outlier_z` = 3.5" in markdown
    assert "| R6 |" in markdown
