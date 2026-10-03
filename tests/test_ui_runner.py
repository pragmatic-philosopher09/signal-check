"""Fetch + analyse orchestration for the UI (signalcheck.ui.runner); no network."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from signalcheck.adapters.base import AdapterError
from signalcheck.config import Config
from signalcheck.narrate import TEMPLATE_LABEL, Narration, ProviderSettings
from signalcheck.ui import runner
from tests.ui_fakes import FakeAdapter, factory, no_network_factories, snapshot_as_live

GOOGLE_TRENDS_CSV = b"""Category: All categories

Week,chatgpt: (Worldwide),claude: (Worldwide)
""" + b"\n".join(
    f"2025-{(i // 4) + 1:02d}-{(i % 4) * 7 + 1:02d},{40 + i % 5},{10 + i % 3}".encode()
    for i in range(40)
)


def test_sample_sources_lists_committed_snapshots(cfg: Config) -> None:
    assert runner.sample_sources("chatgpt", cfg) == ["wikipedia", "hackernews"]
    assert runner.sample_sources("no such topic", cfg) == []


def test_sample_run_reads_snapshots_only(cfg: Config) -> None:
    def explode(_cfg: Config) -> Any:
        raise AssertionError("sample mode must not build live adapters")

    factories = dict.fromkeys(runner.TOPIC_SOURCES, explode)
    result = runner.run_topic(
        "rust programming", ["wikipedia", "hackernews"], cfg, sample=True, factories=factories
    )
    assert result.sample
    assert [o.source for o in result.outcomes] == ["wikipedia", "hackernews"]
    assert all(o.analysis is not None and o.message is None for o in result.outcomes)
    assert all(o.series is not None and o.series.meta["snapshot"] for o in result.outcomes)
    assert result.summary is not None and result.summary.n_compared == 2
    wiki = result.outcomes[0].verdict
    assert wiki is not None and wiki.label == "TREND" and wiki.rule_fired == "R4"


def test_one_failing_adapter_degrades_only_its_card(cfg: Config) -> None:
    factories = no_network_factories(
        wikipedia=FakeAdapter("wikipedia", error=AdapterError("HTTP 503 from Wikimedia")),
        hackernews=FakeAdapter("hackernews", series=snapshot_as_live("hackernews", "chatgpt")),
    )
    result = runner.run_topic("chatgpt", ["wikipedia", "hackernews"], cfg, factories=factories)
    wiki, hn = result.outcomes
    assert wiki.analysis is None
    assert wiki.message == "Couldn't fetch Wikipedia: HTTP 503 from Wikimedia"
    assert not wiki.disabled
    assert hn.verdict is not None and hn.message is None
    assert result.summary is not None
    assert result.summary.unavailable == ["wikipedia"]
    assert result.summary.comparable == ["hackernews"]


def test_disabled_source_is_left_out_of_the_summary(cfg: Config) -> None:
    factories = no_network_factories(
        hackernews=FakeAdapter("hackernews", series=snapshot_as_live("hackernews", "chatgpt")),
    )
    result = runner.run_topic("chatgpt", ["hackernews", "x"], cfg, factories=factories)
    x = result.outcomes[1]
    assert x.disabled and x.message == "X disabled (planned)"
    assert result.summary is not None and "x" not in result.summary.unavailable


def test_unexpected_adapter_exception_is_reported_generically(cfg: Config) -> None:
    factories = no_network_factories(
        hackernews=FakeAdapter("hackernews", error=KeyError("boom")),
    )
    result = runner.run_topic("chatgpt", ["hackernews"], cfg, factories=factories)
    assert result.outcomes[0].message == "Couldn't fetch Hacker News: unexpected error (KeyError)"


def test_analysis_error_degrades_to_a_card_message(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("bad series")

    monkeypatch.setattr(runner, "analyse_detailed", broken)
    result = runner.run_topic("chatgpt", ["hackernews"], cfg, sample=True)
    outcome = result.outcomes[0]
    assert outcome.analysis is None and outcome.series is not None
    assert outcome.message == "Couldn't analyse Hacker News: unexpected error (ValueError)"


def test_params_forward_timeframe_and_wiki_override(cfg: Config) -> None:
    wiki = FakeAdapter("wikipedia", series=snapshot_as_live("wikipedia", "chatgpt"))
    hn = FakeAdapter("hackernews", series=snapshot_as_live("hackernews", "chatgpt"))
    factories = no_network_factories(wikipedia=wiki, hackernews=hn)
    runner.run_topic(
        "chatgpt",
        ["wikipedia", "hackernews"],
        cfg,
        days=180,
        wiki_article="ChatGPT",
        factories=factories,
    )
    assert wiki.calls == [("chatgpt", {"days": 180, "article": "ChatGPT"})]
    assert hn.calls == [("chatgpt", {"days": 180})]


def test_wiki_override_on_a_sample_fetches_wikipedia_live_only(cfg: Config) -> None:
    wiki = FakeAdapter(
        "wikipedia",
        series=snapshot_as_live("wikipedia", "chatgpt", article="ChatGPT", article_overridden=True),
    )
    factories = no_network_factories(wikipedia=wiki)
    result = runner.run_topic(
        "chatgpt",
        ["wikipedia", "hackernews"],
        cfg,
        sample=True,
        wiki_article="ChatGPT",
        factories=factories,
    )
    assert wiki.calls == [("chatgpt", {"article": "ChatGPT"})]
    wiki_out, hn_out = result.outcomes
    assert wiki_out.series is not None and wiki_out.series.meta["article_overridden"]
    assert hn_out.series is not None and hn_out.series.meta["snapshot"]


def test_no_sources_gives_no_summary(cfg: Config) -> None:
    result = runner.run_topic("chatgpt", [], cfg, factories=no_network_factories())
    assert result.outcomes == [] and result.summary is None


def test_probe_reports_why_a_source_is_unavailable(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("signalcheck.adapters.reddit.get_secret", lambda *_a, **_k: None)
    monkeypatch.setattr("signalcheck.adapters.x_twitter.secret_flag", lambda _name: None)
    cfg["sources"]["google_trends"] = False
    statuses = {s.source: s for s in runner.source_statuses(cfg)}
    assert list(statuses) == list(runner.TOPIC_SOURCES)
    assert statuses["wikipedia"].available and statuses["wikipedia"].reason is None
    assert statuses["reddit"].reason == "API credentials not configured"
    assert statuses["x"].reason == "planned"
    assert not statuses["google_trends"].available
    assert statuses["google_trends"].reason == "switched off in config.yaml"


def test_probe_x_needs_a_token_when_enabled(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("signalcheck.adapters.x_twitter.secret_flag", lambda _name: True)
    monkeypatch.setattr("signalcheck.adapters.x_twitter.get_secret", lambda *_a, **_k: None)
    status = runner.probe_source("x", cfg)
    assert not status.available and status.reason == "API credentials not configured"


def test_probe_with_injected_factory(cfg: Config) -> None:
    status = runner.probe_source("x", cfg, factory(FakeAdapter("x")))
    assert status.available


def test_csv_with_several_columns_asks_for_one(cfg: Config) -> None:
    out = runner.run_csv(GOOGLE_TRENDS_CSV, "trends.csv", cfg)
    assert out.result is None
    assert out.columns == ["chatgpt: (Worldwide)", "claude: (Worldwide)"]


def test_csv_with_a_chosen_column_is_analysed(cfg: Config) -> None:
    out = runner.run_csv(GOOGLE_TRENDS_CSV, "trends.csv", cfg, value_column="claude: (Worldwide)")
    assert out.columns is None and out.result is not None
    (outcome,) = out.result.outcomes
    assert outcome.source == "csv" and outcome.verdict is not None
    assert out.result.summary is not None


def test_csv_incomplete_toggle_drops_the_last_period(cfg: Config) -> None:
    data = b"date,value\n" + b"\n".join(
        f"2025-01-{d:02d},{10 + d % 3}".encode() for d in range(1, 31)
    )
    kept = runner.run_csv(data, "x.csv", cfg).result
    dropped = runner.run_csv(data, "x.csv", cfg, last_period_incomplete=True).result
    assert kept is not None and dropped is not None
    kept_a = kept.outcomes[0].analysis
    dropped_a = dropped.outcomes[0].analysis
    assert kept_a is not None and dropped_a is not None
    assert kept_a.pre.series.meta.get("dropped_partial") is None
    assert dropped_a.pre.series.meta["dropped_partial"]["ts"].startswith("2025-01-30")


def test_unreadable_csv_is_a_card_message(cfg: Config) -> None:
    out = runner.run_csv(b"just some text", "bad.csv", cfg)
    assert out.result is not None
    (outcome,) = out.result.outcomes
    assert outcome.analysis is None
    assert outcome.message is not None and outcome.message.startswith("Couldn't fetch CSV upload:")


ENABLED = ProviderSettings("openai", "sk-test", None, None)


def sample_outcomes(cfg: Config) -> list[runner.SourceOutcome]:
    result = runner.run_topic("rust programming", ["wikipedia", "hackernews"], cfg, sample=True)
    failed = runner.SourceOutcome("reddit", "Reddit", message="Couldn't fetch Reddit")
    return [*result.outcomes, failed]


def test_narrations_are_templates_when_no_provider(cfg: Config) -> None:
    def never(*_a: Any, **_k: Any) -> Narration:
        raise AssertionError("narrate must not be called")

    result = runner.narrate_outcomes(
        sample_outcomes(cfg),
        cfg,
        settings=ProviderSettings(None, None, None, None),
        narrate_fn=never,
    )
    assert set(result) == {"wikipedia", "hackernews"}
    assert all(n.path == "template" and n.label == TEMPLATE_LABEL for n in result.values())
    assert result["wikipedia"].text.startswith("Wikipedia shows a sustained upward trend")


def test_narrations_run_concurrently_with_fallbacks(cfg: Config) -> None:
    def fake(verdict: Any, meta: dict[str, Any], **_k: Any) -> Narration:
        if meta["source"] == "hackernews":
            raise RuntimeError("boom")
        assert meta == {"source": "wikipedia", "source_label": "Wikipedia", "freq": "D"}
        return Narration(f"AI says {verdict.label}", "llm")

    result = runner.narrate_outcomes(sample_outcomes(cfg), cfg, settings=ENABLED, narrate_fn=fake)
    assert result["wikipedia"] == Narration("AI says TREND", "llm")
    assert (result["hackernews"].path, result["hackernews"].failure) == ("template", "error")


def test_slow_narration_times_out_to_template(cfg: Config) -> None:
    release = threading.Event()

    def slow(*_a: Any, **_k: Any) -> Narration:
        release.wait(5)
        return Narration("late", "llm")

    outcomes = sample_outcomes(cfg)
    started = time.monotonic()
    result = runner.narrate_outcomes(
        outcomes, cfg, settings=ENABLED, narrate_fn=slow, timeout_s=0.05
    )
    release.set()
    assert time.monotonic() - started < 2
    assert {n.failure for n in result.values()} == {"timeout"}
    assert all(n.path == "template" for n in result.values())
