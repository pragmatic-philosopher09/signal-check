"""Tests for signalcheck.narrate: payload, validator, template, providers (no network)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import requests
import responses

from eval.run_eval import load_cases
from signalcheck.cache import Cache
from signalcheck.config import APP_NAME, Config
from signalcheck.engine import analyse
from signalcheck.engine.change_my_mind import Condition
from signalcheck.models import LABELS, Evidence, Verdict
from signalcheck.narrate import (
    AI_LABEL,
    PROVIDERS,
    SYSTEM_PROMPT,
    TEMPLATE_LABEL,
    Narration,
    NarrationError,
    OpenAICompatibleNarrator,
    ProviderSettings,
    build_payload,
    make_narrator,
    narrate,
    payload_json,
    system_prompt,
    template_narration,
    validate_narration,
)

URL = "https://api.openai.com/v1/chat/completions"
KEY = "sk-test-secret-key"
META = {"source": "wikipedia", "freq": "D"}


def make_verdict(label: str = "TREND", direction: str | None = "up") -> Verdict:
    """A hand-built verdict with realistic summaries and the numbers they display."""
    evidence = [
        Evidence(
            "persistence",
            "supports_trend",
            "Values rose 3.0% of the baseline level per day over the last 31 days "
            "(2024-03-30 to 2024-04-29); the Mann-Kendall trend test is significant "
            "(p < 0.001); the last 25 days sat above the baseline band (44.8 to 55.9).",
            {
                "slope_norm_pct": 3.0,
                "n_tested": 31,
                "p_value": 0.0,
                "p_value_bound": 0.001,
                "run_length": 25,
                "band_lower": 44.8,
                "band_upper": 55.9,
                "start": "2024-03-30",
                "end": "2024-04-29",
                "significant": np.bool_(True),
                "test": "hamed_rao",
            },
            None,
        ),
        Evidence(
            "concentration",
            "neutral",
            "Spread out: the largest day (2024-04-29, 93.5) carries 7.0% of the upward excess "
            "and the top 2 carry 13.8%, across 1,234 events.",
            {
                "top1_value": np.float64(93.5),
                "top1_share_pct": 7.0,
                "top2_share_pct": 13.8,
                "top_k": np.int64(2),
                "total_events": 1234,
                "top_share": 0.123,
            },
            None,
        ),
        Evidence(
            "level_shift",
            "supports_trend",
            "Level shifted up on 2024-04-21: median 50.9 before vs 86.3 after (1.7x), held for "
            "9 days.",
            {
                "before_median": 50.9,
                "after_median": 86.3,
                "ratio": 1.7,
                "n_post": 9,
                "change_point_dates": ["2024-04-05", "2024-04-21"],
                "min_change_pct": -4.2,
                "unused": float("nan"),
            },
            None,
        ),
        Evidence(
            "seasonality",
            "skipped",
            "Skipped: annual seasonality needs at least 2 years of history.",
            {"min_years": 2},
            None,
            skip_reason="needs 2 years",
        ),
        Evidence("breadth", "skipped", "Skipped: no contributor data.", {}, None, "no data"),
    ]
    conditions = [
        Condition(
            "If the next 7 days fall back below 53.1 (baseline median + 1 MAD), this was a fluke.",
            {"periods": 7, "threshold": 53.1, "mads": 1},
        )
    ]
    window = {
        "baseline_start": "2024-01-01",
        "baseline_end": "2024-04-05",
        "recent_start": "2024-04-06",
        "recent_end": "2024-04-29",
        "recent_end_exclusive": "2024-04-30",
        "n_baseline": 96,
        "n_recent": 24,
    }
    return Verdict(
        label=label,
        direction=direction,
        confidence="medium",
        rule_fired="R4",
        evidence=evidence,
        change_my_mind=list(conditions),
        caveats=["Wikipedia data fetched on 2024-04-30 at 12:18 UTC."],
        window=window,
        reason="a significant, practically large trend that is not concentrated in a single spike",
    )


GOOD = (
    "Wikipedia shows a TREND with medium confidence: values rose about 3% of the baseline "
    "level per day over the last 31 days (p < 0.001), and the level shifted up on 21 April "
    "2024, from a median of 50.9 to 86.3. Two checks support the trend and one origin does "
    "not dominate. If the next seven days fall back below 53.1, this was a fluke."
)


@pytest.fixture
def payload() -> dict[str, Any]:
    return build_payload(make_verdict(), META)


def issues(text: str, payload: dict[str, Any], cfg: Config) -> list[str]:
    return validate_narration(text, payload, cfg).reasons


# --- payload -------------------------------------------------------------------


def test_payload_holds_only_the_brief_fields(payload: dict[str, Any]) -> None:
    assert set(payload) == {"source", "verdict", "evidence", "change_my_mind", "caveats", "window"}
    assert payload["source"] == "Wikipedia"
    assert payload["verdict"] == {
        "label": "TREND",
        "direction": "up",
        "confidence": "medium",
        "rule_fired": "R4",
        "reason": make_verdict().reason,
    }
    assert set(payload["evidence"][0]) == {"check", "stance", "summary", "numbers"}
    assert payload["change_my_mind"][0]["numbers"] == {"periods": 7, "threshold": 53.1, "mads": 1}
    assert payload["window"]["period"] == "day"
    assert payload["window"]["recent_start"] == "2024-04-06"


def test_payload_is_json_safe_and_deterministic(payload: dict[str, Any]) -> None:
    text = payload_json(payload)
    assert json.loads(text) == payload
    assert payload_json(build_payload(make_verdict(), META)) == text
    numbers = payload["evidence"][1]["numbers"]
    assert type(numbers["top1_value"]) is float and type(numbers["top_k"]) is int
    assert payload["evidence"][0]["numbers"]["significant"] is True
    assert payload["evidence"][2]["numbers"]["unused"] is None


def test_payload_source_label_override_and_unknown_freq() -> None:
    payload = build_payload(make_verdict(), {"source": "csv", "source_label": "my_data.csv"})
    assert payload["source"] == "my_data.csv"
    assert "period" not in payload["window"]


# --- validator: passing --------------------------------------------------------


def test_realistic_narration_passes(payload: dict[str, Any], cfg: Config) -> None:
    assert issues(GOOD, payload, cfg) == []


@pytest.mark.parametrize(
    "fragment",
    [
        "rose 3.0% per day",  # as displayed
        "rose 3% per day",  # rounded to fewer decimals
        "a band from 45 to 56",  # 44.8 and 55.9 rounded
        "a median of 86",  # 86.3 rounded
        "the band reached 55.90",  # trailing zero
        "1,234 events",  # thousands separator
        "1234 events",
        "a 1.7x jump",
        "a jump of x1.7",
        "a jump of \u00d71.7",
        "roughly 2x",  # 1.7 rounded
        "p < 0.001",
        "12.3% of the volume",  # fraction 0.123 as a percent
        "a share of 0.07",  # percent 7.0 as a fraction
        "a change of -4.2%",
        "a change of \u22124.2%",
        "a fall of 4.2%",  # unsigned matches the magnitude
        "over the 31st day",  # digit ordinal
        "on 21st April 2024",
        "on April 21, 2024",
        "from 6 to 29 April 2024",  # day range written out
        "from 6\u201329 Apr 2024",  # en-dash range
        "in April 2024",
        "in April",
        "since 2024",
        "one source, a single origin, no one spike",  # "one" is exempt
        "two checks support it",  # count of supports_trend stances
        "five checks, three of which ran",  # 5 checks, 3 of them ran
        "seven days",
        "fetched at 12:18 UTC",
        "the trend may continue",  # "may" as a verb
    ],
)
def test_normalised_forms_pass(fragment: str, payload: dict[str, Any], cfg: Config) -> None:
    assert issues(f"TREND: {fragment}.", payload, cfg) == []


def test_no_change_label_with_space(cfg: Config) -> None:
    payload = build_payload(make_verdict("NO_CHANGE", None), META)
    assert issues("The verdict is NO CHANGE.", payload, cfg) == []
    assert issues("The verdict is NO_CHANGE.", payload, cfg) == []


def test_ranges_across_a_month(cfg: Config) -> None:
    verdict = make_verdict()
    window = {**verdict.window, "recent_start": "2026-09-01", "recent_end": "2026-09-28"}
    payload = build_payload(
        Verdict(**{**verdict.__dict__, "window": window}),
        META,
    )
    assert issues("TREND over 1\u201328 Sep 2026.", payload, cfg) == []
    assert issues("TREND over Sep 1-28, 2026.", payload, cfg) == []
    assert issues("TREND over 1\u201329 Sep 2026.", payload, cfg) == ["date: 1\u201329 Sep 2026"]
    assert issues("TREND over 1\u201328 Sep 2025.", payload, cfg) == ["date: 1\u201328 Sep 2025"]


# --- validator: failing --------------------------------------------------------


@pytest.mark.parametrize(
    ("fragment", "expected"),
    [
        ("rose 62% per day", "number: 62%"),  # invented number
        ("a band from 44.9 upwards", "number: 44.9"),  # wrong rounding
        ("rose 3.05% per day", "number: 3.05%"),  # more precise than the JSON
        ("a jump of 8x", "number: 8"),
        ("p < 0.01", "number: 0.01"),
        ("a change of -3.0%", "number: -3.0%"),  # sign must match a negative value
        ("1,235 events", "number: 1,235"),
        ("over the 40th day", "number: 40"),
        ("seventeen checks", "number_word: seventeen"),
        ("Six checks ran", "number_word: Six"),
        ("on 2024-05-01", "date: 2024-05-01"),
        ("on 3 May 2024", "date: 3 May 2024"),
        ("on April 22", "date: April 22"),
        ("in June", "date: June"),
        ("since 2023", "date: 2023"),
        ("in March 2023", "date: March 2023"),
    ],
)
def test_invented_facts_fail(
    fragment: str, expected: str, payload: dict[str, Any], cfg: Config
) -> None:
    assert issues(f"TREND: {fragment}.", payload, cfg) == [expected]


def test_wrong_and_missing_label(payload: dict[str, Any], cfg: Config) -> None:
    assert issues("This looks like a FLUKE.", payload, cfg) == [
        "label_missing: TREND",
        "wrong_label: FLUKE",
    ]
    assert issues("A TREND, not a FLUKE.", payload, cfg) == ["wrong_label: FLUKE"]
    assert issues("There is an upward trend.", payload, cfg) == ["label_missing: TREND"]


def test_word_cap_and_empty(payload: dict[str, Any], cfg: Config) -> None:
    long_text = "TREND " + "word " * 120
    assert issues(long_text, payload, cfg) == ["too_long: 121"]
    assert issues("TREND " + "word " * 119, payload, cfg) == []
    assert issues("   ", payload, cfg) == ["empty"]


def test_result_exposes_codes(payload: dict[str, Any], cfg: Config) -> None:
    result = validate_narration("FLUKE in 2023 with 62 points", payload, cfg)
    assert not result.ok
    assert result.codes == ("date", "label_missing", "number", "wrong_label")


# --- template ------------------------------------------------------------------


@pytest.mark.parametrize("label", LABELS)
def test_template_reads_well_for_every_label(label: str, cfg: Config) -> None:
    direction = "down" if label in ("TREND", "FLUKE") else None
    payload = build_payload(make_verdict(label, direction), META)
    text = template_narration(payload, cfg)
    assert label in text
    assert "medium confidence (rule R4)" in text
    assert validate_narration(text, payload, cfg).ok
    if label == "TREND":
        assert text.startswith("Wikipedia shows a sustained downward trend")
    if label == "FLUKE":
        assert "short-lived dip" in text


def test_template_respects_word_cap(payload: dict[str, Any], cfg: Config) -> None:
    cfg["narration"]["max_words"] = 40
    text = template_narration(payload, cfg)
    assert len(text.split()) <= 40
    assert text.startswith("Wikipedia shows a sustained upward trend: the verdict is TREND")


@pytest.mark.parametrize("case", load_cases(), ids=lambda c: c.id)
def test_templates_validate_on_every_synthetic_case(case: Any, cfg: Config) -> None:
    series = case.build()
    verdict = analyse(series, cfg)
    payload = build_payload(verdict, {"source": series.source, "freq": series.freq})
    text = template_narration(payload, cfg)
    result = validate_narration(text, payload, cfg)
    assert result.ok, (result.reasons, text)


# --- providers -----------------------------------------------------------------


def settings(**overrides: str | None) -> ProviderSettings:
    values: dict[str, str | None] = {
        "provider": "openai",
        "api_key": KEY,
        "model": "gpt-test",
        "base_url": None,
    }
    values.update(overrides)
    return ProviderSettings(**values)


def reply(content: str | None, finish_reason: str = "stop") -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


@pytest.fixture
def narr_cfg(cfg: Config) -> Config:
    cfg["narration"]["max_retries"] = 0
    return cfg


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[Cache]:
    store = Cache(tmp_path / "cache", 3600)
    yield store
    store.close()


@responses.activate
def test_llm_success(narr_cfg: Config, cache: Cache) -> None:
    responses.add(responses.POST, URL, json=reply(GOOD))
    result = narrate(make_verdict(), META, cfg=narr_cfg, settings=settings(), cache=cache)
    assert result == Narration(GOOD, "llm")
    assert result.label == AI_LABEL
    request = responses.calls[0].request
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert request.headers["User-Agent"].startswith(APP_NAME.replace(" ", ""))
    body = json.loads(request.body or b"")
    assert body["model"] == "gpt-test"
    assert body["temperature"] == 0
    assert body["max_tokens"] == narr_cfg["narration"]["max_tokens"]
    assert body["messages"][0] == {"role": "system", "content": system_prompt(narr_cfg)}
    user = body["messages"][1]["content"]
    assert json.loads(user.split("\n", 1)[1]) == build_payload(make_verdict(), META)


@responses.activate
def test_invented_number_falls_back_to_template(narr_cfg: Config, cache: Cache) -> None:
    responses.add(responses.POST, URL, json=reply(GOOD.replace("50.9", "48.0")))
    result = narrate(make_verdict(), META, cfg=narr_cfg, settings=settings(), cache=cache)
    assert result.path == "template" and result.failure == "validation"
    assert result.label == TEMPLATE_LABEL
    assert result.text == template_narration(build_payload(make_verdict(), META), narr_cfg)


@pytest.mark.parametrize(
    ("mock", "category"),
    [
        ({"status": 429, "headers": {"Retry-After": "600"}}, "rate_limited"),
        ({"status": 401}, "auth"),
        ({"status": 500}, "http_error"),
        ({"body": requests.exceptions.ReadTimeout()}, "timeout"),
        ({"body": requests.exceptions.ConnectionError()}, "connection"),
        ({"body": "not json"}, "bad_response"),
        ({"json": {"choices": []}}, "bad_response"),
        ({"json": reply(None)}, "bad_response"),
        ({"json": reply(GOOD, "length")}, "truncated"),
        ({"json": reply("")}, "validation"),
    ],
)
@responses.activate
def test_failures_fall_back_to_template(
    mock: dict[str, Any], category: str, narr_cfg: Config, cache: Cache
) -> None:
    responses.add(responses.POST, URL, **mock)
    result = narrate(make_verdict(), META, cfg=narr_cfg, settings=settings(), cache=cache)
    assert result.path == "template"
    assert result.failure == category


@responses.activate
def test_retry_then_success_uses_narration_retries(narr_cfg: Config) -> None:
    narr_cfg["narration"]["max_retries"] = 1
    responses.add(responses.POST, URL, status=429)
    responses.add(responses.POST, URL, json=reply(GOOD))
    narrator = OpenAICompatibleNarrator(
        KEY, "gpt-test", "https://api.openai.com/v1", narr_cfg, sleep=lambda _s: None
    )
    assert narrator.complete("system", "user") == GOOD
    assert len(responses.calls) == 2


@responses.activate
@pytest.mark.parametrize(
    ("overrides", "category"),
    [
        ({"api_key": None}, "no_key"),
        ({"api_key": "  "}, "no_key"),
        ({"provider": None}, "disabled"),
        ({"provider": ""}, "disabled"),
        ({"provider": "acme"}, "unknown_provider"),
    ],
)
def test_unusable_settings_make_no_http_call(
    overrides: dict[str, str | None], category: str, narr_cfg: Config, cache: Cache
) -> None:
    result = narrate(
        make_verdict(), META, cfg=narr_cfg, settings=settings(**overrides), cache=cache
    )
    assert result.path == "template" and result.failure == category
    assert len(responses.calls) == 0


@responses.activate
def test_missing_key_from_environment(
    monkeypatch: pytest.MonkeyPatch, narr_cfg: Config, cache: Cache
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    result = narrate(make_verdict(), META, cfg=narr_cfg, cache=cache)
    assert (result.path, result.failure) == ("template", "no_key")
    assert len(responses.calls) == 0


@responses.activate
def test_cache_hit_avoids_second_call(narr_cfg: Config, cache: Cache) -> None:
    responses.add(responses.POST, URL, json=reply(GOOD))
    first = narrate(make_verdict(), META, cfg=narr_cfg, settings=settings(), cache=cache)
    second = narrate(make_verdict(), META, cfg=narr_cfg, settings=settings(), cache=cache)
    assert first == second == Narration(GOOD, "llm")
    assert len(responses.calls) == 1
    narrate(make_verdict(), META, cfg=narr_cfg, settings=settings(model="other"), cache=cache)
    assert len(responses.calls) == 2  # the model is part of the key


@responses.activate
def test_cached_reply_is_revalidated(narr_cfg: Config, cache: Cache) -> None:
    responses.add(responses.POST, URL, json=reply(GOOD.replace("50.9", "48.0")))
    for _ in range(2):
        result = narrate(make_verdict(), META, cfg=narr_cfg, settings=settings(), cache=cache)
        assert result.failure == "validation"
    assert len(responses.calls) == 1


@responses.activate
def test_failures_are_not_cached(narr_cfg: Config, cache: Cache) -> None:
    responses.add(responses.POST, URL, status=500)
    responses.add(responses.POST, URL, json=reply(GOOD))
    assert narrate(make_verdict(), META, cfg=narr_cfg, settings=settings(), cache=cache).failure
    assert narrate(make_verdict(), META, cfg=narr_cfg, settings=settings(), cache=cache).path == (
        "llm"
    )


@responses.activate
def test_custom_base_url_and_token_param(narr_cfg: Config, cache: Cache) -> None:
    base = "https://models.github.ai/inference/"
    responses.add(
        responses.POST, "https://models.github.ai/inference/chat/completions", json=reply(GOOD)
    )
    narr_cfg["narration"]["max_tokens_param"] = "max_completion_tokens"
    result = narrate(
        make_verdict(), META, cfg=narr_cfg, settings=settings(base_url=base), cache=cache
    )
    assert result.path == "llm"
    body = json.loads(responses.calls[0].request.body or b"")
    assert "max_tokens" not in body and body["max_completion_tokens"] > 0


@responses.activate
def test_logs_only_the_failure_category(
    narr_cfg: Config, cache: Cache, caplog: pytest.LogCaptureFixture
) -> None:
    responses.add(responses.POST, URL, json=reply(GOOD.replace("50.9", "48.0")))
    with caplog.at_level(logging.DEBUG):
        narrate(make_verdict(), META, cfg=narr_cfg, settings=settings(), cache=cache)
    logged = caplog.text
    assert "validation" in logged
    assert KEY not in logged and "48.0" not in logged and "Wikipedia" not in logged


def test_unexpected_provider_error_falls_back(narr_cfg: Config) -> None:
    class Broken:
        name, model, base_url = "broken", "m", "https://example.invalid"

        def complete(self, system: str, user: str) -> str:
            raise RuntimeError("boom")

    result = narrate(make_verdict(), META, cfg=narr_cfg, narrator=Broken(), cache=None)
    assert (result.path, result.failure) == ("template", "error")


def test_registry_and_settings(monkeypatch: pytest.MonkeyPatch, narr_cfg: Config) -> None:
    assert {"openai", "openai_compatible"} <= set(PROVIDERS)
    monkeypatch.setenv("LLM_PROVIDER", "OpenAI")
    monkeypatch.setenv("LLM_API_KEY", KEY)
    loaded = ProviderSettings.from_secrets()
    narrator = make_narrator(loaded, narr_cfg)
    assert isinstance(narrator, OpenAICompatibleNarrator)
    assert narrator.model == narr_cfg["narration"]["default_model"]
    assert narrator.url == URL
    with pytest.raises(NarrationError, match="disabled"):
        make_narrator(ProviderSettings(None, None, None, None), narr_cfg)


def test_system_prompt_states_the_rules(cfg: Config) -> None:
    prompt = system_prompt(cfg)
    assert "{max_words}" in SYSTEM_PROMPT
    assert "at most 120 words" in prompt
    assert "Use only facts in the JSON" in prompt
    assert "exactly" in prompt and "verdict.label" in prompt
