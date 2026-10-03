"""Tests for the X adapter: feature flag, counts pagination, spend ledger and budget cap."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
import responses

from signalcheck import config as config_mod
from signalcheck.adapters.base import fetch_safely
from signalcheck.adapters.x_twitter import (
    API_BASE,
    COUNTS_PATHS,
    MATCH_CAVEAT,
    SEARCH_RECENT_PATH,
    SpendLedger,
    XAdapter,
)
from signalcheck.engine import analyse

COUNTS_ALL = API_BASE + COUNTS_PATHS["all"]
COUNTS_RECENT = API_BASE + COUNTS_PATHS["recent"]
SEARCH_RECENT = API_BASE + SEARCH_RECENT_PATH
SECRET_TEXT = "ZZ confidential x post text 4b8d"
SECRET_AUTHOR_ID = "998877665544332211"


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def mock_counts(url: str = COUNTS_ALL, page_days: int = 31, skip: set[str] | None = None) -> None:
    """Day buckets over [start_time, end_time); value = day-of-month; next_token paging."""

    def handler(request: Any) -> tuple[int, dict[str, str], str]:
        q = {k: v[0] for k, v in parse_qs(urlparse(request.url).query).items()}
        start, end = _parse(q["start_time"]), _parse(q["end_time"])
        offset = int(q.get("next_token", "0"))
        buckets = []
        day = start.date() + timedelta(days=offset)
        while datetime.combine(day, datetime.min.time(), UTC) < end:
            buckets.append(day)
            day += timedelta(days=1)
        page = buckets[:page_days]
        data = [
            {
                "start": f"{d.isoformat()}T00:00:00.000Z",
                "end": f"{(d + timedelta(days=1)).isoformat()}T00:00:00.000Z",
                "tweet_count": d.day,
            }
            for d in page
            if d.isoformat() not in (skip or set())
        ]
        meta: dict[str, Any] = {"total_tweet_count": sum(b["tweet_count"] for b in data)}
        if len(buckets) > page_days:
            meta["next_token"] = str(offset + page_days)
        return 200, {}, json.dumps({"data": data, "meta": meta})

    responses.add_callback(responses.GET, url, callback=handler)


@pytest.fixture
def x_env(monkeypatch: pytest.MonkeyPatch, adapter_deps: dict[str, Any]) -> dict[str, Any]:
    monkeypatch.setattr(config_mod, "_streamlit_secret", lambda _name: None)
    monkeypatch.setenv("ENABLE_X", "true")
    monkeypatch.setenv("X_BEARER_TOKEN", "xtok")
    monkeypatch.delenv("X_MAX_SPEND_USD", raising=False)
    return adapter_deps


def ledger_of(env: dict[str, Any]) -> SpendLedger:
    return XAdapter(**env).ledger


def test_disabled_by_default_reads_planned(
    monkeypatch: pytest.MonkeyPatch, adapter_deps: dict[str, Any]
) -> None:
    monkeypatch.setattr(config_mod, "_streamlit_secret", lambda _name: None)
    monkeypatch.delenv("ENABLE_X", raising=False)
    result = fetch_safely(XAdapter(**adapter_deps), "chatgpt", {})
    assert result.disabled and result.message == "X disabled (planned)"


def test_env_flag_overrides_config(x_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    x_env["cfg"]["sources"]["x"] = True
    monkeypatch.setenv("ENABLE_X", "false")
    result = fetch_safely(XAdapter(**x_env), "chatgpt", {})
    assert result.message == "X disabled (planned)"


def test_missing_token_is_disabled(x_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X_BEARER_TOKEN")
    result = fetch_safely(XAdapter(**x_env), "chatgpt", {})
    assert result.disabled
    assert result.message == "X disabled (API credentials not configured)"


@responses.activate
def test_counts_all_paginates_and_records_spend(x_env: dict[str, Any]) -> None:
    mock_counts()
    adapter = XAdapter(**x_env)
    series = adapter.fetch("Perplexity AI", {"days": 40})

    assert series.scale == "count" and series.freq == "D" and len(series.points) == 40
    assert series.points["ts"].iloc[-1] == pd.Timestamp("2026-10-03")
    pts = series.points.set_index("ts")
    assert pts.loc[pd.Timestamp("2026-09-15"), "value"] == 15
    assert not series.points["imputed"].any()
    assert MATCH_CAVEAT in series.caveats
    assert series.meta["billed_requests"] == 2 and series.meta["counts_endpoint"] == "all"

    assert len(responses.calls) == 2
    req = responses.calls[0].request
    assert req.headers["Authorization"] == "Bearer xtok"
    q = parse_qs(urlparse(req.url).query)
    assert q["query"] == ["Perplexity AI -is:retweet"] and q["granularity"] == ["day"]
    assert q["start_time"] == ["2026-08-25T00:00:00Z"]
    assert q["end_time"] == ["2026-10-03T11:59:30Z"]  # now minus the 30s margin
    assert parse_qs(urlparse(responses.calls[1].request.url).query)["next_token"] == ["31"]

    data = json.loads(adapter.ledger.path.read_text())
    assert data["total_usd"] == pytest.approx(0.02)
    assert [e["endpoint"] for e in data["entries"]] == ["counts/all", "counts/all"]
    assert "Perplexity" not in adapter.ledger.path.read_text()

    verdict = analyse(series, x_env["cfg"])
    assert verdict.label in {"NO_CHANGE", "INCONCLUSIVE", "TREND", "FLUKE", "SEASONAL"}
    assert MATCH_CAVEAT in verdict.caveats


@responses.activate
def test_cached_result_spends_nothing(x_env: dict[str, Any]) -> None:
    mock_counts()
    XAdapter(**x_env).fetch("chatgpt", {"days": 20})
    again = XAdapter(**x_env)
    again.fetch("chatgpt", {"days": 20})
    assert len(responses.calls) == 1
    assert again.ledger.total() == pytest.approx(0.01)


@responses.activate
def test_budget_cap_refuses_before_any_request(
    x_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_counts()
    monkeypatch.setenv("X_MAX_SPEND_USD", "0.015")
    result = fetch_safely(XAdapter(**x_env), "chatgpt", {"days": 40})  # needs 2 pages = $0.02
    assert not result.ok and not result.disabled
    assert result.message is not None and "budget cap reached" in result.message
    assert len(responses.calls) == 0
    assert not ledger_of(x_env).path.exists()


@responses.activate
def test_budget_cap_counts_previous_spend(
    x_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_counts()
    monkeypatch.setenv("X_MAX_SPEND_USD", "1")
    ledger = ledger_of(x_env)
    ledger.record("counts/all", 0.995, 1, datetime(2026, 10, 1, tzinfo=UTC))
    result = fetch_safely(XAdapter(**x_env), "chatgpt", {"days": 10})
    assert result.message is not None and "$0.995 of $1.00" in result.message
    assert len(responses.calls) == 0 and ledger.total() == pytest.approx(0.995)


@responses.activate
def test_default_budget_comes_from_config(x_env: dict[str, Any]) -> None:
    x_env["cfg"]["adapters"]["x"]["default_max_spend_usd"] = 0.0
    result = fetch_safely(XAdapter(**x_env), "chatgpt", {"days": 10})
    assert result.message is not None and "budget cap reached" in result.message


@pytest.mark.parametrize("raw", ["five", "-1", "nan"])
def test_invalid_budget_fails_closed(
    x_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("X_MAX_SPEND_USD", raw)
    result = fetch_safely(XAdapter(**x_env), "chatgpt", {"days": 10})
    assert not result.ok and result.message is not None and "X_MAX_SPEND_USD" in result.message


@responses.activate
def test_corrupt_ledger_fails_closed(x_env: dict[str, Any]) -> None:
    mock_counts()
    ledger = ledger_of(x_env)
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    ledger.path.write_text("{not json")
    result = fetch_safely(XAdapter(**x_env), "chatgpt", {"days": 10})
    assert result.message is not None and "ledger" in result.message
    assert len(responses.calls) == 0


@responses.activate
def test_rate_limit_exhaustion_is_graceful_and_unbilled(x_env: dict[str, Any]) -> None:
    responses.get(COUNTS_ALL, status=429, json={"title": "Too Many Requests"})
    adapter = XAdapter(**x_env)
    result = fetch_safely(adapter, "chatgpt", {"days": 10})
    assert result.message == "Couldn't fetch X: HTTP 429 after 3 attempt(s)"
    assert len(responses.calls) == 3
    assert adapter.ledger.total() == 0


@responses.activate
@pytest.mark.parametrize(("status", "text"), [(401, "bearer token"), (402, "no API credits")])
def test_auth_and_credit_errors(x_env: dict[str, Any], status: int, text: str) -> None:
    responses.get(COUNTS_ALL, status=status, json={"title": "nope"})
    result = fetch_safely(XAdapter(**x_env), "chatgpt", {"days": 10})
    assert result.message is not None and text in result.message


@responses.activate
def test_missing_buckets_become_imputed_zeros(x_env: dict[str, Any]) -> None:
    mock_counts(skip={"2026-09-30"})
    series = XAdapter(**x_env).fetch("chatgpt", {"days": 10})
    pts = series.points.set_index("ts")
    assert pts.loc[pd.Timestamp("2026-09-30"), "value"] == 0
    assert bool(pts.loc[pd.Timestamp("2026-09-30"), "imputed"])
    assert any("no bucket" in c for c in series.caveats)


@responses.activate
def test_recent_endpoint_clips_to_seven_days(x_env: dict[str, Any]) -> None:
    x_env["cfg"]["adapters"]["x"]["counts_endpoint"] = "recent"
    mock_counts(url=COUNTS_RECENT)
    adapter = XAdapter(**x_env)
    series = adapter.fetch("chatgpt", {"days": 30})
    assert series.points["ts"].iloc[0] == pd.Timestamp("2026-09-27")
    assert len(series.points) == 7
    assert any("counts/recent covers only the last 7 days" in c for c in series.caveats)
    assert adapter.ledger.total() == pytest.approx(0.005)


@responses.activate
def test_breadth_samples_recent_days_without_persisting_ids(
    x_env: dict[str, Any], tmp_path: Path
) -> None:
    x_env["cfg"]["adapters"]["x"]["breadth_enabled"] = True
    x_env["cfg"]["adapters"]["x"]["breadth_posts_per_day"] = 10
    mock_counts()

    def search(request: Any) -> tuple[int, dict[str, str], str]:
        posts = [
            {"id": str(i), "text": SECRET_TEXT, "author_id": SECRET_AUTHOR_ID if i < 3 else str(i)}
            for i in range(4)
        ]
        return 200, {}, json.dumps({"data": posts, "meta": {"result_count": 4}})

    responses.add_callback(responses.GET, SEARCH_RECENT, callback=search)
    adapter = XAdapter(**x_env)
    series = adapter.fetch("chatgpt", {"days": 30})

    searches = [c for c in responses.calls if c.request.url.startswith(SEARCH_RECENT)]
    # Recent window for 29 complete days is 7 days; only those within 7 days of now
    # (2026-09-27..10-02) plus the partial 10-03 qualify.
    assert len(searches) == 7
    q = parse_qs(urlparse(searches[0].request.url).query)
    assert q["tweet.fields"] == ["author_id"] and q["max_results"] == ["10"]
    pts = series.points.set_index("ts")
    oct2 = pts.loc[pd.Timestamp("2026-10-02")]
    assert (oct2["contributors"], oct2["top_share"]) == (2, 0.75)
    assert pd.isna(pts.loc[pd.Timestamp("2026-09-20"), "contributors"])
    assert any("Breadth on X is sampled" in c for c in series.caveats)
    # 1 counts page ($0.01) + 7 days x 4 posts x $0.005
    assert adapter.ledger.total() == pytest.approx(0.01 + 7 * 4 * 0.005)

    for path in tmp_path.rglob("*"):
        if path.is_file():
            blob = path.read_bytes()
            assert SECRET_TEXT.encode() not in blob, path
            assert SECRET_AUTHOR_ID.encode() not in blob, path


@responses.activate
def test_breadth_projection_counts_toward_budget(
    x_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    x_env["cfg"]["adapters"]["x"]["breadth_enabled"] = True
    monkeypatch.setenv("X_MAX_SPEND_USD", "0.5")  # 1 page + 7 x 20 x $0.005 = $0.71
    mock_counts()
    result = fetch_safely(XAdapter(**x_env), "chatgpt", {"days": 30})
    assert result.message is not None and "budget cap reached" in result.message
    assert len(responses.calls) == 0
