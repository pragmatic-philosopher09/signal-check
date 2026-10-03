"""Tests for the Reddit adapter and scripts/collect_reddit.py (HTTP mocked)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
import responses

from scripts import collect_reddit
from signalcheck import config as config_mod
from signalcheck.adapters.base import fetch_safely
from signalcheck.adapters.reddit import (
    POSTS_ONLY_CAVEAT,
    SEARCH_URL,
    TOKEN_URL,
    RedditAdapter,
    read_collected,
    write_collected,
)
from signalcheck.engine import analyse

NOW = datetime(2026, 10, 3, 12, tzinfo=UTC)
SECRET_AUTHOR = "zz_reddit_author_7f3a"
SECRET_TITLE = "ZZ confidential reddit title 91c2"


def post(hours_ago: float, author: str = "alice", subreddit: str = "tech") -> dict[str, Any]:
    """A Reddit listing child with text fields that must never reach disk."""
    return {
        "kind": "t3",
        "data": {
            "created_utc": (NOW - timedelta(hours=hours_ago)).timestamp(),
            "author": author,
            "subreddit": subreddit,
            "title": SECRET_TITLE,
            "selftext": f"{SECRET_TITLE} body",
        },
    }


def mock_reddit(posts: list[dict[str, Any]], page: int = 100) -> None:
    """Token endpoint + search listing paginated with ``after`` = next index."""
    responses.post(TOKEN_URL, json={"access_token": "tok-123", "token_type": "bearer"})

    def search(request: Any) -> tuple[int, dict[str, str], str]:
        params = {k: v[0] for k, v in parse_qs(urlparse(request.url).query).items()}
        start = int(params.get("after", "0"))
        limit = min(int(params["limit"]), page)
        chunk = posts[start : start + limit]
        nxt = start + limit
        after = str(nxt) if nxt < len(posts) else None
        headers = {"X-Ratelimit-Remaining": "99", "X-Ratelimit-Reset": "300"}
        return 200, headers, json.dumps({"data": {"children": chunk, "after": after}})

    responses.add_callback(responses.GET, SEARCH_URL, callback=search)


@pytest.fixture
def reddit_env(
    monkeypatch: pytest.MonkeyPatch, adapter_deps: dict[str, Any], tmp_path: Path
) -> dict[str, Any]:
    monkeypatch.setattr(config_mod, "_streamlit_secret", lambda _name: None)
    monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("REDDIT_USERNAME", "signalbot")
    cfg = adapter_deps["cfg"]
    cfg["adapters"]["reddit"]["collected_dir"] = str(tmp_path / "collected")
    return adapter_deps


@responses.activate
def test_daily_counts_and_breadth(reddit_env: dict[str, Any]) -> None:
    posts = [
        post(1, "alice", "a"),  # 2026-10-03 (partial day)
        post(13, "bob", "a"),  # 2026-10-02
        post(14, "bob", "b"),
        post(15, "[deleted]", "c"),
        post(16, "carol", "b"),
        post(40, "dan", "x"),  # 2026-10-01
        post(24 * 10, "eve", "y"),  # outside a 5-day range -> stops pagination
        post(24 * 11, "fay", "y"),
    ]
    mock_reddit(posts, page=3)
    series = RedditAdapter(**reddit_env).fetch("Perplexity AI", {"days": 5})

    assert series.scale == "count" and series.freq == "D"
    assert series.meta["truncated_before"] is None
    assert POSTS_ONLY_CAVEAT in series.caveats
    pts = series.points.set_index("ts")
    oct2 = pts.loc[pd.Timestamp("2026-10-02")]
    # 4 posts, 2 known authors (deleted is an item, not a contributor);
    # top author share bob 2/4 = 0.5 beats top subreddit share 2/4 -> 0.5.
    assert (oct2["value"], oct2["contributors"], oct2["top_share"]) == (4, 2, 0.5)
    assert pts.loc[pd.Timestamp("2026-09-30"), "value"] == 0
    assert pd.isna(pts.loc[pd.Timestamp("2026-09-30"), "top_share"])
    assert len(series.points) == 5

    searches = [c for c in responses.calls if c.request.url.startswith(SEARCH_URL)]
    assert len(searches) == 3  # third page holds the first post older than the range
    req = searches[0].request
    assert req.headers["Authorization"] == "bearer tok-123"
    assert req.headers["User-Agent"].endswith("(by /u/signalbot)")
    q = parse_qs(urlparse(req.url).query)
    assert q["sort"] == ["new"] and q["t"] == ["all"] and q["q"] == ["Perplexity AI"]
    assert sum(c.request.url == TOKEN_URL for c in responses.calls) == 1


@responses.activate
def test_result_cap_truncates_and_merges_collected(reddit_env: dict[str, Any]) -> None:
    cfg = reddit_env["cfg"]
    cfg["adapters"]["reddit"].update(result_cap=20, page_limit=10)
    # 30 posts every 3 hours from now back: the cap stops after 20 (back to 57h ago).
    posts = [post(3 * i, f"user{i % 4}", f"sub{i % 2}") for i in range(30)]
    mock_reddit(posts)
    collected = Path(cfg["adapters"]["reddit"]["collected_dir"]) / "chatgpt.csv"
    write_collected(
        collected,
        {
            f"2026-09-{d:02d}": {"count": 50.0, "contributors": 40.0, "top_share": 0.1}
            for d in range(25, 32)
            if d <= 30
        }
        | {"2026-10-01": {"count": 999.0, "contributors": 1.0, "top_share": 1.0}},
    )
    series = RedditAdapter(**reddit_env).fetch("ChatGPT", {"days": 10})

    # Oldest fetched post: 57h before noon 10-03 -> 2026-10-01 03:00; first full day 10-02.
    assert series.meta["live_coverage_start"] == "2026-10-02"
    assert series.meta["truncated_before"] == "2026-09-25"
    assert series.meta["collected_days"] == 7
    pts = series.points.set_index("ts")
    assert pts.loc[pd.Timestamp("2026-10-01"), "value"] == 999  # collected fills the dropped day
    assert pts.loc[pd.Timestamp("2026-10-02"), "value"] == 8  # live, posts 15..36h ago
    assert any("stopped at its 20-result limit" in c for c in series.caveats)
    verdict = analyse(series, cfg)
    assert verdict.label


@responses.activate
def test_cap_within_a_day_without_history_is_graceful(reddit_env: dict[str, Any]) -> None:
    reddit_env["cfg"]["adapters"]["reddit"].update(result_cap=10, page_limit=10)
    mock_reddit([post(i * 0.1) for i in range(30)])
    result = fetch_safely(RedditAdapter(**reddit_env), "nvidia", {"days": 7})
    assert not result.ok and "1,000-result limit within one day" in (result.error or "")


@responses.activate
def test_empty_results(reddit_env: dict[str, Any]) -> None:
    mock_reddit([])
    series = RedditAdapter(**reddit_env).fetch("obscure topic", {"days": 3})
    assert series.points["value"].tolist() == [0, 0, 0]
    assert series.meta["truncated_before"] is None


def test_missing_credentials_disable_the_source(
    monkeypatch: pytest.MonkeyPatch, adapter_deps: dict[str, Any]
) -> None:
    monkeypatch.setattr(config_mod, "_streamlit_secret", lambda _name: None)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    result = fetch_safely(RedditAdapter(**adapter_deps), "chatgpt", {})
    assert result.disabled
    assert result.message == "Reddit disabled (API credentials not configured)"


@responses.activate
def test_rejected_credentials(reddit_env: dict[str, Any]) -> None:
    responses.post(TOKEN_URL, status=401, json={"error": 401})
    result = fetch_safely(RedditAdapter(**reddit_env), "chatgpt", {"days": 3})
    assert result.message == "Couldn't fetch Reddit: Reddit rejected the API credentials (HTTP 401)"


@responses.activate
def test_5xx_exhaustion_is_graceful(reddit_env: dict[str, Any]) -> None:
    responses.post(TOKEN_URL, json={"access_token": "t"})
    responses.get(SEARCH_URL, status=502)
    result = fetch_safely(RedditAdapter(**reddit_env), "chatgpt", {"days": 3})
    assert result.message == "Couldn't fetch Reddit: HTTP 502 after 3 attempt(s)"


@responses.activate
def test_rate_limit_headers_pause(reddit_env: dict[str, Any]) -> None:
    sleeps: list[float] = []
    reddit_env["sleep"] = sleeps.append
    responses.post(TOKEN_URL, json={"access_token": "t"})
    responses.get(
        SEARCH_URL,
        json={"data": {"children": [post(1)], "after": None}},
        headers={"X-Ratelimit-Remaining": "0", "X-Ratelimit-Reset": "7"},
    )
    RedditAdapter(**reddit_env).fetch("chatgpt", {"days": 2})
    assert 7.0 in sleeps


@responses.activate
def test_malformed_collected_file_is_a_caveat(reddit_env: dict[str, Any]) -> None:
    path = Path(reddit_env["cfg"]["adapters"]["reddit"]["collected_dir"]) / "chatgpt.csv"
    path.parent.mkdir(parents=True)
    path.write_text("date,count\nnot-a-date,3\n", encoding="utf-8")
    mock_reddit([post(1)])
    series = RedditAdapter(**reddit_env).fetch("chatgpt", {"days": 2})
    assert any(c.startswith("Collected history ignored") for c in series.caveats)


@responses.activate
def test_collector_upserts_complete_days(reddit_env: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_collected(
        out / "rust-programming.csv",
        {"2026-09-01": {"count": 3.0, "contributors": 2.0, "top_share": float("nan")}},
    )
    mock_reddit([post(2, "a"), post(13, "b"), post(14, "c", "rust"), post(24 * 3, "d")])
    adapter = RedditAdapter(**reddit_env)
    ok, errors = collect_reddit.collect(["rust programming"], reddit_env["cfg"], out, adapter)
    assert (ok, errors) == (1, [])
    rows = read_collected(out / "rust-programming.csv")
    assert "2026-10-03" not in rows  # today is incomplete
    assert rows["2026-10-02"]["count"] == 2 and rows["2026-09-30"]["count"] == 1
    assert rows["2026-09-01"]["count"] == 3  # existing history kept
    assert min(rows) == "2026-09-01" and "2026-09-26" in rows
    text = (out / "rust-programming.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "date,count,contributors,top_share"


def test_collector_exit_code_when_all_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config_mod, "_streamlit_secret", lambda _name: None)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    assert collect_reddit.main(["--topics", "chatgpt", "--out", str(tmp_path)]) == 1
    assert not any(tmp_path.iterdir())
