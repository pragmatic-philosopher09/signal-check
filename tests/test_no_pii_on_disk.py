"""Privacy guard: no author names, ids or post text reach disk.

Runs the Hacker News, Reddit and X adapters on mocked responses full of distinctive
author names and text, writes everything those flows persist (disk cache, sample
snapshots, collected Reddit history, X spend ledger), then scans every file under
the temporary directory for those markers.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import responses

from scripts import collect_reddit
from signalcheck import config as config_mod
from signalcheck.adapters.hackernews import SEARCH_URL as HN_URL
from signalcheck.adapters.hackernews import HackerNewsAdapter
from signalcheck.adapters.reddit import SEARCH_URL as REDDIT_URL
from signalcheck.adapters.reddit import TOKEN_URL, RedditAdapter
from signalcheck.adapters.x_twitter import API_BASE, COUNTS_PATHS, SEARCH_RECENT_PATH, XAdapter
from signalcheck.config import Config
from signalcheck.engine import analyse
from signalcheck.snapshots import read_snapshot, write_snapshot
from tests.conftest import FIXED_NOW

MARKERS = {
    "hn_author": "zzhnauthor_c41f",
    "hn_text": "ZZ private hn comment text 0d2e",
    "reddit_author": "zzredditauthor_9a7b",
    "reddit_sub": "zzsecretsub_51aa",
    "reddit_text": "ZZ private reddit selftext 6c3f",
    "x_author": "771166554433221100",
    "x_text": "ZZ private x post text 2e9a",
}


def find_markers(root: Path) -> list[tuple[str, str]]:
    """``(file, marker)`` pairs for every marker found in any file under ``root``."""
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        blob = path.read_bytes()
        for name, marker in MARKERS.items():
            for encoded in (marker.encode(), marker.encode("utf-16-le")):
                if encoded in blob:
                    found.append((str(path.relative_to(root)), name))
    return found


def mock_hn() -> None:
    def handler(request: Any) -> tuple[int, dict[str, str], str]:
        per_page = int(parse_qs(urlparse(request.url).query)["hitsPerPage"][0])
        hits = [
            {
                "author": f"{MARKERS['hn_author']}{i % 2}",
                "comment_text": MARKERS["hn_text"],
                "title": MARKERS["hn_text"],
                "created_at_i": 0,
                "objectID": str(i),
            }
            for i in range(min(per_page, 3))
        ]
        return 200, {}, json.dumps({"nbHits": 3, "exhaustiveNbHits": True, "hits": hits})

    responses.add_callback(responses.GET, HN_URL, callback=handler)


def mock_reddit() -> None:
    responses.post(TOKEN_URL, json={"access_token": "tok", "token_type": "bearer"})
    children = [
        {
            "kind": "t3",
            "data": {
                "created_utc": (FIXED_NOW - timedelta(hours=6 * i + 13)).timestamp(),
                "author": f"{MARKERS['reddit_author']}{i % 3}",
                "subreddit": MARKERS["reddit_sub"],
                "title": MARKERS["reddit_text"],
                "selftext": MARKERS["reddit_text"],
            },
        }
        for i in range(30)
    ]
    responses.get(REDDIT_URL, json={"data": {"children": children, "after": None}})


def mock_x() -> None:
    def counts(request: Any) -> tuple[int, dict[str, str], str]:
        data = [
            {"start": f"2026-09-{d:02d}T00:00:00.000Z", "tweet_count": d} for d in range(4, 31)
        ] + [{"start": f"2026-10-0{d}T00:00:00.000Z", "tweet_count": d} for d in range(1, 4)]
        return 200, {}, json.dumps({"data": data, "meta": {}})

    posts = [
        {"id": str(i), "text": MARKERS["x_text"], "author_id": MARKERS["x_author"]}
        for i in range(5)
    ]
    responses.add_callback(responses.GET, API_BASE + COUNTS_PATHS["all"], callback=counts)
    responses.get(API_BASE + SEARCH_RECENT_PATH, json={"data": posts, "meta": {}})


@responses.activate
def test_no_author_names_or_text_on_disk(
    adapter_deps: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_mod, "_streamlit_secret", lambda _name: None)
    monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("ENABLE_X", "true")
    monkeypatch.setenv("X_BEARER_TOKEN", "xtok")
    monkeypatch.delenv("X_MAX_SPEND_USD", raising=False)
    cfg = adapter_deps["cfg"]
    cfg["adapters"]["reddit"]["collected_dir"] = str(tmp_path / "collected")
    cfg["adapters"]["x"]["breadth_enabled"] = True
    samples = tmp_path / "samples"
    mock_hn()
    mock_reddit()
    mock_x()

    hn = HackerNewsAdapter(**adapter_deps).fetch("privacy", {"days": 30})
    reddit_adapter = RedditAdapter(**adapter_deps)
    reddit = reddit_adapter.fetch("privacy", {"days": 10})
    x = XAdapter(**adapter_deps).fetch("privacy", {"days": 30})
    ok, errors = collect_reddit.collect(["privacy"], cfg, adapter=reddit_adapter)
    assert (ok, errors) == (1, [])
    for series in (hn, reddit, x):
        assert series.points["contributors"].notna().any()  # breadth was really computed
        write_snapshot(series, samples)
    adapter_deps["cache"].close()

    written = {p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*") if p.is_file()}
    assert {"cache", "samples", "collected"} <= written
    assert (tmp_path / "cache" / cfg["adapters"]["x"]["ledger_file"]).exists()
    assert len(list(samples.rglob("*.json"))) == 3
    assert find_markers(tmp_path) == []


def test_scanner_detects_a_leak(tmp_path: Path) -> None:
    (tmp_path / "leak.bin").write_bytes(b"\x00prefix" + MARKERS["reddit_author"].encode())
    assert find_markers(tmp_path) == [("leak.bin", "reddit_author")]


COMMITTED = sorted((Path(__file__).resolve().parents[1] / "data" / "samples").rglob("*.json"))
AGGREGATE_COLUMNS = {"ts", "value", "contributors", "top_share", "raw_count", "imputed"}


@pytest.mark.parametrize("path", COMMITTED, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_committed_snapshots_hold_aggregates_only(path: Path, cfg: Config) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert {k for point in data["points"] for k in point} <= AGGREGATE_COLUMNS
    series = read_snapshot(path)
    assert series.meta["fetched_at"]
    assert analyse(series, cfg).label
