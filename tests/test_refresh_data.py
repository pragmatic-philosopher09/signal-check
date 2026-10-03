"""Tests for scripts/refresh_data.py and the snapshot merge/writer (no network: fake adapters)."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts import refresh_data as rd
from signalcheck.adapters.base import AdapterDisabled, AdapterError
from signalcheck.config import Config, get_config
from signalcheck.http import HttpError
from signalcheck.models import Series
from signalcheck.snapshots import (
    merge_history,
    read_snapshot,
    series_to_dict,
    snapshot_path,
    snapshot_text,
    write_snapshot_if_changed,
    write_snapshot_to,
)
from signalcheck.watchlist import WatchItem

T1 = "2026-10-01T03:00:00+00:00"
T2 = "2026-10-02T03:00:00+00:00"
NOW1 = datetime(2026, 10, 1, 3, 0, tzinfo=UTC)
NOW2 = datetime(2026, 10, 2, 3, 0, tzinfo=UTC)


def make_series(
    source: str,
    query: str,
    start: str,
    values: list[float],
    fetched_at: str = T1,
    scale: str = "pageviews",
    **meta: Any,
) -> Series:
    points = pd.DataFrame(
        {
            "ts": pd.date_range(start, periods=len(values), freq="D").astype("datetime64[ns]"),
            "value": [float(v) for v in values],
        }
    )
    full_meta = {"fetched_at": fetched_at, "caveats": ["base caveat"], "resolved_query": query}
    full_meta.update(meta)
    return Series(source, query, "D", points, scale, full_meta)


class FakeAdapter:
    """Returns canned series (or raises) per query; records every call."""

    def __init__(self, source: str, data: Mapping[str, Series | Exception]) -> None:
        self.source = source
        self.data = dict(data)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def fetch(self, query: str, params: Mapping[str, Any]) -> Series:
        self.calls.append((query, dict(params)))
        value = self.data[query]
        if isinstance(value, Exception):
            raise value
        return value


class FakeReddit(FakeAdapter):
    def __init__(self, data: Mapping[str, Series | Exception], creds: bool = True) -> None:
        super().__init__("reddit", data)
        self.creds = creds

    def ensure_enabled(self) -> None:
        return None

    def credentials(self) -> tuple[str, str]:
        if not self.creds:
            raise AdapterDisabled("API credentials not configured")
        return "id", "secret"


def item(topic: str, **sources: Any) -> WatchItem:
    wiki = sources.get("wikipedia", False)
    return WatchItem(
        topic=topic,
        wikipedia=bool(wiki),
        wiki_article=wiki if isinstance(wiki, str) else None,
        hackernews=bool(sources.get("hackernews", False)),
        reddit=bool(sources.get("reddit", False)),
    )


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    data = copy.deepcopy(get_config())
    data["samples"]["dir"] = str(tmp_path / "samples")
    data["refresh"]["manifest"] = str(tmp_path / "manifest.json")
    data["refresh"]["max_history_days"] = 30
    return data


def factories(**adapters: Any) -> dict[str, Any]:
    return {source: (lambda _cfg, a=a: a) for source, a in adapters.items()}


# --- snapshot writer --------------------------------------------------------------


def test_snapshot_text_is_deterministic_compact_and_roundtrips(tmp_path: Path) -> None:
    series = make_series("wikipedia", "chatgpt", "2026-09-01", [3, 1, 2], article="ChatGPT")
    data = series_to_dict(series)
    text = snapshot_text(data)
    assert text == snapshot_text(json.loads(json.dumps(data)))
    assert json.loads(text) == data
    lines = text.splitlines()
    assert lines[0].startswith('{"freq":"D","meta":{"article":"ChatGPT","caveats":')
    assert lines[1] == '{"ts":"2026-09-01","value":3.0},'
    assert lines[-1] == "]}"
    assert len(lines) == 3 + 2  # header, three points, closing line
    assert ": " not in text and ", " not in lines[1]
    path = write_snapshot_to(series, tmp_path / "s.json")
    back = read_snapshot(path)
    pd.testing.assert_frame_equal(back.points, series.points)


def test_write_if_changed_ignores_only_fetched_at(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    first = make_series("hackernews", "rust", "2026-09-01", [1, 2], fetched_at=T1, scale="count")
    assert write_snapshot_if_changed(first, path)
    before = path.read_bytes()
    same = make_series("hackernews", "rust", "2026-09-01", [1, 2], fetched_at=T2, scale="count")
    assert not write_snapshot_if_changed(same, path)
    assert path.read_bytes() == before
    changed = make_series("hackernews", "rust", "2026-09-01", [1, 3], fetched_at=T2, scale="count")
    assert write_snapshot_if_changed(changed, path)
    assert json.loads(path.read_text())["meta"]["fetched_at"] == T2


# --- history merge ----------------------------------------------------------------


def test_merge_keeps_older_points_and_fresh_wins() -> None:
    previous = make_series("wikipedia", "q", "2026-09-01", [1] * 10, article="Q")
    fresh = make_series("wikipedia", "q", "2026-09-06", [5] * 10, fetched_at=T2, article="Q")
    merged = merge_history(previous, fresh, max_days=100)
    assert len(merged.points) == 15
    assert merged.points["ts"].iloc[0] == pd.Timestamp("2026-09-01")
    assert list(merged.points["value"]) == [1.0] * 5 + [5.0] * 10
    assert merged.meta["merged_days"] == 5
    assert merged.meta["fetched_at"] == T2
    assert merged.caveats == [
        "base caveat",
        "5 earlier day(s) come from previous daily refreshes of this snapshot.",
    ]
    # Merging again with the merged snapshot as history does not stack caveats.
    again = merge_history(merged, fresh, max_days=100)
    assert again.caveats == merged.caveats
    pd.testing.assert_frame_equal(again.points, merged.points)


def test_merge_trims_to_max_days() -> None:
    previous = make_series("wikipedia", "q", "2026-08-01", [1] * 40, article="Q")
    fresh = make_series("wikipedia", "q", "2026-09-05", [2] * 10, article="Q")
    merged = merge_history(previous, fresh, max_days=20)
    assert len(merged.points) == 20
    assert merged.points["ts"].iloc[-1] == pd.Timestamp("2026-09-14")
    assert merged.meta["merged_days"] == 10


@pytest.mark.parametrize(
    "previous",
    [
        make_series("wikipedia", "q", "2026-09-01", [1] * 5, article="Other article"),
        make_series("hackernews", "q", "2026-09-01", [1] * 5, scale="count", tags="(story)"),
        make_series("google_trends", "q", "2026-09-01", [1] * 5, scale="relative_0_100"),
        None,
    ],
)
def test_merge_refuses_different_series(previous: Series | None) -> None:
    source = previous.source if previous is not None else "wikipedia"
    scale = previous.scale if previous is not None else "pageviews"
    article = "Q" if source == "wikipedia" else None
    fresh = make_series(source, "q", "2026-09-04", [7] * 3, scale=scale, article=article)
    merged = merge_history(previous, fresh, max_days=100)
    assert len(merged.points) == 3
    assert "merged_days" not in merged.meta


def test_merge_fills_imputed_flags() -> None:
    previous = make_series("wikipedia", "q", "2026-09-01", [1] * 3, article="Q")
    previous.points["imputed"] = [False, True, False]
    fresh = make_series("wikipedia", "q", "2026-09-04", [2] * 2, article="Q")
    merged = merge_history(previous, fresh, max_days=100)
    assert merged.points["imputed"].dtype == bool
    assert list(merged.points["imputed"]) == [False, True, False, False, False]


# --- refresh ----------------------------------------------------------------------


def test_refresh_writes_merges_and_passes_article_override(cfg: Config, tmp_path: Path) -> None:
    base = tmp_path / "samples"
    old = make_series("wikipedia", "claude ai", "2026-09-01", [1] * 5, article="Claude (LM)")
    write_snapshot_to(old, snapshot_path("wikipedia", "claude ai", base))
    wiki = FakeAdapter(
        "wikipedia",
        {
            "claude ai": make_series(
                "wikipedia", "claude ai", "2026-09-04", [9] * 4, T2, **{"article": "Claude (LM)"}
            )
        },
    )
    hn = FakeAdapter(
        "hackernews",
        {"claude ai": make_series("hackernews", "claude ai", "2026-09-01", [2] * 3, scale="count")},
    )
    items = [item("claude ai", wikipedia="Claude (LM)", hackernews=True)]
    entries = rd.refresh(items, cfg, factories=factories(wikipedia=wiki, hackernews=hn))
    assert [(e.source, e.status) for e in entries] == [
        ("wikipedia", rd.UPDATED),
        ("hackernews", rd.UPDATED),
    ]
    assert wiki.calls == [("claude ai", {"article": "Claude (LM)"})]
    assert hn.calls == [("claude ai", {})]
    merged = read_snapshot(snapshot_path("wikipedia", "claude ai", base))
    assert len(merged.points) == 7
    assert merged.meta["merged_days"] == 3
    assert entries[0].fetched_at == T2 and entries[0].points == 7


def test_refresh_is_idempotent_without_churn(cfg: Config, tmp_path: Path) -> None:
    base = tmp_path / "samples"
    items = [item("rust", wikipedia=True, hackernews=True)]

    def run(fetched_at: str) -> list[rd.Entry]:
        wiki = FakeAdapter(
            "wikipedia",
            {
                "rust": make_series(
                    "wikipedia", "rust", "2026-09-01", [4, 5, 6], fetched_at, article="Rust"
                )
            },
        )
        hn = FakeAdapter(
            "hackernews",
            {
                "rust": make_series(
                    "hackernews", "rust", "2026-09-01", [1, 2], fetched_at, scale="count"
                )
            },
        )
        return rd.refresh(items, cfg, factories=factories(wikipedia=wiki, hackernews=hn))

    run(T1)
    files = sorted(base.rglob("*.json"))
    before = {p: p.read_bytes() for p in files}
    second = run(T2)
    assert [e.status for e in second] == [rd.UNCHANGED, rd.UNCHANGED]
    assert [e.fetched_at for e in second] == [T1, T1]
    assert {p: p.read_bytes() for p in sorted(base.rglob("*.json"))} == before


def test_partial_failure_keeps_previous_snapshot(cfg: Config, tmp_path: Path) -> None:
    base = tmp_path / "samples"
    old = make_series("wikipedia", "nvidia", "2026-09-01", [1, 2, 3], article="Nvidia")
    path = write_snapshot_to(old, snapshot_path("wikipedia", "nvidia", base))
    before = path.read_bytes()
    wiki = FakeAdapter("wikipedia", {"nvidia": HttpError("HTTP 503 from Wikimedia")})
    hn = FakeAdapter(
        "hackernews",
        {"nvidia": make_series("hackernews", "nvidia", "2026-09-01", [1], scale="count")},
    )
    items = [item("nvidia", wikipedia=True, hackernews=True)]
    entries = rd.refresh(items, cfg, factories=factories(wikipedia=wiki, hackernews=hn))
    failed, ok = entries
    assert failed.status == rd.FAILED
    assert failed.error == "Couldn't fetch Wikipedia: HTTP 503 from Wikimedia"
    assert failed.fetched_at == T1 and failed.points == 3
    assert path.read_bytes() == before
    assert ok.status == rd.UPDATED
    assert rd.exit_code(entries) == 0


def test_every_source_failed_exits_nonzero(cfg: Config) -> None:
    wiki = FakeAdapter("wikipedia", {"a": AdapterError("boom"), "b": AdapterError("boom")})
    items = [item("a", wikipedia=True), item("b", wikipedia=True)]
    entries = rd.refresh(items, cfg, factories=factories(wikipedia=wiki))
    assert [e.status for e in entries] == [rd.FAILED, rd.FAILED]
    assert rd.exit_code(entries) == 1
    # Skipped sources don't count as attempts.
    skipped = rd.Entry("c", "reddit", rd.SKIPPED)
    assert rd.exit_code([*entries, skipped]) == 1
    assert rd.exit_code([skipped]) == 0


def test_refuses_to_snapshot_a_snapshot(cfg: Config) -> None:
    snap = make_series("wikipedia", "a", "2026-09-01", [1], snapshot=True)
    entries = rd.refresh(
        [item("a", wikipedia=True)],
        cfg,
        factories=factories(wikipedia=FakeAdapter("wikipedia", {"a": snap})),
    )
    assert entries[0].status == rd.FAILED
    assert "refusing to snapshot a snapshot" in (entries[0].error or "")


def test_reddit_skipped_without_credentials(cfg: Config) -> None:
    reddit = FakeReddit({}, creds=False)
    collected: list[str] = []
    entries = rd.refresh(
        [item("chatgpt", reddit=True)],
        cfg,
        factories=factories(reddit=reddit),
        collect=lambda _a, topic: collected.append(topic) or 0,
    )
    assert entries == [
        rd.Entry(
            "chatgpt",
            "reddit",
            rd.SKIPPED,
            error="Couldn't fetch Reddit: API credentials not configured",
        )
    ]
    assert collected == [] and reddit.calls == []
    assert rd.exit_code(entries) == 0


def test_reddit_collects_then_snapshots_without_merge(cfg: Config, tmp_path: Path) -> None:
    base = tmp_path / "samples"
    old = make_series("reddit", "chatgpt", "2026-08-01", [1] * 3, scale="count")
    write_snapshot_to(old, snapshot_path("reddit", "chatgpt", base))
    fresh = make_series("reddit", "chatgpt", "2026-09-01", [4] * 3, T2, scale="count")
    reddit = FakeReddit({"chatgpt": fresh})
    order: list[str] = []

    def collect(_adapter: Any, topic: str) -> int:
        order.append(f"collect {topic}")
        return 3

    entries = rd.refresh(
        [item("chatgpt", reddit=True)], cfg, factories=factories(reddit=reddit), collect=collect
    )
    assert order == ["collect chatgpt"]
    assert entries[0].status == rd.UPDATED
    # History comes from data/collected via the adapter, not from the old snapshot.
    assert len(read_snapshot(snapshot_path("reddit", "chatgpt", base)).points) == 3


def test_reddit_collect_failure_is_recorded(cfg: Config) -> None:
    def collect(_adapter: Any, _topic: str) -> int:
        raise HttpError("rate limited")

    entries = rd.refresh(
        [item("chatgpt", reddit=True)],
        cfg,
        factories=factories(reddit=FakeReddit({})),
        collect=collect,
    )
    assert entries[0].status == rd.FAILED
    assert entries[0].error == "Couldn't fetch Reddit: rate limited"


def test_sources_filter_and_no_x(cfg: Config) -> None:
    assert set(rd.ADAPTERS) == {"wikipedia", "hackernews", "reddit"}
    hn = FakeAdapter(
        "hackernews", {"a": make_series("hackernews", "a", "2026-09-01", [1], scale="count")}
    )
    entries = rd.refresh(
        [item("a", wikipedia=True, hackernews=True)],
        cfg,
        sources=["hackernews"],
        factories=factories(hackernews=hn),
    )
    assert [e.source for e in entries] == ["hackernews"]


# --- manifest and summary ---------------------------------------------------------


def test_manifest_records_status_and_merges_previous_runs() -> None:
    first = [
        rd.Entry("chatgpt", "wikipedia", rd.UPDATED, T1, 800),
        rd.Entry("chatgpt", "reddit", rd.SKIPPED, error="Couldn't fetch Reddit: no creds"),
        rd.Entry("old topic", "hackernews", rd.UPDATED, T1, 90),
    ]
    manifest = rd.manifest_json(first, NOW1)
    assert manifest["refreshed_at"] == "2026-10-01T03:00:00+00:00"
    assert manifest["status"] == "ok"
    assert manifest["sources"]["wikipedia"] == {
        "status": "ok",
        "updated": 1,
        "unchanged": 0,
        "failed": 0,
        "skipped": 0,
    }
    assert manifest["sources"]["reddit"]["status"] == rd.SKIPPED
    assert manifest["topics"]["chatgpt"] == {
        "query": "chatgpt",
        "sources": {
            "wikipedia": {"status": "updated", "fetched_at": T1, "points": 800},
            "reddit": {"status": "skipped", "error": "Couldn't fetch Reddit: no creds"},
        },
    }
    previous = json.loads(rd.manifest_text(manifest))
    second = rd.manifest_json(
        [rd.Entry("chatgpt", "wikipedia", rd.FAILED, T1, 800, "Couldn't fetch Wikipedia: 503")],
        NOW2,
        previous,
        topics=["chatgpt"],
    )
    assert set(second["topics"]) == {"chatgpt"}  # dropped from the watchlist
    assert second["topics"]["chatgpt"]["sources"]["reddit"]["status"] == "skipped"  # kept
    assert second["sources"]["wikipedia"]["status"] == rd.FAILED
    assert second["status"] == rd.FAILED


def test_manifest_text_is_stable() -> None:
    entries = [
        rd.Entry("b", "hackernews", rd.UPDATED, T1, 3),
        rd.Entry("a", "wikipedia", rd.UNCHANGED, T1, 2),
    ]
    text = rd.manifest_text(rd.manifest_json(entries, NOW1))
    assert text == rd.manifest_text(rd.manifest_json(list(reversed(entries)), NOW1))
    assert text.endswith("}\n") and '": ' not in text


def test_summary_markdown_table() -> None:
    entries = [
        rd.Entry("chatgpt", "wikipedia", rd.UPDATED, T1, 800),
        rd.Entry("chatgpt", "hackernews", rd.UNCHANGED, T1, 90),
        rd.Entry("chatgpt", "reddit", rd.SKIPPED, error="Couldn't fetch Reddit: no creds"),
        rd.Entry("black friday", "wikipedia", rd.FAILED, error="Couldn't fetch Wikipedia: 503"),
    ]
    text = rd.summary_markdown(entries, NOW1)
    assert "| Topic | Wikipedia | Hacker News | Reddit |" in text
    assert "| chatgpt | ✅ ok | ⚪ unchanged | ⏭️ skipped |" in text
    assert "| black friday | ❌ failed | — | — |" in text
    assert "1 updated, 1 unchanged, 1 failed, 1 skipped" in text
    assert "- black friday / Wikipedia: Couldn't fetch Wikipedia: 503" in text


def test_main_writes_manifest_and_step_summary(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg["watchlist"] = [{"topic": "rust", "wikipedia": True, "hackernews": True}]
    monkeypatch.setattr(rd, "get_config", lambda: cfg)
    wiki = FakeAdapter(
        "wikipedia",
        {"rust": make_series("wikipedia", "rust", "2026-09-01", [1, 2], article="Rust")},
    )
    hn = FakeAdapter("hackernews", {"rust": HttpError("timeout")})
    monkeypatch.setattr(rd, "ADAPTERS", factories(wikipedia=wiki, hackernews=hn))
    step = tmp_path / "step.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step))
    assert rd.main([], now=NOW1) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["refreshed_at"] == "2026-10-01T03:00:00+00:00"
    assert manifest["status"] == "partial"
    assert (
        manifest["topics"]["rust"]["sources"]["hackernews"]["error"]
        == "Couldn't fetch Hacker News: timeout"
    )
    assert "| rust | ✅ ok | ❌ failed |" in step.read_text()

    # A second run with identical data: snapshots untouched, manifest differs only in time.
    hn.data["rust"] = HttpError("timeout")
    snap = snapshot_path("wikipedia", "rust", tmp_path / "samples")
    before = snap.read_bytes()
    assert rd.main([], now=NOW2) == 0
    assert snap.read_bytes() == before
    again = json.loads((tmp_path / "manifest.json").read_text())
    assert again["topics"]["rust"]["sources"]["wikipedia"]["status"] == "unchanged"
    assert again["refreshed_at"] == "2026-10-02T03:00:00+00:00"


def test_main_all_failed_keeps_manifest(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg["watchlist"] = [{"topic": "rust", "wikipedia": True}]
    monkeypatch.setattr(rd, "get_config", lambda: cfg)
    monkeypatch.setattr(
        rd, "ADAPTERS", factories(wikipedia=FakeAdapter("wikipedia", {"rust": HttpError("down")}))
    )
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert rd.main([], now=NOW1) == 1
    assert not (tmp_path / "manifest.json").exists()


def test_main_rejects_unknown_topics(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rd, "get_config", lambda: cfg)
    with pytest.raises(SystemExit):
        rd.main(["--topics", "not on the list"])
