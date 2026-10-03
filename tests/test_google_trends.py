"""Tests for the Google Trends adapter and scripts/refresh_samples.py (no network)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import responses

from scripts import refresh_samples
from signalcheck.adapters.base import AdapterError, fetch_safely
from signalcheck.adapters.csv_upload import TRENDS_CAVEATS
from signalcheck.adapters.google_trends import (
    UNKNOWN_FETCH_TIME_CAVEAT,
    GoogleTrendsAdapter,
    PytrendsFetcher,
    TrendsRequest,
    granularity,
)
from signalcheck.adapters.hackernews import SEARCH_URL
from signalcheck.engine import analyse
from signalcheck.engine.preprocess import preprocess
from signalcheck.snapshots import read_snapshot, write_snapshot_to

TRENDS_CSV = (
    "Category: All categories\n\n"
    "Day,perplexity ai: (Worldwide)\n"
    + "".join(f"2026-09-{d:02d},{40 + d % 5}\n" for d in range(1, 31))
    + "2026-10-01,<1\n2026-10-02,55\n2026-10-03,20\n"
)


class FakeFetcher:
    def __init__(self, rows: list[tuple[date, float | str]] | Exception) -> None:
        self.rows = rows
        self.requests: list[TrendsRequest] = []

    def __call__(self, request: TrendsRequest) -> list[tuple[date, float | str]]:
        self.requests.append(request)
        if isinstance(self.rows, Exception):
            raise self.rows
        return self.rows


def _adapter(deps: dict[str, Any], fetcher: Any, snap: Path) -> GoogleTrendsAdapter:
    return GoogleTrendsAdapter(fetcher=fetcher, snapshot_dir=snap, **deps)


def _rows(n: int = 30) -> list[tuple[date, float | str]]:
    start = date(2026, 10, 3) - timedelta(days=n - 1)
    return [(start + timedelta(days=i), "<1" if i == 3 else float(50 + i % 4)) for i in range(n)]


def test_live_fetch(adapter_deps: dict[str, Any], tmp_path: Path) -> None:
    fetcher = FakeFetcher(_rows())
    adapter = _adapter(adapter_deps, fetcher, tmp_path / "snap")
    series = adapter.fetch("Perplexity AI", {"days": 30, "geo": "US"})
    assert fetcher.requests == [
        TrendsRequest("Perplexity AI", date(2026, 9, 4), date(2026, 10, 3), "US")
    ]
    assert series.scale == "relative_0_100" and series.freq == "D"
    assert series.points.loc[3, "value"] == 0.5 and series.points.loc[3, "imputed"]
    assert series.meta["partial_rule"] == "fetched_at"
    assert all(c in series.caveats for c in TRENDS_CAVEATS)
    assert any("'<1'" in c for c in series.caveats)
    # Cached: a second fetch does not call the live fetcher again.
    adapter.fetch("perplexity ai", {"days": 30, "geo": "US"})
    assert len(fetcher.requests) == 1


def test_granularity(cfg: dict[str, Any]) -> None:
    assert [granularity(d, cfg) for d in (7, 90, 91, 1825, 1826)] == ["D", "D", "W", "W", "M"]


def test_live_failure_falls_back_to_json_snapshot(
    adapter_deps: dict[str, Any], tmp_path: Path
) -> None:
    snap = tmp_path / "snap"
    live = _adapter(adapter_deps, FakeFetcher(_rows()), snap)
    write_snapshot_to(live.fetch_live("perplexity ai", {"days": 30}), snap / "perplexity-ai.json")

    later = datetime(2026, 10, 25, tzinfo=UTC)
    adapter_deps["now"] = lambda: later
    failing = _adapter(adapter_deps, FakeFetcher(RuntimeError("429 Too Many Requests")), snap)
    series = failing.fetch("Perplexity AI", {})
    assert series.meta["snapshot"] is True and series.meta["snapshot_age_days"] == 21
    text = " | ".join(series.caveats)
    assert "Live Google Trends unavailable: live fetch failed (RuntimeError)" in text
    assert "Snapshot is 21 days old" in text
    assert series.meta["fetched_at"].startswith("2026-10-03T12:00")
    # Partial last day (2026-10-03, fetched at noon) is still dropped by preprocessing.
    prepared = preprocess(series, adapter_deps["cfg"])
    assert prepared.series.meta["dropped_partial"]["ts"] == "2026-10-03"


def test_csv_snapshot_with_and_without_sidecar(
    adapter_deps: dict[str, Any], tmp_path: Path
) -> None:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "perplexity-ai.csv").write_text(TRENDS_CSV, encoding="utf-8")
    adapter_deps["cfg"]["adapters"]["google_trends"]["live_enabled"] = False
    adapter = _adapter(adapter_deps, FakeFetcher([]), snap)

    series = adapter.fetch("perplexity ai", {})
    assert UNKNOWN_FETCH_TIME_CAVEAT in series.caveats
    assert series.meta["last_period_incomplete"] is True
    assert "Live Google Trends unavailable: live fetching is switched off." in series.caveats
    assert series.points["imputed"].sum() == 1

    (snap / "perplexity-ai.meta.json").write_text(
        json.dumps({"fetched_at": "2026-10-03T08:00:00+00:00"}), encoding="utf-8"
    )
    series = adapter.fetch("perplexity ai", {})
    assert UNKNOWN_FETCH_TIME_CAVEAT not in series.caveats
    assert series.meta["partial_rule"] == "fetched_at"
    verdict = analyse(series, adapter_deps["cfg"])
    assert verdict.label


def test_no_live_and_no_snapshot_is_graceful(adapter_deps: dict[str, Any], tmp_path: Path) -> None:
    adapter = _adapter(adapter_deps, FakeFetcher(AdapterError("blocked")), tmp_path)
    result = fetch_safely(adapter, "unknown topic", {})
    assert result.message == (
        "Couldn't fetch Google Trends: blocked; no Google Trends snapshot for 'unknown topic'"
    )


def test_pytrends_missing_is_an_adapter_error(
    cfg: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    def fail(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.raises(AdapterError, match="pytrends not installed"):
        PytrendsFetcher(cfg)(TrendsRequest("x", date(2026, 9, 1), date(2026, 10, 1), ""))


def test_refresh_samples_writes_snapshots(adapter_deps: dict[str, Any], tmp_path: Path) -> None:
    cfg = adapter_deps["cfg"]

    @responses.activate
    def run() -> tuple[int, list[str]]:
        responses.get(SEARCH_URL, json={"nbHits": 3, "exhaustiveNbHits": True, "hits": []})
        factories = {
            "hackernews": lambda c: refresh_samples.HackerNewsAdapter(
                c,
                cache=refresh_samples.NoCache(),
                sleep=adapter_deps["sleep"],
                now=adapter_deps["now"],
            ),
            "google_trends": lambda c: GoogleTrendsAdapter(
                c, fetcher=FakeFetcher(AdapterError("blocked")), now=adapter_deps["now"]
            ),
        }
        cfg["adapters"]["hackernews"]["history_days"] = 5
        return refresh_samples.refresh(
            ["rust programming"], ["hackernews", "google_trends"], cfg, tmp_path, factories
        )

    written, errors = run()
    assert written == 1
    assert errors == ["'rust programming': Couldn't fetch Google Trends: blocked"]
    series = read_snapshot(tmp_path / "hackernews" / "rust-programming.json")
    assert series.points["value"].tolist() == [3.0] * 5
    assert series.meta["fetched_at"].startswith("2026-10-03")


def test_import_trends_csv(cfg: dict[str, Any], tmp_path: Path) -> None:
    csv_path = tmp_path / "multiTimeline.csv"
    csv_path.write_text(TRENDS_CSV, encoding="utf-8")
    out = refresh_samples.import_trends_csv(
        csv_path, "Perplexity AI", datetime(2026, 10, 3, 8, tzinfo=UTC), cfg, tmp_path
    )
    assert out == tmp_path / "google_trends" / "perplexity-ai.json"
    series = read_snapshot(out)
    assert series.source == "google_trends" and series.meta["partial_rule"] == "fetched_at"
    assert (
        refresh_samples.main(
            [
                "--import-trends-csv",
                str(csv_path),
                "--query",
                "Perplexity AI",
                "--out",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    assert (tmp_path / "cli" / "google_trends" / "perplexity-ai.json").exists()


def test_import_rejects_non_trends_csv(cfg: dict[str, Any], tmp_path: Path) -> None:
    csv_path = tmp_path / "x.csv"
    csv_path.write_text("date,value\n2026-09-01,1\n2026-09-02,2\n", encoding="utf-8")
    with pytest.raises(AdapterError, match="not a Google Trends download"):
        refresh_samples.import_trends_csv(csv_path, "x", None, cfg, tmp_path)


def test_default_sources_skip_paid_and_disabled(cfg: dict[str, Any]) -> None:
    cfg["sources"]["wikipedia"] = False
    assert "wikipedia" not in refresh_samples.default_sources(cfg)
    assert "x" not in refresh_samples.default_sources(cfg)
