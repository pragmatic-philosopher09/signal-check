"""Tests for signalcheck.cache: key normalisation, TTL and fetch-through (HTTP mocked)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import responses

from signalcheck import config as config_mod
from signalcheck import http
from signalcheck.cache import (
    Cache,
    MemoryStore,
    get_cache,
    install_cache,
    make_key,
    normalise_params,
    normalise_query,
)
from signalcheck.config import Config
from tests.conftest import build_series

URL = "https://api.example.test/counts"


@pytest.fixture
def cache(tmp_path: Path, cfg: Config) -> Iterator[Cache]:
    cfg["cache"]["dir"] = str(tmp_path / "cache")
    c = Cache.from_config(cfg)
    yield c
    c.close()


def test_query_and_params_are_normalised() -> None:
    assert normalise_query("  Perplexity   AI ") == "perplexity ai"
    a = make_key("reddit", "Perplexity AI", {"b": 2, "a": "x ", "skip": None})
    b = make_key("reddit", " perplexity  ai", {"a": "x", "b": 2})
    assert a == b
    assert a.startswith("reddit:")
    assert make_key("hackernews", "perplexity ai", {"a": "x", "b": 2}) != a
    assert make_key("reddit", "perplexity ai", {"a": "x", "b": 3}) != a
    assert normalise_params({"d": date(2026, 10, 3), "t": (1, 2), "s": {"b", "a"}}) == (
        '{"d": "2026-10-03", "s": ["a", "b"], "t": [1, 2]}'
    )


def test_ttl_from_config(cache: Cache, cfg: Config) -> None:
    assert cache.ttl_seconds == 6 * 3600
    assert cache.directory.is_absolute()


def test_relative_dir_resolves_under_repo_root(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_PATH", tmp_path / "config.yaml")
    cfg["cache"]["dir"] = ".cache/signalcheck"
    c = Cache.from_config(cfg)
    try:
        assert c.directory == tmp_path / ".cache/signalcheck"
    finally:
        c.close()


@responses.activate
def test_get_or_fetch_hits_network_once(cache: Cache, cfg: Config) -> None:
    responses.get(URL, json={"nbHits": 42})

    def fetch() -> int:
        data = http.get_json(
            URL, cfg=cfg, session=http.build_session(), rng=np.random.default_rng(0)
        )
        return int(data["nbHits"])

    assert cache.get_or_fetch("hackernews", "Perplexity", {"day": "2026-10-01"}, fetch) == 42
    assert cache.get_or_fetch("hackernews", "perplexity ", {"day": "2026-10-01"}, fetch) == 42
    assert len(responses.calls) == 1


def test_failures_are_not_cached(cache: Cache) -> None:
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise http.HttpError("HTTP 503")
        return "ok"

    with pytest.raises(http.HttpError):
        cache.get_or_fetch("wikipedia", "q", {}, flaky)
    assert cache.get_or_fetch("wikipedia", "q", {}, flaky) == "ok"
    assert len(calls) == 2


def test_falsy_values_are_cached(cache: Cache) -> None:
    calls: list[int] = []

    def zero() -> int:
        calls.append(1)
        return 0

    assert cache.get_or_fetch("hackernews", "q", None, zero) == 0
    assert cache.get_or_fetch("hackernews", "q", None, zero) == 0
    assert len(calls) == 1


def test_entries_expire_after_ttl(cache: Cache, monkeypatch: pytest.MonkeyPatch) -> None:
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    cache.set("k", "v")
    _, expire_at = cache._cache.get("k", expire_time=True)
    assert expire_at == pytest.approx(now + 6 * 3600)
    monkeypatch.setattr(time, "time", lambda: now + 6 * 3600 - 1)
    assert cache.get("k") == "v"
    monkeypatch.setattr(time, "time", lambda: now + 6 * 3600 + 1)
    assert cache.get("k") is None


def test_series_round_trip(cache: Cache) -> None:
    series = build_series(np.arange(5.0), scale="count")
    out = cache.get_or_fetch("csv", "q", {}, lambda: series)
    again = cache.get_or_fetch("csv", "q", {}, lambda: None)
    assert out is series
    assert again is not None
    assert again.points.equals(series.points) and again.meta == series.meta


def test_memory_store_ttl_persist_hook_and_reload() -> None:
    clock = [1000.0]
    stored: list[tuple[str, float | None, object]] = []
    mem = Cache.in_memory(60, now=lambda: clock[0], on_store=lambda *e: stored.append(e))
    calls: list[int] = []

    def fetch() -> dict[str, int]:
        calls.append(1)
        return {"n": len(calls)}

    assert mem.get_or_fetch("hackernews", "Rust ", {"a": 1}, fetch) == {"n": 1}
    assert mem.get_or_fetch("hackernews", "rust", {"a": 1}, fetch) == {"n": 1}
    assert len(calls) == 1
    key = make_key("hackernews", "rust", {"a": 1})
    assert stored == [(key, 1060.0, {"n": 1})]

    fresh = MemoryStore(now=lambda: clock[0])
    assert fresh.load([(key, 1060.0, {"n": 1}), ("old", 999.0, 1), ("forever", None, 2)]) == 2
    assert fresh.get(key) == {"n": 1} and fresh.get("old") is None and fresh.get("forever") == 2
    clock[0] = 1061.0
    assert fresh.get(key, "gone") == "gone"
    assert mem.get_or_fetch("hackernews", "rust", {"a": 1}, fetch) == {"n": 2}


def test_installed_cache_replaces_the_disk_cache() -> None:
    mem = Cache.in_memory(60)
    install_cache(mem)
    try:
        assert get_cache() is mem
    finally:
        install_cache(None)
    assert get_cache() is not mem
