"""Pyodide worker glue: the XHR transport, browser cache boot and JSON entry points.

No browser is needed: a fake ``XMLHttpRequest`` stands in for ``js.XMLHttpRequest``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from signalcheck import http
from signalcheck.adapters.hackernews import HackerNewsAdapter
from signalcheck.cache import get_cache, install_cache
from signalcheck.web import worker

Handler = Callable[[str, str, dict[str, str]], tuple[int, dict[str, str], str]]


class FakeXhr:
    """Records what a page script would do with a synchronous XMLHttpRequest."""

    def __init__(self, handler: Handler, log: list[FakeXhr], error: Exception | None = None):
        self.handler = handler
        self.error = error
        self.headers: dict[str, str] = {}
        self.timeout = 0
        self.status = 0
        self.statusText = ""
        self.responseText = ""
        self._response_headers = ""
        log.append(self)

    def open(self, method: str, url: str, is_async: bool) -> None:
        assert is_async is False
        self.method, self.url = method, url

    def setRequestHeader(self, name: str, value: str) -> None:
        self.headers[name] = value

    def send(self, body: Any = None) -> None:
        if self.error is not None:
            raise self.error
        self.status, headers, self.responseText = self.handler(self.method, self.url, self.headers)
        self.statusText = "OK" if self.status == 200 else "Error"
        self._response_headers = "".join(f"{k}: {v}\r\n" for k, v in headers.items())

    def getAllResponseHeaders(self) -> str:
        return self._response_headers


def factory(handler: Handler, error: Exception | None = None) -> tuple[Any, list[FakeXhr]]:
    log: list[FakeXhr] = []
    return (lambda: FakeXhr(handler, log, error)), log


def ok(body: Any, headers: dict[str, str] | None = None) -> Handler:
    sent = headers or {"Content-Type": "application/json"}
    return lambda _m, _u, _h: (200, sent, json.dumps(body))


@pytest.fixture(autouse=True)
def reset_installed() -> Iterator[None]:
    """Never leak an installed session or cache into other tests."""
    yield
    http.install_session(None)
    install_cache(None)


# --- URL / header rules ---------------------------------------------------------


def test_browser_url_adds_origin_only_for_the_action_api() -> None:
    api = "https://en.wikipedia.org/w/api.php?action=query&format=json"
    assert parse_qs(urlsplit(worker.browser_url(api)).query)["origin"] == ["*"]
    already = api + "&origin=*"
    assert worker.browser_url(already) == already
    rest = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/x"
    assert worker.browser_url(rest) == rest
    hn = "https://hn.algolia.com/api/v1/search_by_date?query=x"
    assert worker.browser_url(hn) == hn


def test_browser_headers_drop_forbidden_and_identify_on_wikipedia_only() -> None:
    sent = {"User-Agent": "x", "Accept-Encoding": "gzip", "Connection": "keep-alive", "X-A": "1"}
    wiki = worker.browser_headers("https://en.wikipedia.org/w/api.php", sent, "SC/1")
    assert wiki == {"X-A": "1", "Api-User-Agent": "SC/1"}
    # The pageviews REST preflight rejects custom headers: send a simple request.
    rest = worker.browser_headers("https://wikimedia.org/api/rest_v1/x", sent, "SC/1")
    assert rest == {"X-A": "1"}
    assert worker.browser_headers("https://hn.algolia.com/api/v1/x", {}, "SC/1") == {}
    assert worker.browser_headers("https://evilwikipedia.org/x", {}, "SC/1") == {}


def test_parse_response_headers_is_case_insensitive() -> None:
    headers = worker.parse_response_headers("content-type: application/json\r\nRetry-After: 5\r\n")
    assert headers["Content-Type"] == "application/json" and headers["retry-after"] == "5"


# --- XhrSession -------------------------------------------------------------------


def test_xhr_session_returns_a_real_response() -> None:
    xhr, log = factory(ok({"a": 1}, {"Content-Type": "application/json", "Retry-After": "3"}))
    hosts: list[str] = []
    session = worker.XhrSession(xhr, on_request=hosts.append, api_user_agent="SC/1")
    resp = session.get(
        "https://en.wikipedia.org/w/api.php", params={"q": "a b"}, timeout=(3.05, 20)
    )
    assert isinstance(resp, requests.Response)
    assert resp.status_code == 200 and resp.json() == {"a": 1}
    assert resp.headers["retry-after"] == "3"
    (call,) = log
    assert call.method == "GET"
    assert parse_qs(urlsplit(call.url).query) == {"q": ["a b"], "origin": ["*"]}
    assert call.timeout == 23050
    assert call.headers == {"Api-User-Agent": "SC/1"}
    assert hosts == ["en.wikipedia.org"]


def test_xhr_session_sends_no_headers_to_hacker_news() -> None:
    xhr, log = factory(ok({}))
    worker.XhrSession(xhr).get("https://hn.algolia.com/api/v1/search_by_date", timeout=5)
    assert log[0].headers == {} and log[0].timeout == 5000


def test_status_zero_is_a_connection_error() -> None:
    xhr, _ = factory(lambda _m, _u, _h: (0, {}, ""))
    with pytest.raises(requests.ConnectionError, match="CORS"):
        worker.XhrSession(xhr).get("https://hn.algolia.com/api/v1/x")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("NetworkError: Failed to execute 'send'", requests.ConnectionError),
        ("TimeoutError: timeout", requests.Timeout),
    ],
)
def test_send_exceptions_map_to_requests_errors(message: str, expected: type[Exception]) -> None:
    xhr, _ = factory(ok({}), RuntimeError(message))
    with pytest.raises(expected):
        worker.XhrSession(xhr).get("https://hn.algolia.com/api/v1/x")


def test_error_status_flows_through_signalcheck_http(cfg: dict[str, Any]) -> None:
    xhr, _ = factory(lambda _m, _u, _h: (404, {}, "{}"))
    with pytest.raises(http.HttpError) as err:
        http.request("GET", "https://hn.algolia.com/x", session=worker.XhrSession(xhr), cfg=cfg)
    assert err.value.status == 404


def test_hacker_news_adapter_runs_over_the_xhr_transport(adapter_deps: dict[str, Any]) -> None:
    def handler(_m: str, url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], str]:
        assert headers == {}
        params = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        lower = int(params["numericFilters"].split(",")[0].split(">=")[1])
        day = datetime.fromtimestamp(lower, tz=UTC).day
        return 200, {}, json.dumps({"nbHits": day, "exhaustiveNbHits": True, "hits": []})

    xhr, log = factory(handler)
    adapter_deps["session"] = worker.XhrSession(xhr)
    series = HackerNewsAdapter(**adapter_deps).fetch("rust", {"days": 10})
    assert len(log) == 10 and len(series.points) == 10
    assert series.points["value"].iloc[0] == series.points["ts"].iloc[0].day


# --- boot / browser cache ---------------------------------------------------------


def test_boot_installs_session_and_persisting_cache() -> None:
    writes: list[tuple[str, float | None, str]] = []
    rows = [["k1", 4_102_444_800.0, json.dumps({"v": 1})], ["bad"], ["k2", 1.0, "{}"]]
    xhr, _ = factory(ok({}))
    status = worker.boot(json.dumps(rows), lambda *a: writes.append(a), xhr_factory=xhr)
    assert status == {"cached_entries": 1, "ttl_hours": 6.0}
    assert isinstance(http.get_session(), worker.XhrSession)
    cache = get_cache()
    assert cache.get("k1") == {"v": 1} and cache.get("k2") is None
    cache.set("k3", {"days": [1, 2]})
    ((key, expires_at, value),) = writes
    assert key == "k3" and expires_at is not None and json.loads(value) == {"days": [1, 2]}


# --- JSON entry points --------------------------------------------------------------


def test_run_topic_json_degrades_per_source() -> None:
    xhr, _ = factory(lambda _m, _u, _h: (404, {}, "{}"))
    worker.boot(None, None, xhr_factory=xhr)
    out = json.loads(worker.run_topic_json("zzqx nothing here", 30))
    cards = {c["source"]: c for c in out["cards"]}
    assert out["kind"] == "topic" and out["sample"] is False
    for source in ("wikipedia", "hackernews"):
        assert cards[source]["message"].startswith("Couldn't fetch")
        assert cards[source]["chart"] is None
    for source in ("reddit", "x"):
        assert cards[source]["disabled"] is True
        assert "needs server-side credentials" in cards[source]["message"]
    assert "google_trends" in cards and cards["google_trends"]["disabled"] is True


def test_topic_sources_add_trends_only_with_a_snapshot(cfg: Any, tmp_path: Any) -> None:
    cfg["samples"]["dir"] = str(tmp_path)
    assert worker.topic_sources("chatgpt", cfg) == ["wikipedia", "hackernews"]
    (tmp_path / "google_trends").mkdir()
    (tmp_path / "google_trends" / "chatgpt.json").write_text("{}")
    assert worker.topic_sources("chatgpt", cfg) == ["wikipedia", "hackernews", "google_trends"]


def test_run_sample_json_is_strict_json() -> None:
    xhr, log = factory(lambda _m, _u, _h: (404, {}, "{}"))
    worker.boot(None, None, xhr_factory=xhr)
    text = worker.run_sample_json("chatgpt")
    out = json.loads(text)
    assert "NaN" not in text and out["sample"] is True
    assert not log  # samples come from bundled snapshots: no network
    assert any(c["chart"] for c in out["cards"])


def test_run_csv_json_asks_for_a_column_then_analyses() -> None:
    rows = ["date,a,b"] + [f"2026-01-{d:02d},{d},{100 - d}" for d in range(1, 29)]
    data = ("\n".join(rows) + "\n").encode()
    first = json.loads(worker.run_csv_json(data, "x.csv"))
    assert first == {"schema": 1, "kind": "columns", "columns": ["a", "b"]}
    second = json.loads(worker.run_csv_json(data, "x.csv", "b", True))
    assert second["kind"] == "csv"
    (card,) = second["cards"]
    assert card["badge"]["key"].startswith(("TREND", "NO_CHANGE", "INCONCLUSIVE", "FLUKE"))
    assert card["chart"]["dropped_partial"] is not None
