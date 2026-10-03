"""Tests for signalcheck.http: User-Agent, timeouts, retry/backoff (HTTP mocked)."""

from __future__ import annotations

import email.utils
from typing import Any

import numpy as np
import pytest
import requests
import responses

from signalcheck import http
from signalcheck.config import Config

URL = "https://api.example.test/data"


class SleepRecorder:
    """Stands in for time.sleep and records requested delays."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


@pytest.fixture
def sleep() -> SleepRecorder:
    return SleepRecorder()


def _call(
    cfg: Config, rng: np.random.Generator, sleep: SleepRecorder, **kwargs: Any
) -> requests.Response:
    return http.request(
        "GET", URL, cfg=cfg, rng=rng, sleep=sleep, session=http.build_session(), **kwargs
    )


def test_user_agent_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNALCHECK_REPO_URL", "https://github.com/me/signal-check")
    monkeypatch.setenv("SIGNALCHECK_CONTACT_EMAIL", "me@example.com")
    assert (
        http.user_agent()
        == "SignalCheck/0.1.0 (+https://github.com/me/signal-check; me@example.com)"
    )
    monkeypatch.setenv("SIGNALCHECK_CONTACT_EMAIL", "")
    assert http.user_agent() == "SignalCheck/0.1.0 (+https://github.com/me/signal-check)"


def test_shared_session_is_reused() -> None:
    assert http.get_session() is http.get_session()
    assert str(http.get_session().headers["User-Agent"]).startswith("SignalCheck/")


@responses.activate
def test_success_sends_ua_timeouts_and_params(
    cfg: Config, rng: np.random.Generator, sleep: SleepRecorder
) -> None:
    responses.get(URL, json={"ok": True})
    resp = _call(cfg, rng, sleep, params={"q": "perplexity ai"})
    assert resp.json() == {"ok": True}
    call = responses.calls[0]
    assert call.request.headers["User-Agent"].startswith("SignalCheck/")
    assert call.request.req_kwargs["timeout"] == (5.0, 20.0)  # type: ignore[attr-defined]
    assert "q=perplexity+ai" in str(call.request.url)
    assert sleep.delays == []


@responses.activate
def test_retries_5xx_then_succeeds(
    cfg: Config, rng: np.random.Generator, sleep: SleepRecorder
) -> None:
    responses.get(URL, status=503)
    responses.get(URL, status=502)
    responses.get(URL, json={"ok": True})
    assert _call(cfg, rng, sleep).status_code == 200
    assert len(responses.calls) == 3
    # Equal jitter: attempt k waits in [cap/2, cap] with cap = base * 2**k.
    assert 0.5 <= sleep.delays[0] <= 1.0
    assert 1.0 <= sleep.delays[1] <= 2.0


@responses.activate
def test_gives_up_after_max_retries(
    cfg: Config, rng: np.random.Generator, sleep: SleepRecorder
) -> None:
    responses.get(URL, status=500)
    with pytest.raises(http.HttpError, match="HTTP 500 after 5 attempt") as exc_info:
        _call(cfg, rng, sleep)
    assert exc_info.value.status == 500
    assert len(responses.calls) == cfg["http"]["max_retries"] + 1
    assert len(sleep.delays) == cfg["http"]["max_retries"]


@responses.activate
def test_retry_count_from_config(
    cfg: Config, rng: np.random.Generator, sleep: SleepRecorder
) -> None:
    cfg["http"]["max_retries"] = 1
    responses.get(URL, status=429)
    with pytest.raises(http.HttpError):
        _call(cfg, rng, sleep)
    assert len(responses.calls) == 2


@responses.activate
def test_non_retryable_status_fails_fast(
    cfg: Config, rng: np.random.Generator, sleep: SleepRecorder
) -> None:
    responses.get(URL, status=404)
    with pytest.raises(http.HttpError, match="HTTP 404") as exc_info:
        _call(cfg, rng, sleep)
    assert exc_info.value.status == 404
    assert len(responses.calls) == 1 and sleep.delays == []


@responses.activate
def test_retry_after_seconds_is_honoured(
    cfg: Config, rng: np.random.Generator, sleep: SleepRecorder
) -> None:
    responses.get(URL, status=429, headers={"Retry-After": "7"})
    responses.get(URL, json={})
    _call(cfg, rng, sleep)
    assert sleep.delays == [7.0]


@responses.activate
def test_retry_after_too_long_gives_up(
    cfg: Config, rng: np.random.Generator, sleep: SleepRecorder
) -> None:
    responses.get(URL, status=429, headers={"Retry-After": "3600"})
    with pytest.raises(http.HttpError, match="retry after 3600s"):
        _call(cfg, rng, sleep)
    assert len(responses.calls) == 1 and sleep.delays == []


def test_parse_retry_after_http_date() -> None:
    now = 1_790_000_000.0
    header = email.utils.formatdate(now + 12, usegmt=True)
    assert http.parse_retry_after(header, now=now) == pytest.approx(12)
    assert http.parse_retry_after(email.utils.formatdate(now - 5, usegmt=True), now=now) == 0
    assert http.parse_retry_after(None) is None
    assert http.parse_retry_after("soon") is None


@responses.activate
def test_connection_errors_are_retried(
    cfg: Config, rng: np.random.Generator, sleep: SleepRecorder
) -> None:
    responses.get(URL, body=requests.ConnectionError("boom"))
    responses.get(URL, body=requests.Timeout("slow"))
    responses.get(URL, json={"ok": 1})
    assert _call(cfg, rng, sleep).json() == {"ok": 1}
    assert len(sleep.delays) == 2


@responses.activate
def test_timeouts_exhausted(cfg: Config, rng: np.random.Generator, sleep: SleepRecorder) -> None:
    cfg["http"]["max_retries"] = 2
    responses.get(URL, body=requests.Timeout("slow"))
    with pytest.raises(http.HttpError, match="timed out after 3 attempt"):
        _call(cfg, rng, sleep)


def test_backoff_is_capped_and_seeded(cfg: Config) -> None:
    a = [http.backoff_delay(k, cfg, np.random.default_rng(1)) for k in range(10)]
    b = [http.backoff_delay(k, cfg, np.random.default_rng(1)) for k in range(10)]
    assert a == b
    cap = cfg["http"]["backoff_max_s"]
    assert all(cap / 2 <= d <= cap for d in a[6:])


@responses.activate
def test_get_json(cfg: Config, rng: np.random.Generator, sleep: SleepRecorder) -> None:
    responses.get(URL, json={"n": 3})
    responses.get(URL, body="<html>")
    kwargs = {"cfg": cfg, "rng": rng, "sleep": sleep, "session": http.build_session()}
    assert http.get_json(URL, **kwargs) == {"n": 3}
    with pytest.raises(http.HttpError, match="not valid JSON"):
        http.get_json(URL, **kwargs)
