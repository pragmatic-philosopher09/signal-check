"""Shared HTTP session: identifying User-Agent, timeouts, bounded retries.

Every external call goes through :func:`request`, which applies the rules from
COPILOT_BRIEF.md section 1: connect/read timeouts, and retries with exponential
backoff plus jitter on HTTP 429 / 5xx and connection errors, honouring
``Retry-After``. Failures surface as :class:`HttpError` with a short,
user-presentable reason so adapters can degrade to a per-card message.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import Any

import numpy as np
import requests

from signalcheck.config import APP_NAME, APP_VERSION, Config, get_config, get_secret

DEFAULT_REPO_URL = "https://github.com/pragmatic-philosopher09/signal-check"


class HttpError(RuntimeError):
    """An HTTP call failed for good (non-retryable status or retries exhausted)."""

    def __init__(self, reason: str, status: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


def user_agent() -> str:
    """``SignalCheck/<version> (+<repo-url>; <contact-email>)`` from env/secrets."""
    repo = get_secret("SIGNALCHECK_REPO_URL", DEFAULT_REPO_URL)
    email = get_secret("SIGNALCHECK_CONTACT_EMAIL")
    contact = f"+{repo}; {email}" if email else f"+{repo}"
    return f"{APP_NAME.replace(' ', '')}/{APP_VERSION} ({contact})"


def build_session() -> requests.Session:
    """A new session carrying the identifying User-Agent."""
    session = requests.Session()
    session.headers["User-Agent"] = user_agent()
    return session


_installed: list[requests.Session] = []


def install_session(session: requests.Session | None) -> None:
    """Route every request through ``session`` (``None`` restores the default).

    The transport is injectable so the same adapters run unchanged in the browser
    build, where the web worker installs a session that sends requests with the
    browser's XMLHttpRequest instead of sockets.
    """
    _installed.clear()
    if session is not None:
        _installed.append(session)


@lru_cache(maxsize=1)
def _shared_session() -> requests.Session:
    return build_session()


def get_session() -> requests.Session:
    """The installed session, else the process-wide shared one (pooling, one User-Agent)."""
    if _installed:
        return _installed[0]
    return _shared_session()


def is_retryable(status: int) -> bool:
    """429 (rate limited) and 5xx (server trouble) are worth retrying."""
    return status == 429 or 500 <= status <= 599


def parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header (delta-seconds or HTTP-date)."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    current = time.time() if now is None else now
    return max(when.timestamp() - current, 0.0)


def backoff_delay(attempt: int, cfg: Config, rng: np.random.Generator) -> float:
    """Exponential backoff with equal jitter for retry number ``attempt`` (0-based).

    ``cap = min(backoff_max_s, backoff_base_s * 2**attempt)``; the delay is drawn
    uniformly from ``[cap/2, cap]`` so concurrent clients spread out but still back
    off.
    """
    http = cfg["http"]
    cap = min(float(http["backoff_max_s"]), float(http["backoff_base_s"]) * 2**attempt)
    return float(cap / 2 + rng.uniform(0.0, cap / 2))


def request(
    method: str,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    session: requests.Session | None = None,
    cfg: Config | None = None,
    rng: np.random.Generator | None = None,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> requests.Response:
    """Send a request with timeouts and bounded retries; return a 2xx/3xx response.

    Retries up to ``http.max_retries`` times on 429/5xx, timeouts and connection
    errors. A ``Retry-After`` header sets the minimum wait; if it asks for longer
    than ``http.backoff_max_s`` the call gives up immediately rather than blocking
    the app. Raises :class:`HttpError` on any final failure.
    """
    cfg = cfg if cfg is not None else get_config()
    http = cfg["http"]
    session = session if session is not None else get_session()
    rng = rng if rng is not None else np.random.default_rng()
    timeout = (float(http["connect_timeout_s"]), float(http["read_timeout_s"]))
    max_retries = int(http["max_retries"])

    for attempt in range(max_retries + 1):
        last_try = attempt == max_retries
        try:
            response = session.request(
                method, url, params=params, headers=headers, timeout=timeout, **kwargs
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            if last_try:
                kind = "timed out" if isinstance(exc, requests.Timeout) else "connection failed"
                raise HttpError(f"request {kind} after {attempt + 1} attempt(s)") from exc
            sleep(backoff_delay(attempt, cfg, rng))
            continue

        status = response.status_code
        if status < 400:
            return response
        if not is_retryable(status):
            raise HttpError(f"HTTP {status} {response.reason or ''}".strip(), status)
        if last_try:
            raise HttpError(f"HTTP {status} after {attempt + 1} attempt(s)", status)
        delay = backoff_delay(attempt, cfg, rng)
        retry_after = parse_retry_after(response.headers.get("Retry-After"))
        if retry_after is not None:
            if retry_after > float(http["backoff_max_s"]):
                raise HttpError(
                    f"rate limited; server asks to retry after {retry_after:.0f}s", status
                )
            delay = max(delay, retry_after)
        sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def get_json(url: str, **kwargs: Any) -> Any:
    """GET ``url`` via :func:`request` and decode JSON (bad JSON -> :class:`HttpError`)."""
    response = request("GET", url, **kwargs)
    try:
        return response.json()
    except ValueError as exc:
        raise HttpError("response was not valid JSON", response.status_code) from exc
