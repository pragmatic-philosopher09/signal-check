"""Entry points for the Pyodide web worker (the static GitHub Pages build).

The worker (``web/src/engine/worker.ts``) loads this package into Pyodide and
calls :func:`boot` once, then :func:`run_topic_json` / :func:`run_csv_json`.
Adapters are unchanged: their HTTP goes through :class:`XhrSession`, a
``requests.Session`` that sends each request with the browser's synchronous
``XMLHttpRequest`` (allowed in workers), and their 6h cache is an in-memory
store that the worker mirrors to IndexedDB. Sources are fetched one after
another (Pyodide has no threads). Narration is template-only: no LLM key is ever
shipped to the browser.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from requests.structures import CaseInsensitiveDict

from signalcheck import http
from signalcheck.cache import Cache, MemoryStore, install_cache
from signalcheck.config import Config, get_config
from signalcheck.snapshots import samples_dir, snapshot_path
from signalcheck.ui import runner
from signalcheck.web.export import LIVE_SOURCES, csv_json, result_json, with_static_cards

log = logging.getLogger(__name__)

# Headers a page may not set on XMLHttpRequest (the browser refuses and logs an error).
FORBIDDEN_HEADERS = frozenset(
    {
        "accept-charset",
        "accept-encoding",
        "connection",
        "content-length",
        "cookie",
        "date",
        "host",
        "keep-alive",
        "origin",
        "referer",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "user-agent",
        "via",
    }
)
WIKIMEDIA_HOSTS = ("wikipedia.org", "wikimedia.org")
# Hosts whose CORS preflight allows ``Api-User-Agent``. The pageviews REST API on
# ``wikimedia.org`` answers the preflight with 405, so it gets a simple request
# (no custom headers) and is identified by the browser's Origin/Referer instead.
API_USER_AGENT_HOSTS = ("wikipedia.org",)

XhrFactory = Callable[[], Any]
OnRequest = Callable[[str], None]


def _host_in(url: str, domains: tuple[str, ...]) -> bool:
    host = urlsplit(url).hostname or ""
    return any(host == d or host.endswith("." + d) for d in domains)


def is_wikimedia(url: str) -> bool:
    """Whether ``url`` points at a Wikimedia host."""
    return _host_in(url, WIKIMEDIA_HOSTS)


def browser_url(url: str) -> str:
    """``url`` with ``origin=*`` added for the MediaWiki Action API.

    Anonymous cross-origin requests to ``api.php`` only get CORS headers when they
    say ``origin=*``; the REST pageviews API needs nothing extra.
    """
    parts = urlsplit(url)
    if not (is_wikimedia(url) and parts.path.endswith("/api.php")):
        return url
    query = parse_qsl(parts.query, keep_blank_values=True)
    if any(k == "origin" for k, _ in query):
        return url
    return urlunsplit(parts._replace(query=urlencode([*query, ("origin", "*")])))


def browser_headers(url: str, headers: Mapping[str, str], api_user_agent: str) -> dict[str, str]:
    """Headers safe to set from a page: forbidden names dropped, ``Api-User-Agent`` added.

    Browsers do not let scripts set ``User-Agent``, so Wikimedia's documented
    alternative identifies the app on ``*.wikipedia.org`` (whose preflight allows
    it). Other hosts get no extra headers, so no CORS preflight is needed.
    """
    out = {k: v for k, v in headers.items() if k.lower() not in FORBIDDEN_HEADERS}
    if _host_in(url, API_USER_AGENT_HOSTS):
        out["Api-User-Agent"] = api_user_agent
    return out


def parse_response_headers(raw: str) -> CaseInsensitiveDict[str]:
    """``getAllResponseHeaders()`` text as a case-insensitive dict."""
    headers: CaseInsensitiveDict[str] = CaseInsensitiveDict()
    for line in raw.splitlines():
        name, sep, value = line.partition(":")
        if sep and name.strip():
            headers[name.strip()] = value.strip()
    return headers


def _default_xhr() -> Any:  # pragma: no cover - only exists inside Pyodide
    import js  # type: ignore[import-not-found]

    return js.XMLHttpRequest.new()


class XhrSession(requests.Session):
    """A ``requests.Session`` whose transport is the browser's synchronous XHR.

    Returns real :class:`requests.Response` objects, so :mod:`signalcheck.http`
    (status handling, retries, ``HttpError`` reasons) works unchanged. A status of
    0 means the browser blocked or could not complete the request (network error
    or CORS) and is raised as :class:`requests.ConnectionError`.
    """

    def __init__(
        self,
        xhr_factory: XhrFactory | None = None,
        *,
        on_request: OnRequest | None = None,
        api_user_agent: str | None = None,
    ) -> None:
        super().__init__()
        self.headers.clear()
        self._xhr_factory = xhr_factory or _default_xhr
        self._on_request = on_request
        self._api_user_agent = api_user_agent or http.user_agent()

    def request(  # type: ignore[override]
        self,
        method: str,
        url: str,
        params: Any = None,
        data: Any = None,
        headers: Mapping[str, str] | None = None,
        timeout: Any = None,
        **_kwargs: Any,
    ) -> requests.Response:
        """Send one request synchronously and wrap the result as a ``requests.Response``."""
        prepared = requests.Request(
            method.upper(), url, params=params, data=data, headers=dict(headers or {})
        ).prepare()
        full_url = browser_url(str(prepared.url))
        if self._on_request is not None:
            self._on_request(urlsplit(full_url).hostname or "")
        xhr = self._xhr_factory()
        xhr.open(prepared.method, full_url, False)
        if timeout is not None:
            seconds = sum(timeout) if isinstance(timeout, tuple) else float(timeout)
            xhr.timeout = int(seconds * 1000)
        merged = {k: str(v) for k, v in {**self.headers, **prepared.headers}.items()}
        for name, value in browser_headers(full_url, merged, self._api_user_agent).items():
            xhr.setRequestHeader(name, value)
        try:
            xhr.send(prepared.body)
        except Exception as exc:
            if "timeout" in str(exc).lower():
                raise requests.Timeout(f"browser request timed out: {full_url}") from exc
            raise requests.ConnectionError("blocked by the browser (network or CORS)") from exc
        status = int(xhr.status)
        if status == 0:
            raise requests.ConnectionError("blocked by the browser (network or CORS)")
        response = requests.Response()
        response.status_code = status
        response.reason = str(xhr.statusText or "")
        response.headers = parse_response_headers(str(xhr.getAllResponseHeaders() or ""))
        response._content = str(xhr.responseText or "").encode("utf-8")
        response.encoding = "utf-8"
        response.url = full_url
        return response


PersistFn = Callable[[str, float | None, str], None]


def boot(
    entries_json: str | None = None,
    persist: PersistFn | None = None,
    *,
    xhr_factory: XhrFactory | None = None,
    on_request: OnRequest | None = None,
) -> dict[str, Any]:
    """Install the XHR transport and the browser cache; return a small status dict.

    ``entries_json`` is a JSON list of ``[key, expires_at, value_json]`` read from
    IndexedDB; ``persist(key, expires_at, value_json)`` is called on every write.
    """
    cfg = get_config()
    http.install_session(XhrSession(xhr_factory, on_request=on_request))

    def on_store(key: str, expires_at: float | None, value: Any) -> None:
        if persist is not None:
            persist(key, expires_at, json.dumps(value))

    cache = Cache.in_memory(float(cfg["cache"]["ttl_hours"]) * 3600, on_store=on_store)
    loaded = 0
    if entries_json:
        store = cache.store
        assert isinstance(store, MemoryStore)
        loaded = store.load(_decode_entries(json.loads(entries_json)))
    install_cache(cache)
    return {"cached_entries": loaded, "ttl_hours": float(cfg["cache"]["ttl_hours"])}


def _decode_entries(rows: Iterable[Any]) -> Iterable[tuple[str, float | None, Any]]:
    for row in rows:
        try:
            key, expires_at, value_json = row
            yield (
                str(key),
                (None if expires_at is None else float(expires_at)),
                json.loads(value_json),
            )
        except (TypeError, ValueError):
            continue


def topic_sources(query: str, cfg: Config) -> list[str]:
    """Live sources plus Google Trends when a committed snapshot exists for ``query``."""
    sources = list(LIVE_SOURCES)
    if snapshot_path("google_trends", query, samples_dir(cfg)).exists():
        sources.append("google_trends")
    return sources


def run_topic_json(query: str, days: int | None = None, wiki_article: str | None = None) -> str:
    """Fetch and analyse ``query`` live (Wikipedia, Hacker News) and return result JSON."""
    cfg = get_config()
    sources = topic_sources(query, cfg)
    factories = dict(runner.ADAPTER_FACTORIES)
    base = samples_dir(cfg)
    factories["google_trends"] = lambda _cfg: runner.SnapshotAdapter("google_trends", base)
    result = runner.run_topic(
        query,
        sources,
        cfg,
        days=days,
        wiki_article=wiki_article or None,
        factories=factories,
        max_workers=1,
    )
    result = with_static_cards(result, cfg, sources)
    return json.dumps(result_json(result, cfg), allow_nan=False)


def run_sample_json(query: str, wiki_article: str | None = None) -> str:
    """A sample topic from the bundled snapshots (Wikipedia live with an article override)."""
    cfg = get_config()
    sources = runner.sample_sources(query, cfg)
    result = runner.run_topic(
        query, sources, cfg, sample=True, wiki_article=wiki_article or None, max_workers=1
    )
    result = with_static_cards(result, cfg, sources)
    return json.dumps(result_json(result, cfg), allow_nan=False)


def run_csv_json(
    data: bytes,
    name: str,
    value_column: str | None = None,
    last_period_incomplete: bool = False,
) -> str:
    """Analyse an uploaded CSV (incl. Google Trends exports); may ask for a column."""
    cfg = get_config()
    result = runner.run_csv(
        bytes(data),
        name,
        cfg,
        value_column=value_column or None,
        last_period_incomplete=bool(last_period_incomplete),
    )
    return json.dumps(csv_json(result, cfg), allow_nan=False)
