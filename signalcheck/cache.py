"""Cache for external calls (6h TTL by default, see ``config.yaml``).

On a server the store is :mod:`diskcache`; in the browser build (Pyodide, where
there is no persistent disk) :func:`install_cache` swaps in an in-memory
:class:`MemoryStore` whose writes the web worker mirrors to IndexedDB.

Keys are ``source + normalised query + normalised params`` so the same logical
request hits the same entry regardless of whitespace, letter case or parameter
order. Only successful results are cached. Callers must store **aggregates only**
(counts, contributor counts, shares): never author names or post text.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, TypeVar

from signalcheck.config import Config, get_config, resolve_path

T = TypeVar("T")
_MISSING = object()


def normalise_query(query: str) -> str:
    """Case-fold and collapse whitespace: ``"  Perplexity  AI "`` -> ``"perplexity ai"``."""
    return " ".join(query.split()).casefold()


def _normalise_value(value: Any) -> Any:
    """JSON-stable form of a parameter value (dates as ISO, tuples as lists)."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _normalise_value(v) for k, v in value.items() if v is not None}
    if isinstance(value, list | tuple):
        return [_normalise_value(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted(_normalise_value(v) for v in value)
    if isinstance(value, str):
        return value.strip()
    return value


def normalise_params(params: Mapping[str, Any] | None) -> str:
    """Canonical JSON of ``params``: keys sorted, ``None`` values dropped."""
    return json.dumps(_normalise_value(dict(params or {})), sort_keys=True, default=str)


def make_key(source: str, query: str, params: Mapping[str, Any] | None = None) -> str:
    """Cache key ``<source>:<sha256(normalised query + params)>``."""
    payload = json.dumps([normalise_query(query), normalise_params(params)])
    return f"{source}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


class Store(Protocol):
    """The subset of :class:`diskcache.Cache` that :class:`Cache` uses."""

    def get(self, key: str, default: Any = None) -> Any: ...

    def set(self, key: str, value: Any, expire: float | None = None) -> Any: ...

    def clear(self) -> Any: ...

    def close(self) -> None: ...


OnStore = Callable[[str, float | None, Any], None]


class MemoryStore:
    """In-memory TTL store with the :class:`Store` interface (browser build).

    ``on_store(key, expires_at, value)`` is called after every write so a host
    (the web worker) can persist entries; :meth:`load` re-hydrates them, skipping
    expired ones. ``expires_at`` is a Unix timestamp or ``None`` (never expires).
    """

    def __init__(self, now: Callable[[], float] = time.time, on_store: OnStore | None = None):
        self._now = now
        self._on_store = on_store
        self._entries: dict[str, tuple[float | None, Any]] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Value for ``key`` or ``default`` when absent or expired."""
        entry = self._entries.get(key)
        if entry is None:
            return default
        expires_at, value = entry
        if expires_at is not None and expires_at <= self._now():
            del self._entries[key]
            return default
        return value

    def set(self, key: str, value: Any, expire: float | None = None) -> None:
        """Store ``value``; ``expire`` is a lifetime in seconds."""
        expires_at = None if expire is None else self._now() + float(expire)
        self._entries[key] = (expires_at, value)
        if self._on_store is not None:
            self._on_store(key, expires_at, value)

    def load(self, entries: Iterable[tuple[str, float | None, Any]]) -> int:
        """Add persisted ``(key, expires_at, value)`` entries; return how many were live."""
        now = self._now()
        loaded = 0
        for key, expires_at, value in entries:
            if expires_at is None or expires_at > now:
                self._entries[key] = (expires_at, value)
                loaded += 1
        return loaded

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Remove every entry."""
        self._entries.clear()

    def close(self) -> None:
        """Nothing to release."""


class Cache:
    """Fixed-TTL cache over a :class:`Store` (a :class:`diskcache.Cache` by default)."""

    def __init__(
        self, directory: str | Path | None, ttl_seconds: float, store: Store | None = None
    ) -> None:
        self.directory = Path(directory) if directory is not None else None
        self.ttl_seconds = float(ttl_seconds)
        if store is None:
            if self.directory is None:
                raise ValueError("a directory is required for the disk cache")
            import diskcache  # imported lazily: unavailable/unneeded in the browser build

            store = diskcache.Cache(str(self.directory))
        self._cache: Store = store

    @classmethod
    def in_memory(
        cls,
        ttl_seconds: float,
        *,
        now: Callable[[], float] = time.time,
        on_store: OnStore | None = None,
    ) -> Cache:
        """A cache backed by a :class:`MemoryStore` (no disk access)."""
        return cls(None, ttl_seconds, MemoryStore(now=now, on_store=on_store))

    @property
    def store(self) -> Store:
        """The underlying store."""
        return self._cache

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> Cache:
        """Cache at ``cache.dir`` (relative paths are under the repo root), ``ttl_hours``."""
        cfg = cfg if cfg is not None else get_config()
        return cls(resolve_path(cfg["cache"]["dir"]), float(cfg["cache"]["ttl_hours"]) * 3600)

    def get(self, key: str, default: Any = None) -> Any:
        """Cached value for ``key`` or ``default`` if absent/expired."""
        return self._cache.get(key, default=default)

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` for ``ttl_seconds``."""
        self._cache.set(key, value, expire=self.ttl_seconds)

    def get_or_fetch(
        self,
        source: str,
        query: str,
        params: Mapping[str, Any] | None,
        fetch: Callable[[], T],
    ) -> T:
        """Return the cached result for this request, calling ``fetch`` on a miss.

        Exceptions from ``fetch`` propagate and nothing is cached, so failures are
        retried on the next call.
        """
        key = make_key(source, query, params)
        hit = self._cache.get(key, default=_MISSING)
        if hit is not _MISSING:
            return hit
        value = fetch()
        self.set(key, value)
        return value

    def clear(self) -> None:
        """Remove every entry."""
        self._cache.clear()

    def close(self) -> None:
        """Close the underlying database handle."""
        self._cache.close()


class NoCache:
    """Pass-through stand-in for :class:`Cache` (always fetches, stores nothing).

    Used by the refresh/collect scripts, which must see fresh data.
    """

    def get_or_fetch(
        self,
        source: str,
        query: str,
        params: Mapping[str, Any] | None,
        fetch: Callable[[], T],
    ) -> T:
        """Call ``fetch`` and return its result."""
        return fetch()


CacheLike = Cache | NoCache


_installed: list[Cache] = []


def install_cache(cache: Cache | None) -> None:
    """Make ``cache`` the process-wide cache (``None`` restores the disk cache)."""
    _installed.clear()
    if cache is not None:
        _installed.append(cache)


@lru_cache(maxsize=1)
def _disk_cache() -> Cache:
    return Cache.from_config()


def get_cache() -> Cache:
    """The installed cache, else the process-wide disk cache built from ``config.yaml``."""
    if _installed:
        return _installed[0]
    return _disk_cache()
