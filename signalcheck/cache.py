"""Disk cache for external calls (6h TTL by default, see ``config.yaml``).

Keys are ``source + normalised query + normalised params`` so the same logical
request hits the same entry regardless of whitespace, letter case or parameter
order. Only successful results are cached. Callers must store **aggregates only**
(counts, contributor counts, shares): never author names or post text.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import diskcache

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


class Cache:
    """Thin wrapper over :class:`diskcache.Cache` with a fixed TTL."""

    def __init__(self, directory: str | Path, ttl_seconds: float) -> None:
        self.directory = Path(directory)
        self.ttl_seconds = float(ttl_seconds)
        self._cache = diskcache.Cache(str(self.directory))

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


@lru_cache(maxsize=1)
def get_cache() -> Cache:
    """The process-wide cache built from ``config.yaml``."""
    return Cache.from_config()
