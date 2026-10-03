"""Adapter protocol: every source turns ``fetch(query, params)`` into a ``Series``.

Adapters raise :class:`AdapterError` (or let :class:`signalcheck.http.HttpError`
propagate) with a short reason; the UI shows "Couldn't fetch <source>: <reason>"
on that source's card instead of crashing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from signalcheck.models import Series


class AdapterError(RuntimeError):
    """A source could not produce a series; ``reason`` is safe to show to users."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@runtime_checkable
class Adapter(Protocol):
    """A data source. ``source`` is one of :data:`signalcheck.models.SOURCES`."""

    source: str

    def fetch(self, query: str, params: Mapping[str, Any]) -> Series:
        """Return the series for ``query``; raise :class:`AdapterError` on failure."""
        ...
