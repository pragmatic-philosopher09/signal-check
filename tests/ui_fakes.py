"""Fake adapters for UI tests (no network)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from signalcheck.adapters.base import AdapterDisabled, AdapterError
from signalcheck.config import Config
from signalcheck.models import Series
from signalcheck.snapshots import load_snapshot


def snapshot_as_live(source: str, query: str, **meta: Any) -> Series:
    """A committed snapshot dressed up as a live fetch (no snapshot flag)."""
    series = load_snapshot(source, query)
    clean = {k: v for k, v in series.meta.items() if k not in ("snapshot", "snapshot_age_days")}
    clean["caveats"] = [c for c in clean.get("caveats", []) if "snapshot" not in c]
    clean.update(meta)
    return replace(series, meta=clean)


class FakeAdapter:
    """Records calls; returns ``series``, raises ``error`` or is disabled."""

    def __init__(
        self,
        source: str,
        *,
        series: Series | None = None,
        error: BaseException | None = None,
        disabled: str | None = None,
    ) -> None:
        self.source = source
        self.series = series
        self.error = error
        self.disabled = disabled
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def ensure_enabled(self) -> None:
        if self.disabled:
            raise AdapterDisabled(self.disabled)

    def fetch(self, query: str, params: Mapping[str, Any]) -> Series:
        self.calls.append((query, dict(params)))
        self.ensure_enabled()
        if self.error is not None:
            raise self.error
        if self.series is None:
            raise AdapterError("no fake data")
        return self.series


def factory(adapter: FakeAdapter) -> Callable[[Config], FakeAdapter]:
    """Factory returning the same fake (so tests can inspect its calls)."""
    return lambda _cfg: adapter


def no_network_factories(**overrides: FakeAdapter) -> dict[str, Callable[[Config], FakeAdapter]]:
    """Fakes for every topic source: unconfigured ones fail loudly if fetched."""
    defaults = {
        "wikipedia": FakeAdapter("wikipedia", error=AdapterError("unexpected live fetch")),
        "hackernews": FakeAdapter("hackernews", error=AdapterError("unexpected live fetch")),
        "google_trends": FakeAdapter("google_trends", error=AdapterError("unexpected live fetch")),
        "reddit": FakeAdapter("reddit", disabled="API credentials not configured"),
        "x": FakeAdapter("x", disabled="planned"),
    }
    defaults.update(overrides)
    return {name: factory(adapter) for name, adapter in defaults.items()}
