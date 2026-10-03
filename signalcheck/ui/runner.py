"""Fetch + analyse orchestration for the Streamlit app (no Streamlit imports).

Every source is fetched through :func:`fetch_safely` and analysed in isolation,
so one failing adapter (or an unexpected analysis error) only degrades its own
card. Sources are fetched concurrently with a small thread pool (or inline with
``max_workers=1``, as in the browser build); the shared HTTP session and disk
cache are created on the calling thread first.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signalcheck import http
from signalcheck.adapters.base import (
    SOURCE_LABELS,
    Adapter,
    AdapterError,
    FetchResult,
    fetch_safely,
)
from signalcheck.adapters.csv_upload import ColumnChoiceRequired, CsvUploadAdapter
from signalcheck.adapters.google_trends import GoogleTrendsAdapter
from signalcheck.adapters.hackernews import HackerNewsAdapter
from signalcheck.adapters.reddit import RedditAdapter
from signalcheck.adapters.wikipedia import WikipediaAdapter
from signalcheck.adapters.x_twitter import XAdapter
from signalcheck.cache import get_cache
from signalcheck.config import Config
from signalcheck.engine.cross_source import CrossSourceSummary, compare_sources
from signalcheck.engine.pipeline import Analysis, analyse_detailed
from signalcheck.models import Series, Verdict
from signalcheck.narrate import Narration, ProviderSettings, narrate, template_only
from signalcheck.snapshots import load_snapshot, samples_dir, snapshot_path

log = logging.getLogger(__name__)

# Sources offered for a topic search, in display order.
TOPIC_SOURCES: tuple[str, ...] = ("wikipedia", "hackernews", "google_trends", "reddit", "x")

AdapterFactory = Callable[[Config], Adapter]

ADAPTER_FACTORIES: dict[str, AdapterFactory] = {
    "wikipedia": WikipediaAdapter,
    "hackernews": HackerNewsAdapter,
    "google_trends": GoogleTrendsAdapter,
    "reddit": RedditAdapter,
    "x": XAdapter,
}

MAX_WORKERS = 4


@dataclass(frozen=True)
class SourceStatus:
    """Whether a source can be selected, and why not (``reason``) when it cannot."""

    source: str
    label: str
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class SourceOutcome:
    """One source's card data: an analysis, or a failure/disabled message."""

    source: str
    label: str
    series: Series | None = None
    analysis: Analysis | None = None
    message: str | None = None
    disabled: bool = False

    @property
    def verdict(self) -> Verdict | None:
        """The verdict, when the source was fetched and analysed."""
        return self.analysis.verdict if self.analysis is not None else None


@dataclass(frozen=True)
class TopicResult:
    """All cards for one request plus the cross-source summary.

    ``summary`` is ``None`` when no enabled source was requested.
    """

    query: str
    outcomes: list[SourceOutcome]
    summary: CrossSourceSummary | None
    sample: bool = False


@dataclass(frozen=True)
class CsvResult:
    """CSV run: either a result or the value columns the user must choose from."""

    result: TopicResult | None = None
    columns: list[str] | None = None


class SnapshotAdapter:
    """Adapter that only reads committed snapshots (sample chips; never touches the network)."""

    def __init__(self, source: str, base: Path) -> None:
        self.source = source
        self.base = base

    def fetch(self, query: str, params: Mapping[str, Any]) -> Series:
        """The snapshot for ``query`` (params are ignored: snapshots are fixed)."""
        return load_snapshot(self.source, query, self.base)


class _ColumnCapture:
    """Wraps the CSV adapter to remember the columns offered by :class:`ColumnChoiceRequired`."""

    source = "csv"

    def __init__(self, inner: CsvUploadAdapter) -> None:
        self.inner = inner
        self.columns: list[str] | None = None

    def fetch(self, query: str, params: Mapping[str, Any]) -> Series:
        try:
            return self.inner.fetch(query, params)
        except ColumnChoiceRequired as exc:
            self.columns = list(exc.columns)
            raise


def source_label(source: str) -> str:
    """Display name of a source."""
    return SOURCE_LABELS.get(source, source)


def probe_source(source: str, cfg: Config, factory: AdapterFactory | None = None) -> SourceStatus:
    """Availability without network: config switch, then credentials where needed."""
    label = source_label(source)
    try:
        adapter = (factory or ADAPTER_FACTORIES[source])(cfg)
        ensure = getattr(adapter, "ensure_enabled", None)
        if ensure is not None:
            ensure()
        if isinstance(adapter, RedditAdapter):
            adapter.credentials()
        if isinstance(adapter, XAdapter):
            adapter.token()
    except AdapterError as exc:
        return SourceStatus(source, label, False, exc.reason)
    return SourceStatus(source, label, True)


def source_statuses(
    cfg: Config, factories: Mapping[str, AdapterFactory] | None = None
) -> list[SourceStatus]:
    """:func:`probe_source` for every topic source, in display order."""
    factories = factories or ADAPTER_FACTORIES
    return [probe_source(s, cfg, factories.get(s)) for s in TOPIC_SOURCES]


def sample_sources(query: str, cfg: Config) -> list[str]:
    """Topic sources with a committed snapshot for ``query``."""
    base = samples_dir(cfg)
    return [s for s in TOPIC_SOURCES if snapshot_path(s, query, base).exists()]


def analyse_result(fetched: FetchResult, cfg: Config) -> SourceOutcome:
    """Analyse a fetch result; an engine error degrades to a per-card message."""
    label = source_label(fetched.source)
    if fetched.series is None:
        return SourceOutcome(
            fetched.source, label, message=fetched.message, disabled=fetched.disabled
        )
    try:
        analysis = analyse_detailed(fetched.series, cfg)
    except Exception as exc:
        log.error("analysis of %s failed with %s", fetched.source, type(exc).__name__)
        return SourceOutcome(
            fetched.source,
            label,
            series=fetched.series,
            message=f"Couldn't analyse {label}: unexpected error ({type(exc).__name__})",
        )
    return SourceOutcome(fetched.source, label, series=fetched.series, analysis=analysis)


def _fetch_and_analyse(
    adapter: Adapter, query: str, params: Mapping[str, Any], cfg: Config
) -> SourceOutcome:
    return analyse_result(fetch_safely(adapter, query, params), cfg)


def summarise(outcomes: Sequence[SourceOutcome], cfg: Config) -> CrossSourceSummary | None:
    """Cross-source summary; failed sources count as unavailable, disabled ones are left out."""
    verdicts = {o.source: o.verdict for o in outcomes if not o.disabled}
    if not verdicts:
        return None
    return compare_sources(verdicts, cfg)


def warm_shared_clients() -> None:
    """Create the shared HTTP session and cache on this thread (their factories aren't locked)."""
    http.get_session()
    get_cache()


def source_params(source: str, days: int | None, wiki_article: str | None) -> dict[str, Any]:
    """Adapter params for one source from the UI choices."""
    params: dict[str, Any] = {}
    if days is not None:
        params["days"] = int(days)
    if source == "wikipedia" and wiki_article:
        params["article"] = wiki_article
    return params


def disabled_outcome(status: SourceStatus) -> SourceOutcome:
    """Greyed-out card for a source that can't be used (no fetch is attempted)."""
    return SourceOutcome(
        status.source,
        status.label,
        message=f"{status.label} disabled ({status.reason})",
        disabled=True,
    )


def _guarded(source: str, call: Callable[..., SourceOutcome], *args: Any) -> SourceOutcome:
    """``call(*args)``, turning any unexpected exception into a per-card failure."""
    try:
        return call(*args)
    except Exception as exc:
        log.error("source %s failed with %s", source, type(exc).__name__)
        label = source_label(source)
        return SourceOutcome(
            source,
            label,
            message=f"Couldn't fetch {label}: unexpected error ({type(exc).__name__})",
        )


def run_topic(
    query: str,
    sources: Sequence[str],
    cfg: Config,
    *,
    days: int | None = None,
    sample: bool = False,
    wiki_article: str | None = None,
    factories: Mapping[str, AdapterFactory] | None = None,
    max_workers: int = MAX_WORKERS,
) -> TopicResult:
    """Fetch and analyse ``query`` for each source; never raises for a source failure.

    ``sample=True`` reads committed snapshots only. A Wikipedia article override
    always fetches Wikipedia live, even for a sample topic.
    """
    factories = factories or ADAPTER_FACTORIES
    base = samples_dir(cfg)
    adapters: list[tuple[Adapter, dict[str, Any]]] = []
    for source in sources:
        live = not sample or (source == "wikipedia" and bool(wiki_article))
        adapter: Adapter = factories[source](cfg) if live else SnapshotAdapter(source, base)
        adapters.append((adapter, source_params(source, days, wiki_article)))
    if any(not isinstance(a, SnapshotAdapter) for a, _ in adapters):
        warm_shared_clients()
    outcomes: list[SourceOutcome] = []
    if adapters:
        workers = max(1, min(max_workers, len(adapters)))
        if workers == 1:
            # Inline, no threads: also the path for the browser build (Pyodide has none).
            for adapter, params in adapters:
                outcomes.append(
                    _guarded(adapter.source, _fetch_and_analyse, adapter, query, params, cfg)
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    (adapter.source, pool.submit(_fetch_and_analyse, adapter, query, params, cfg))
                    for adapter, params in adapters
                ]
                for source, future in futures:
                    outcomes.append(_guarded(source, future.result))
    return TopicResult(query, outcomes, summarise(outcomes, cfg), sample=sample)


NarrateFn = Callable[..., Narration]


def series_meta(outcome: SourceOutcome) -> dict[str, Any]:
    """The source facts the narration payload needs (id, display label, frequency)."""
    freq = outcome.series.freq if outcome.series is not None else None
    return {"source": outcome.source, "source_label": outcome.label, "freq": freq}


def template_narrations(outcomes: Sequence[SourceOutcome], cfg: Config) -> dict[str, Narration]:
    """Template narration for every analysed outcome (instant; no provider involved)."""
    return {
        o.source: template_only(o.analysis.verdict, series_meta(o), cfg)
        for o in outcomes
        if o.analysis is not None
    }


def narrate_outcomes(
    outcomes: Sequence[SourceOutcome],
    cfg: Config,
    *,
    settings: ProviderSettings | None = None,
    narrate_fn: NarrateFn | None = None,
    timeout_s: float | None = None,
) -> dict[str, Narration]:
    """Narration per analysed outcome, keyed by source; never raises or blocks for long.

    With no provider configured every card gets its template immediately. Otherwise
    the LLM calls run concurrently (``narration.max_workers``) and the page waits at
    most ``narration.total_timeout_s``; unfinished or failed narrations fall back to
    the template (category ``timeout``/``error``).
    """
    settings = settings if settings is not None else ProviderSettings.from_secrets()
    templates = template_narrations(outcomes, cfg)
    if not settings.enabled or not templates:
        return templates
    n = cfg["narration"]
    timeout = float(n["total_timeout_s"]) if timeout_s is None else timeout_s
    narrate_fn = narrate_fn if narrate_fn is not None else narrate
    warm_shared_clients()
    pool = ThreadPoolExecutor(max_workers=max(1, min(int(n["max_workers"]), len(templates))))
    futures: dict[str, Future[Narration]] = {
        o.source: pool.submit(
            narrate_fn, o.analysis.verdict, series_meta(o), cfg=cfg, settings=settings
        )
        for o in outcomes
        if o.analysis is not None
    }
    wait(futures.values(), timeout=timeout)
    pool.shutdown(wait=False, cancel_futures=True)
    results: dict[str, Narration] = {}
    for source, future in futures.items():
        fallback = templates[source]
        if not future.done():
            log.info("narration fell back to template: timeout")
            results[source] = Narration(fallback.text, "template", "timeout")
            continue
        try:
            results[source] = future.result()
        except Exception:
            log.info("narration fell back to template: error")
            results[source] = Narration(fallback.text, "template", "error")
    return results


def run_csv(
    data: bytes,
    name: str,
    cfg: Config,
    *,
    value_column: str | None = None,
    last_period_incomplete: bool = False,
) -> CsvResult:
    """Parse and analyse an uploaded CSV; ask for a column when several are present."""
    adapter = _ColumnCapture(CsvUploadAdapter(cfg))
    params: dict[str, Any] = {"data": data, "last_period_incomplete": last_period_incomplete}
    if value_column:
        params["value_column"] = value_column
    fetched = fetch_safely(adapter, name, params)
    if adapter.columns is not None:
        return CsvResult(columns=adapter.columns)
    outcome = analyse_result(fetched, cfg)
    return CsvResult(result=TopicResult(name, [outcome], summarise([outcome], cfg)))
