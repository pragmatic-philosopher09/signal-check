"""Daily refresh of the watchlist snapshots (run by ``.github/workflows/refresh.yml``).

Usage::

    python -m scripts.refresh_data                         # every watchlist topic x source
    python -m scripts.refresh_data --topics chatgpt --sources wikipedia
    python -m scripts.refresh_data --summary summary.md    # also write the Markdown table

For each ``watchlist`` entry in ``config.yaml`` and each source it enables
(Wikipedia, Hacker News, Reddit), the adapter is called without the cache and the
series is written to ``data/samples/<source>/<slug>.json``. Only per-bucket
aggregates and ``meta`` are written, never author names or post text.

* **History grows**: points older than the fresh fetch are kept from the previous
  snapshot (fresh points win; trimmed to ``refresh.max_history_days``; counted in
  a caveat). Reddit history instead accumulates in ``data/collected/reddit/`` (the
  last ``adapters.reddit.collect_days`` days are upserted, then the adapter merges
  them into the snapshot).
* **No churn**: files are deterministic (sorted keys, compact, one point per line)
  and left untouched when nothing but ``fetched_at`` would change.
* **Failures degrade**: a failed source keeps its previous snapshot and the error is
  recorded. Reddit is skipped (not failed) without ``REDDIT_CLIENT_ID``/``SECRET``.
  X is never refreshed here.
* ``refresh.manifest`` (``data/manifest.json``) records the run time and every
  topic x source status, error and ``fetched_at``; the Pages build reads it.

Exits 1 only when every attempted source failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.collect_reddit import collect_topic
from signalcheck.adapters.base import (
    Adapter,
    AdapterDisabled,
    AdapterError,
    fetch_safely,
    slugify,
)
from signalcheck.adapters.hackernews import HackerNewsAdapter
from signalcheck.adapters.reddit import RedditAdapter
from signalcheck.adapters.wikipedia import WikipediaAdapter
from signalcheck.cache import NoCache
from signalcheck.config import WATCHLIST_SOURCES, Config, get_config, resolve_path
from signalcheck.http import HttpError
from signalcheck.snapshots import (
    merge_history,
    read_snapshot,
    samples_dir,
    snapshot_path,
    write_snapshot_if_changed,
)
from signalcheck.watchlist import WatchItem, watchlist

log = logging.getLogger("refresh_data")

MANIFEST_SCHEMA = 1
LABELS: dict[str, str] = {
    "wikipedia": "Wikipedia",
    "hackernews": "Hacker News",
    "reddit": "Reddit",
}
# Status of one topic x source in a run.
UPDATED, UNCHANGED, FAILED, SKIPPED = "updated", "unchanged", "failed", "skipped"
SUMMARY_CELLS: dict[str, str] = {
    UPDATED: "✅ ok",
    UNCHANGED: "⚪ unchanged",
    FAILED: "❌ failed",
    SKIPPED: "⏭️ skipped",
}

AdapterFactory = Callable[[Config], Adapter]
Collector = Callable[[Any, str], int]

ADAPTERS: dict[str, AdapterFactory] = {
    "wikipedia": lambda cfg: WikipediaAdapter(cfg, cache=NoCache()),
    "hackernews": lambda cfg: HackerNewsAdapter(cfg, cache=NoCache()),
    "reddit": lambda cfg: RedditAdapter(cfg, cache=NoCache()),
}


@dataclass(frozen=True)
class Entry:
    """Outcome of one topic x source: status, the snapshot's ``fetched_at`` and any error."""

    topic: str
    source: str
    status: str
    fetched_at: str | None = None
    points: int | None = None
    error: str | None = None

    def as_json(self) -> dict[str, Any]:
        """Manifest form (``None`` fields omitted, so unchanged runs stay byte-stable)."""
        data = {
            "status": self.status,
            "fetched_at": self.fetched_at,
            "points": self.points,
            "error": self.error,
        }
        return {k: v for k, v in data.items() if v is not None}


def _previous(path: Path) -> tuple[Any, str | None, int | None]:
    """``(series, fetched_at, n_points)`` of the snapshot at ``path`` (``None``s if absent)."""
    try:
        series = read_snapshot(path)
    except AdapterError:
        return None, None, None
    return series, str(series.meta["fetched_at"]), len(series.points)


def refresh_snapshot(
    adapter: Adapter,
    item: WatchItem,
    source: str,
    cfg: Config,
    base: Path,
    *,
    merge: bool = True,
) -> Entry:
    """Fetch one topic x source, merge history and write the snapshot if it changed."""
    path = snapshot_path(source, item.topic, base)
    previous, prev_fetched, prev_points = _previous(path)
    result = fetch_safely(adapter, item.topic, item.params(source))
    if result.series is None:
        status = SKIPPED if result.disabled else FAILED
        error = f"Couldn't fetch {LABELS[source]}: {result.error}"
        return Entry(item.topic, source, status, prev_fetched, prev_points, error)
    fresh = result.series
    if fresh.meta.get("snapshot"):
        error = f"{LABELS[source]}: refusing to snapshot a snapshot"
        return Entry(item.topic, source, FAILED, prev_fetched, prev_points, error)
    max_days = int(cfg["refresh"]["max_history_days"])
    series = merge_history(previous if merge else None, fresh, max_days)
    if write_snapshot_if_changed(series, path):
        log.info("%s/%s: wrote %d points", source, slugify(item.topic), len(series.points))
        fetched_at = str(series.meta["fetched_at"])
        return Entry(item.topic, source, UPDATED, fetched_at, len(series.points))
    log.info("%s/%s: unchanged", source, slugify(item.topic))
    return Entry(item.topic, source, UNCHANGED, prev_fetched, prev_points)


def reddit_skip_reason(adapter: Any) -> str | None:
    """Why Reddit can't run (switched off or no credentials), else ``None``."""
    try:
        adapter.ensure_enabled()
        adapter.credentials()
    except AdapterDisabled as exc:
        return str(exc.reason)
    return None


def refresh_reddit(
    adapter: Any, item: WatchItem, cfg: Config, base: Path, collect: Collector
) -> Entry:
    """Upsert the collected daily history, then rebuild the Reddit snapshot from it."""
    path = snapshot_path("reddit", item.topic, base)
    reason = reddit_skip_reason(adapter)
    if reason is not None:
        _, prev_fetched, prev_points = _previous(path)
        error = f"Couldn't fetch Reddit: {reason}"
        return Entry(item.topic, "reddit", SKIPPED, prev_fetched, prev_points, error)
    try:
        collect(adapter, item.topic)
    except (AdapterError, HttpError) as exc:
        _, prev_fetched, prev_points = _previous(path)
        error = f"Couldn't fetch Reddit: {exc.reason}"
        return Entry(item.topic, "reddit", FAILED, prev_fetched, prev_points, error)
    # The adapter merges data/collected into the series, so no snapshot merge here.
    return refresh_snapshot(adapter, item, "reddit", cfg, base, merge=False)


def refresh(
    items: Sequence[WatchItem],
    cfg: Config,
    *,
    sources: Sequence[str] = WATCHLIST_SOURCES,
    base: Path | None = None,
    factories: Mapping[str, AdapterFactory] | None = None,
    collect: Collector = collect_topic,
) -> list[Entry]:
    """Refresh every watchlist topic x enabled source; never raises for a source failure."""
    factories = factories if factories is not None else ADAPTERS
    root = base if base is not None else samples_dir(cfg)
    adapters: dict[str, Adapter] = {}
    entries: list[Entry] = []
    for item in items:
        for source in item.sources:
            if source not in sources:
                continue
            if source not in adapters:
                adapters[source] = factories[source](cfg)
            adapter = adapters[source]
            if source == "reddit":
                entry = refresh_reddit(adapter, item, cfg, root, collect)
            else:
                entry = refresh_snapshot(adapter, item, source, cfg, root)
            if entry.error:
                log.warning("%s", entry.error)
            entries.append(entry)
    return entries


def source_status(entries: Sequence[Entry]) -> str:
    """``ok`` / ``partial`` / ``failed`` for attempted entries, ``skipped`` if none ran."""
    attempted = [e for e in entries if e.status != SKIPPED]
    if not attempted:
        return SKIPPED
    failed = sum(e.status == FAILED for e in attempted)
    if failed == len(attempted):
        return FAILED
    return "partial" if failed else "ok"


def _entries_from_manifest(manifest: Mapping[str, Any]) -> list[Entry]:
    """The topic x source entries recorded in a previous manifest."""
    entries = []
    for topic in (manifest.get("topics") or {}).values():
        for source, data in (topic.get("sources") or {}).items():
            fields = {k: data.get(k) for k in ("fetched_at", "points", "error")}
            entries.append(Entry(topic["query"], source, data["status"], **fields))
    return entries


def read_manifest(path: Path) -> dict[str, Any]:
    """The previous manifest, or ``{}`` when missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def manifest_json(
    entries: Sequence[Entry],
    now: datetime,
    previous: Mapping[str, Any] | None = None,
    topics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """``data/manifest.json``: run time, per-source status and every topic x source entry.

    Entries from ``previous`` that this run did not refresh (a ``--topics`` or
    ``--sources`` subset) are kept; topics no longer in ``topics`` are dropped.
    """
    merged: dict[tuple[str, str], Entry] = {}
    for entry in [*_entries_from_manifest(previous or {}), *entries]:
        merged[(slugify(entry.topic), entry.source)] = entry
    keep = {slugify(t) for t in topics} if topics is not None else None
    final = [e for (slug, _), e in merged.items() if keep is None or slug in keep]
    by_source: dict[str, list[Entry]] = {}
    out_topics: dict[str, dict[str, Any]] = {}
    for entry in final:
        by_source.setdefault(entry.source, []).append(entry)
        topic = out_topics.setdefault(slugify(entry.topic), {"query": entry.topic, "sources": {}})
        topic["sources"][entry.source] = entry.as_json()
    sources = {
        source: {
            "status": source_status(group),
            **{
                status: sum(e.status == status for e in group)
                for status in (UPDATED, UNCHANGED, FAILED, SKIPPED)
            },
        }
        for source, group in by_source.items()
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "refreshed_at": now.astimezone(UTC).isoformat(timespec="seconds"),
        "status": source_status(final),
        "sources": sources,
        "topics": out_topics,
    }


def manifest_text(manifest: Mapping[str, Any]) -> str:
    """Deterministic manifest JSON (sorted keys, no spaces, one key per line)."""
    return json.dumps(manifest, sort_keys=True, indent=1, separators=(",", ":")) + "\n"


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write the manifest atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(manifest_text(manifest), encoding="utf-8")
    os.replace(tmp, path)


def summary_markdown(entries: Sequence[Entry], now: datetime) -> str:
    """A Markdown topic x source table (for ``$GITHUB_STEP_SUMMARY``) plus the errors."""
    sources = [s for s in WATCHLIST_SOURCES if any(e.source == s for e in entries)]
    lines = [
        f"### Data refresh {now.astimezone(UTC):%Y-%m-%d %H:%M} UTC",
        "",
        "| Topic | " + " | ".join(LABELS[s] for s in sources) + " |",
        "|---|" + "---|" * len(sources),
    ]
    cells: dict[tuple[str, str], str] = {(e.topic, e.source): e.status for e in entries}
    for topic in dict.fromkeys(e.topic for e in entries):
        row = [SUMMARY_CELLS.get(cells.get((topic, s), ""), "—") for s in sources]
        lines.append(f"| {topic} | " + " | ".join(row) + " |")
    counts = {s: sum(e.status == s for e in entries) for s in SUMMARY_CELLS}
    lines += ["", ", ".join(f"{n} {status}" for status, n in counts.items())]
    problems = [e for e in entries if e.error]
    if problems:
        lines += ["", "**Not refreshed** (previous snapshot kept):", ""]
        lines += [f"- {e.topic} / {LABELS[e.source]}: {e.error}" for e in problems]
    return "\n".join(lines) + "\n"


def exit_code(entries: Sequence[Entry]) -> int:
    """1 when every attempted source failed, else 0."""
    return 1 if source_status(entries) == FAILED else 0


def main(argv: Sequence[str] | None = None, now: datetime | None = None) -> int:
    """CLI entry point."""
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--topics", nargs="+", help="watchlist topics (default: all)")
    parser.add_argument("--sources", nargs="+", choices=WATCHLIST_SOURCES, help="sources")
    parser.add_argument("--summary", type=Path, help="also write the Markdown summary here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    items = watchlist(cfg)
    if args.topics:
        wanted = {slugify(t) for t in args.topics}
        items = [i for i in items if slugify(i.topic) in wanted]
        if not items:
            parser.error("none of --topics is on the watchlist")
    entries = refresh(items, cfg, sources=args.sources or WATCHLIST_SOURCES)
    when = now if now is not None else datetime.now(UTC)
    if source_status(entries) != FAILED:
        path = resolve_path(cfg["refresh"]["manifest"])
        topics = [i.topic for i in watchlist(cfg)]
        write_manifest(path, manifest_json(entries, when, read_manifest(path), topics))
    summary = summary_markdown(entries, when)
    print(summary)
    targets = [args.summary, os.environ.get("GITHUB_STEP_SUMMARY")]
    for target in filter(None, targets):
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(summary)
    return exit_code(entries)


if __name__ == "__main__":
    sys.exit(main())
