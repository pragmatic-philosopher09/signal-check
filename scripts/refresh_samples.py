"""Refresh aggregate-only snapshots in ``data/samples/`` from a developer machine.

Usage::

    python -m scripts.refresh_samples                       # samples.topics x free sources
    python -m scripts.refresh_samples --sources wikipedia hackernews --topics "rust programming"
    python -m scripts.refresh_samples --import-trends-csv ~/Downloads/multiTimeline.csv \\
        --query "perplexity ai" [--fetched-at 2026-10-03T09:00:00Z]

For each topic (``samples.topics`` in ``config.yaml`` unless ``--topics``) and
each selected source, the adapter is called without the cache and the resulting
series is written to ``data/samples/<source>/<slug>.json``: per-bucket aggregates
and ``meta`` (including ``fetched_at``) only, never author names or post text.

* Google Trends uses the live fetcher only (needs the optional ``pytrends``
  package and a residential IP); snapshots are never refreshed from snapshots.
  Alternatively download the CSV from trends.google.com and use
  ``--import-trends-csv`` (``--fetched-at`` defaults to now; pass the download time
  if it was earlier so the partial-last-period rule stays correct).
* Sources that need credentials (Reddit, X) are skipped with a reason when not
  configured. X costs money and is only refreshed when listed in ``--sources``.

Exits 1 when every requested refresh failed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from signalcheck.adapters.base import Adapter, LiveAdapter, fetch_safely, slugify
from signalcheck.adapters.google_trends import GoogleTrendsAdapter, series_from_trends_csv
from signalcheck.adapters.hackernews import HackerNewsAdapter
from signalcheck.adapters.reddit import RedditAdapter
from signalcheck.adapters.wikipedia import WikipediaAdapter
from signalcheck.cache import NoCache
from signalcheck.config import Config, get_config, resolve_path
from signalcheck.snapshots import samples_dir, write_snapshot_to

log = logging.getLogger("refresh_samples")

AdapterFactory = Callable[[Config], LiveAdapter]

ADAPTERS: dict[str, AdapterFactory] = {
    "wikipedia": lambda cfg: WikipediaAdapter(cfg, cache=NoCache()),
    "hackernews": lambda cfg: HackerNewsAdapter(cfg, cache=NoCache()),
    "google_trends": lambda cfg: GoogleTrendsAdapter(cfg, cache=NoCache()),
    "reddit": lambda cfg: RedditAdapter(cfg, cache=NoCache()),
}
# Sources refreshed only when named explicitly (they cost money).
OPT_IN_SOURCES: frozenset[str] = frozenset({"x"})


def default_sources(cfg: Config) -> list[str]:
    """Sources switched on in ``config.yaml``, excluding paid opt-in ones."""
    return [s for s in ADAPTERS if cfg["sources"].get(s, False) and s not in OPT_IN_SOURCES]


def snapshot_target(source: str, query: str, cfg: Config, base: Path | None) -> Path:
    """Where the snapshot for ``source``/``query`` is written."""
    if source == "google_trends" and base is None:
        return (
            resolve_path(cfg["adapters"]["google_trends"]["snapshot_dir"])
            / f"{slugify(query)}.json"
        )
    return (base if base is not None else samples_dir(cfg)) / source / f"{slugify(query)}.json"


def refresh_one(adapter: Adapter, query: str, cfg: Config, base: Path | None) -> str | None:
    """Fetch and write one snapshot; return an error message or ``None``."""
    params = {"live": True}
    if isinstance(adapter, GoogleTrendsAdapter):
        try:
            series = adapter.fetch_live(query, params)
        except Exception as exc:
            reason = getattr(exc, "reason", type(exc).__name__)
            return f"Couldn't fetch Google Trends: {reason}"
    else:
        result = fetch_safely(adapter, query, params)
        if result.series is None:
            return result.message
        series = result.series
    if series.meta.get("snapshot"):
        return f"{adapter.source}: refusing to snapshot a snapshot"
    path = write_snapshot_to(series, snapshot_target(adapter.source, query, cfg, base))
    log.info("wrote %s (%d points)", path, len(series.points))
    return None


def refresh(
    topics: Sequence[str],
    sources: Sequence[str],
    cfg: Config,
    base: Path | None = None,
    factories: dict[str, AdapterFactory] | None = None,
) -> tuple[int, list[str]]:
    """Refresh every topic x source; return ``(n_written, errors)``."""
    factories = factories if factories is not None else ADAPTERS
    written = 0
    errors: list[str] = []
    for source in sources:
        adapter = factories[source](cfg)
        for topic in topics:
            error = refresh_one(adapter, topic, cfg, base)
            if error:
                errors.append(f"{topic!r}: {error}")
            else:
                written += 1
    return written, errors


def import_trends_csv(
    path: Path, query: str, fetched_at: datetime | None, cfg: Config, base: Path | None = None
) -> Path:
    """Convert a Trends-website CSV download into a native JSON snapshot."""
    when = fetched_at or datetime.now(UTC)
    series = series_from_trends_csv(path.read_bytes(), query, when, cfg)
    return write_snapshot_to(series, snapshot_target("google_trends", query, cfg, base))


def _parse_time(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--topics", nargs="+", help="topics (default: samples.topics)")
    parser.add_argument("--sources", nargs="+", choices=sorted(ADAPTERS), help="sources")
    parser.add_argument("--out", type=Path, help="base dir (default: samples.dir)")
    parser.add_argument("--import-trends-csv", type=Path, metavar="CSV")
    parser.add_argument("--query", help="topic for --import-trends-csv")
    parser.add_argument("--fetched-at", type=_parse_time, help="ISO time the CSV was downloaded")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.import_trends_csv:
        if not args.query:
            parser.error("--import-trends-csv needs --query")
        out = import_trends_csv(args.import_trends_csv, args.query, args.fetched_at, cfg, args.out)
        log.info("wrote %s", out)
        return 0

    topics = args.topics or list(cfg["samples"]["topics"])
    sources = args.sources or default_sources(cfg)
    written, errors = refresh(topics, sources, cfg, args.out)
    for error in errors:
        log.warning("skipped %s", error)
    log.info("%d snapshot(s) written, %d skipped", written, len(errors))
    return 1 if errors and not written else 0


if __name__ == "__main__":
    sys.exit(main())
