"""Accumulate daily Reddit aggregates for a fixed watchlist (run daily in CI).

Reddit search has no time-series endpoint, no historical backfill and stops at
about 1,000 results, so long histories must be collected over time. Each run
re-fetches the last ``adapters.reddit.collect_days`` complete UTC days for every
``watchlist`` topic with ``reddit: true`` and upserts them into
``data/collected/reddit/<slug>.csv`` (columns ``date,count,contributors,top_share``).
Only aggregates are written; author names and post text never reach disk. Days
before the earliest fully covered day of a capped search are not written.

The daily ``scripts/refresh_data.py`` run calls :func:`collect_topic` for every
such topic; this CLI is for one-off backfills.

Usage::

    python -m scripts.collect_reddit [--topics "perplexity ai" ...] [--out DIR]

Needs ``REDDIT_CLIENT_ID`` and ``REDDIT_CLIENT_SECRET`` (env, ``.env`` or
``st.secrets``). Exits 1 only when every topic failed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from signalcheck.adapters.base import AdapterError, slugify
from signalcheck.adapters.reddit import (
    RedditAdapter,
    collected_path,
    read_collected,
    write_collected,
)
from signalcheck.cache import NoCache
from signalcheck.config import Config, get_config
from signalcheck.http import HttpError
from signalcheck.watchlist import topics_for

log = logging.getLogger("collect_reddit")


def collect_topic(adapter: RedditAdapter, topic: str, out: Path | None = None) -> int:
    """Upsert the last ``collect_days`` complete days for ``topic``; return days written."""
    cfg = adapter.cfg
    today = adapter.now().date()
    end = today - timedelta(days=1)
    start = today - timedelta(days=int(cfg["adapters"]["reddit"]["collect_days"]))
    payload = adapter.live_days(topic, start, end)
    fresh = {d: row for d, row in payload["days"].items() if d < today.isoformat()}
    path = out / f"{slugify(topic)}.csv" if out is not None else collected_path(topic, cfg)
    days = read_collected(path)
    days.update(fresh)
    if fresh:
        write_collected(path, days)
    if payload["capped"]:
        log.warning(
            "%s: search hit the result cap; complete days from %s",
            topic,
            payload["coverage_start"],
        )
    return len(fresh)


def collect(
    topics: Sequence[str],
    cfg: Config,
    out: Path | None = None,
    adapter: RedditAdapter | None = None,
) -> tuple[int, list[str]]:
    """Collect every topic; return ``(n_topics_ok, errors)``."""
    adapter = adapter if adapter is not None else RedditAdapter(cfg, cache=NoCache())
    ok = 0
    errors: list[str] = []
    for topic in topics:
        try:
            adapter.ensure_enabled()
            n = collect_topic(adapter, topic, out)
        except (AdapterError, HttpError) as exc:
            errors.append(f"{topic!r}: {exc.reason}")
            continue
        log.info("%s: %d day(s) upserted", topic, n)
        ok += 1
    return ok, errors


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--topics", nargs="+", help="topics (default: watchlist topics with reddit: true)"
    )
    parser.add_argument(
        "--out", type=Path, help="directory (default: adapters.reddit.collected_dir)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    topics = args.topics or topics_for("reddit", cfg)
    ok, errors = collect(topics, cfg, args.out)
    for error in errors:
        log.error("failed %s", error)
    return 1 if errors and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
