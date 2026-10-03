"""Fetch the real eval cases in ``eval/real_cases.yaml`` into aggregate-only snapshots.

Usage::

    python -m eval.fetch_real                  # fetch cases without a snapshot yet
    python -m eval.fetch_real --force          # re-fetch every case
    python -m eval.fetch_real --ids suez_blockage halloween_daily

Each case is fetched once from its live adapter (no cache) over ``start``..``end``
and written to ``eval/real_data/<id>.json`` in the ``data/samples`` snapshot
format (per-bucket aggregates + ``meta`` with ``fetched_at``; never author names or
post text), so the eval runs offline and reproducibly. Weekly (``freq: W``) cases
are Monday-Sunday sums of the daily series; only complete weeks are kept. This
script only collects data: it never runs the engine (labels are pre-registered).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from signalcheck.adapters.base import AdapterError, LiveAdapter
from signalcheck.adapters.hackernews import HackerNewsAdapter
from signalcheck.adapters.wikipedia import WikipediaAdapter
from signalcheck.cache import NoCache
from signalcheck.config import Config, get_config
from signalcheck.http import HttpError
from signalcheck.models import Series
from signalcheck.snapshots import write_snapshot_to

log = logging.getLogger("fetch_real")

EVAL_DIR = Path(__file__).resolve().parent
REAL_CASES_PATH = EVAL_DIR / "real_cases.yaml"
REAL_DATA_DIR = EVAL_DIR / "real_data"
REAL_SOURCES: tuple[str, ...] = ("wikipedia", "hackernews")
REAL_FREQS: tuple[str, ...] = ("D", "W")
WEEKLY_CAVEAT = (
    "Weekly values are Monday-Sunday sums of daily values; only complete weeks are kept."
)

AdapterFactory = Callable[[Config], LiveAdapter]
ADAPTERS: dict[str, AdapterFactory] = {
    "wikipedia": lambda cfg: WikipediaAdapter(cfg, cache=NoCache()),
    "hackernews": lambda cfg: HackerNewsAdapter(cfg, cache=NoCache()),
}


def real_data_path(case_id: str, base: Path = REAL_DATA_DIR) -> Path:
    """Snapshot file of a real case."""
    return base / f"{case_id}.json"


def load_entries(path: Path = REAL_CASES_PATH) -> list[dict[str, Any]]:
    """Raw case entries with dates as ISO strings; validates source/freq/ids."""
    raw = yaml.safe_load(path.read_text()) or {}
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in raw.get("cases") or []:
        case_id = str(entry["id"])
        if case_id in seen:
            raise ValueError(f"duplicate real case id {case_id!r}")
        seen.add(case_id)
        if entry.get("source") not in REAL_SOURCES:
            raise ValueError(f"{case_id}: source must be one of {REAL_SOURCES}")
        if entry.get("freq") not in REAL_FREQS:
            raise ValueError(f"{case_id}: freq must be one of {REAL_FREQS}")
        out = dict(entry)
        for key in ("start", "end", "as_of", "event_date", "labelled_on"):
            if out.get(key) is not None:
                out[key] = str(out[key])
        if not out["start"] <= out["as_of"] <= out["end"]:
            raise ValueError(f"{case_id}: need start <= as_of <= end")
        entries.append(out)
    return entries


def weekly_sums(series: Series) -> Series:
    """Monday-Sunday weekly sums of a daily series (complete weeks only).

    A week is ``imputed`` if any of its days was; contributor/breadth columns are
    dropped (they do not add up across days).
    """
    if series.freq != "D":
        raise ValueError("weekly_sums needs a daily series")
    df = series.points.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df["week"] = df["ts"] - pd.to_timedelta(df["ts"].dt.weekday, unit="D")
    imputed = (
        df["imputed"].astype(bool) if "imputed" in df.columns else pd.Series(False, index=df.index)
    )
    grouped = df.assign(imputed=imputed).groupby("week")
    weeks = pd.DataFrame(
        {
            "value": grouped["value"].sum(min_count=1),
            "days": grouped["value"].count(),
            "imputed": grouped["imputed"].any(),
        }
    )
    weeks = weeks[weeks["days"] == 7].drop(columns="days").reset_index(names="ts")
    meta = dict(series.meta)
    meta["caveats"] = [*meta.get("caveats", []), WEEKLY_CAVEAT]
    meta["aggregated_from"] = "D"
    return Series(series.source, series.query, "W", weeks, series.scale, meta)


def fetch_case(entry: Mapping[str, Any], cfg: Config, adapters: Mapping[str, Any]) -> Series:
    """Fetch one case over ``start``..``end`` and aggregate to its ``freq``."""
    adapter = adapters[entry["source"]]
    params: dict[str, Any] = {"start": entry["start"], "end": entry["end"]}
    if entry.get("article"):
        params["article"] = entry["article"]
    series = adapter.fetch(str(entry["query"]), params)
    if entry.get("article") and series.meta.get("article") != entry["article"]:
        raise AdapterError(
            f"article resolved to {series.meta.get('article')!r}, expected {entry['article']!r}"
        )
    return weekly_sums(series) if entry["freq"] == "W" else series


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; exits 1 if any requested case failed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ids", nargs="*", help="only these case ids")
    parser.add_argument("--force", action="store_true", help="overwrite existing snapshots")
    parser.add_argument("--cases", type=Path, default=REAL_CASES_PATH)
    parser.add_argument("--out", type=Path, default=REAL_DATA_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = get_config()
    adapters = {name: factory(cfg) for name, factory in ADAPTERS.items()}
    failed = 0
    for entry in load_entries(args.cases):
        if args.ids and entry["id"] not in args.ids:
            continue
        path = real_data_path(entry["id"], args.out)
        if path.exists() and not args.force:
            log.info("%s: snapshot exists, skipping", entry["id"])
            continue
        try:
            series = fetch_case(entry, cfg, adapters)
        except (AdapterError, HttpError) as exc:
            failed += 1
            log.error("%s: failed: %s", entry["id"], exc)
            continue
        write_snapshot_to(series, path)
        log.info("%s: %d points -> %s", entry["id"], len(series.points), path.name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
