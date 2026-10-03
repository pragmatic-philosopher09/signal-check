"""Aggregate-only JSON snapshots of a :class:`Series` (``data/samples/<source>/<slug>.json``).

Snapshots let the demo work without live APIs and are written by
``scripts/refresh_samples.py``. They store only per-bucket aggregates (value,
contributors, top_share, raw_count, imputed) plus ``meta`` (fetched_at, caveats,
resolved query, truncation), never author names or post text. ``fetched_at`` is
kept so the partial-last-period rule still applies when a snapshot is loaded.
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from signalcheck.adapters.base import AdapterError, slugify
from signalcheck.config import Config, get_config, resolve_path
from signalcheck.models import OPTIONAL_COLUMNS, Series

SCHEMA_VERSION = 1
# Meta keys recomputed by preprocessing; never persisted.
_TRANSIENT_META: tuple[str, ...] = ("dropped_partial",)


def samples_dir(cfg: Config | None = None) -> Path:
    """Absolute ``samples.dir``."""
    cfg = cfg if cfg is not None else get_config()
    return resolve_path(cfg["samples"]["dir"])


def snapshot_path(source: str, query: str, base: Path | None = None, ext: str = "json") -> Path:
    """``<base>/<source>/<slug>.<ext>``; ``base`` defaults to ``samples.dir``."""
    root = base if base is not None else samples_dir()
    return root / source / f"{slugify(query)}.{ext}"


def _json_value(value: Any) -> Any:
    """JSON-safe scalar: NaN -> None, numpy/pandas scalars -> Python, dates -> ISO."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp | datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def series_to_dict(series: Series) -> dict[str, Any]:
    """Serialisable dict of a series (aggregate columns only)."""
    columns = ["value", *[c for c in OPTIONAL_COLUMNS if c in series.points.columns]]
    points = []
    for ts, (_, row) in zip(
        pd.to_datetime(series.points["ts"]), series.points.iterrows(), strict=True
    ):
        record: dict[str, Any] = {"ts": ts.date().isoformat()}
        for column in columns:
            record[column] = _json_value(row[column])
        points.append(record)
    meta = {k: v for k, v in series.meta.items() if k not in _TRANSIENT_META}
    return {
        "schema": SCHEMA_VERSION,
        "source": series.source,
        "query": series.query,
        "freq": series.freq,
        "scale": series.scale,
        "meta": json.loads(json.dumps(meta, default=_json_value)),
        "points": points,
    }


def series_from_dict(data: dict[str, Any]) -> Series:
    """Inverse of :func:`series_to_dict` (raises :class:`AdapterError` if malformed)."""
    try:
        frame = pd.DataFrame(data["points"])
        if frame.empty:
            frame = pd.DataFrame({"ts": [], "value": []})
        frame["ts"] = pd.to_datetime(frame["ts"]).astype("datetime64[ns]")
        frame["value"] = frame["value"].astype(float)
        if "imputed" in frame.columns:
            frame["imputed"] = frame["imputed"].astype("boolean").fillna(False).astype(bool)
        for column in ("contributors", "top_share", "raw_count"):
            if column in frame.columns:
                frame[column] = frame[column].astype(float)
        meta = dict(data.get("meta") or {})
        if not meta.get("fetched_at"):
            raise AdapterError("snapshot has no fetched_at")
        meta.setdefault("caveats", [])
        meta["dropped_partial"] = None
        return Series(
            source=data["source"],
            query=data["query"],
            freq=data["freq"],
            points=frame,
            scale=data["scale"],
            meta=meta,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AdapterError(f"snapshot is malformed ({type(exc).__name__}: {exc})") from exc


def write_snapshot(series: Series, base: Path | None = None) -> Path:
    """Write ``series`` atomically to its snapshot path under ``base``; return the path."""
    return write_snapshot_to(series, snapshot_path(series.source, series.query, base))


def _compact(value: Any, sort_keys: bool = True) -> str:
    return json.dumps(value, sort_keys=sort_keys, separators=(",", ":"), allow_nan=False)


def snapshot_text(data: dict[str, Any]) -> str:
    """Deterministic snapshot JSON: sorted keys, compact, one point per line.

    Point keys keep :func:`series_to_dict`'s fixed column order (``ts``, ``value``,
    then the optional columns), which is already deterministic.

    The same data always gives the same bytes, and a daily refresh that appends a
    day changes only a few lines, so git diffs (and repo growth) stay small.
    """
    head = _compact({k: v for k, v in data.items() if k != "points"})
    points = ",\n".join(_compact(p, sort_keys=False) for p in data["points"])
    return f'{head[:-1]},"points":[\n{points}\n]}}\n'


def _without_fetch_time(data: dict[str, Any]) -> dict[str, Any]:
    meta = {k: v for k, v in (data.get("meta") or {}).items() if k != "fetched_at"}
    return {**data, "meta": meta}


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_snapshot_to(series: Series, path: Path) -> Path:
    """Write ``series`` atomically to ``path`` (aggregates + meta only)."""
    _write_atomic(path, snapshot_text(series_to_dict(series)))
    return path


def write_snapshot_if_changed(series: Series, path: Path) -> bool:
    """Write ``series`` to ``path`` unless only ``meta.fetched_at`` would change.

    Returns ``True`` when the file was (re)written. Leaving an unchanged snapshot
    alone keeps the daily refresh free of churn; its older ``fetched_at`` still
    truthfully dates the identical data.
    """
    text = snapshot_text(series_to_dict(series))
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = None
    if isinstance(previous, dict) and _without_fetch_time(previous) == _without_fetch_time(
        json.loads(text)
    ):
        return False
    _write_atomic(path, text)
    return True


# Meta keys that identify what a series measures; history is merged only when they match.
_IDENTITY_META: tuple[str, ...] = ("article", "project", "agent", "tags", "resolved_query")


def can_merge(previous: Series, fresh: Series) -> bool:
    """Whether ``previous`` measures the same thing as ``fresh`` on the same scale.

    Relative (0-100) series are rescaled per request, so they are never merged; nor
    are series of a different article, project, agent filter or resolved query.
    """
    return (
        previous.source == fresh.source
        and previous.freq == fresh.freq
        and previous.scale == fresh.scale
        and fresh.scale != "relative_0_100"
        and all(previous.meta.get(k) == fresh.meta.get(k) for k in _IDENTITY_META)
    )


def merge_history(previous: Series | None, fresh: Series, max_days: int) -> Series:
    """``fresh`` extended back in time with ``previous`` points it no longer covers.

    Fresh points always win: only previous points dated before the first fresh
    point are kept, so a partial last day stored earlier is replaced by its complete
    value. History is trimmed to the ``max_days`` days ending at the newest point.
    Kept days are counted in ``meta.merged_days`` and named in a caveat (never
    silently mixed in); gaps between them are left for preprocessing to report.
    """
    points = fresh.points.sort_values("ts", kind="stable").reset_index(drop=True)
    meta = dict(fresh.meta)
    if previous is not None and can_merge(previous, fresh) and not points.empty:
        first = points["ts"].iloc[0]
        older = previous.points[previous.points["ts"] < first]
        if not older.empty:
            points = pd.concat([older, points], ignore_index=True)
            points = points.sort_values("ts", kind="stable").reset_index(drop=True)
    if not points.empty:
        cutoff = points["ts"].iloc[-1] - pd.Timedelta(days=max_days - 1)
        points = points[points["ts"] >= cutoff].reset_index(drop=True)
    if "imputed" in points.columns:
        points["imputed"] = points["imputed"].astype("boolean").fillna(False).astype(bool)
    merged = int((points["ts"] < fresh.points["ts"].min()).sum()) if len(fresh.points) else 0
    meta.pop("merged_days", None)
    if merged:
        meta["merged_days"] = merged
        meta["caveats"] = [
            *meta.get("caveats", []),
            f"{merged} earlier day(s) come from previous daily refreshes of this snapshot.",
        ]
    return Series(fresh.source, fresh.query, fresh.freq, points, fresh.scale, meta)


def read_snapshot(path: Path) -> Series:
    """Load a snapshot file (raises :class:`AdapterError` when missing or malformed)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AdapterError(f"no snapshot at {path.name}") from exc
    except (OSError, ValueError) as exc:
        raise AdapterError(f"could not read snapshot {path.name}: {exc}") from exc
    return series_from_dict(data)


def load_snapshot(source: str, query: str, base: Path | None = None) -> Series:
    """Snapshot for ``source``/``query`` with a caveat naming its fetch date."""
    series = read_snapshot(snapshot_path(source, query, base))
    return mark_snapshot(series)


def mark_snapshot(series: Series, now: datetime | None = None) -> Series:
    """Flag a series as loaded from a snapshot (``meta.snapshot`` + caveat with its age)."""
    meta = dict(series.meta)
    fetched = pd.Timestamp(meta["fetched_at"])
    fetched = fetched.tz_localize(UTC) if fetched.tzinfo is None else fetched.tz_convert(UTC)
    current = now if now is not None else datetime.now(UTC)
    age_days = max((pd.Timestamp(current) - fetched).days, 0)
    caveats = list(meta.get("caveats", []))
    caveats.append(
        f"Cached snapshot fetched on {fetched:%Y-%m-%d} ({age_days} day(s) ago), not live data."
    )
    meta.update(caveats=caveats, snapshot=True, snapshot_age_days=age_days)
    return Series(series.source, series.query, series.freq, series.points, series.scale, meta)
