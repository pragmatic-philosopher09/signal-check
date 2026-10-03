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


def write_snapshot_to(series: Series, path: Path) -> Path:
    """Write ``series`` atomically to ``path`` (aggregates + meta only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(series_to_dict(series), indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


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
