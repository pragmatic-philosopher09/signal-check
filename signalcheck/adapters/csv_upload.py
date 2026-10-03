"""CSV upload adapter (generic two-column files and Google Trends downloads).

Accepted inputs:

* **Generic**: a date column and one or more value columns with flexible header
  names (``date``/``day``/``week``/``ts``..., ``value``/``count``/``views``...),
  comma, semicolon or tab separated, with or without a header row.
* **Google Trends download**: leading metadata lines (``Category: All
  categories``), headers like ``Week`` and ``perplexity ai: (Worldwide)``, ``<1``
  cells (mapped to ``adapters.csv.trends_lt1_value`` and flagged ``imputed``)
  and several query columns, of which the user must pick one
  (:class:`ColumnChoiceRequired` lists them).

Dates must be ISO-like (``YYYY-MM-DD``, ``YYYY/MM/DD``, ``YYYY-MM``, optional time,
or a Trends range ``YYYY-MM-DD - YYYY-MM-DD`` which uses the start date) so day
and month are never ambiguous.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

from signalcheck.adapters.base import AdapterError
from signalcheck.config import FREQS, SCALES, Config, get_config
from signalcheck.models import Series

DELIMITERS: tuple[str, ...] = (",", ";", "\t")
DATE_HEADERS: dict[str, str | None] = {
    "date": None,
    "ts": None,
    "time": None,
    "timestamp": None,
    "period": None,
    "datetime": None,
    "day": "D",
    "week": "W",
    "month": "M",
}
TRENDS_COLUMN = re.compile(r"^(?P<query>.+?):\s*\((?P<geo>[^()]*)\)\s*$")
TRENDS_LT1 = "<1"
TRENDS_CAVEATS: tuple[str, ...] = (
    "Google Trends values are on a relative 0-100 scale, not search volume.",
    "Google samples this data; small moves can be sampling noise.",
)
_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?(?:Z|[+-]\d{2}:?\d{2})?$"
    ),
    re.compile(r"^\d{4}-\d{1,2}$"),
)
_RANGE = re.compile(r"^(\d{4}-\d{1,2}-\d{1,2})\s+-\s+\d{4}-\d{1,2}-\d{1,2}$")
_SLASH_DATE = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$")
_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
# Median spacing (days) between consecutive dates for each frequency.
_SPACING_DAYS: dict[str, tuple[float, float]] = {"D": (1, 1), "W": (7, 7), "M": (28, 31)}
DEFAULT_VALUE_NAME = "value"


class CsvError(AdapterError):
    """The uploaded file could not be read as a time series."""


class ColumnChoiceRequired(CsvError):
    """Several value columns were found; the user must pick one of ``columns``."""

    def __init__(self, columns: list[str]) -> None:
        super().__init__(f"Choose one value column: {', '.join(columns)}")
        self.columns = columns


@dataclass(frozen=True)
class CsvTable:
    """A located table: raw string cells plus what was learnt about its layout."""

    frame: pd.DataFrame
    date_column: str
    value_columns: list[str]
    metadata: list[str]
    freq_hint: str | None
    first_data_line: int


def decode(data: bytes | str) -> str:
    """Decode uploaded bytes (UTF-8 with/without BOM, UTF-16, else Latin-1)."""
    if isinstance(data, str):
        return data.lstrip("\ufeff")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def normalise_date_text(text: str) -> str | None:
    """ISO form of a date cell, or ``None`` if it is not an unambiguous date."""
    text = text.strip().strip('"').strip()
    if match := _RANGE.match(text):
        text = match.group(1)
    if match := _SLASH_DATE.match(text):
        text = "-".join(match.groups())
    return text if any(p.match(text) for p in _DATE_PATTERNS) else None


def parse_dates(cells: pd.Series, first_line: int) -> pd.Series:
    """Parse a column of date cells into naive UTC timestamps (raises on bad rows)."""
    normalised = cells.map(normalise_date_text)
    bad = normalised.isna()
    if bad.any():
        pos = int(np.flatnonzero(bad.to_numpy())[0])
        raise CsvError(
            f"Line {first_line + pos}: {cells.iloc[pos]!r} is not a date "
            "(use YYYY-MM-DD, YYYY/MM/DD or YYYY-MM)."
        )
    try:
        parsed = pd.to_datetime(normalised, format="ISO8601", utc=True)
    except (ValueError, TypeError) as exc:
        raise CsvError(f"Could not parse dates: {exc}") from exc
    return parsed.dt.tz_localize(None).astype("datetime64[ns]")


def _split(line: str, delimiter: str) -> list[str]:
    """Split one CSV line, honouring quotes."""
    return next(csv.reader([line], delimiter=delimiter, skipinitialspace=True), [])


def _best_delimiter(line: str) -> str:
    """Delimiter producing the most fields for ``line`` (ties prefer comma)."""
    return max(DELIMITERS, key=lambda d: (len(_split(line, d)), d == ","))


def _date_field_index(fields: list[str]) -> int | None:
    """Index of the first field that is a date (``None`` if none is)."""
    for i, field in enumerate(fields):
        if normalise_date_text(field) is not None:
            return i
    return None


def locate_table(text: str) -> CsvTable:
    """Find the header (if any) and data rows, skipping leading metadata lines.

    A line is the header when it has >= 2 fields, none of them a date, and the
    next non-blank line has a date field. A line whose fields already contain a
    date starts a header-less table (columns ``date``, ``value``, ``value_2``...).
    """
    lines = text.splitlines()
    nonblank = [(i, line) for i, line in enumerate(lines) if line.strip()]
    for k, (lineno, line) in enumerate(nonblank):
        delimiter = _best_delimiter(line)
        fields = _split(line, delimiter)
        if len(fields) < 2:
            continue
        metadata = [ln.strip() for _, ln in nonblank[:k]]
        date_idx = _date_field_index(fields)
        if date_idx is not None:
            names = ["date", DEFAULT_VALUE_NAME] + [
                f"{DEFAULT_VALUE_NAME}_{j}" for j in range(2, len(fields))
            ]
            names[0], names[date_idx] = names[date_idx], names[0]
            body = "\n".join(ln for _, ln in nonblank[k:])
            frame = _read(body, delimiter, header=None, names=names)
            return _table(frame, "date", metadata, None, lineno + 1)
        if k + 1 >= len(nonblank):
            break
        next_fields = _split(nonblank[k + 1][1], delimiter)
        date_idx = _date_field_index(next_fields)
        if date_idx is None or date_idx >= len(fields):
            continue
        body = "\n".join(ln for _, ln in nonblank[k:])
        frame = _read(body, delimiter, header=0, names=None)
        date_column = str(frame.columns[date_idx])
        hint = DATE_HEADERS.get(date_column.strip().casefold())
        return _table(frame, date_column, metadata, hint, nonblank[k + 1][0] + 1)
    raise CsvError("Could not find a date column followed by value column(s) in the file.")


def _read(body: str, delimiter: str, header: int | None, names: list[str] | None) -> pd.DataFrame:
    """Read the table body as strings (no NA inference; we parse cells ourselves)."""
    try:
        frame = pd.read_csv(
            io.StringIO(body),
            sep=delimiter,
            header=header,
            names=names,
            dtype=str,
            keep_default_na=False,
            skipinitialspace=True,
        )
    except (pd.errors.ParserError, ValueError) as exc:
        raise CsvError(f"Could not read the table: {exc}") from exc
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.columns.duplicated().any():
        raise CsvError("Column names must be unique.")
    return frame


def _table(
    frame: pd.DataFrame, date_column: str, metadata: list[str], hint: str | None, first: int
) -> CsvTable:
    """Assemble a :class:`CsvTable`, dropping blank-date rows and empty columns."""
    frame = frame[frame[date_column].str.strip() != ""].reset_index(drop=True)
    values = [
        str(c) for c in frame.columns if c != date_column and (frame[c].str.strip() != "").any()
    ]
    if not values:
        raise CsvError("No value column found next to the date column.")
    return CsvTable(frame, date_column, values, metadata, hint, first)


def is_trends_format(table: CsvTable) -> bool:
    """Google Trends download: metadata lines, ``query: (geo)`` headers or ``<1`` cells."""
    if any(line.casefold().startswith("category:") for line in table.metadata):
        return True
    if any(TRENDS_COLUMN.match(c) for c in table.value_columns):
        return True
    return any((table.frame[c].str.strip() == TRENDS_LT1).any() for c in table.value_columns)


def _clean_name(column: str) -> str:
    """Query name from a Trends header (``"chatgpt: (Worldwide)"`` -> ``"chatgpt"``)."""
    match = TRENDS_COLUMN.match(column)
    return match.group("query").strip() if match else column.strip()


def list_value_columns(data: bytes | str) -> list[str]:
    """Value columns available in an upload, for the UI's column picker."""
    return locate_table(decode(data)).value_columns


def choose_column(columns: list[str], wanted: str | None) -> str:
    """Pick the value column: the only one, or the one the user named.

    ``wanted`` matches a header exactly or by its query name (case-insensitive),
    so ``"chatgpt"`` selects ``"chatgpt: (Worldwide)"``.
    """
    if wanted:
        target = wanted.strip().casefold()
        for column in columns:
            if target in (column.casefold(), _clean_name(column).casefold()):
                return column
        raise CsvError(f"Column {wanted!r} not found; available: {', '.join(columns)}")
    if len(columns) == 1:
        return columns[0]
    raise ColumnChoiceRequired(columns)


def parse_values(cells: pd.Series, column: str, lt1_value: float, first: int) -> pd.DataFrame:
    """Parse value cells: blanks -> NaN (gap), ``<1`` -> ``lt1_value`` (imputed).

    Thousands separators (``1,234``) are removed; anything else non-numeric is an
    error naming the line.
    """
    values = np.full(len(cells), np.nan)
    imputed = np.zeros(len(cells), dtype=bool)
    for i, raw in enumerate(cells.tolist()):
        cell = str(raw).strip()
        if cell == "":
            continue
        if cell == TRENDS_LT1:
            values[i], imputed[i] = lt1_value, True
            continue
        if _THOUSANDS.match(cell):
            cell = cell.replace(",", "")
        try:
            values[i] = float(cell)
        except ValueError as exc:
            raise CsvError(f"Line {first + i}: {raw!r} in {column!r} is not a number.") from exc
        if not np.isfinite(values[i]):
            raise CsvError(f"Line {first + i}: {raw!r} in {column!r} is not a finite number.")
    return pd.DataFrame({"value": values, "imputed": imputed})


def infer_freq(ts: pd.Series, date_cells: pd.Series, hint: str | None) -> str:
    """Frequency from the header (Day/Week/Month), ``YYYY-MM`` cells, or date spacing.

    Spacing uses the median gap between consecutive distinct dates: 1 day -> D,
    7 days -> W, 28-31 days -> M.
    """
    if hint:
        return hint
    if date_cells.str.strip().str.fullmatch(r"\d{4}-\d{1,2}").all():
        return "M"
    unique = ts.drop_duplicates().sort_values()
    if len(unique) < 2:
        raise CsvError("Cannot infer the frequency from a single date; choose D, W or M.")
    spacing = float(unique.diff().dropna().dt.days.median())
    for freq, (lo, hi) in _SPACING_DAYS.items():
        if lo <= spacing <= hi:
            return freq
    raise CsvError(f"Dates are {spacing:g} days apart; only daily, weekly or monthly data work.")


def infer_scale(values: pd.Series, trends: bool) -> str:
    """``relative_0_100`` for Trends, ``count`` for non-negative integers, else ``value``."""
    if trends:
        return "relative_0_100"
    observed = values.dropna()
    if len(observed) and (observed >= 0).all() and (observed == np.round(observed)).all():
        return "count"
    return "value"


def parse_csv(
    data: bytes | str,
    *,
    query: str | None = None,
    value_column: str | None = None,
    scale: str | None = None,
    freq: str | None = None,
    upload_date: date | str | None = None,
    last_period_incomplete: bool = False,
    source: str = "csv",
    now: datetime | None = None,
    cfg: Config | None = None,
) -> Series:
    """Parse an uploaded CSV into a :class:`Series` (no preprocessing yet).

    The series uses the CSV partial-period rule: preprocessing drops the last
    period only if it contains ``upload_date`` (default: today, UTC) or the user
    set ``last_period_incomplete``.
    """
    cfg = cfg if cfg is not None else get_config()
    if freq is not None and freq not in FREQS:
        raise CsvError(f"freq must be one of {FREQS}")
    if scale is not None and scale not in SCALES:
        raise CsvError(f"scale must be one of {SCALES}")
    now = now if now is not None else datetime.now(UTC)
    table = locate_table(decode(data))
    trends = is_trends_format(table)
    column = choose_column(table.value_columns, value_column)

    ts = parse_dates(table.frame[table.date_column], table.first_data_line)
    parsed = parse_values(
        table.frame[column],
        column,
        float(cfg["adapters"]["csv"]["trends_lt1_value"]),
        table.first_data_line,
    )
    if parsed["value"].notna().sum() == 0:
        raise CsvError(f"Column {column!r} has no numeric values.")
    points = pd.DataFrame({"ts": ts, "value": parsed["value"], "imputed": parsed["imputed"]})
    freq = freq or infer_freq(ts, table.frame[table.date_column], table.freq_hint)
    scale = scale or infer_scale(points["value"], trends)
    if scale in cfg["preprocess"]["log_scales"] and (points["value"].dropna() < 0).any():
        raise CsvError(f"Scale {scale!r} needs non-negative values.")

    caveats: list[str] = []
    if trends:
        caveats.extend(TRENDS_CAVEATS)
    n_lt1 = int(parsed["imputed"].sum())
    if n_lt1:
        lt1 = cfg["adapters"]["csv"]["trends_lt1_value"]
        noun = "value" if n_lt1 == 1 else "values"
        caveats.append(f"{n_lt1} {noun} reported as '<1' set to {lt1:g} and marked imputed.")

    name = query.strip() if query and query.strip() else _clean_name(column)
    geo_match = TRENDS_COLUMN.match(column)
    upload = upload_date if upload_date is not None else now.astimezone(UTC).date()
    meta: dict[str, Any] = {
        "fetched_at": now.astimezone(UTC).isoformat(),
        "caveats": caveats,
        "resolved_query": name,
        "dropped_partial": None,
        "truncated_before": None,
        "partial_rule": "upload_date",
        "upload_date": upload if isinstance(upload, str) else upload.isoformat(),
        "last_period_incomplete": bool(last_period_incomplete),
        "csv_format": "google_trends" if trends else "generic",
        "value_column": column,
        "available_columns": list(table.value_columns),
        "csv_metadata": list(table.metadata),
    }
    if geo_match:
        meta["geo"] = geo_match.group("geo").strip()
    return Series(source=source, query=name, freq=freq, points=points, scale=scale, meta=meta)


class CsvUploadAdapter:
    """Adapter wrapper: ``params["data"]`` holds the uploaded bytes or text.

    Other recognised params: ``value_column``, ``scale``, ``freq``,
    ``upload_date`` and ``last_period_incomplete`` (see :func:`parse_csv`).
    """

    source = "csv"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg

    def fetch(self, query: str, params: Mapping[str, Any]) -> Series:
        """Parse the uploaded CSV; raise :class:`CsvError` with a readable reason."""
        if "data" not in params:
            raise CsvError("No file uploaded.")
        return parse_csv(
            params["data"],
            query=query or None,
            value_column=params.get("value_column"),
            scale=params.get("scale"),
            freq=params.get("freq"),
            upload_date=params.get("upload_date"),
            last_period_incomplete=bool(params.get("last_period_incomplete", False)),
            cfg=self.cfg,
        )
