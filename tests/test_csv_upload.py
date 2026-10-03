"""Tests for the CSV upload adapter, including Google Trends download quirks."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from signalcheck.adapters.base import Adapter, AdapterError
from signalcheck.adapters.csv_upload import (
    ColumnChoiceRequired,
    CsvError,
    CsvUploadAdapter,
    list_value_columns,
    parse_csv,
)
from signalcheck.config import Config
from signalcheck.engine.preprocess import preprocess

NOW = datetime(2026, 10, 3, 9, 30, tzinfo=UTC)


def _parse(data: bytes | str, cfg: Config, **kwargs: Any) -> Any:
    return parse_csv(data, now=NOW, cfg=cfg, **kwargs)


def _daily_rows(n: int, start: str = "2026-08-01", sep: str = ",") -> list[str]:
    days = pd.date_range(start, periods=n)
    return [f"{d:%Y-%m-%d}{sep}{i * 3}" for i, d in enumerate(days)]


def _trends_weekly(columns: Mapping[str, list[str]], weeks: int = 30) -> str:
    """A Google Trends 'Interest over time' download (weeks start on Sunday)."""
    sundays = pd.date_range(end="2026-09-27", periods=weeks, freq="W-SUN")
    header = "Week," + ",".join(columns)
    rows = [
        f"{d:%Y-%m-%d}," + ",".join(vals[i] for vals in columns.values())
        for i, d in enumerate(sundays)
    ]
    return "\n".join(["Category: All categories", "", header, *rows]) + "\n"


# --- generic two-column CSVs -------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "sep"),
    [("date,value", ","), ("Date,Count", ","), ("Day;Views", ";"), ("timestamp\tpageviews", "\t")],
)
def test_flexible_headers_and_delimiters(cfg: Config, header: str, sep: str) -> None:
    text = "\n".join([header, *_daily_rows(30, sep=sep)])
    s = _parse(text, cfg)
    assert s.freq == "D" and s.source == "csv"
    assert len(s.points) == 30
    assert s.points["value"].iloc[:3].tolist() == [0.0, 3.0, 6.0]
    assert s.scale == "count"  # non-negative integers
    assert s.meta["csv_format"] == "generic"
    assert s.meta["value_column"] == header.split(sep)[1]


def test_headerless_file(cfg: Config) -> None:
    s = _parse("\n".join(_daily_rows(30)), cfg, query="my metric")
    assert s.query == "my metric" and len(s.points) == 30


def test_value_before_date_column(cfg: Config) -> None:
    text = "views,day\n" + "\n".join(f"{i},2026-08-{i + 1:02d}" for i in range(30))
    s = _parse(text, cfg)
    assert s.points["ts"].iloc[0] == pd.Timestamp("2026-08-01")
    assert s.points["value"].iloc[29] == 29.0
    assert s.query == "views"


def test_blank_values_become_gaps_and_thousands_parse(cfg: Config) -> None:
    text = 'date,views\n2026-08-01,"1,234"\n2026-08-02,\n2026-08-03,1500.5\n'
    s = _parse(text, cfg)
    assert s.points["value"].iloc[0] == 1234.0
    assert np.isnan(s.points["value"].iloc[1])
    assert s.scale == "value"  # non-integer values


def test_scale_and_freq_overrides(cfg: Config) -> None:
    s = _parse("\n".join(["d,v", *_daily_rows(30)]), cfg, scale="pageviews", freq="D")
    assert s.scale == "pageviews"
    with pytest.raises(CsvError, match="non-negative"):
        _parse("date,v\n2026-08-01,-1\n2026-08-02,3\n", cfg, scale="count")
    with pytest.raises(CsvError, match="freq"):
        _parse("date,v\n2026-08-01,1\n", cfg, freq="H")
    with pytest.raises(CsvError, match="scale"):
        _parse("date,v\n2026-08-01,1\n", cfg, scale="bogus")


@pytest.mark.parametrize(
    ("dates", "freq"),
    [
        (pd.date_range("2026-01-05", periods=30, freq="W-MON"), "W"),
        (pd.date_range("2024-01-01", periods=30, freq="MS"), "M"),
        (pd.date_range("2026-01-01", periods=30, freq="D"), "D"),
    ],
)
def test_freq_inferred_from_spacing(cfg: Config, dates: pd.DatetimeIndex, freq: str) -> None:
    text = "when,amount\n" + "\n".join(f"{d:%Y-%m-%d},{i}" for i, d in enumerate(dates))
    assert _parse(text, cfg).freq == freq


def test_irregular_spacing_rejected(cfg: Config) -> None:
    dates = pd.date_range("2026-01-01", periods=10, freq="3D")
    text = "date,value\n" + "\n".join(f"{d:%Y-%m-%d},1" for d in dates)
    with pytest.raises(CsvError, match="3 days apart"):
        _parse(text, cfg)


def test_single_row_needs_freq(cfg: Config) -> None:
    with pytest.raises(CsvError, match="single date"):
        _parse("date,value\n2026-08-01,1\n", cfg)
    assert _parse("date,value\n2026-08-01,1\n", cfg, freq="D").freq == "D"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("date,value\n2026-08-01,1\n03/04/2026,2\n", r"Line 3: '03/04/2026' is not a date"),
        (
            "date,value\n2026-08-01,1\n2026-08-02,lots\n",
            r"Line 3: 'lots' in 'value' is not a number",
        ),
        ("date,value\n2026-08-01,1\n2026-08-02,inf\n", "not a finite number"),
        ("just one column\n2026-08-01\n", "Could not find a date column"),
        ("date,value\n2026-08-01,\n2026-08-02,\n", "No value column"),
    ],
)
def test_bad_files_raise_readable_errors(cfg: Config, text: str, message: str) -> None:
    with pytest.raises(CsvError, match=message):
        _parse(text, cfg)


def test_slash_iso_dates_and_times(cfg: Config) -> None:
    text = "date,value\n2026/08/01,1\n2026-08-02T00:00:00Z,2\n2026-08-03 00:00,3\n"
    s = _parse(text, cfg)
    assert s.points["ts"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-08-01",
        "2026-08-02",
        "2026-08-03",
    ]


# --- Google Trends downloads -------------------------------------------------------------


def test_trends_single_query(cfg: Config) -> None:
    values = [str(v) for v in range(30)]
    values[3] = "<1"
    values[4] = "<1"
    text = _trends_weekly({"perplexity ai: (Worldwide)": values})
    s = _parse(text, cfg)
    assert s.query == "perplexity ai"
    assert s.meta["geo"] == "Worldwide"
    assert s.meta["csv_format"] == "google_trends"
    assert s.meta["csv_metadata"] == ["Category: All categories"]
    assert (s.freq, s.scale) == ("W", "relative_0_100")
    assert s.points.loc[[3, 4], "value"].tolist() == [0.5, 0.5]
    assert s.points["imputed"].tolist() == [i in (3, 4) for i in range(30)]
    caveats = s.meta["caveats"]
    assert "Google Trends values are on a relative 0-100 scale, not search volume." in caveats
    assert "Google samples this data; small moves can be sampling noise." in caveats
    assert "2 values reported as '<1' set to 0.5 and marked imputed." in caveats


def test_trends_lt1_value_from_config(cfg: Config) -> None:
    cfg["adapters"]["csv"]["trends_lt1_value"] = 0.25
    text = _trends_weekly({"x: (US)": ["<1"] + ["5"] * 29})
    assert _parse(text, cfg).points["value"].iloc[0] == 0.25


def test_trends_multiple_queries_require_choice(cfg: Config) -> None:
    cols = {
        "chatgpt: (Worldwide)": [str(50 + i) for i in range(30)],
        "perplexity ai: (Worldwide)": [str(i) for i in range(30)],
    }
    text = _trends_weekly(cols)
    assert list_value_columns(text) == list(cols)
    with pytest.raises(ColumnChoiceRequired) as exc_info:
        _parse(text, cfg)
    assert exc_info.value.columns == list(cols)
    by_name = _parse(text, cfg, value_column="Perplexity AI")
    by_header = _parse(text, cfg, value_column="chatgpt: (Worldwide)")
    assert by_name.query == "perplexity ai" and by_name.points["value"].iloc[-1] == 29
    assert by_header.query == "chatgpt" and by_header.points["value"].iloc[0] == 50
    assert by_name.meta["available_columns"] == list(cols)
    with pytest.raises(CsvError, match="not found"):
        _parse(text, cfg, value_column="gemini")


def test_trends_daily_and_monthly_headers(cfg: Config) -> None:
    days = pd.date_range("2026-07-01", periods=40)
    daily = "Category: All categories\n\nDay,claude: (United Kingdom)\n" + "\n".join(
        f"{d:%Y-%m-%d},{i % 100}" for i, d in enumerate(days)
    )
    s = _parse(daily, cfg)
    assert (s.freq, s.meta["geo"]) == ("D", "United Kingdom")
    months = pd.date_range("2004-01-01", periods=30, freq="MS")
    monthly = "Category: All categories\n\nMonth,rust: (Worldwide)\n" + "\n".join(
        f"{d:%Y-%m},{i}" for i, d in enumerate(months)
    )
    m = _parse(monthly, cfg)
    assert m.freq == "M"
    assert m.points["ts"].iloc[1] == pd.Timestamp("2004-02-01")


def test_trends_metadata_with_commas_and_week_ranges(cfg: Config) -> None:
    sundays = pd.date_range("2026-03-01", periods=30, freq="W-SUN")
    rows = [
        f"{d:%Y-%m-%d} - {d + pd.Timedelta(days=6):%Y-%m-%d},{i}" for i, d in enumerate(sundays)
    ]
    text = "Category: Arts, Entertainment\n\nWeek,film: (Worldwide)\n" + "\n".join(rows)
    s = _parse(text, cfg)
    assert s.freq == "W"
    assert s.points["ts"].iloc[0] == pd.Timestamp("2026-03-01")


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16"])
def test_trends_encodings(cfg: Config, encoding: str) -> None:
    text = _trends_weekly({"café: (France)": [str(i) for i in range(30)]})
    s = _parse(text.encode(encoding), cfg)
    assert s.query == "café" and len(s.points) == 30


# --- partial period (CSV rule) end to end -----------------------------------------------------


def test_csv_partial_rule_drops_period_containing_upload_date(cfg: Config) -> None:
    # Last Trends week starts Sun 27 Sep 2026 and contains the upload date (3 Oct).
    text = _trends_weekly({"q: (Worldwide)": [str(i) for i in range(30)]})
    pre = preprocess(_parse(text, cfg), cfg)
    assert pre.series.meta["dropped_partial"] == {"ts": "2026-09-27", "value": 29.0}
    assert len(pre.series.points) == 29


def test_csv_partial_rule_keeps_complete_last_period(cfg: Config) -> None:
    text = "\n".join(["date,value", *_daily_rows(30)])  # ends 30 Aug
    s = _parse(text, cfg)
    assert s.meta["upload_date"] == "2026-10-03"
    assert preprocess(s, cfg).series.meta["dropped_partial"] is None
    flagged = _parse(text, cfg, last_period_incomplete=True)
    assert preprocess(flagged, cfg).series.meta["dropped_partial"] == {
        "ts": "2026-08-30",
        "value": 87.0,
    }


def test_csv_partial_rule_custom_upload_date(cfg: Config) -> None:
    text = "\n".join(["date,value", *_daily_rows(30)])
    s = _parse(text, cfg, upload_date="2026-08-30")
    assert preprocess(s, cfg).series.meta["dropped_partial"] is not None


# --- adapter wrapper ---------------------------------------------------------------------------


def test_adapter_protocol_and_fetch(cfg: Config) -> None:
    adapter = CsvUploadAdapter(cfg)
    assert isinstance(adapter, Adapter)
    data = "\n".join(["date,value", *_daily_rows(30)]).encode()
    s = adapter.fetch("signups", {"data": data, "scale": "count"})
    assert s.query == "signups" and s.scale == "count"
    with pytest.raises(AdapterError, match="No file uploaded"):
        adapter.fetch("x", {})
