"""Static-site JSON export: strict JSON that mirrors the Streamlit cards and charts."""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from signalcheck.config import Config
from signalcheck.ui import runner
from signalcheck.web.export import (
    LIVE_SOURCES,
    SCHEMA_VERSION,
    STATIC_UNAVAILABLE,
    csv_json,
    jsonable,
    result_json,
    site_json,
    with_static_cards,
)

WHEN = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)


def test_jsonable_is_strict() -> None:
    value = {
        "nan": float("nan"),
        "inf": np.float64(math.inf),
        "i": np.int64(3),
        "b": np.bool_(True),
        "ts": pd.Timestamp("2026-01-02 13:00"),
        "dt": datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
        "d": date(2026, 1, 2),
        "t": (1, 2.5),
        "nat": pd.NaT,
        1: "int key",
    }
    out = jsonable(value)
    assert out == {
        "nan": None,
        "inf": None,
        "i": 3,
        "b": True,
        "ts": "2026-01-02",
        "dt": "2026-01-02T03:04:00+00:00",
        "d": "2026-01-02",
        "t": [1, 2.5],
        "nat": None,
        "1": "int key",
    }
    json.dumps(out, allow_nan=False)


@pytest.fixture(scope="module")
def sample() -> dict[str, Any]:
    from signalcheck.config import get_config

    cfg = get_config()
    sources = runner.sample_sources("chatgpt", cfg)
    result = runner.run_topic("chatgpt", sources, cfg, sample=True, max_workers=1)
    return result_json(with_static_cards(result, cfg, sources), cfg, generated_at=WHEN)


def test_sample_result_shape(sample: dict[str, Any]) -> None:
    json.dumps(sample, allow_nan=False)
    assert sample["schema"] == SCHEMA_VERSION and sample["kind"] == "topic"
    assert sample["query"] == "chatgpt" and sample["sample"] is True
    assert sample["generated_at"] == "2026-10-03T12:00:00+00:00"
    sources = [c["source"] for c in sample["cards"]]
    assert sources == ["wikipedia", "hackernews", *STATIC_UNAVAILABLE]


def test_static_cards_explain_why(sample: dict[str, Any]) -> None:
    cards = {c["source"]: c for c in sample["cards"]}
    for source, reason in STATIC_UNAVAILABLE.items():
        card = cards[source]
        assert card["disabled"] is True and card["chart"] is None and card["badge"] is None
        assert reason in card["message"]
    # Disabled sources are not "unavailable" (failed) in the cross-source summary.
    assert sample["summary"]["unavailable"] == []


def test_analysed_cards_carry_verdict_narration_and_chart(sample: dict[str, Any]) -> None:
    for card in sample["cards"][:2]:
        assert card["badge"]["color"] in {"green", "red", "orange", "violet", "blue", "gray"}
        assert card["rule"] and card["reason"] and card["evidence"]
        assert card["narration"]["label"] == "Summary (template)"
        chart = card["chart"]
        assert chart["freq"] == "D" and chart["observed"]
        day, value = chart["observed"][0]
        assert len(day) == 10 and (value is None or isinstance(value, float))
        kinds = {"spans", "bands", "points", "segments", "lines", "vlines"}
        assert chart["annotations"] and all(set(a) <= kinds for a in chart["annotations"])
    chips = sample["summary"]["chips"]
    assert {c["source"] for c in chips} <= set(LIVE_SOURCES)
    assert all(c["badge"]["key"] for c in chips)


def test_csv_json_columns_and_result(cfg: Config) -> None:
    rows = ["date,a,b"] + [f"2026-01-{d:02d},{d},{100 - d}" for d in range(1, 29)]
    data = ("\n".join(rows) + "\n").encode()
    ask = csv_json(runner.run_csv(data, "x.csv", cfg), cfg)
    assert ask == {"schema": SCHEMA_VERSION, "kind": "columns", "columns": ["a", "b"]}
    done = csv_json(runner.run_csv(data, "x.csv", cfg, value_column="a"), cfg)
    assert done["kind"] == "csv" and len(done["cards"]) == 1
    assert done["cards"][0]["chart"]["observed"][0] == ["2026-01-01", 1.0]


def test_site_json_reads_the_shipped_config(cfg: Config) -> None:
    cfg["ui"]["threshold_headroom"] = 2.0
    site = site_json(cfg, [{"query": "chatgpt", "slug": "chatgpt", "sources": ["wikipedia"]}])
    json.dumps(site, allow_nan=False)
    titles = [s["title"] for s in site["methodology"]]
    assert titles and all(s["body"] for s in site["methodology"])
    assert set(site["rules"]) == {"R1", "R2", "R3", "R4", "R5", "R6"}
    assert site["ui"]["threshold_headroom"] == 2.0
    assert site["ui"]["template_label"] == "Summary (template)"
    live = {s["source"]: s for s in site["sources"]}
    assert live["wikipedia"]["live"] and live["hackernews"]["live"]
    assert not live["reddit"]["live"] and "server-side" in live["reddit"]["reason"]
    assert site["labels"]["TREND_up"]["color"] == "green"
    assert site["labels"]["FLUKE"]["color"] == "orange"
    body = "\n".join(s["body"] for s in site["methodology"])
    assert "R6" in body
