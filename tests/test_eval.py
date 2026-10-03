"""Eval harness smoke tests (eval.run_eval, eval/cases.yaml, eval/real_cases.yaml)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import yaml

from eval import run_eval
from eval.run_eval import (
    CASES_PATH,
    REAL_CASES_PATH,
    load_cases,
    load_real_cases,
    noise_sweep,
    truncate_as_of,
)
from signalcheck.config import Config
from signalcheck.models import Series

LABELS = ("TREND", "FLUKE", "SEASONAL", "NO_CHANGE", "INCONCLUSIVE")


def test_cases_are_stratified_about_70_30() -> None:
    cases = load_cases()
    assert 35 <= len(cases) <= 45
    for label in LABELS:
        of_label = [c for c in cases if c.label == label]
        n_test = sum(c.split == "test" for c in of_label)
        assert n_test >= 1, label
        assert n_test <= len(of_label) - 1, label
    share = sum(c.split == "test" for c in cases) / len(cases)
    assert 0.25 <= share <= 0.4
    assert any(c.noise for c in cases if c.split == "tune")
    assert any(c.noise for c in cases if c.split == "test")
    assert all(c.label == "NO_CHANGE" for c in cases if c.noise)


def test_cases_vary_frequency_and_direction() -> None:
    raw = yaml.safe_load(CASES_PATH.read_text())["cases"]
    freqs = Counter((e.get("params") or {}).get("freq", "default") for e in raw)
    assert {"D", "W", "M"} <= set(freqs)
    dirs = Counter(e.get("direction") for e in raw if e["label"] == "TREND")
    assert dirs["up"] >= 2 and dirs["down"] >= 2


def test_load_cases_rejects_bad_entries(tmp_path: Path) -> None:
    bad = tmp_path / "cases.yaml"
    bad.write_text(
        "cases:\n  - {id: a, label: TREND, split: tune, generator: flat_noise, seed: 1}\n"
    )
    with pytest.raises(ValueError, match="direction"):
        load_cases(bad)
    bad.write_text(
        "cases:\n  - {id: a, label: NO_CHANGE, split: dev, generator: flat_noise, seed: 1}\n"
    )
    with pytest.raises(ValueError, match="split"):
        load_cases(bad)
    bad.write_text("cases:\n  - {id: a, label: NO_CHANGE, split: tune, generator: nope, seed: 1}\n")
    with pytest.raises(ValueError, match="generator"):
        load_cases(bad)


def test_real_cases_load_from_snapshots_cut_at_as_of(tmp_path: Path) -> None:
    cases, skipped = load_real_cases(REAL_CASES_PATH)
    raw = {e["id"]: e for e in yaml.safe_load(REAL_CASES_PATH.read_text())["cases"]}
    assert skipped == [] and len(cases) == len(raw)
    assert {c.split for c in cases} == {"tune", "test"}
    for case in cases:
        series = case.build()
        assert case.kind == "real"
        last = pd.Timestamp(series.points["ts"].iloc[-1])
        as_of = pd.Timestamp(raw[case.id]["as_of"])
        span = pd.Timedelta(days=6 if series.freq == "W" else 0)
        assert last + span <= as_of < last + span + pd.Timedelta(days=7 if span.days else 1)
        assert series.meta["as_of"] == raw[case.id]["as_of"]
    _, missing = load_real_cases(REAL_CASES_PATH, tmp_path)
    assert len(missing) == len(raw)


def test_truncate_as_of_weekly_keeps_whole_weeks() -> None:
    ts = pd.date_range("2024-01-01", periods=4, freq="7D")
    points = pd.DataFrame({"ts": ts, "value": [1.0, 2.0, 3.0, 4.0]})
    series = Series("wikipedia", "q", "W", points, "pageviews", {"fetched_at": "x", "caveats": []})
    # The week starting 2024-01-15 ends on 2024-01-21, after as_of.
    cut = truncate_as_of(series, "2024-01-20")
    assert cut.points["value"].tolist() == [1.0, 2.0]
    daily = Series("wikipedia", "q", "D", points, "pageviews", {"fetched_at": "x", "caveats": []})
    assert truncate_as_of(daily, "2024-01-15").points["value"].tolist() == [1.0, 2.0, 3.0]


def test_noise_sweep_is_deterministic() -> None:
    cases = [c for c in load_cases() if c.noise][:2]
    a = [c.build().points for c in noise_sweep(cases, 3)]
    b = [c.build().points for c in noise_sweep(cases, 3)]
    assert len(a) == 6
    for x, y in zip(a, b, strict=True):
        assert x.equals(y)
    assert not a[0].equals(a[1])


def test_run_report_structure(cfg: Config, tmp_path: Path) -> None:
    report = run_eval.run("tune", sweep=1, cfg=cfg)
    assert list(report["splits"]) == ["tune"]
    kinds = report["splits"]["tune"]
    assert set(kinds) == {"synthetic", "real", "combined"}
    assert kinds["combined"]["n"] == kinds["synthetic"]["n"] + kinds["real"]["n"]
    assert kinds["real"]["n"] >= 1 and len(kinds["real"]["cases"]) == kinds["real"]["n"]
    assert kinds["real"]["false_trend"]["n"] == 0
    s = kinds["combined"]
    assert s["n"] == sum(sum(row.values()) for row in s["confusion"].values())
    assert 0.0 <= s["accuracy"] <= 1.0
    assert set(s["recall"]) == set(LABELS)
    assert s["false_trend"]["n"] >= 1
    assert s["noise_sweep"]["n"] == s["false_trend"]["n"]
    for f in s["failures"]:
        assert f["rule"] in {"R1", "R2", "R3", "R4", "R5", "R6"}
    assert report["target_false_trend_rate"] == cfg["eval"]["target_false_trend_rate"]
    json_path, md_path = run_eval.save(report, "smoke", tmp_path)
    assert json.loads(json_path.read_text()) == report
    md = md_path.read_text()
    assert "| tune | real |" in md and "## Real cases (tune)" in md


def test_run_is_deterministic(cfg: Config) -> None:
    assert run_eval.run("tune", cfg=cfg) == run_eval.run("tune", cfg=cfg)


def test_cli_json_and_text(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_eval.main(["--json", "--split", "tune", "--sweep", "0"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert list(report["splits"]) == ["tune"]
    assert run_eval.main(["--split", "tune", "--sweep", "0"]) == 0
    text = capsys.readouterr().out
    assert (
        "=== tune / synthetic:" in text
        and "=== tune / real:" in text
        and "accuracy:" in text
        and "false-TREND rate" in text
    )
