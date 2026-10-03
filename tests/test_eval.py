"""Eval harness smoke tests (eval.run_eval, eval/cases.yaml, eval/real_cases.yaml)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from eval import run_eval
from eval.run_eval import CASES_PATH, REAL_CASES_PATH, load_cases, load_real_cases, noise_sweep
from signalcheck.config import Config

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


def test_real_template_is_skipped_until_filled(cfg: Config) -> None:
    raw = yaml.safe_load(REAL_CASES_PATH.read_text())
    entries = raw["cases"]
    assert len(entries) == 10
    cases, skipped = load_real_cases(REAL_CASES_PATH, cfg)
    assert cases == [] and len(skipped) == 10


def test_noise_sweep_is_deterministic() -> None:
    cases = [c for c in load_cases() if c.noise][:2]
    a = [c.build().points for c in noise_sweep(cases, 3)]
    b = [c.build().points for c in noise_sweep(cases, 3)]
    assert len(a) == 6
    for x, y in zip(a, b, strict=True):
        assert x.equals(y)
    assert not a[0].equals(a[1])


def test_run_report_structure(cfg: Config) -> None:
    report = run_eval.run("test", sweep=1, cfg=cfg)
    assert list(report["splits"]) == ["test"]
    s = report["splits"]["test"]
    assert s["n"] == sum(sum(row.values()) for row in s["confusion"].values())
    assert 0.0 <= s["accuracy"] <= 1.0
    assert set(s["recall"]) == set(LABELS)
    assert s["false_trend"]["n"] >= 1
    assert s["noise_sweep"]["n"] == s["false_trend"]["n"]
    for f in s["failures"]:
        assert f["rule"] in {"R1", "R2", "R3", "R4", "R5", "R6"}
    assert report["target_false_trend_rate"] == cfg["eval"]["target_false_trend_rate"]


def test_run_is_deterministic(cfg: Config) -> None:
    assert run_eval.run("test", cfg=cfg) == run_eval.run("test", cfg=cfg)


def test_cli_json_and_text(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_eval.main(["--json", "--split", "test", "--sweep", "0"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert list(report["splits"]) == ["test"]
    assert run_eval.main(["--split", "tune", "--sweep", "0"]) == 0
    text = capsys.readouterr().out
    assert "=== tune:" in text and "accuracy:" in text and "false-TREND rate" in text
