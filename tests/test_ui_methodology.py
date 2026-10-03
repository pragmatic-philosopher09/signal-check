"""Methodology text is generated from the live config (signalcheck.ui.methodology)."""

from __future__ import annotations

from signalcheck.config import Config
from signalcheck.engine.checks import CHECKS
from signalcheck.engine.verdict import RULES
from signalcheck.ui.methodology import format_value, methodology_sections


def body(cfg: Config, title: str) -> str:
    return next(s.body for s in methodology_sections(cfg) if s.title == title)


def test_sections_in_reading_order(cfg: Config) -> None:
    assert [s.title for s in methodology_sections(cfg)] == [
        "Preprocessing",
        "Checks",
        "Decision table",
        "Confidence",
        "What would change my mind",
        "Cross-source summary",
    ]


def test_every_check_threshold_is_listed(cfg: Config) -> None:
    text = body(cfg, "Checks")
    for check in CHECKS:
        for key, value in cfg["checks"][check].items():
            assert f"`{key}` = {format_value(value)}" in text, (check, key)


def test_thresholds_are_read_live(cfg: Config) -> None:
    cfg["checks"]["outliers"]["outlier_z"] = 7.25
    cfg["confidence"]["conf_high"] = 5
    assert "`outlier_z` = 7.25" in body(cfg, "Checks")
    assert "`conf_high` = 5" in body(cfg, "Confidence")


def test_decision_table_has_every_rule(cfg: Config) -> None:
    text = body(cfg, "Decision table")
    for rule in RULES:
        assert f"| {rule} |" in text
    assert "->" not in text


def test_confidence_formula_from_engine_docstring(cfg: Config) -> None:
    text = body(cfg, "Confidence")
    assert "score" in text and "```text" in text
    assert "| TREND |" in text


def test_format_value() -> None:
    assert format_value({"D": 28, "W": 26}) == "D 28 \u00b7 W 26"
    assert format_value(True) == "yes"
    assert format_value(0.5) == "0.5"
