"""Tests for signalcheck.config: loading, fail-fast validation and get_secret."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from signalcheck import config as cfg_mod
from signalcheck.config import (
    APP_NAME,
    DEFAULT_CONFIG_PATH,
    ConfigError,
    get_config,
    get_secret,
    load_config,
    validate_config,
)


@pytest.fixture
def raw_config() -> dict[str, Any]:
    """A fresh, mutable copy of the repo's config.yaml."""
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return copy.deepcopy(data)


def _write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_app_name() -> None:
    assert APP_NAME == "Signal Check"


def test_default_config_loads_and_is_valid() -> None:
    cfg = load_config()
    assert cfg["preprocess"]["min_history"] == {"D": 28, "W": 26, "M": 24}
    assert cfg["preprocess"]["recent_frac"] == 0.2
    assert cfg["checks"]["low_count"]["low_count_threshold"] == 5
    assert cfg["eval"]["target_false_trend_rate"] == 0.05
    assert get_config() == cfg


def test_missing_top_level_key_fails_fast(tmp_path: Path, raw_config: dict[str, Any]) -> None:
    del raw_config["verdict"]
    with pytest.raises(ConfigError, match=r"verdict: missing"):
        load_config(_write(tmp_path, raw_config))


def test_missing_nested_keys_are_all_reported(raw_config: dict[str, Any]) -> None:
    del raw_config["checks"]["persistence"]["mk_alpha"]
    del raw_config["preprocess"]["min_history"]["W"]
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw_config)
    message = str(exc_info.value)
    assert "checks.persistence.mk_alpha: missing" in message
    assert "preprocess.min_history.W: missing" in message


def test_wrong_type_fails(raw_config: dict[str, Any]) -> None:
    raw_config["checks"]["outliers"]["outlier_z"] = "high"
    raw_config["preprocess"]["recent_min"]["D"] = True
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw_config)
    message = str(exc_info.value)
    assert "checks.outliers.outlier_z: expected int or float" in message
    assert "preprocess.recent_min.D: expected int" in message


def test_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Cannot read config file"):
        load_config(tmp_path / "nope.yaml")


def test_empty_file_fails(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_config(path)


def test_get_secret_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNALCHECK_TEST_SECRET", "s3cret")
    assert get_secret("SIGNALCHECK_TEST_SECRET") == "s3cret"


def test_get_secret_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIGNALCHECK_TEST_SECRET", raising=False)
    monkeypatch.delitem(sys.modules, "streamlit", raising=False)
    assert get_secret("SIGNALCHECK_TEST_SECRET") is None
    assert get_secret("SIGNALCHECK_TEST_SECRET", "fallback") == "fallback"


def test_get_secret_falls_back_to_streamlit_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIGNALCHECK_TEST_SECRET", raising=False)
    fake_st = SimpleNamespace(secrets={"SIGNALCHECK_TEST_SECRET": "from-st"})
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    assert get_secret("SIGNALCHECK_TEST_SECRET") == "from-st"


def test_get_secret_env_wins_over_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNALCHECK_TEST_SECRET", "from-env")
    fake_st = SimpleNamespace(secrets={"SIGNALCHECK_TEST_SECRET": "from-st"})
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    assert get_secret("SIGNALCHECK_TEST_SECRET") == "from-env"


def test_get_secret_tolerates_broken_streamlit_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenSecrets:
        def __contains__(self, key: object) -> bool:
            raise FileNotFoundError("no secrets.toml")

    monkeypatch.delenv("SIGNALCHECK_TEST_SECRET", raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets=BrokenSecrets()))
    assert get_secret("SIGNALCHECK_TEST_SECRET", "dflt") == "dflt"


def test_get_secret_loads_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SIGNALCHECK_DOTENV_SECRET=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("SIGNALCHECK_DOTENV_SECRET", raising=False)
    monkeypatch.setattr(
        cfg_mod, "_load_dotenv_once", lambda: cfg_mod.load_dotenv(env_file, override=False)
    )
    try:
        assert get_secret("SIGNALCHECK_DOTENV_SECRET") == "from-dotenv"
    finally:
        os.environ.pop("SIGNALCHECK_DOTENV_SECRET", None)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["preprocess"].update(recent_frac=1.0), "recent_frac: must be in"),
        (
            lambda c: c["preprocess"]["recent_min"].update(D=30),
            "recent_min.D: must be <= recent_max",
        ),
        (lambda c: c["preprocess"]["recent_min"].update(M=0), "recent_min.M: must be >= 1"),
        (lambda c: c["preprocess"].update(recent_frac=0.99), "leaves no baseline"),
        (lambda c: c["preprocess"].update(mad_floor=0), "mad_floor: must be > 0"),
        (lambda c: c["preprocess"].update(log_scales=["count", "bytes"]), "unknown scale"),
        (lambda c: c["http"].update(read_timeout_s=0), "read_timeout_s: must be > 0"),
        (lambda c: c["http"].update(backoff_max_s=0.1), "backoff_max_s: must be >="),
        (lambda c: c["cache"].update(ttl_hours=0), "ttl_hours: must be > 0"),
        (lambda c: c["adapters"]["csv"].update(trends_lt1_value=1), "trends_lt1_value"),
    ],
)
def test_out_of_range_values_fail(raw_config: dict[str, Any], mutate: Any, message: str) -> None:
    mutate(raw_config)
    with pytest.raises(ConfigError, match=message):
        validate_config(raw_config)


def test_phase1_keys_present() -> None:
    cfg = load_config()
    assert cfg["preprocess"]["log_scales"] == ["count", "pageviews"]
    assert cfg["adapters"]["csv"]["trends_lt1_value"] == 0.5


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["checks"]["persistence"].update(mk_alpha=1.0), "mk_alpha: must be in"),
        (lambda c: c["checks"]["persistence"].update(min_points=2), "min_points: must be >= 3"),
        (
            lambda c: c["checks"]["persistence"].update(slope_level_floor=0),
            "slope_level_floor: must be > 0",
        ),
        (lambda c: c["checks"]["concentration"].update(conc_top1_max=1.5), "conc_top1_max"),
        (
            lambda c: c["checks"]["seasonality"]["annual_min_points"].update(W=60),
            "annual_min_points.W: must cover two annual periods",
        ),
        (
            lambda c: c["checks"]["seasonality"]["annual_min_points"].update(D=400),
            "annual_min_points.D: must cover two annual periods",
        ),
        (
            lambda c: c["checks"]["outliers"].update(isolated_max_points=0),
            "isolated_max_points: must be >= 1",
        ),
        (lambda c: c["checks"]["level_shift"].update(hold_fraction=0.5), "hold_fraction"),
        (
            lambda c: c["checks"]["level_shift"]["recent_tolerance"].update(D=-1),
            "recent_tolerance.D: must be >= 0",
        ),
        (lambda c: c["checks"]["low_count"].update(ci_level=1.0), "ci_level: must be in"),
        (lambda c: c["display"].update(p_decimals=0), "p_decimals"),
        (
            lambda c: c["checks"]["seasonality"].update(stl_seasonal_deg=2),
            "stl_seasonal_deg: must be 0 or 1",
        ),
    ],
)
def test_check_thresholds_out_of_range_fail(
    raw_config: dict[str, Any], mutate: Any, message: str
) -> None:
    mutate(raw_config)
    with pytest.raises(ConfigError, match=message):
        validate_config(raw_config)


def test_missing_phase2_keys_fail(raw_config: dict[str, Any]) -> None:
    del raw_config["checks"]["outliers"]["isolated_max_points"]
    del raw_config["display"]
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw_config)
    assert "isolated_max_points" in str(exc_info.value)
    assert "display" in str(exc_info.value)


def test_phase2_keys_present() -> None:
    cfg = load_config()
    assert cfg["checks"]["level_shift"]["recent_tolerance"] == {"D": 7, "W": 3, "M": 2}
    assert cfg["display"]["p_decimals"] == 3


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda c: c["verdict"].update(seasonal_explained_min=0),
            "seasonal_explained_min: must be in",
        ),
        (lambda c: c["confidence"].update(conf_high=0), "conf_high: must be >= "),
        (lambda c: c["confidence"].update(history_multiple=0.5), "history_multiple: must be >= 1"),
        (lambda c: c["change_my_mind"].update(exceed_mads=0), "exceed_mads: must be > 0"),
        (lambda c: c["change_my_mind"].update(fallback_mads=3), "fallback_mads: must be <="),
        (lambda c: c["change_my_mind"].update(max_conditions=0), "max_conditions: must be >= 1"),
        (lambda c: c["eval"].update(target_false_trend_rate=1.0), "target_false_trend_rate"),
        (lambda c: c["cross_source"].update(min_overlap=0), "min_overlap: must be in"),
    ],
)
def test_verdict_settings_out_of_range_fail(
    raw_config: dict[str, Any], mutate: Any, message: str
) -> None:
    mutate(raw_config)
    with pytest.raises(ConfigError, match=message):
        validate_config(raw_config)


def test_phase3_keys_present() -> None:
    cfg = load_config()
    assert cfg["verdict"]["fluke_from_isolated_outliers"] is True
    assert cfg["confidence"]["conf_high"] >= cfg["confidence"]["conf_medium"]
    assert cfg["change_my_mind"]["max_conditions"] == 3
    assert cfg["cross_source"]["min_overlap"] == 0.5


def test_fluke_toggle_must_be_bool(raw_config: dict[str, Any]) -> None:
    raw_config["verdict"]["fluke_from_isolated_outliers"] = "yes"
    with pytest.raises(ConfigError, match="fluke_from_isolated_outliers"):
        validate_config(raw_config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["adapters"]["wikipedia"].update(history_days=0), "history_days: must be >= 1"),
        (lambda c: c["adapters"]["hackernews"].update(max_hits_per_day=5000), "max_hits_per_day"),
        (lambda c: c["adapters"]["reddit"].update(page_limit=101), "page_limit: must be in"),
        (lambda c: c["adapters"]["reddit"].update(min_interval_s=-1), "min_interval_s"),
        (
            lambda c: c["adapters"]["google_trends"].update(daily_max_days=4000),
            "daily_max_days",
        ),
        (lambda c: c["adapters"]["x"].update(counts_endpoint="full"), "counts_endpoint"),
        (lambda c: c["adapters"]["x"].update(cost_counts_all_usd=-0.01), "cost_counts_all_usd"),
        (lambda c: c["adapters"]["x"].update(breadth_posts_per_day=500), "breadth_posts_per_day"),
        (lambda c: c.update(watchlist=[]), "watchlist: must list at least one topic"),
        (lambda c: c["refresh"].update(max_history_days=0), "refresh.max_history_days"),
        (lambda c: c["refresh"].update(manifest=""), "refresh.manifest"),
        (lambda c: c["ui"].update(timeframe_days=[]), "ui.timeframe_days"),
        (lambda c: c["ui"].update(timeframe_days=[90, 0]), "ui.timeframe_days"),
        (lambda c: c["ui"].update(chart_height_px=50), "ui.chart_height_px"),
        (lambda c: c["ui"].update(threshold_headroom=0.5), "ui.threshold_headroom"),
    ],
)
def test_adapter_settings_out_of_range_fail(
    raw_config: dict[str, Any], mutate: Any, message: str
) -> None:
    mutate(raw_config)
    with pytest.raises(ConfigError, match=message):
        validate_config(raw_config)


def test_phase4_keys_present() -> None:
    cfg = load_config()
    assert cfg["adapters"]["wikipedia"]["agent"] == "user"
    assert cfg["adapters"]["reddit"]["result_cap"] == 1000
    assert cfg["adapters"]["x"]["counts_endpoint"] == "all"
    assert cfg["watchlist"]


def test_secret_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg_mod, "_streamlit_secret", lambda _name: None)
    monkeypatch.delenv("SC_TEST_FLAG", raising=False)
    assert cfg_mod.secret_flag("SC_TEST_FLAG") is None
    for raw, expected in (("true", True), ("ON", True), ("1", True), ("false", False)):
        monkeypatch.setenv("SC_TEST_FLAG", raw)
        assert cfg_mod.secret_flag("SC_TEST_FLAG") is expected


def test_resolve_path_is_repo_relative(tmp_path: Path) -> None:
    assert cfg_mod.resolve_path("data/samples") == DEFAULT_CONFIG_PATH.parent / "data/samples"
    assert cfg_mod.resolve_path(tmp_path) == tmp_path


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["narration"].update(max_words=121), "narration.max_words"),
        (lambda c: c["narration"].update(max_words=5), "narration.max_words"),
        (lambda c: c["narration"].update(max_tokens=50), "narration.max_tokens"),
        (lambda c: c["narration"].update(max_tokens_param="tokens"), "max_tokens_param"),
        (lambda c: c["narration"].update(default_base_url="api.openai.com"), "default_base_url"),
        (lambda c: c["narration"].update(default_model=" "), "default_model"),
        (lambda c: c["narration"].update(connect_timeout_s=0), "connect_timeout_s"),
        (lambda c: c["narration"].update(read_timeout_s=-1), "read_timeout_s"),
        (lambda c: c["narration"].update(max_retries=-1), "narration.max_retries"),
        (lambda c: c["narration"].update(total_timeout_s=10), "total_timeout_s"),
        (lambda c: c["narration"].update(max_workers=0), "narration.max_workers"),
    ],
)
def test_narration_settings_out_of_range_fail(
    raw_config: dict[str, Any], mutate: Any, message: str
) -> None:
    mutate(raw_config)
    with pytest.raises(ConfigError, match=message):
        validate_config(raw_config)


def test_narration_keys_present() -> None:
    n = load_config()["narration"]
    assert n["max_words"] == 120
    assert n["max_tokens"] >= n["max_words"]
    assert n["default_base_url"] == "https://api.openai.com/v1"


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("chatgpt", r"watchlist\[0\]: expected a mapping"),
        ({"wikipedia": True}, r"watchlist\[0\]\.topic: must be a non-empty string"),
        ({"topic": "  ", "wikipedia": True}, r"\.topic: must be a non-empty string"),
        ({"topic": "!!", "wikipedia": True}, "must contain a letter or digit"),
        ({"topic": "x", "wikipedia": ""}, "wikipedia: must be true, false or an article title"),
        ({"topic": "x", "wikipedia": 3}, "wikipedia: must be true, false or an article title"),
        ({"topic": "x", "hackernews": "yes"}, "hackernews: must be a boolean"),
        ({"topic": "x", "reddit": 1}, "reddit: must be a boolean"),
        ({"topic": "x", "hackernew": True}, r"unknown key\(s\) \['hackernew'\]"),
        ({"topic": "x", "x": True}, r"unknown key\(s\) \['x'\]"),
        ({"topic": "x", "wikipedia": False}, "enable at least one of"),
        ({"topic": "x"}, "enable at least one of"),
    ],
)
def test_watchlist_entry_invalid(raw_config: dict[str, Any], entry: Any, message: str) -> None:
    raw_config["watchlist"] = [entry]
    with pytest.raises(ConfigError, match=message):
        validate_config(raw_config)


def test_watchlist_duplicate_slug_fails(raw_config: dict[str, Any]) -> None:
    raw_config["watchlist"] = [
        {"topic": "Rust programming", "hackernews": True},
        {"topic": "rust  programming!", "wikipedia": True},
    ]
    with pytest.raises(ConfigError, match="duplicate topic \\(slug 'rust-programming'\\)"):
        validate_config(raw_config)


def test_watchlist_valid_shapes(raw_config: dict[str, Any]) -> None:
    raw_config["watchlist"] = [
        {"topic": "a", "wikipedia": True},
        {"topic": "b", "wikipedia": "Some article", "hackernews": False, "reddit": True},
        {"topic": "c", "hackernews": True},
    ]
    validate_config(raw_config)


def test_shipped_watchlist() -> None:
    from signalcheck.watchlist import topics_for, watchlist, watchlist_topics

    cfg = load_config()
    topics = watchlist_topics(cfg)
    assert 8 <= len(topics) <= 12
    # The original four sample topics stay on the watchlist.
    assert {"perplexity ai", "chatgpt", "taylor swift", "rust programming"} <= set(topics)
    items = {i.topic: i for i in watchlist(cfg)}
    assert items["claude ai"].wiki_article == "Claude (language model)"
    assert items["claude ai"].params("wikipedia") == {"article": "Claude (language model)"}
    assert items["chatgpt"].params("wikipedia") == {}
    assert items["chatgpt"].sources == ["wikipedia", "hackernews", "reddit"]
    assert "taylor swift" not in topics_for("reddit", cfg)
