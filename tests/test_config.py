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
