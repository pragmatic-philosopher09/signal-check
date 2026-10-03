"""Application constants, ``config.yaml`` loading/validation, and secret lookup.

All engine thresholds live in ``config.yaml``. :func:`load_config` validates the
file against :data:`SCHEMA` and fails fast, listing every missing or mistyped key,
so a bad config can never silently fall back to a hidden default.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

APP_NAME = "Signal Check"
APP_VERSION = "0.1.0"

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
FREQS: tuple[str, ...] = ("D", "W", "M")
SCALES: tuple[str, ...] = ("relative_0_100", "count", "pageviews", "value")

Config = dict[str, Any]


class ConfigError(ValueError):
    """Raised when ``config.yaml`` is missing, unreadable or fails validation."""


@dataclass(frozen=True)
class FreqMap:
    """Schema marker: a mapping with one value per frequency in :data:`FREQS`."""

    value_type: type | tuple[type, ...]


NUMBER: tuple[type, ...] = (int, float)

SCHEMA: dict[str, Any] = {
    "sources": {
        "csv": bool,
        "google_trends": bool,
        "reddit": bool,
        "wikipedia": bool,
        "hackernews": bool,
        "x": bool,
    },
    "http": {
        "connect_timeout_s": NUMBER,
        "read_timeout_s": NUMBER,
        "max_retries": int,
        "backoff_base_s": NUMBER,
        "backoff_max_s": NUMBER,
    },
    "cache": {
        "ttl_hours": NUMBER,
        "dir": str,
    },
    "adapters": {
        "csv": {
            "trends_lt1_value": NUMBER,
        },
    },
    "preprocess": {
        "min_history": FreqMap(int),
        "recent_frac": NUMBER,
        "recent_min": FreqMap(int),
        "recent_max": FreqMap(int),
        "gap_fill_max_consecutive": int,
        "mad_scale": NUMBER,
        "mad_floor": NUMBER,
        "log_scales": list,
    },
    "checks": {
        "persistence": {
            "persistence_context": int,
            "hamed_rao_min_n": int,
            "mk_alpha": NUMBER,
            "min_slope": NUMBER,
            "band_mads": NUMBER,
        },
        "concentration": {
            "excess_min_mads": NUMBER,
            "conc_top1_max": NUMBER,
            "conc_top2_max": NUMBER,
        },
        "seasonality": {
            "weekly_period": int,
            "weekly_min_weeks": int,
            "annual_min_points": FreqMap(int),
            "annual_period": FreqMap(int),
            "stl_robust": bool,
            "excess_min_mads": NUMBER,
        },
        "outliers": {
            "outlier_z": NUMBER,
        },
        "level_shift": {
            "pen_beta": NUMBER,
            "min_persist": FreqMap(int),
            "hold_fraction": NUMBER,
        },
        "low_count": {
            "low_count_threshold": NUMBER,
            "ci_level": NUMBER,
        },
        "breadth": {
            "breadth_top_share_max": NUMBER,
            "breadth_min_contrib_ratio": NUMBER,
        },
    },
    "verdict": {
        "seasonal_explained_min": NUMBER,
    },
    "confidence": {
        "conf_high": int,
        "conf_medium": int,
        "history_multiple": NUMBER,
    },
    "eval": {
        "target_false_trend_rate": NUMBER,
    },
    "cross_source": {
        "min_overlap": NUMBER,
    },
}


def _type_ok(value: object, expected: type | tuple[type, ...]) -> bool:
    """Return True if ``value`` matches ``expected``; bools never count as numbers."""
    if isinstance(value, bool) and expected is not bool:
        return False
    return isinstance(value, expected)


def _type_name(expected: type | tuple[type, ...]) -> str:
    """Human-readable name for an expected type, used in error messages."""
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    return expected.__name__


def _validate(node: object, schema: object, path: str, errors: list[str]) -> None:
    """Recursively check ``node`` against ``schema``, appending problems to ``errors``."""
    if isinstance(schema, dict):
        if not isinstance(node, dict):
            errors.append(f"{path or '<root>'}: expected a mapping")
            return
        for key, sub_schema in schema.items():
            sub_path = f"{path}.{key}" if path else key
            if key not in node:
                errors.append(f"{sub_path}: missing")
            else:
                _validate(node[key], sub_schema, sub_path, errors)
        return
    if isinstance(schema, FreqMap):
        if not isinstance(node, dict):
            errors.append(f"{path}: expected a mapping with keys {', '.join(FREQS)}")
            return
        for freq in FREQS:
            sub_path = f"{path}.{freq}"
            if freq not in node:
                errors.append(f"{sub_path}: missing")
            elif not _type_ok(node[freq], schema.value_type):
                errors.append(f"{sub_path}: expected {_type_name(schema.value_type)}")
        return
    if isinstance(schema, type | tuple) and not _type_ok(node, schema):
        errors.append(f"{path}: expected {_type_name(schema)}")


def _check_constraints(cfg: Config, errors: list[str]) -> None:
    """Check value ranges and cross-key consistency of a schema-valid config.

    These guarantee the preprocessing maths is well defined: a recent window of at
    least one point that always leaves a non-empty baseline, a positive MAD floor,
    and positive HTTP timeouts and cache TTL.
    """

    def require(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    pre = cfg["preprocess"]
    require(0 < pre["recent_frac"] < 1, "preprocess.recent_frac: must be in (0, 1)")
    for freq in FREQS:
        lo, hi, hist = pre["recent_min"][freq], pre["recent_max"][freq], pre["min_history"][freq]
        require(lo >= 1, f"preprocess.recent_min.{freq}: must be >= 1")
        require(lo <= hi, f"preprocess.recent_min.{freq}: must be <= recent_max.{freq}")
        require(lo < hist, f"preprocess.recent_min.{freq}: must be < min_history.{freq}")
        # round-half-up(recent_frac * n) < n for every n >= min_history, so the baseline
        # window is never empty once the history is long enough.
        require(
            pre["recent_frac"] * hist < hist - 0.5,
            f"preprocess.recent_frac: leaves no baseline at min_history.{freq}",
        )
    require(pre["gap_fill_max_consecutive"] >= 0, "preprocess.gap_fill_max_consecutive: >= 0")
    require(pre["mad_scale"] > 0, "preprocess.mad_scale: must be > 0")
    require(pre["mad_floor"] > 0, "preprocess.mad_floor: must be > 0")
    unknown = [s for s in pre["log_scales"] if s not in SCALES]
    require(not unknown, f"preprocess.log_scales: unknown scale(s) {unknown}; known: {SCALES}")

    http = cfg["http"]
    require(http["connect_timeout_s"] > 0, "http.connect_timeout_s: must be > 0")
    require(http["read_timeout_s"] > 0, "http.read_timeout_s: must be > 0")
    require(http["max_retries"] >= 0, "http.max_retries: must be >= 0")
    require(http["backoff_base_s"] >= 0, "http.backoff_base_s: must be >= 0")
    require(
        http["backoff_max_s"] >= http["backoff_base_s"],
        "http.backoff_max_s: must be >= backoff_base_s",
    )
    require(cfg["cache"]["ttl_hours"] > 0, "cache.ttl_hours: must be > 0")
    lt1 = cfg["adapters"]["csv"]["trends_lt1_value"]
    require(0 < lt1 < 1, "adapters.csv.trends_lt1_value: must be in (0, 1)")


def validate_config(raw: object) -> Config:
    """Validate a parsed config against :data:`SCHEMA` and return it.

    Raises :class:`ConfigError` listing every missing or mistyped key at once, then
    every out-of-range or inconsistent value (see :func:`_check_constraints`).
    """
    errors: list[str] = []
    _validate(raw, SCHEMA, "", errors)
    if not errors:
        assert isinstance(raw, dict)
        _check_constraints(raw, errors)
    if errors:
        raise ConfigError("Invalid config.yaml:\n  " + "\n  ".join(errors))
    assert isinstance(raw, dict)
    return raw


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate ``config.yaml`` (defaults to the repo-root file)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {config_path}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Cannot parse config file {config_path}: {exc}") from exc
    return validate_config(raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the validated default config, loaded once per process."""
    return load_config()


@lru_cache(maxsize=1)
def _load_dotenv_once() -> None:
    """Load a local ``.env`` into ``os.environ`` without overriding existing values."""
    load_dotenv(override=False)


def _streamlit_secret(name: str) -> str | None:
    """Read ``name`` from ``st.secrets`` if Streamlit is already running in-process."""
    st = sys.modules.get("streamlit")
    if st is None:
        return None
    try:
        secrets = st.secrets
        if name in secrets:
            return str(secrets[name])
    except Exception:  # no secrets.toml, or not inside a Streamlit runtime
        return None
    return None


def get_secret(name: str, default: str | None = None) -> str | None:
    """Return a secret from the environment (incl. ``.env``) or ``st.secrets``.

    Environment variables win; empty strings count as unset. Returns ``default``
    when the secret is not found anywhere.
    """
    _load_dotenv_once()
    value = os.environ.get(name)
    if value:
        return value
    st_value = _streamlit_secret(name)
    if st_value:
        return st_value
    return default
