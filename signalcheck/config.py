"""Application constants, ``config.yaml`` loading/validation, and secret lookup.

All engine thresholds live in ``config.yaml``. :func:`load_config` validates the
file against :data:`SCHEMA` and fails fast, listing every missing or mistyped key,
so a bad config can never silently fall back to a hidden default.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
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
        "wikipedia": {
            "project": str,
            "access": str,
            "agent": str,
            "history_days": int,
            "search_limit": int,
        },
        "hackernews": {
            "history_days": int,
            "tags": str,
            "typo_tolerance": bool,
            "max_hits_per_day": int,
            "min_interval_s": NUMBER,
        },
        "google_trends": {
            "live_enabled": bool,
            "timeframe_days": int,
            "daily_max_days": int,
            "weekly_max_days": int,
            "geo": str,
            "snapshot_dir": str,
            "snapshot_stale_days": NUMBER,
        },
        "reddit": {
            "history_days": int,
            "page_limit": int,
            "result_cap": int,
            "min_interval_s": NUMBER,
            "collected_dir": str,
            "collect_days": int,
            "watchlist": list,
        },
        "x": {
            "counts_endpoint": str,
            "history_days": int,
            "query_suffix": str,
            "cost_counts_recent_usd": NUMBER,
            "cost_counts_all_usd": NUMBER,
            "cost_post_read_usd": NUMBER,
            "counts_all_page_days": int,
            "default_max_spend_usd": NUMBER,
            "ledger_file": str,
            "breadth_enabled": bool,
            "breadth_posts_per_day": int,
        },
    },
    "samples": {
        "dir": str,
        "topics": list,
    },
    "ui": {
        "timeframe_days": list,
        "chart_height_px": int,
        "threshold_headroom": NUMBER,
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
            "negligible_slope": NUMBER,
            "band_mads": NUMBER,
            "min_points": int,
            "slope_level_floor": NUMBER,
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
            "stl_seasonal_deg": int,
            "excess_min_mads": NUMBER,
        },
        "outliers": {
            "outlier_z": NUMBER,
            "isolated_max_points": int,
        },
        "level_shift": {
            "pen_beta": NUMBER,
            "min_persist": FreqMap(int),
            "hold_fraction": NUMBER,
            "recent_tolerance": FreqMap(int),
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
    "display": {
        "value_decimals": int,
        "pct_decimals": int,
        "stat_decimals": int,
        "p_decimals": int,
    },
    "verdict": {
        "seasonal_explained_min": NUMBER,
        "fluke_from_isolated_outliers": bool,
    },
    "confidence": {
        "conf_high": int,
        "conf_medium": int,
        "history_multiple": NUMBER,
    },
    "change_my_mind": {
        "exceed_mads": NUMBER,
        "fallback_mads": NUMBER,
        "seasonal_excess_mads": NUMBER,
        "max_conditions": int,
    },
    "eval": {
        "target_false_trend_rate": NUMBER,
    },
    "cross_source": {
        "min_overlap": NUMBER,
    },
    "narration": {
        "max_words": int,
        "max_tokens": int,
        "max_tokens_param": str,
        "default_base_url": str,
        "default_model": str,
        "connect_timeout_s": NUMBER,
        "read_timeout_s": NUMBER,
        "max_retries": int,
        "total_timeout_s": NUMBER,
        "max_workers": int,
    },
}

MAX_TOKENS_PARAMS: tuple[str, ...] = ("max_tokens", "max_completion_tokens")


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
    _check_adapter_constraints(cfg, require)
    _check_check_constraints(cfg, require)


def _check_adapter_constraints(cfg: Config, require: Callable[[bool, str], None]) -> None:
    """Range checks for the live adapters (section 4) and sample topics.

    History lengths, page sizes and caps must be positive; pacing intervals and
    costs non-negative; Google Trends granularity cut-offs ordered; the X counts
    endpoint one of the two the API offers; topic lists non-empty strings.
    """
    ad = cfg["adapters"]
    for name in ("wikipedia", "hackernews", "reddit", "x"):
        require(ad[name]["history_days"] >= 1, f"adapters.{name}.history_days: must be >= 1")
    for name in ("hackernews", "reddit"):
        require(ad[name]["min_interval_s"] >= 0, f"adapters.{name}.min_interval_s: must be >= 0")
    wiki = ad["wikipedia"]
    require(1 <= wiki["search_limit"] <= 50, "adapters.wikipedia.search_limit: must be in [1, 50]")
    require(bool(wiki["project"]), "adapters.wikipedia.project: must not be empty")
    hn = ad["hackernews"]
    require(
        1 <= hn["max_hits_per_day"] <= 1000,
        "adapters.hackernews.max_hits_per_day: must be in [1, 1000]",
    )
    gt = ad["google_trends"]
    require(gt["timeframe_days"] >= 1, "adapters.google_trends.timeframe_days: must be >= 1")
    require(
        1 <= gt["daily_max_days"] <= gt["weekly_max_days"],
        "adapters.google_trends.daily_max_days: must be >= 1 and <= weekly_max_days",
    )
    require(gt["snapshot_stale_days"] > 0, "adapters.google_trends.snapshot_stale_days: > 0")
    rd = ad["reddit"]
    require(1 <= rd["page_limit"] <= 100, "adapters.reddit.page_limit: must be in [1, 100]")
    require(
        rd["result_cap"] >= rd["page_limit"], "adapters.reddit.result_cap: must be >= page_limit"
    )
    require(rd["collect_days"] >= 1, "adapters.reddit.collect_days: must be >= 1")
    x = ad["x"]
    require(
        x["counts_endpoint"] in ("all", "recent"),
        "adapters.x.counts_endpoint: must be 'all' or 'recent'",
    )
    for key in ("cost_counts_recent_usd", "cost_counts_all_usd", "cost_post_read_usd"):
        require(x[key] >= 0, f"adapters.x.{key}: must be >= 0")
    require(x["default_max_spend_usd"] >= 0, "adapters.x.default_max_spend_usd: must be >= 0")
    require(
        1 <= x["counts_all_page_days"] <= 31, "adapters.x.counts_all_page_days: must be in [1, 31]"
    )
    require(
        10 <= x["breadth_posts_per_day"] <= 100,
        "adapters.x.breadth_posts_per_day: must be in [10, 100]",
    )
    require(bool(x["ledger_file"]), "adapters.x.ledger_file: must not be empty")
    ui = cfg["ui"]
    require(
        bool(ui["timeframe_days"])
        and all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 1 for d in ui["timeframe_days"]
        ),
        "ui.timeframe_days: must be a non-empty list of positive integers",
    )
    require(ui["chart_height_px"] >= 120, "ui.chart_height_px: must be >= 120")
    require(ui["threshold_headroom"] >= 1, "ui.threshold_headroom: must be >= 1")
    for path, topics in (
        ("adapters.reddit.watchlist", rd["watchlist"]),
        ("samples.topics", cfg["samples"]["topics"]),
    ):
        require(
            bool(topics) and all(isinstance(t, str) and t.strip() for t in topics),
            f"{path}: must be a non-empty list of non-empty strings",
        )


def _check_check_constraints(cfg: Config, require: Callable[[bool, str], None]) -> None:
    """Range checks for the section 6 check thresholds and display rounding.

    Probabilities and shares must lie in their open/closed unit intervals, window
    lengths and counts must be positive, and STL periods must be at least 2 with
    two full periods available at the annual minimum history.
    """
    checks = cfg["checks"]
    per = checks["persistence"]
    require(per["persistence_context"] >= 0, "checks.persistence.persistence_context: >= 0")
    require(per["hamed_rao_min_n"] >= 3, "checks.persistence.hamed_rao_min_n: must be >= 3")
    require(0 < per["mk_alpha"] < 1, "checks.persistence.mk_alpha: must be in (0, 1)")
    require(per["min_slope"] >= 0, "checks.persistence.min_slope: must be >= 0")
    require(
        0 <= per["negligible_slope"] <= per["min_slope"],
        "checks.persistence.negligible_slope: must be between 0 and min_slope",
    )
    require(per["band_mads"] > 0, "checks.persistence.band_mads: must be > 0")
    require(per["min_points"] >= 3, "checks.persistence.min_points: must be >= 3")
    require(per["slope_level_floor"] > 0, "checks.persistence.slope_level_floor: must be > 0")

    conc = checks["concentration"]
    require(conc["excess_min_mads"] >= 0, "checks.concentration.excess_min_mads: must be >= 0")
    for key in ("conc_top1_max", "conc_top2_max"):
        require(0 < conc[key] <= 1, f"checks.concentration.{key}: must be in (0, 1]")

    seas = checks["seasonality"]
    require(seas["weekly_period"] >= 2, "checks.seasonality.weekly_period: must be >= 2")
    require(
        seas["stl_seasonal_deg"] in (0, 1), "checks.seasonality.stl_seasonal_deg: must be 0 or 1"
    )
    require(seas["weekly_min_weeks"] >= 2, "checks.seasonality.weekly_min_weeks: must be >= 2")
    require(seas["excess_min_mads"] >= 0, "checks.seasonality.excess_min_mads: must be >= 0")
    for freq in FREQS:
        period = seas["annual_period"][freq]
        require(period >= 2, f"checks.seasonality.annual_period.{freq}: must be >= 2")
        # Daily data is aggregated to weeks of weekly_period days before annual STL.
        days = seas["weekly_period"] if freq == "D" else 1
        require(
            seas["annual_min_points"][freq] >= 2 * period * days,
            f"checks.seasonality.annual_min_points.{freq}: must cover two annual periods",
        )

    out = checks["outliers"]
    require(out["outlier_z"] > 0, "checks.outliers.outlier_z: must be > 0")
    require(out["isolated_max_points"] >= 1, "checks.outliers.isolated_max_points: must be >= 1")

    ls = checks["level_shift"]
    require(ls["pen_beta"] > 0, "checks.level_shift.pen_beta: must be > 0")
    require(0.5 < ls["hold_fraction"] <= 1, "checks.level_shift.hold_fraction: must be in (0.5, 1]")
    for freq in FREQS:
        require(
            ls["min_persist"][freq] >= 1, f"checks.level_shift.min_persist.{freq}: must be >= 1"
        )
        require(
            ls["recent_tolerance"][freq] >= 0,
            f"checks.level_shift.recent_tolerance.{freq}: must be >= 0",
        )

    low = checks["low_count"]
    require(low["low_count_threshold"] > 0, "checks.low_count.low_count_threshold: must be > 0")
    require(0 < low["ci_level"] < 1, "checks.low_count.ci_level: must be in (0, 1)")

    br = checks["breadth"]
    require(
        0 < br["breadth_top_share_max"] <= 1,
        "checks.breadth.breadth_top_share_max: must be in (0, 1]",
    )
    require(
        br["breadth_min_contrib_ratio"] >= 0,
        "checks.breadth.breadth_min_contrib_ratio: must be >= 0",
    )

    for key, value in cfg["display"].items():
        require(value >= 0, f"display.{key}: must be >= 0")
    # With 0 decimals every p-value would display as "p < 1".
    require(cfg["display"]["p_decimals"] >= 1, "display.p_decimals: must be >= 1")
    _check_verdict_constraints(cfg, require)


def _check_verdict_constraints(cfg: Config, require: Callable[[bool, str], None]) -> None:
    """Range checks for section 7-8 settings (verdict, confidence, change-my-mind, eval).

    Shares and rates are fractions in (0, 1]; the confidence cut-offs must be
    ordered (``conf_high >= conf_medium``) so "high" always implies "medium"; MAD
    multipliers must be positive and the fluke fallback must sit inside the trend
    band (``fallback_mads <= exceed_mads``); at least one condition is generated.
    """
    seasonal = cfg["verdict"]["seasonal_explained_min"]
    require(0 < seasonal <= 1, "verdict.seasonal_explained_min: must be in (0, 1]")

    conf = cfg["confidence"]
    require(
        conf["conf_high"] >= conf["conf_medium"],
        "confidence.conf_high: must be >= confidence.conf_medium",
    )
    require(conf["history_multiple"] >= 1, "confidence.history_multiple: must be >= 1")

    cmm = cfg["change_my_mind"]
    for key in ("exceed_mads", "fallback_mads", "seasonal_excess_mads"):
        require(cmm[key] > 0, f"change_my_mind.{key}: must be > 0")
    require(
        cmm["fallback_mads"] <= cmm["exceed_mads"],
        "change_my_mind.fallback_mads: must be <= change_my_mind.exceed_mads",
    )
    require(cmm["max_conditions"] >= 1, "change_my_mind.max_conditions: must be >= 1")

    rate = cfg["eval"]["target_false_trend_rate"]
    require(0 < rate < 1, "eval.target_false_trend_rate: must be in (0, 1)")
    overlap = cfg["cross_source"]["min_overlap"]
    require(0 < overlap <= 1, "cross_source.min_overlap: must be in (0, 1]")
    _check_narration_constraints(cfg, require)


def _check_narration_constraints(cfg: Config, require: Callable[[bool, str], None]) -> None:
    """Range checks for the optional LLM narration (section 9.1).

    The word cap must leave room for a useful paragraph and stay within the
    brief's 120 words; the token budget must cover the word cap (a word is at
    least one token); timeouts are positive and the UI's overall wait is at least
    one request's worth; the base URL is an http(s) URL.
    """
    n = cfg["narration"]
    require(20 <= n["max_words"] <= 120, "narration.max_words: must be in [20, 120]")
    require(
        n["max_tokens"] >= n["max_words"], "narration.max_tokens: must be >= narration.max_words"
    )
    require(
        n["max_tokens_param"] in MAX_TOKENS_PARAMS,
        f"narration.max_tokens_param: must be one of {MAX_TOKENS_PARAMS}",
    )
    require(
        str(n["default_base_url"]).startswith(("https://", "http://")),
        "narration.default_base_url: must be an http(s) URL",
    )
    require(bool(n["default_model"].strip()), "narration.default_model: must not be empty")
    require(n["connect_timeout_s"] > 0, "narration.connect_timeout_s: must be > 0")
    require(n["read_timeout_s"] > 0, "narration.read_timeout_s: must be > 0")
    require(n["max_retries"] >= 0, "narration.max_retries: must be >= 0")
    require(
        n["total_timeout_s"] >= n["connect_timeout_s"] + n["read_timeout_s"],
        "narration.total_timeout_s: must be >= connect_timeout_s + read_timeout_s",
    )
    require(n["max_workers"] >= 1, "narration.max_workers: must be >= 1")


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


def resolve_path(path: str | Path) -> Path:
    """Absolute path for a config path; relative paths are under the repo root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else DEFAULT_CONFIG_PATH.parent / candidate


def secret_flag(name: str) -> bool | None:
    """Boolean env/secret flag (``1/true/yes/on`` vs ``0/false/no/off``); ``None`` if unset."""
    value = get_secret(name)
    if value is None:
        return None
    return value.strip().casefold() in ("1", "true", "yes", "on")


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
