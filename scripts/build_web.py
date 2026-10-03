"""Build the static assets for the React frontend (GitHub Pages).

Usage::

    python -m scripts.build_web                 # writes web/public/{results,py,site.json}
    python -m scripts.build_web --no-wheels     # skip downloading pure-Python wheels

Steps (all deterministic except ``generated_at`` and snapshot ages):

1. Run the engine natively over the committed snapshots in ``data/samples/`` for
   every ``watchlist`` topic and write ``results/<slug>.json`` (the exact
   JSON the in-browser engine produces), so sample chips render instantly.
2. Write ``site.json``: methodology built from the shipped ``config.yaml``, the
   R1-R6 table, label colours, source availability, the sample index and the
   last daily data refresh from ``data/manifest.json`` (``refresh.manifest``).
3. Zip the ``signalcheck`` package, ``config.yaml`` and ``data/samples`` into
   ``py/signalcheck-<hash>.zip`` for Pyodide, and download the pure-Python
   wheels Pyodide does not ship (``pymannkendall``, verified by SHA-256).
   ``py/manifest.json`` tells the web worker what to load.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signalcheck.adapters.base import slugify
from signalcheck.config import DEFAULT_CONFIG_PATH, Config, get_config, resolve_path
from signalcheck.snapshots import samples_dir
from signalcheck.ui import runner
from signalcheck.watchlist import watchlist_topics
from signalcheck.web.export import result_json, site_json, with_static_cards

log = logging.getLogger("build_web")

ROOT = DEFAULT_CONFIG_PATH.parent
DEFAULT_OUT = ROOT / "web" / "public"
# Packages the worker loads from the Pyodide distribution.
PYODIDE_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "pyyaml",
    "requests",
    "micropip",
)
# Pure-Python wheels installed with micropip (deps=False): name -> pinned version.
EXTRA_WHEELS: dict[str, str] = {"pymannkendall": "1.4.3"}
ZIP_DATE = (2020, 1, 1, 0, 0, 0)


def build_results(cfg: Config, out: Path, now: datetime) -> list[dict[str, Any]]:
    """Precompute every sample topic; return the sample index for ``site.json``."""
    results_dir = out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for topic in watchlist_topics(cfg):
        sources = runner.sample_sources(topic, cfg)
        if not sources:
            log.warning("no snapshots for sample topic %r; skipped", topic)
            continue
        result = runner.run_topic(topic, sources, cfg, sample=True, max_workers=1)
        result = with_static_cards(result, cfg, sources)
        payload = result_json(result, cfg, generated_at=now)
        slug = slugify(topic)
        write_json(results_dir / f"{slug}.json", payload)
        index.append({"query": topic, "slug": slug, "sources": sources})
        log.info("sample %s: %s", topic, ", ".join(sources))
    return index


def data_freshness(manifest_path: Path) -> dict[str, Any] | None:
    """Last daily refresh (time, overall and per-source status) from ``data/manifest.json``."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "refreshed_at": str(manifest["refreshed_at"]),
            "status": str(manifest["status"]),
            "sources": {s: str(v["status"]) for s, v in sorted(manifest["sources"].items())},
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def bundle_files(root: Path, cfg: Config) -> list[Path]:
    """Files shipped to Pyodide: the package, the config and the sample snapshots."""
    package = sorted(
        p for p in (root / "signalcheck").rglob("*.py") if "__pycache__" not in p.parts
    )
    samples = sorted(p for p in samples_dir(cfg).rglob("*") if p.is_file())
    return [*package, root / "config.yaml", *samples]


def build_zip(files: Iterable[Path], root: Path) -> bytes:
    """A reproducible zip (fixed timestamps, sorted entries) of ``files`` relative to ``root``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files):
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    return buffer.getvalue()


def fetch_wheel(name: str, version: str, dest: Path) -> str:
    """Download the pure-Python wheel ``name==version`` from PyPI, verifying its SHA-256."""
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        meta = json.load(resp)
    files = [f for f in meta["urls"] if f["filename"].endswith("-py3-none-any.whl")]
    if not files:
        raise RuntimeError(f"{name} {version} has no pure-Python wheel on PyPI")
    info = files[0]
    with urllib.request.urlopen(info["url"], timeout=60) as resp:
        data = resp.read()
    if hashlib.sha256(data).hexdigest() != info["digests"]["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {info['filename']}")
    (dest / info["filename"]).write_bytes(data)
    return str(info["filename"])


def build_python(cfg: Config, out: Path, root: Path, wheels: bool) -> dict[str, Any]:
    """Write the Pyodide bundle (+ wheels) under ``out/py`` and return its manifest."""
    py_dir = out / "py"
    py_dir.mkdir(parents=True, exist_ok=True)
    for stale in py_dir.glob("signalcheck-*.zip"):
        stale.unlink()
    data = build_zip(bundle_files(root, cfg), root)
    bundle = f"signalcheck-{hashlib.sha256(data).hexdigest()[:12]}.zip"
    (py_dir / bundle).write_bytes(data)
    wheel_files: list[str] = []
    if wheels:
        wheel_files = [fetch_wheel(n, v, py_dir) for n, v in EXTRA_WHEELS.items()]
    else:
        wheel_files = sorted(p.name for p in py_dir.glob("*.whl"))
    manifest = {
        "bundle": bundle,
        "bundle_bytes": len(data),
        "wheels": wheel_files,
        "pyodide_packages": list(PYODIDE_PACKAGES),
    }
    write_json(py_dir / "manifest.json", manifest)
    return manifest


def write_json(path: Path, payload: Any) -> None:
    """Compact, strict JSON (no NaN) with a trailing newline."""
    path.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")


def build(out: Path, *, wheels: bool = True, now: datetime | None = None) -> dict[str, Any]:
    """Run every step; return a short report."""
    cfg = get_config()
    when = now if now is not None else datetime.now(UTC)
    out.mkdir(parents=True, exist_ok=True)
    samples = build_results(cfg, out, when)
    site = site_json(cfg, samples)
    site["generated_at"] = when.isoformat(timespec="seconds")
    site["data"] = data_freshness(resolve_path(cfg["refresh"]["manifest"]))
    write_json(out / "site.json", site)
    manifest = build_python(cfg, out, ROOT, wheels)
    return {"samples": [s["slug"] for s in samples], **manifest}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output dir (web/public)")
    parser.add_argument(
        "--no-wheels", action="store_true", help="don't download wheels (reuse existing ones)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = build(args.out, wheels=not args.no_wheels)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
