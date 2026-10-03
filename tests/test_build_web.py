"""``scripts/build_web.py``: precomputed samples, site metadata and the Pyodide bundle."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from scripts import build_web

WHEN = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)


def test_build_writes_results_site_and_bundle(tmp_path: Path) -> None:
    report = build_web.build(tmp_path, wheels=False, now=WHEN)
    assert report["samples"] == ["perplexity-ai", "chatgpt", "taylor-swift", "rust-programming"]

    for slug in report["samples"]:
        result = json.loads((tmp_path / "results" / f"{slug}.json").read_text())
        assert result["sample"] is True and result["generated_at"].startswith("2026-10-03")

    site = json.loads((tmp_path / "site.json").read_text())
    assert [s["slug"] for s in site["samples"]] == report["samples"]
    assert site["generated_at"] == "2026-10-03T12:00:00+00:00"

    manifest = json.loads((tmp_path / "py" / "manifest.json").read_text())
    bundle = tmp_path / "py" / manifest["bundle"]
    assert manifest["bundle_bytes"] == bundle.stat().st_size
    assert manifest["wheels"] == []
    assert "pandas" in manifest["pyodide_packages"]
    names = zipfile.ZipFile(bundle).namelist()
    assert "config.yaml" in names
    assert "signalcheck/web/worker.py" in names
    assert "signalcheck/engine/pelt.py" in names
    assert any(n.startswith("data/samples/wikipedia/") for n in names)
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)


def test_bundle_is_reproducible_and_content_addressed(tmp_path: Path) -> None:
    cfg_root = build_web.ROOT
    from signalcheck.config import get_config

    files = build_web.bundle_files(cfg_root, get_config())
    first = build_web.build_zip(files, cfg_root)
    assert first == build_web.build_zip(list(reversed(files)), cfg_root)
    info = zipfile.ZipFile(io.BytesIO(first)).infolist()[0]
    assert info.date_time == build_web.ZIP_DATE

    a = build_web.build_python(get_config(), tmp_path, cfg_root, wheels=False)
    b = build_web.build_python(get_config(), tmp_path, cfg_root, wheels=False)
    assert a["bundle"] == b["bundle"]
    assert len(list((tmp_path / "py").glob("signalcheck-*.zip"))) == 1


def test_build_reuses_existing_wheels_offline(tmp_path: Path) -> None:
    py_dir = tmp_path / "py"
    py_dir.mkdir()
    (py_dir / "pymannkendall-1.4.3-py3-none-any.whl").write_bytes(b"wheel")
    from signalcheck.config import get_config

    manifest = build_web.build_python(get_config(), tmp_path, build_web.ROOT, wheels=False)
    assert manifest["wheels"] == ["pymannkendall-1.4.3-py3-none-any.whl"]
