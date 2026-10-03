"""Run the verdict engine over the labelled eval cases and report how it does.

Usage::

    python -m eval.run_eval                 # both splits, text report
    python -m eval.run_eval --split tune    # one split
    python -m eval.run_eval --json          # machine-readable output
    python -m eval.run_eval --sweep 0       # skip the extra pure-noise seeds (default 20)
    python -m eval.run_eval --split test --save final_test   # also write eval/results/*.json|md

For each split (``tune``/``test``) it reports the synthetic cases
(``eval/cases.yaml``), the real cases (``eval/real_cases.yaml``, loaded offline
from the aggregate snapshots in ``eval/real_data/`` and cut at each case's
``as_of``) and both combined: confusion matrix (rows = expected label, columns =
verdict), accuracy (a TREND counts as correct only with the right direction),
per-label recall, the false-TREND rate on pure-noise cases (target from
``eval.target_false_trend_rate``) and every failing case with the rule that
fired and its reason.

Everything is deterministic: each synthetic case has a fixed seed, and
``--sweep`` derives its extra seeds from the case seed with
:class:`numpy.random.SeedSequence`. This script reports; it never changes thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from eval.fetch_real import REAL_DATA_DIR, load_entries, real_data_path
from eval.generators import GENERATORS
from signalcheck.config import Config, get_config
from signalcheck.engine import analyse
from signalcheck.models import DIRECTIONS, LABELS, Series
from signalcheck.snapshots import read_snapshot

EVAL_DIR = Path(__file__).resolve().parent
CASES_PATH = EVAL_DIR / "cases.yaml"
REAL_CASES_PATH = EVAL_DIR / "real_cases.yaml"
RESULTS_DIR = EVAL_DIR / "results"
SPLITS: tuple[str, ...] = ("tune", "test")
KINDS: tuple[str, ...] = ("synthetic", "real", "combined")
SHORT: dict[str, str] = {
    "TREND": "TREND",
    "FLUKE": "FLUKE",
    "SEASONAL": "SEAS",
    "NO_CHANGE": "NO_CH",
    "INCONCLUSIVE": "INCON",
}


@dataclass(frozen=True)
class Case:
    """One labelled series; ``build`` creates it on demand."""

    id: str
    label: str
    direction: str | None
    split: str
    noise: bool
    kind: str
    build: Callable[[], Series]


@dataclass(frozen=True)
class Result:
    """The engine's verdict for one case."""

    case: Case
    label: str
    direction: str | None
    confidence: str
    rule: str
    reason: str

    @property
    def correct(self) -> bool:
        """Label matches, and for TREND the direction matches too."""
        if self.label != self.case.label:
            return False
        return self.label != "TREND" or self.direction == self.case.direction


def _validate(entry: dict[str, Any], where: str) -> None:
    label, direction, split = entry.get("label"), entry.get("direction"), entry.get("split")
    if label not in LABELS:
        raise ValueError(f"{where}: label must be one of {LABELS}, got {label!r}")
    if split not in SPLITS:
        raise ValueError(f"{where}: split must be one of {SPLITS}, got {split!r}")
    if label == "TREND" and direction not in DIRECTIONS:
        raise ValueError(f"{where}: a TREND case needs direction up/down")
    if label != "TREND" and direction is not None:
        raise ValueError(f"{where}: only TREND cases have a direction")


def synthetic_builder(generator: str, params: dict[str, Any], seed: Any) -> Callable[[], Series]:
    """A zero-argument builder calling ``GENERATORS[generator]`` with a fresh seeded rng."""
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}")
    build = GENERATORS[generator]
    return lambda: build(np.random.default_rng(seed), **params)


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    """Load and validate the synthetic cases."""
    raw = yaml.safe_load(path.read_text())
    cases: list[Case] = []
    seen: set[str] = set()
    for entry in raw["cases"]:
        case_id = str(entry["id"])
        if case_id in seen:
            raise ValueError(f"duplicate case id {case_id!r}")
        seen.add(case_id)
        _validate(entry, case_id)
        cases.append(
            Case(
                id=case_id,
                label=entry["label"],
                direction=entry.get("direction"),
                split=entry["split"],
                noise=bool(entry.get("noise", False)),
                kind="synthetic",
                build=synthetic_builder(
                    entry["generator"], dict(entry.get("params") or {}), int(entry["seed"])
                ),
            )
        )
    return cases


def truncate_as_of(series: Series, as_of: str) -> Series:
    """Keep only periods that end on or before ``as_of`` (what the engine may see)."""
    ts = pd.to_datetime(series.points["ts"])
    span = pd.Timedelta(days=6) if series.freq == "W" else pd.Timedelta(0)
    keep = (ts + span) <= pd.Timestamp(as_of)
    meta = dict(series.meta)
    meta["as_of"] = as_of
    points = series.points.loc[keep.to_numpy()].reset_index(drop=True)
    return Series(series.source, series.query, series.freq, points, series.scale, meta)


def real_builder(snapshot: Path, as_of: str) -> Callable[[], Series]:
    """A builder reading a real snapshot (offline) and cutting it at ``as_of``."""
    return lambda: truncate_as_of(read_snapshot(snapshot), as_of)


def load_real_cases(
    path: Path = REAL_CASES_PATH, data_dir: Path = REAL_DATA_DIR
) -> tuple[list[Case], list[str]]:
    """Labelled real cases whose snapshot exists, plus the ids of cases without one."""
    if not path.exists():
        return [], []
    cases: list[Case] = []
    skipped: list[str] = []
    for entry in load_entries(path):
        file = real_data_path(entry["id"], data_dir)
        if not file.exists():
            skipped.append(entry["id"])
            continue
        _validate(entry, entry["id"])
        cases.append(
            Case(
                id=entry["id"],
                label=entry["label"],
                direction=entry.get("direction"),
                split=entry["split"],
                noise=False,
                kind="real",
                build=real_builder(file, entry["as_of"]),
            )
        )
    return cases, skipped


def noise_sweep(cases: Sequence[Case], n_seeds: int, path: Path = CASES_PATH) -> list[Case]:
    """Extra seeds of every pure-noise synthetic case (for a finer false-TREND rate)."""
    if n_seeds <= 0:
        return []
    raw = {str(e["id"]): e for e in yaml.safe_load(path.read_text())["cases"]}
    out: list[Case] = []
    for case in cases:
        if not case.noise or case.kind != "synthetic":
            continue
        entry = raw[case.id]
        for k in range(1, n_seeds + 1):
            seed = np.random.SeedSequence([int(entry["seed"]), k])
            out.append(
                Case(
                    id=f"{case.id}#{k}",
                    label=case.label,
                    direction=None,
                    split=case.split,
                    noise=True,
                    kind="sweep",
                    build=synthetic_builder(
                        entry["generator"], dict(entry.get("params") or {}), seed
                    ),
                )
            )
    return out


def evaluate(cases: Sequence[Case], cfg: Config) -> list[Result]:
    """Run :func:`signalcheck.engine.analyse` on every case."""
    results: list[Result] = []
    for case in cases:
        verdict = analyse(case.build(), cfg)
        results.append(
            Result(
                case,
                verdict.label,
                verdict.direction,
                verdict.confidence,
                verdict.rule_fired,
                verdict.reason,
            )
        )
    return results


def false_trend(results: Sequence[Result]) -> dict[str, Any]:
    """Share of pure-noise results called TREND."""
    noise = [r for r in results if r.case.noise]
    hits = [r for r in noise if r.label == "TREND"]
    rate = len(hits) / len(noise) if noise else None
    return {"n": len(noise), "false_trend": len(hits), "rate": rate}


def describe(r: Result) -> dict[str, Any]:
    """One result as a report row."""
    return {
        "id": r.case.id,
        "expected": r.case.label + (f" {r.case.direction}" if r.case.direction else ""),
        "got": r.label + (f" {r.direction}" if r.direction else ""),
        "correct": r.correct,
        "confidence": r.confidence,
        "rule": r.rule,
        "reason": r.reason,
    }


def summarise(results: Sequence[Result], sweep: Sequence[Result] = ()) -> dict[str, Any]:
    """Confusion matrix, accuracy, recall, false-TREND rate, cases and failures for a group."""
    confusion = {e: {p: 0 for p in LABELS} for e in LABELS}
    for r in results:
        confusion[r.case.label][r.label] += 1
    recall: dict[str, float | None] = {}
    for label in LABELS:
        of_label = [r for r in results if r.case.label == label]
        recall[label] = sum(r.correct for r in of_label) / len(of_label) if of_label else None
    n_correct = sum(r.correct for r in results)
    rows = [describe(r) for r in results]
    out: dict[str, Any] = {
        "n": len(results),
        "correct": n_correct,
        "accuracy": n_correct / len(results) if results else None,
        "confusion": confusion,
        "recall": recall,
        "false_trend": false_trend(results),
        "cases": rows,
        "failures": [
            {k: v for k, v in row.items() if k != "correct"} for row in rows if not row["correct"]
        ],
    }
    if sweep:
        out["noise_sweep"] = false_trend(sweep)
        out["noise_sweep"]["trend_ids"] = [r.case.id for r in sweep if r.label == "TREND"]
    return out


def run(
    split: str = "all",
    sweep: int = 0,
    cases_path: Path = CASES_PATH,
    real_path: Path = REAL_CASES_PATH,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Evaluate the requested split(s); returns the full report as a dict."""
    cfg = cfg if cfg is not None else get_config()
    cases = load_cases(cases_path)
    real, skipped = load_real_cases(real_path)
    wanted = SPLITS if split == "all" else (split,)
    selected = [c for c in [*cases, *real] if c.split in wanted]
    results = evaluate(selected, cfg)
    sweep_results = evaluate(
        noise_sweep([c for c in cases if c.split in wanted], sweep, cases_path), cfg
    )
    splits: dict[str, Any] = {}
    for name in wanted:
        in_split = [r for r in results if r.case.split == name]
        split_sweep = [r for r in sweep_results if r.case.split == name]
        synthetic = [r for r in in_split if r.case.kind == "synthetic"]
        real_results = [r for r in in_split if r.case.kind == "real"]
        splits[name] = {
            "synthetic": summarise(synthetic, split_sweep),
            "real": summarise(real_results),
            "combined": summarise(in_split, split_sweep),
        }
    return {
        "target_false_trend_rate": float(cfg["eval"]["target_false_trend_rate"]),
        "sweep_seeds": sweep,
        "splits": splits,
        "real_cases_skipped": skipped,
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def format_split(name: str, s: dict[str, Any], target: float) -> list[str]:
    """Text block for one split."""
    lines = [f"=== {name}: {s['n']} cases ==="]
    width = max(len(v) for v in SHORT.values()) + 1
    lines.append("expected \\ got".ljust(16) + "".join(SHORT[p].rjust(width) for p in LABELS))
    for e in LABELS:
        row = s["confusion"][e]
        if sum(row.values()) == 0:
            continue
        lines.append(e.ljust(16) + "".join(str(row[p]).rjust(width) for p in LABELS))
    lines.append(f"accuracy: {_pct(s['accuracy'])} ({s['correct']}/{s['n']})")
    recall = ", ".join(f"{label} {_pct(v)}" for label, v in s["recall"].items() if v is not None)
    lines.append(f"recall: {recall}")
    ft = s["false_trend"]
    lines.append(
        f"false-TREND rate on pure noise: {_pct(ft['rate'])} ({ft['false_trend']}/{ft['n']}; "
        f"target <= {_pct(target)})"
    )
    if "noise_sweep" in s:
        sw = s["noise_sweep"]
        lines.append(
            f"false-TREND rate, noise sweep: {_pct(sw['rate'])} ({sw['false_trend']}/{sw['n']})"
        )
    if s["failures"]:
        lines.append("failing cases:")
        for f in s["failures"]:
            lines.append(
                f"  - {f['id']}: expected {f['expected']}, got {f['got']} "
                f"({f['confidence']}) via {f['rule']}: {f['reason']}"
            )
    else:
        lines.append("failing cases: none")
    return lines


def format_report(report: dict[str, Any]) -> str:
    """The full text report: synthetic, real and combined blocks per split."""
    lines: list[str] = []
    target = report["target_false_trend_rate"]
    for name, kinds in report["splits"].items():
        for kind in KINDS:
            s = kinds[kind]
            if s["n"] == 0:
                continue
            lines.extend(format_split(f"{name} / {kind}", s, target))
            if kind == "real":
                lines.append("real cases:")
                lines.extend(
                    f"  - {c['id']}: expected {c['expected']}, got {c['got']} "
                    f"({c['confidence']}) via {c['rule']}"
                    for c in s["cases"]
                )
            lines.append("")
    if report["real_cases_skipped"]:
        lines.append(f"real cases missing a snapshot: {', '.join(report['real_cases_skipped'])}")
    return "\n".join(lines).rstrip() + "\n"


def _md_row(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def format_markdown(report: dict[str, Any], title: str = "Eval results") -> str:
    """Markdown tables: summary per split and kind, per-label recall, real cases, failures."""
    target = report["target_false_trend_rate"]
    lines = [
        f"# {title}",
        "",
        f"False-TREND target: <= {_pct(target)} on pure noise; noise sweep: "
        f"{report['sweep_seeds']} extra seeds per pure-noise case.",
        "",
        _md_row(["Split", "Cases", "n", "Accuracy", "False-TREND (cases)", "False-TREND (sweep)"]),
        _md_row(["---"] * 6),
    ]
    for name, kinds in report["splits"].items():
        for kind in KINDS:
            s = kinds[kind]
            if s["n"] == 0:
                continue
            ft = s["false_trend"]
            sw = s.get("noise_sweep")
            lines.append(
                _md_row(
                    [
                        name,
                        kind,
                        s["n"],
                        f"{_pct(s['accuracy'])} ({s['correct']}/{s['n']})",
                        f"{_pct(ft['rate'])} ({ft['false_trend']}/{ft['n']})" if ft["n"] else "-",
                        f"{_pct(sw['rate'])} ({sw['false_trend']}/{sw['n']})" if sw else "-",
                    ]
                )
            )
    lines += ["", "## Per-label recall (combined)", ""]
    lines.append(_md_row(["Split", *LABELS]))
    lines.append(_md_row(["---"] * (len(LABELS) + 1)))
    for name, kinds in report["splits"].items():
        recall = kinds["combined"]["recall"]
        lines.append(_md_row([name, *(_pct(recall[label]) for label in LABELS)]))
    for name, kinds in report["splits"].items():
        real = kinds["real"]
        if real["n"]:
            lines += ["", f"## Real cases ({name})", ""]
            lines.append(_md_row(["Case", "Expected", "Got", "Confidence", "Rule", "Correct"]))
            lines.append(_md_row(["---"] * 6))
            for c in real["cases"]:
                lines.append(
                    _md_row(
                        [
                            c["id"],
                            c["expected"],
                            c["got"],
                            c["confidence"],
                            c["rule"],
                            "yes" if c["correct"] else "no",
                        ]
                    )
                )
        failures = kinds["combined"]["failures"]
        lines += ["", f"## Failing cases ({name})", ""]
        if not failures:
            lines.append("None.")
        for f in failures:
            lines.append(
                f"- `{f['id']}`: expected {f['expected']}, got {f['got']} "
                f"({f['confidence']}) via {f['rule']}: {f['reason']}"
            )
    return "\n".join(lines) + "\n"


def save(report: dict[str, Any], name: str, out_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    """Write ``<name>.json`` and ``<name>.md`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{name}.json", out_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_markdown(report, f"Eval results: {name}"), encoding="utf-8")
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    parser.add_argument(
        "--sweep", type=int, default=20, help="extra seeds per pure-noise case (default 20)"
    )
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--real", type=Path, default=REAL_CASES_PATH)
    parser.add_argument("--save", metavar="NAME", help="also write eval/results/NAME.json and .md")
    args = parser.parse_args(argv)
    report = run(args.split, args.sweep, args.cases, args.real)
    if args.save:
        save(report, args.save)
    if args.json:
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    else:
        sys.stdout.write(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
