"""Run the verdict engine over the labelled eval cases and report how it does.

Usage::

    python -m eval.run_eval                 # both splits, text report
    python -m eval.run_eval --split test    # one split
    python -m eval.run_eval --json          # machine-readable output
    python -m eval.run_eval --sweep 0       # skip the extra pure-noise seeds (default 20)

For each split (``tune``/``test``; real snapshots from ``eval/real_cases.yaml``
are reported as ``real-tune``/``real-test`` once labelled) it prints the
confusion matrix (rows = expected label, columns = verdict), accuracy (a TREND
counts as correct only with the right direction), per-label recall, the
false-TREND rate on pure-noise cases (target from ``eval.target_false_trend_rate``)
and every failing case with the rule that fired and its reason.

Everything is deterministic: each case has a fixed seed, and ``--sweep`` derives
its extra seeds from the case seed with :class:`numpy.random.SeedSequence`.
This script reports; it never changes thresholds.
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
import yaml

from eval.generators import GENERATORS
from signalcheck.adapters.csv_upload import parse_csv
from signalcheck.config import Config, get_config
from signalcheck.engine import analyse
from signalcheck.models import DIRECTIONS, LABELS, Series

EVAL_DIR = Path(__file__).resolve().parent
CASES_PATH = EVAL_DIR / "cases.yaml"
REAL_CASES_PATH = EVAL_DIR / "real_cases.yaml"
SPLITS: tuple[str, ...] = ("tune", "test")
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

    @property
    def group(self) -> str:
        """Report section: ``tune``/``test`` for synthetic, ``real-<split>`` for real."""
        return self.split if self.kind == "synthetic" else f"{self.kind}-{self.split}"


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


def real_builder(entry: dict[str, Any], snapshot: Path, cfg: Config) -> Callable[[], Series]:
    """A builder parsing a real CSV snapshot with the entry's options."""

    def build() -> Series:
        return parse_csv(
            snapshot.read_bytes(),
            query=entry.get("query"),
            value_column=entry.get("value_column"),
            scale=entry.get("scale"),
            freq=entry.get("freq"),
            upload_date=entry.get("upload_date"),
            last_period_incomplete=bool(entry.get("last_period_incomplete", False)),
            cfg=cfg,
        )

    return build


def load_real_cases(path: Path, cfg: Config) -> tuple[list[Case], list[str]]:
    """Labelled real cases whose snapshot exists, plus the ids of incomplete entries."""
    if not path.exists():
        return [], []
    raw = yaml.safe_load(path.read_text()) or {}
    cases: list[Case] = []
    skipped: list[str] = []
    for entry in raw.get("cases") or []:
        case_id = str(entry.get("id"))
        snapshot = entry.get("snapshot")
        file = path.parent / snapshot if snapshot else None
        if entry.get("label") is None or file is None or not file.exists():
            skipped.append(case_id)
            continue
        _validate(entry, case_id)
        cases.append(
            Case(
                id=case_id,
                label=entry["label"],
                direction=entry.get("direction"),
                split=entry["split"],
                noise=False,
                kind="real",
                build=real_builder(entry, file, cfg),
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


def summarise(results: Sequence[Result], sweep: Sequence[Result] = ()) -> dict[str, Any]:
    """Confusion matrix, accuracy, recall, false-TREND rate and failures for one split."""
    confusion = {e: {p: 0 for p in LABELS} for e in LABELS}
    for r in results:
        confusion[r.case.label][r.label] += 1
    recall: dict[str, float | None] = {}
    for label in LABELS:
        of_label = [r for r in results if r.case.label == label]
        recall[label] = sum(r.correct for r in of_label) / len(of_label) if of_label else None
    n_correct = sum(r.correct for r in results)
    failures = [
        {
            "id": r.case.id,
            "expected": r.case.label + (f" {r.case.direction}" if r.case.direction else ""),
            "got": r.label + (f" {r.direction}" if r.direction else ""),
            "confidence": r.confidence,
            "rule": r.rule,
            "reason": r.reason,
        }
        for r in results
        if not r.correct
    ]
    out: dict[str, Any] = {
        "n": len(results),
        "correct": n_correct,
        "accuracy": n_correct / len(results) if results else None,
        "confusion": confusion,
        "recall": recall,
        "false_trend": false_trend(results),
        "failures": failures,
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
    real, skipped = load_real_cases(real_path, cfg)
    wanted = SPLITS if split == "all" else (split,)
    selected = [c for c in [*cases, *real] if c.split in wanted]
    results = evaluate(selected, cfg)
    sweep_results = evaluate(
        noise_sweep([c for c in cases if c.split in wanted], sweep, cases_path), cfg
    )
    groups: dict[str, Any] = {}
    for group in dict.fromkeys(c.group for c in selected):
        in_group = [r for r in results if r.case.group == group]
        group_sweep = [r for r in sweep_results if r.case.split == group]
        groups[group] = summarise(in_group, group_sweep)
    return {
        "target_false_trend_rate": float(cfg["eval"]["target_false_trend_rate"]),
        "sweep_seeds": sweep,
        "splits": groups,
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
    """The full text report."""
    lines: list[str] = []
    for name, s in report["splits"].items():
        lines.extend(format_split(name, s, report["target_false_trend_rate"]))
        lines.append("")
    if report["real_cases_skipped"]:
        lines.append(
            f"real cases not yet labelled or missing a snapshot: "
            f"{', '.join(report['real_cases_skipped'])}"
        )
    return "\n".join(lines).rstrip() + "\n"


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
    args = parser.parse_args(argv)
    report = run(args.split, args.sweep, args.cases, args.real)
    if args.json:
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    else:
        sys.stdout.write(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
