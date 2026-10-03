# Signal Check

**Is this change real and lasting, or is it noise?**

A free, non-commercial web app that gives a deterministic, evidence-backed verdict (`TREND`, `FLUKE`, `SEASONAL`, `NO_CHANGE`, `INCONCLUSIVE`) on public attention data from Google Trends, Reddit, Wikipedia, Hacker News and (optionally) X.

Status: Phase 3 (data models, preprocessing, HTTP/cache layer, CSV upload, the seven statistical checks, the R1–R6 decision table, confidence, "what would change my mind", cross-source comparison and the synthetic eval). No UI yet. The build specification is in [`COPILOT_BRIEF.md`](COPILOT_BRIEF.md).

## Development

Requires Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
cp .env.example .env   # fill in any keys you have

ruff check . && ruff format --check . && mypy signalcheck && pytest
```

All engine thresholds live in [`config.yaml`](config.yaml); `signalcheck/config.py` validates it on load and fails fast on missing keys.

## Engine

```python
from signalcheck.engine import analyse, compare_sources

verdict = analyse(series)          # preprocess -> checks -> R1–R6 -> confidence -> change-my-mind
summary = compare_sources({"wikipedia": v1, "reddit": v2})  # calendar-aligned one-liner + counts
```

`analyse` never raises for a short series; it returns `INCONCLUSIVE` via rule R1.

## Evaluation

```bash
python -m eval.run_eval                      # both splits, text report
python -m eval.run_eval --split test --json  # one split, JSON
python -m eval.run_eval --sweep 0            # skip the extra pure-noise seeds
```

[`eval/cases.yaml`](eval/cases.yaml) holds ~40 labelled synthetic series (seeded builders in [`eval/generators.py`](eval/generators.py)), each tagged `split: tune` or `split: test`. The report prints, per split, the confusion matrix, accuracy, per-label recall, the false-TREND rate on pure-noise cases (plus a seeded noise sweep) and every failing case with the rule that fired. Hand-labelled real snapshots go in [`eval/real_cases.yaml`](eval/real_cases.yaml) (see [`eval/snapshots/README.md`](eval/snapshots/README.md)); unlabelled entries are skipped.
