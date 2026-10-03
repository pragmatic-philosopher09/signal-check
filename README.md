# Signal Check

**Is this change real and lasting, or is it noise?**

A free, non-commercial web app that gives a deterministic, evidence-backed verdict (`TREND`, `FLUKE`, `SEASONAL`, `NO_CHANGE`, `INCONCLUSIVE`) on public attention data from Google Trends, Reddit, Wikipedia, Hacker News and (optionally) X.

Status: Phase 1 (data models, preprocessing, HTTP/cache layer, CSV upload incl. Google Trends downloads). The build specification is in [`COPILOT_BRIEF.md`](COPILOT_BRIEF.md).

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
