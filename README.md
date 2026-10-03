# Signal Check

**Is this change real and lasting, or is it noise?**

A free, non-commercial web app that gives a deterministic, evidence-backed verdict (`TREND`, `FLUKE`, `SEASONAL`, `NO_CHANGE`, `INCONCLUSIVE`) on public attention data from Google Trends, Reddit, Wikipedia, Hacker News and (optionally) X.

Status: Phase 4 (data models, preprocessing, HTTP/cache layer, CSV upload, the seven statistical checks, the R1–R6 decision table, confidence, "what would change my mind", cross-source comparison, the synthetic eval, and live adapters for Wikipedia, Hacker News, Google Trends, Reddit and X). No UI yet. The build specification is in [`COPILOT_BRIEF.md`](COPILOT_BRIEF.md).

## Development

Requires Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
cp .env.example .env   # fill in any keys you have

ruff check . && ruff format --check . && mypy signalcheck && pytest
pytest -m live   # opt-in smoke tests against the real Wikipedia and Hacker News APIs
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

## Data sources

Each adapter in [`signalcheck/adapters/`](signalcheck/adapters) returns a `Series` through the shared HTTP session (User-Agent `SignalCheck/<version> (+<repo>; <SIGNALCHECK_CONTACT_EMAIL>)`, timeouts, retry/backoff) and the 6-hour disk cache. Use `fetch_safely(adapter, query, params)` to get a `FetchResult`: either a series or a message such as "Couldn't fetch Reddit: HTTP 503 after 5 attempt(s)" or "Reddit disabled (API credentials not configured)". Caveats are recorded in `series.meta["caveats"]`. Only per-day aggregates are cached or written to disk; no author names, ids or post text.

| Source | Needs | Notes |
| :- | :- | :- |
| Wikipedia | nothing | Pageviews (`agent=user`) for the resolved article; pass `{"article": "..."}` to override. |
| Hacker News | nothing | Algolia `nbHits` per day; breadth from authors on the recent window only. |
| Google Trends | optional `pytrends` | Live fetch is best effort; otherwise the snapshot in `data/samples/google_trends/` is used. |
| Reddit | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` (+ optional `REDDIT_USERNAME`) | App-only OAuth. Reddit now requires [approval](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy) for API access. |
| X | `ENABLE_X=true`, `X_BEARER_TOKEN`, `X_MAX_SPEND_USD` | Off by default ("X disabled (planned)"). Uses the counts endpoint (counts/all at $0.010 per request). Spend is tracked in `<cache.dir>/x_spend_ledger.json`, and any request that would exceed the budget is refused. |

### Sample snapshots

`python -m scripts.refresh_samples` refreshes `data/samples/<source>/<slug>.json` for the topics in `config.yaml` (`samples.topics`) and every enabled source except X; add `--sources x` to opt in, which spends from the X budget. Options include `--topics "a" "b"` and `--sources wikipedia hackernews`.

Google Trends has no official API. To create a Trends snapshot without `pytrends`, download the CSV from trends.google.com (Explore → ⤓) and import it:

```bash
python -m scripts.refresh_samples --import-trends-csv ~/Downloads/multiTimeline.csv \
    --query "perplexity ai" --fetched-at 2026-10-03T12:00:00Z
```

`--fetched-at` should be the download time; the partial-period rule depends on it.

### Reddit history collector

Reddit search stops at about 1,000 results, so long histories for busy topics are built up by a daily collector. `python -m scripts.collect_reddit` appends complete-day aggregates for `adapters.reddit.watchlist` to `data/collected/reddit/<slug>.csv`. The adapter merges these files in front of its live window. [`.github/workflows/collect.yml`](.github/workflows/collect.yml) runs the collector daily and commits the CSVs. It does nothing until these repository secrets exist: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` and optionally `REDDIT_USERNAME` and `SIGNALCHECK_CONTACT_EMAIL`.
