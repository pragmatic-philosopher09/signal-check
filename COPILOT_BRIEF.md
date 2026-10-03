# Signal Check — Build Brief for GitHub Copilot (v2)

> Working name: **Signal Check** (a single constant `APP_NAME` in `signalcheck/config.py`; easy to rename).
> How to use this file: keep it in the repo root. `.github/copilot-instructions.md` tells Copilot that sections 0–9 are binding. Paste the prompts from section 10 into Copilot one at a time, in order. Do not ask Copilot to build everything in one go.
>
> v2 changes (from review): precise seasonality definition and rule order, autocorrelation-robust trend test, capped recent window, defined concentration/change-point parameters, Reddit truncation + collector, Trends sampling/CSV quirks, Wikipedia article override, cheap HN counts, real-data eval with tune/test split, false-trend-rate metric, normalised narration validator, calendar-aligned cross-source comparison, CI/lint tooling.

---

## 0. What we are building

A free, non-commercial web app that answers one question about public attention data:

**"Is this change real and lasting, or is it noise?"**

The user enters a topic (e.g. "perplexity ai") or uploads a time series. The app pulls attention data from several sources (Google Trends, Reddit, X, plus a few free extras), and for each source returns a **verdict card**:

- Verdict: `TREND` (up or down), `FLUKE`, `SEASONAL`, `NO_CHANGE`, or `INCONCLUSIVE`
- Confidence: Low / Medium / High
- 3–4 evidence findings, each tied to a mark on the chart
- "What would change my mind": concrete, computed conditions that would flip the verdict
- Caveats specific to that source (scale, history length, partial periods, sampling, truncation)

A top card summarises cross-source agreement over a **shared calendar window** (e.g. "3 of 4 sources show an upward trend over 1–28 Sep").

This is a portfolio project. It must not be replicable by pasting data into a chatbot. The differentiators are:

1. **Deterministic statistics decide the verdict. The LLM never decides, it only narrates.**
2. **Breadth and low-count awareness**: a spike driven by one subreddit/author, or by 2 → 8 mentions, is flagged instead of called a trend.
3. **Cross-source corroboration**: direction is compared across sources over the same dates, never raw values.
4. **Computed "what would change my mind"** conditions.
5. **Measured quality**: a published eval with a held-out test set and an explicit false-trend rate on pure noise.

## 1. Hard rules (apply to all code you write)

- Python 3.12 (3.11 minimum). Type hints everywhere. Small pure functions. Docstrings state the statistical meaning of each check.
- Tooling: `ruff` (lint + format), `mypy` on `signalcheck/` (non-strict to start), `pytest`. GitHub Actions CI runs all three on every push/PR. `pre-commit` runs ruff locally.
- **No ML model for the verdict.** A transparent rule table over check outputs, with **every threshold in `config.yaml`** (never literals in engine code). No forecasting.
- **Determinism**: all randomness (eval generators, tests) uses an explicit seeded `numpy.random.Generator`.
- **The LLM narrates only.** It receives structured JSON of findings and may not introduce any fact, number or date that isn't in that JSON. Enforce with the validator in section 9.1. If no LLM key is set, or validation fails, fall back to template text so the app still works.
- Never interpolate missing data silently. Gaps are filled only if flagged in `meta.caveats`.
- Always drop or flag the **last, incomplete period** (see section 5).
- Never hard-code secrets. Use `.env` + `.env.example` locally and `st.secrets` on Streamlit Cloud (one `get_secret()` helper reads both). Never commit `.env` or `.streamlit/secrets.toml`.
- Every adapter failure degrades gracefully: show "Couldn't fetch <source>: <reason>" in that source's card. The app never crashes because one source failed. All HTTP calls have timeouts (connect 5s, read 20s) and bounded retries with exponential backoff + jitter on 429/5xx, honouring `Retry-After`.
- Cache every external call (disk cache, 6h TTL, keyed by source + query + normalised params).
- Store only **aggregates** (counts per bucket, contributor counts, shares). Author names / post text may be used transiently in memory to compute aggregates, never written to disk, cache, logs or snapshots.
- Respect each API's terms. User-Agent: `SignalCheck/<version> (+<repo-url>; <contact-email>)`. No scraping X. No third-party "unofficial" X scrapers.
- Write tests alongside code (`pytest`). Engine code needs unit tests with synthetic series. HTTP is mocked in tests (`responses` or `requests-mock`); no test hits the network.

## 2. Stack and layout

Python + **Streamlit** + **Plotly**. Libraries: `pandas`, `numpy`, `scipy`, `statsmodels` (STL), `pymannkendall`, `ruptures`, `pyyaml`, `diskcache`, `requests`, `python-dotenv`. Reddit via plain `requests` with OAuth (fewer moving parts than `praw`). Dev: `pytest`, `responses`, `ruff`, `mypy`, `pre-commit`.

Dependencies: `requirements.txt` (runtime, pinned with `==`, what Streamlit Cloud installs) and `requirements-dev.txt` (`-r requirements.txt` + dev tools). Tool config lives in `pyproject.toml`.

```
signal-check/
  app.py                       # Streamlit UI
  config.yaml                  # thresholds, windows, source toggles
  pyproject.toml               # ruff/mypy/pytest config
  requirements.txt
  requirements-dev.txt
  .env.example
  .pre-commit-config.yaml
  .github/
    copilot-instructions.md
    workflows/
      ci.yml                   # ruff, mypy, pytest
      collect.yml              # daily cron: scripts/collect_reddit.py (enabled once Reddit access is approved)
  signalcheck/
    config.py                  # APP_NAME, loads + validates config.yaml, get_secret()
    models.py                  # Series, Evidence, Verdict dataclasses
    cache.py
    http.py                    # shared session: UA, timeouts, retry/backoff
    adapters/
      base.py                  # Adapter protocol: fetch(query, params) -> Series
      csv_upload.py
      google_trends.py
      reddit.py
      x_twitter.py             # feature-flagged OFF by default
      wikipedia.py
      hackernews.py
    engine/
      preprocess.py
      checks/
        persistence.py
        concentration.py
        seasonality.py
        outliers.py
        level_shift.py
        low_count.py
        breadth.py
      verdict.py               # decision table
      confidence.py
      change_my_mind.py
      cross_source.py
    narrate.py
  scripts/
    refresh_samples.py         # run locally to refresh data/samples snapshots
    collect_reddit.py          # appends daily aggregate counts to data/collected/reddit/
  data/
    samples/                   # cached snapshots so the demo never depends on live APIs
    collected/                 # accumulated aggregate history (committed)
  eval/
    generators.py              # synthetic labeled series
    cases.yaml                 # synthetic cases, each tagged split: tune | test
    real_cases.yaml            # hand-labeled real snapshots, each tagged split
    run_eval.py                # confusion matrix, accuracy, false-trend rate, per split
  tests/
```

## 3. Core data contract (the engine never knows which source a series came from)

```python
@dataclass(frozen=True)
class Series:
    source: str        # "google_trends" | "reddit" | "x" | "wikipedia" | "hackernews" | "csv"
    query: str
    freq: str          # "D" | "W" | "M"
    points: pd.DataFrame  # columns: ts (date), value (float)
                          # optional: contributors (int), top_share (float 0-1), raw_count (int), imputed (bool)
    scale: str         # "relative_0_100" | "count" | "pageviews"
    meta: dict         # fetched_at, caveats: list[str], resolved_query (e.g. Wikipedia article),
                       # dropped_partial: {ts, value} | None, truncated_before: date | None

@dataclass(frozen=True)
class Evidence:
    check: str
    stance: str        # "supports_trend" | "supports_fluke" | "supports_seasonal" | "supports_no_change" | "neutral" | "skipped"
    summary: str       # one plain-language sentence, numbers included
    numbers: dict      # every number used in `summary`, with the same rounding
    annotation: dict | None   # what to draw on the chart (band, highlighted points, vertical line)
    skip_reason: str | None = None

@dataclass(frozen=True)
class Verdict:
    label: str         # TREND | FLUKE | SEASONAL | NO_CHANGE | INCONCLUSIVE
    direction: str | None     # "up" | "down" | None
    confidence: str    # "low" | "medium" | "high"
    rule_fired: str    # id of the decision-table row that matched (shown in UI)
    evidence: list[Evidence]
    change_my_mind: list[str]
    caveats: list[str]
    window: dict       # baseline/recent start+end dates, used by cross_source
```

`top_share` is defined per adapter as **the largest share of a bucket's items attributable to a single origin**: Reddit = max(top subreddit share, top author share); Hacker News = top author share; X = top author share. Adapters document which in `meta`.

## 4. Data sources — reality check (as of 3 Oct 2026; **re-verify before building each adapter**)

**Design consequence: all adapters turn a source into a `Series`. Build `csv_upload` first so the engine works even when every live API fails.**

| Source | Current reality | Adapter approach |
|---|---|---|
| **Google Trends** | No open official API (official Trends API is a gated alpha). `pytrends` was archived April 2025 and gets 429s, especially from cloud IPs. Values are **relative (0–100)**, scaled per request, and **sampled** (repeat requests return slightly different numbers). Granularity is set by timeframe: ≤ 90 days → daily, ≤ 5 years → weekly, longer → monthly. | Behind the adapter interface: (a) best-effort live fetch with pacing/backoff/cache; (b) **cached snapshots in `data/samples/`** refreshed by `scripts/refresh_samples.py` from the developer's own machine; (c) user uploads the CSV downloaded from the Trends website. The deployed demo must work from (b)/(c) alone. Always add caveats "relative scale, not volume" and "Google samples this data; small moves can be sampling noise". Map `<1` to `0.5` and set `imputed=True` for those points. |
| **Reddit** | Official Data API is free for **non-commercial** use, OAuth required, ~100 queries/min, app approval required. Listings return max 100 items per page and **search stops at ~1,000 results**. No time-series endpoint, no historical backfill. Search does not return comments. | Daily counts from search (sort=new), bucketed by day. Record `contributors` (unique authors) and `top_share` (see section 3). If the 1,000-result cap is hit, set `meta.truncated_before` to the earliest fully covered day, drop earlier days and add a caveat. **`scripts/collect_reddit.py` is required, not optional**: run daily (GitHub Actions cron, secrets in repo secrets) for a fixed watchlist and append aggregates to `data/collected/reddit/<slug>.csv`; the adapter merges collected history with live results. **Developer action: apply for API access today.** |
| **X (Twitter)** | **No free tier.** Pay-per-use since Feb 2026 (~$0.005 per post read); recent search covers ~7 days. | Behind `ENABLE_X=false`. Official API only. **Prefer the counts endpoint** (aggregate counts, no post reads) if available on the plan; fall back to post reads only for breadth on the recent window. Hard budget cap `X_MAX_SPEND_USD` (default 5) tracked in a persisted ledger; refuse requests that would exceed it. Public demo uses cached snapshots only. If disabled: card shows "X disabled (planned)". |
| **Wikipedia pageviews** | Free Wikimedia REST API, daily data, long history. Requires descriptive User-Agent. | Resolve query → article via the search API, **show the resolved title in the card and let the user override it**. Use `agent=user` (excludes known bots) and add caveat "unflagged automated traffic can still cause spikes". Best-behaved series for testing. |
| **Hacker News** | Free Algolia search API, no key, date-range filters via `numericFilters=created_at_i`. | **Counts**: one request per day with `hitsPerPage=0`, read `nbHits` (cheap, any history length). **Breadth**: fetch hits (≤ 1,000/day) only for the recent window to compute `contributors` and `top_share`; baseline days leave them null. |

Other candidates to evaluate later (verify access first): YouTube Data API (free quota), Bluesky, Mastodon.

## 5. Preprocessing rules

- Sort, dedupe (keep last), regularise to `freq`. Gaps: leave as NaN and add a caveat listing their count; only fill (linear, max 2 consecutive) when a check requires a regular series, set `imputed=True` and add a caveat.
- **Partial last period**: drop the last bucket if its period end is ≥ `fetched_at` (UTC). For CSV uploads, drop it if it contains the upload date; otherwise keep it, and expose a UI toggle "Last period is incomplete". Store the dropped point in `meta.dropped_partial` (drawn greyed on the chart).
- Minimum history (configurable): D ≥ 28 points, W ≥ 26, M ≥ 24. Below that → `INCONCLUSIVE` with reason "not enough history".
- **Recent window** = `clamp(round(recent_frac · n), recent_min[freq], recent_max[freq])` with defaults `recent_frac=0.2`, D: 7–28, W: 4–13, M: 3–6. **Baseline window** = all points before it.
- Robust baseline stats: `baseline_median`, `baseline_mad` (scaled by 1.4826). If MAD = 0, use `max(MAD, mad_floor · max(median, 1))`.
- Work on `log1p(values)` for count/pageview series; leave relative-scale series as is. Checks report numbers on the **original scale** in `summary`.

## 6. The checks (each returns an `Evidence`)

Each check takes `(series, windows, cfg)` and returns `Evidence`. A check that cannot run returns stance `skipped` with `skip_reason`.

1. **Persistence** (`persistence.py`):
   - Trend test on the recent window **plus the last `persistence_context` baseline points** (so short windows have power): `pymannkendall.hamed_rao_modification_test` when n ≥ 10, else `original_test`.
   - Theil–Sen slope, normalised as % of baseline level per period (on log scale: % change per period).
   - Count consecutive most-recent periods beyond `baseline_median ± 2·MAD` (direction-aware).
   - Supports trend if p < `mk_alpha` and |normalised slope| ≥ `min_slope`.
2. **Concentration** (`concentration.py`):
   - `excess_t = value_t − baseline_median` over the recent window; direction = sign of the sum.
   - If |Σ excess| < `excess_min_mads · MAD · n_recent` → `skipped` ("no meaningful excess").
   - Otherwise share of directional excess from the top 1 and top 2 periods (only same-signed excess counts). Top-1 share ≥ `conc_top1_max` or top-2 share ≥ `conc_top2_max` → supports fluke. Works for drops as well as rises.
3. **Seasonality** (`seasonality.py`):
   - **Weekly (day-of-week) patterns are a nuisance, not a verdict.** For daily data with ≥ 2 weeks, STL with period 7 is used only to produce a de-weekly-ised series that other checks may use; it never yields `supports_seasonal`.
   - **Annual seasonality** drives the SEASONAL verdict and requires ≥ 2 full years: D ≥ 730 points (period 365, STL on weekly-aggregated data, period 52), W ≥ 104 (period 52), M ≥ 24 (period 12). Otherwise `skipped` ("needs ≥ 2 years of history").
   - Robust STL (`robust=True`). Define `rise = mean(recent) − baseline_median` and `seasonal_rise = mean(S over recent) − mean(S over baseline)`. If |rise| < `excess_min_mads · MAD` → `skipped` ("no rise to explain"). Else `seasonal_share = clip(seasonal_rise / rise, 0, 1)`.
   - Re-run persistence and level shift on the seasonally adjusted series; report both in `numbers`.
4. **Outlier** (`outliers.py`): robust z = (x − baseline_median) / MAD for each recent point; flag points with |z| ≥ `outlier_z`. Reports the count and the max |z|.
5. **Level shift** (`level_shift.py`): `ruptures.Pelt(model="l2", min_size=min_persist)` on the robust-standardised series, penalty `pen_beta · log(n)` (`pen_beta` in config). Report whether a change point falls in or just before the recent window, its date, the before/after medians, and whether the new level has held for ≥ `min_persist` periods (≥ 80% of post-change points on the new side of the midpoint).
6. **Low count** (`low_count.py`): `count` series only. If baseline mean per bucket < `low_count_threshold` (default 5): conditional exact test on the rate ratio (recent total vs baseline total, scaled by window lengths) via `scipy.stats.binomtest`, report the ratio and its 95% CI. If the CI includes 1 → "too few events to call". Always downgrades confidence when active.
7. **Breadth** (`breadth.py`): only if `top_share`/`contributors` exist for the recent window. If volume-weighted mean `top_share` over the recent window ≥ `breadth_top_share_max`, or recent contributors per item fall below `breadth_min_contrib_ratio` → supports fluke ("one origin drives most of the volume").

## 7. Verdict decision table (`verdict.py`)

Evaluate in this order; first match wins. All thresholds come from `config.yaml`. Each row has a stable id (`R1`…`R6`) stored in `Verdict.rule_fired`.

1. **R1** Not enough history → `INCONCLUSIVE` (reason stated).
2. **R2** `seasonal_share ≥ seasonal_explained_min` **and** persistence on the seasonally adjusted series is not significant **and** no sustained level shift on the adjusted series → `SEASONAL`. (If the adjusted series still trends, fall through so a real trend on top of seasonality is called `TREND`; mention seasonality in evidence.)
3. **R3** (concentration supports fluke **and** persistence does not support trend) **or** breadth supports fluke → `FLUKE`.
4. **R4** (persistence supports trend **or** sustained level shift ≥ `min_persist`) **and** concentration does not support fluke → `TREND` with direction. Low-count "too few events to call" blocks this row and falls through.
5. **R5** No significant slope, no outliers, no shift in the recent window → `NO_CHANGE`.
6. **R6** Otherwise → `INCONCLUSIVE`, listing the specific conflicting checks.

**Confidence** (`confidence.py`, documented in code and in the UI "Methodology" expander):

```
score = 0
+1 per non-skipped check whose stance agrees with the verdict
-1 per non-skipped check whose stance contradicts it
-1 if history < 2 × minimum history
-1 if low-count flag active
-1 if seasonality was skipped and freq != "D"   (couldn't rule out annual seasonality)
-1 if source has truncation or imputed points in the recent window
high if score ≥ conf_high, medium if score ≥ conf_medium, else low
INCONCLUSIVE is always low.
```

**Multiple testing**: no per-check correction; instead the eval reports the **false-TREND rate on pure-noise series** and thresholds are tuned to keep it ≤ `target_false_trend_rate` (default 5%).

**Cross-source** (`cross_source.py`):
- Comparison window = the most recent calendar span covered by **all** compared sources (anchored to the latest common complete date, length = the shortest source's recent window in days). Sources whose recent window overlaps it by < 50% are listed as "not comparable (different period)".
- Count verdicts and directions among comparable sources and produce one sentence, e.g. "3 of 4 sources show an upward trend over 1–28 Sep 2026; Reddit looks like a fluke."
- Never compare raw values across sources.

## 8. "What would change my mind" (`change_my_mind.py`, deterministic)

Generate 1–3 conditions from the series itself, for example:

- `FLUKE`: "If the next `min_persist` days stay above X (baseline + 2·MAD), this becomes a trend."
- `TREND`: "If the next `min_persist` days fall back below Y (baseline median + 1·MAD), this was a fluke."
- `SEASONAL`: "If this period's value exceeds last year's same-period value by more than Z, something beyond seasonality is happening."
- `NO_CHANGE`: "A value above X (or below Y) would be unusual (|z| ≥ `outlier_z`)."
- `INCONCLUSIVE`: "Need N more <periods>, or contributor data, to decide." (N computed from the minimum-history gap.)

Use the series frequency for wording (days/weeks/months). Numbers come from the data on the original scale, rounded consistently, never invented. Each condition carries its numbers so the narration validator can see them.

## 9. UI (Streamlit)

- Input: topic text box, source checkboxes (disabled ones show why), timeframe selector, OR CSV upload (with the "last period is incomplete" toggle). A row of sample-topic chips that load cached snapshots instantly.
- Output: top cross-source summary card; then one card per source: verdict badge, confidence, rule fired, evidence bullets, annotated Plotly chart (baseline band, recent-window shading, highlighted spike points, change-point line, dropped partial period greyed, imputed points hollow), "What would change my mind" box, caveats list. Wikipedia card shows the resolved article with an override box.
- Optional LLM narration paragraph (≤ 120 words) under each card, clearly labelled "AI-written summary of the findings above" and generated from the JSON only.
- "Methodology" expander listing every check, threshold (read live from config), the decision table and the confidence formula.
- Clean, minimal, mobile-friendly. Use `st.cache_data` for engine results. No dashboards-of-everything.

### 9.1 Narration validator (`narrate.py`)

- Provider-agnostic: `LLM_PROVIDER` + `LLM_API_KEY` + `LLM_MODEL` env vars; temperature 0; ≤ 120 words.
- Prompt gives the JSON (verdict, rule, evidence summaries + numbers, change-my-mind, caveats) and instructs: use only these facts; copy numbers exactly as written.
- Validation, after normalisation:
  - Extract numbers from the narration: digits with optional `,` `.` `%` `x`/`×` and number words zero–twenty (`three` → 3).
  - Each extracted number must match some JSON number **as displayed** (same rounding as the `summary` strings), or that number rounded to fewer decimals; percentages match `0.123` or `12.3`.
  - Dates and month names must appear in the JSON.
  - Verdict label in the narration must equal `Verdict.label`.
- Any failure → discard and use the template text; log the failure reason (no content) for the eval log.

## 10. Phased prompts (paste one at a time; finish and test each before the next)

**Phase 0 — scaffold**
"Create the project layout from section 2 of COPILOT_BRIEF.md with empty modules, `requirements.txt`, `requirements-dev.txt`, `pyproject.toml` (ruff, mypy, pytest config), `.pre-commit-config.yaml`, `.github/workflows/ci.yml` (ruff, mypy, pytest on Python 3.12), `.env.example`, `.gitignore` (include `.env`, `.streamlit/secrets.toml`, cache dirs), and `config.yaml` containing every threshold named in sections 5–7 with sensible defaults. Implement `config.py` to load and validate `config.yaml` (fail fast on missing keys) and `get_secret()`. No engine logic yet. CI must pass."

**Phase 1 — models, preprocessing, CSV adapter**
"Implement `models.py` (section 3), `engine/preprocess.py` (section 5), `http.py`, `cache.py` and `adapters/csv_upload.py`. CSV must accept two columns (date, value) with flexible header names, and Google Trends' downloaded CSV format (leading metadata lines such as 'Category: All categories', `<1` values, multiple query columns → user picks one). Add tests for gaps, duplicates, partial last period (live fetch and CSV rules), too-short series, recent-window clamping, MAD = 0, and Trends CSV quirks."

**Phase 2 — checks**
"Implement the seven checks in section 6, one file each, returning `Evidence`. Use seeded synthetic series: linear trend (up and down), single spike, single dip, seasonal wave with and without an underlying trend, flat noise, step change, low-count Poisson noise, single-origin spike (high top_share). Each check must have at least one test where it supports the expected stance, one where it doesn't, and one where it is skipped (if it can be)."

**Phase 3 — verdict, confidence, change-my-mind, cross-source, eval harness**
"Implement `verdict.py` (section 7, with rule ids), `confidence.py`, `change_my_mind.py` (section 8) and `cross_source.py` (calendar-aligned). Build `eval/generators.py` and `eval/cases.yaml` with ~40 labeled synthetic series (vary noise, length, frequency, direction), each tagged `split: tune` or `split: test` (~70/30, stratified by label). Create `eval/real_cases.yaml` as a template with fields for ~10 hand-labeled real snapshots (I will fill them in). `eval/run_eval.py` prints, per split: confusion matrix, accuracy, per-label recall, and the false-TREND rate on pure-noise cases. Do not tune thresholds yet; report results and list the failing cases with the rule that fired."

**Phase 4 — adapters, in this order**
"Implement adapters in this order, each with caching, the shared HTTP session and graceful failure: (1) `wikipedia.py` (resolved article + override, `agent=user`), (2) `hackernews.py` (`nbHits` counts, breadth on recent window only), (3) `google_trends.py` with live best-effort + snapshot fallback + `scripts/refresh_samples.py`, (4) `reddit.py` per section 4 including `contributors`, `top_share`, 1,000-result truncation handling and merge with `data/collected/`, plus `scripts/collect_reddit.py` and `.github/workflows/collect.yml` (disabled until secrets exist), (5) `x_twitter.py` behind `ENABLE_X`, preferring the counts endpoint, with a persisted spend ledger and hard budget cap. Follow section 4 exactly and record caveats in `meta.caveats`. Mock HTTP in tests; assert no author names or text are written to disk."

**Phase 5 — UI**
"Build `app.py` per section 9 using the engine and adapters. Include sample-topic chips that load snapshots from `data/samples/`. Handle every adapter failure per card. Show the rule fired and the Methodology expander driven from `config.yaml`."

**Phase 6 — narration**
"Implement `narrate.py` per section 9.1: build the structured JSON from a `Verdict`, call an LLM provider selected by env var (provider-agnostic interface, one concrete implementation), enforce the normalised numbers/dates/label validator, and fall back to templates when no key is set or validation fails. Unit-test the validator with passing and failing narrations (rounded numbers, number words, invented numbers, wrong label, invented dates)."

**Phase 7 — tune and ship**
"Review eval failures on the **tune** split only. Tune thresholds in `config.yaml` (never in code) to maximise accuracy subject to false-TREND rate ≤ `target_false_trend_rate`. Then run once on the **test** split (synthetic + real) and commit that table to the README; do not re-tune after seeing test results. Add a README with the thesis, architecture diagram (Mermaid), methodology, limitations (especially source access limits, Trends sampling, Reddit truncation), eval results, and deployment steps for Streamlit Community Cloud (Python 3.12, secrets via `st.secrets`)."

## 11. Open items for the human (not for Copilot)

- Apply for Reddit API access now; once approved, add secrets to the repo and enable `collect.yml` with a watchlist of demo topics so history accumulates.
- Decide whether to allocate a small X budget (default: leave X disabled and show it as "planned").
- Run `scripts/refresh_samples.py` from your own machine before each demo/deploy so Google Trends snapshots are fresh.
- Fill `eval/real_cases.yaml` with ~10 hand-labeled real events (e.g. a product launch → TREND, Super Bowl → SEASONAL, a one-day viral story → FLUKE, a steady topic → NO_CHANGE). Label them **before** running the engine on them.
- Keep a short log of eval results over time; it becomes the "how I measured quality" section of the portfolio write-up.
