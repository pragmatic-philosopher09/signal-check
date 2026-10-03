# Tuning log (Phase 7)

Each entry is a config diff or engine change, followed by its **tune-split** metrics: synthetic
cases from `eval/cases.yaml` plus real cases from `eval/real_cases.yaml`, replayed offline from
`eval/real_data/`. The objective was to maximise combined tune accuracy while keeping the
false-TREND rate on pure noise at or below `eval.target_false_trend_rate` (5%). That rate is
measured on the noise cases and on a noise sweep of 20 extra seeds per pure-noise case (140 series).

To reproduce: `python -m eval.run_eval --split tune --sweep 20`. Saved outputs are in
[`results/`](results/): `baseline_tune.*`, `final_tune.*` and `final_test.*`.

## Disclosures

- **The synthetic test split had already been seen.** Phase 3 reported all-split results, and the
  first eval run in this session (before any tuning) printed the synthetic test split: 84.6%
  (11/13). No tuning decision below used test-split results. Still, the synthetic test split is
  not a fully blind hold-out. The **real** test cases were never run before the final test run.
- **Real-case labels were pre-registered** in commit `1af24ab` ("eval: pre-register real-case
  labels"), before `analyse` ran on any of them. One case changed before that commit: the planned
  "Large language model" article had no usable pageview history for the chosen range, so
  `openai_rise_weekly` replaced it. The swap was driven by data availability, not results. No
  label has changed since.
- Labels were assigned by the agent from world knowledge and are **pending human review**.
  Questionable labels are listed under the remaining failures; they were left unchanged on purpose.

## Baseline (Phase 6 config, before any change)

| Tune | Accuracy | False-TREND (noise cases) | False-TREND (sweep) |
| --- | --- | --- | --- |
| synthetic | 74.1% (20/27) | 0/7 | 1/140 (0.7%) |
| real | 0.0% (0/6) | - | - |
| combined | 60.6% (20/33) | 0/7 | 1/140 (0.7%) |

Tune failures at baseline:
- **Synthetic:**
  - SEASONAL recall 1/4: seasonality skipped with "no rise to explain".
  - flat_daily and weekday_daily ended INCONCLUSIVE.
  - decaying_spike_daily was INCONCLUSIVE.
- **Real:** all six missed.
  - halloween_daily: TREND, because an adjusted level shift *down* blocked R2.
  - coffee_steady: INCONCLUSIVE, because a significant 0.2%/day slope blocked R5.
  - bluesky_election_surge: TREND down, a decay that was still far above baseline.
  - clubhouse_decline_weekly, chatgpt_launch_hn and suez_blockage: see the remaining failures below.

## Iterations

Metrics are tune-split counts after each step. Each step is cumulative.

| # | Change | Kind | Synthetic | Real | Combined | Sweep false-TREND |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | baseline | - | 20/27 | 0/6 | 20/33 | 1/140 |
| 1 | `checks.seasonality.excess_min_mads` 1.0 → 0.5 | config | 23/27 | 0/6 | 23/33 | 1/140 |
| 2 | E1: seasonal residual must run in the direction of the rise | engine | 23/27 | 1/6 | 24/33 | not recorded |
| 3 | E2: minimum effect size `checks.persistence.negligible_slope: 0.005` | engine + config | 24/27 | 2/6 | 26/33 | not recorded |
| 4 | E3: outliers score the de-weekly-ised daily series (fixed baseline profile) | engine | 25/27 | 2/6 | 27/33 | not recorded |
| 5 | E4: R4 also needs the departure direction to agree | engine | 25/27 | 3/6 | 28/33 | 1/140 |
| 6 | `checks.concentration.conc_top1_max` 0.6 → 0.4 | config | 26/27 | 3/6 | 29/33 | 1/140 |
| 7 | `checks.persistence.min_slope` 0.02 → 0.0125 | config | 26/27 | 4/6 | **30/33** | 2/140 (1.4%) |

### 1. Seasonality `excess_min_mads` 1.0 → 0.5

The scaled MAD (×1.4826) of a series with an annual cycle is roughly the wave's amplitude. A
cosine season's peak-window mean is about 0.86 × amplitude above the median, so it could never
clear 1.0 MAD. The check therefore skipped with "no rise to explain", and SEASONAL was
structurally unreachable. This fixed seasonal_cosine_weekly, seasonal_monthly and seasonal_daily.

Scan: 0.25, 0.4 and 0.5 gave the same result (23/33); 0.6 and 0.75 gave 22/33. I chose 0.5,
the largest value on the plateau. It is a gate on whether there is anything to explain; the 70%
share bar (`verdict.seasonal_explained_min`) still decides.

### 2. E1: residual direction in the seasonality check

The seasonality check used to call a change "not just seasonal" if the adjusted series had *any*
significant trend or sustained shift. In halloween_daily the annual peak lifts the recent window
while the adjusted level dips. A residual running *against* the rise cannot have produced it.

Now only an adjusted trend or shift in the rise's direction blocks the seasonal call. The logic
is in `seasonality.residual_with_change`, and the summary says so explicitly. Tests:
`test_residual_with_change_needs_the_same_direction` and
`test_residual_against_the_rise_leaves_it_seasonal`.

### 3. E2: minimum effect size for "no change"

With 90+ days of data, Hamed-Rao Mann-Kendall makes tiny slopes significant: 0.2%/day on a
steady topic, or on flat_daily. Such a slope made persistence neutral and blocked R5, so steady
series became INCONCLUSIVE. A slope that is significant but below `negligible_slope` (0.5% of
baseline per period) is now practically nil and supports no change. Between `negligible_slope`
and `min_slope` the check stays neutral. The config validates
`0 ≤ negligible_slope ≤ min_slope`.

Scan: 0.0025 and 0.01 gave the same result as 0.005. This fixed flat_daily and coffee_steady.

### 4. E3: outliers on the de-weekly-ised series

Phase 3 built the de-weekly-ised series (section 6.3), but no check used it. In weekday_daily,
weekends were flagged as outliers. The outlier check now scores daily data after subtracting a
**fixed** day-of-week profile: the mean STL weekly component per weekday over the baseline.

Two simpler versions were tried and rejected:
- Plain STL on the whole series leaked a spike into the same weekday a week earlier.
- In-sample STL shrank the baseline MAD and created spurious flags.

Displayed values stay on the observed scale. This fixed weekday_daily.

### 5. E4: R4 departure-direction agreement

bluesky_election_surge, which is a post-surge decay, was called "TREND down": the recent slope was
significantly negative while the level was still far above the baseline. A fall while the
level is still far above the baseline is a decay towards it, not a downward trend.

R4 now requires the direction of the recent excess (concentration's direction, when it ran) to
agree with the trend-supporting checks. Otherwise R4 falls through to R6. Tests:
`test_r4_blocked_when_departure_runs_against_the_trend` and `test_r4_holds_when_departure_agrees`.

### 6. Concentration `conc_top1_max` 0.6 → 0.4

A spike decaying with a half-life of about one period puts at most about 50% of its excess on the
first period: 44.5% in decaying_spike_daily. So 0.6 only caught near-single-point spikes.

Scan: 0.35 broke trend_up_monthly, whose top-1 share is 37.1% because the window has n=6 and a
monthly ramp's last point carries a big share. 0.4 gave 29/33. 0.45 and 0.5 gave 28/33.

**The margin is thin on both sides** (37.1% vs 44.5%). This is structural for short monthly
windows. An n-aware rule would be more robust, but it was not added, to keep the engine change
count small.

Rejected: `conc_top2_max` 0.65. It also fixed decaying_spike_daily but broke trend_up_monthly
(top-2 share 66.8%).

### 7. Persistence `min_slope` 0.02 → 0.0125

clubhouse_decline_weekly declines about 1.3%/week, below the 2% bar.

Scan, on top of iteration 6:

| min_slope | Combined | Sweep false-TREND |
| --- | --- | --- |
| 0.0075 | 30/33 | 3/140 |
| 0.01 | 30/33 | 2/140 |
| 0.0125 | 30/33 | 2/140 |
| 0.015 | 29/33 | 1/140 |
| 0.02 | 29/33 | 1/140 |

0.0125 is the most conservative point of the 30/33 plateau, and its sweep rate (1.4%) is well
inside the 5% target. Following the stated objective (maximise accuracy subject to false-TREND
≤ target), I took it.

Rejected: 0.03. It "fixed" chatgpt_launch_hn only because that case's −2.9%/day slope sits just
under 3%. That would be fitting one case, and it lowers sensitivity to real trends.

### Other scans (no change kept)

- `checks.outliers.isolated_max_points` 1, 3, 4: no change.
- `checks.outliers.outlier_z` 3.0: worse (26/33). 4.0: no change.

## Final tune result

| Tune | Accuracy | False-TREND (noise cases) | False-TREND (sweep) |
| --- | --- | --- | --- |
| synthetic | 96.3% (26/27) | 0/7 | 2/140 (1.4%) |
| real | 66.7% (4/6) | - | - |
| combined | 90.9% (30/33) | 0/7 | 2/140 (1.4%) |

### Remaining tune failures (left as is)

- **low_count_daily** (NO_CHANGE → INCONCLUSIVE, R6): the change point found in the recent window
  did not hold, and low-count says "too few to call". Fixing it would need a low-count-specific
  R5 exception. Not worth the risk.
- **chatgpt_launch_hn** (TREND up → INCONCLUSIVE, R6): the recent window ends over the late-December
  holiday dip. The slope over the window is down (−2.9%/day) while the level shift and departure
  are up, so the directions disagree.
  - *Label note:* the label is right in hindsight, but the chosen `as_of` puts the holiday dip in
    the window. INCONCLUSIVE is a defensible answer from that data.
- **suez_blockage** (FLUKE → TREND up, R4): `as_of` is three weeks after the event and pageviews
  were still about 4× baseline, so a sustained level shift is what the data shows at that cutoff.
  - *Label note:* "fluke" assumes hindsight. A later `as_of` would show the decay. The label was
    not changed (pre-registered).

## Test run (once, after freezing the config above)

Recorded below after a single run of `python -m eval.run_eval --split test --sweep 20 --save
final_test`. No tuning happened after it.
