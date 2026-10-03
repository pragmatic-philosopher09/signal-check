# Eval results: final_test

False-TREND target: <= 5.0% on pure noise; noise sweep: 20 extra seeds per pure-noise case.

| Split | Cases | n | Accuracy | False-TREND (cases) | False-TREND (sweep) |
| --- | --- | --- | --- | --- | --- |
| test | synthetic | 13 | 92.3% (12/13) | 0.0% (0/3) | 1.7% (1/60) |
| test | real | 5 | 60.0% (3/5) | - | - |
| test | combined | 18 | 83.3% (15/18) | 0.0% (0/3) | 1.7% (1/60) |

## Per-label recall (combined)

| Split | TREND | FLUKE | SEASONAL | NO_CHANGE | INCONCLUSIVE |
| --- | --- | --- | --- | --- | --- |
| test | 66.7% | 75.0% | 100.0% | 100.0% | 100.0% |

## Real cases (test)

| Case | Expected | Got | Confidence | Rule | Correct |
| --- | --- | --- | --- | --- | --- |
| openai_rise_weekly | TREND up | INCONCLUSIVE | low | R6 | no |
| nft_decline_weekly | TREND down | TREND down | low | R4 | yes |
| crowdstrike_outage | FLUKE | TREND up | low | R4 | no |
| super_bowl_weekly | SEASONAL | SEASONAL | medium | R2 | yes |
| python_hn_steady | NO_CHANGE | NO_CHANGE | high | R5 | yes |

## Failing cases (test)

- `trend_on_seasonal_weekly`: expected TREND up, got NO_CHANGE (high) via R5: no significant slope, no outliers and no level shift in the recent window
- `openai_rise_weekly`: expected TREND up, got INCONCLUSIVE (low) via R6: the checks do not line up (persistence: trend; concentration: neutral; outliers: fluke; level shift: no change); the trend-supporting checks and the recent departure from the baseline disagree on the direction
- `crowdstrike_outage`: expected FLUKE, got TREND up (low) via R4: a sustained level shift that is not concentrated in a single spike
