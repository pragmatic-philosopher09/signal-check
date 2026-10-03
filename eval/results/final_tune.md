# Eval results: final_tune

False-TREND target: <= 5.0% on pure noise; noise sweep: 20 extra seeds per pure-noise case.

| Split | Cases | n | Accuracy | False-TREND (cases) | False-TREND (sweep) |
| --- | --- | --- | --- | --- | --- |
| tune | synthetic | 27 | 96.3% (26/27) | 0.0% (0/7) | 1.4% (2/140) |
| tune | real | 6 | 66.7% (4/6) | - | - |
| tune | combined | 33 | 90.9% (30/33) | 0.0% (0/7) | 1.4% (2/140) |

## Per-label recall (combined)

| Split | TREND | FLUKE | SEASONAL | NO_CHANGE | INCONCLUSIVE |
| --- | --- | --- | --- | --- | --- |
| tune | 90.0% | 85.7% | 100.0% | 87.5% | 100.0% |

## Real cases (tune)

| Case | Expected | Got | Confidence | Rule | Correct |
| --- | --- | --- | --- | --- | --- |
| chatgpt_launch_hn | TREND up | INCONCLUSIVE | low | R6 | no |
| clubhouse_decline_weekly | TREND down | TREND down | low | R4 | yes |
| suez_blockage | FLUKE | TREND up | low | R4 | no |
| halloween_daily | SEASONAL | SEASONAL | medium | R2 | yes |
| coffee_steady | NO_CHANGE | NO_CHANGE | high | R5 | yes |
| bluesky_election_surge | INCONCLUSIVE | INCONCLUSIVE | low | R6 | yes |

## Failing cases (tune)

- `low_count_daily`: expected NO_CHANGE, got INCONCLUSIVE (low) via R6: the checks do not line up (persistence: no change; outliers: no change; level shift: neutral; low count: neutral)
- `chatgpt_launch_hn`: expected TREND up, got INCONCLUSIVE (low) via R6: the checks do not line up (persistence: trend; concentration: neutral; outliers: neutral; level shift: trend; breadth: neutral); the trend-supporting checks and the recent departure from the baseline disagree on the direction
- `suez_blockage`: expected FLUKE, got TREND up (low) via R4: a sustained level shift that is not concentrated in a single spike
