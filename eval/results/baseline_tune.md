# Eval results: baseline_tune

False-TREND target: <= 5.0% on pure noise; noise sweep: 20 extra seeds per pure-noise case.

| Split | Cases | n | Accuracy | False-TREND (cases) | False-TREND (sweep) |
| --- | --- | --- | --- | --- | --- |
| tune | synthetic | 27 | 74.1% (20/27) | 0.0% (0/7) | 0.7% (1/140) |
| tune | real | 6 | 0.0% (0/6) | - | - |
| tune | combined | 33 | 60.6% (20/33) | 0.0% (0/7) | 0.7% (1/140) |

## Per-label recall (combined)

| Split | TREND | FLUKE | SEASONAL | NO_CHANGE | INCONCLUSIVE |
| --- | --- | --- | --- | --- | --- |
| tune | 80.0% | 71.4% | 20.0% | 50.0% | 66.7% |

## Real cases (tune)

| Case | Expected | Got | Confidence | Rule | Correct |
| --- | --- | --- | --- | --- | --- |
| chatgpt_launch_hn | TREND up | INCONCLUSIVE | low | R6 | no |
| clubhouse_decline_weekly | TREND down | INCONCLUSIVE | low | R6 | no |
| suez_blockage | FLUKE | TREND up | low | R4 | no |
| halloween_daily | SEASONAL | TREND up | medium | R4 | no |
| coffee_steady | NO_CHANGE | INCONCLUSIVE | low | R6 | no |
| bluesky_election_surge | INCONCLUSIVE | TREND down | low | R4 | no |

## Failing cases (tune)

- `decaying_spike_daily`: expected FLUKE, got INCONCLUSIVE (low) via R6: the checks do not line up (persistence: no change; concentration: neutral; outliers: neutral; level shift: neutral)
- `seasonal_cosine_weekly`: expected SEASONAL, got INCONCLUSIVE (low) via R6: the checks do not line up (persistence: neutral; outliers: no change; level shift: no change)
- `seasonal_monthly`: expected SEASONAL, got NO_CHANGE (high) via R5: no significant slope, no outliers and no level shift in the recent window
- `seasonal_daily`: expected SEASONAL, got NO_CHANGE (high) via R5: no significant slope, no outliers and no level shift in the recent window
- `flat_daily`: expected NO_CHANGE, got INCONCLUSIVE (low) via R6: the checks do not line up (persistence: neutral; outliers: no change; level shift: no change)
- `weekday_daily`: expected NO_CHANGE, got INCONCLUSIVE (low) via R6: the checks do not line up (persistence: no change; outliers: neutral; level shift: no change)
- `low_count_daily`: expected NO_CHANGE, got INCONCLUSIVE (low) via R6: the checks do not line up (persistence: no change; outliers: no change; level shift: neutral; low count: neutral)
- `chatgpt_launch_hn`: expected TREND up, got INCONCLUSIVE (low) via R6: the checks do not line up (persistence: trend; concentration: neutral; outliers: neutral; level shift: trend; breadth: neutral); the trend-supporting checks disagree on the direction
- `clubhouse_decline_weekly`: expected TREND down, got INCONCLUSIVE (low) via R6: the checks do not line up (persistence: neutral; concentration: neutral; outliers: no change; level shift: no change)
- `suez_blockage`: expected FLUKE, got TREND up (low) via R4: a sustained level shift that is not concentrated in a single spike
- `halloween_daily`: expected SEASONAL, got TREND up (medium) via R4: a significant, practically large trend and a sustained level shift that is not concentrated in a single spike
- `coffee_steady`: expected NO_CHANGE, got INCONCLUSIVE (low) via R6: the checks do not line up (persistence: neutral; outliers: no change; level shift: no change)
- `bluesky_election_surge`: expected INCONCLUSIVE, got TREND down (low) via R4: a significant, practically large trend that is not concentrated in a single spike
