# Mammoths V2: Single vs Mutual Predator Captures

Source log:
- `/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/mammoths/ray_results/pred_decay_0_20/GRID_30_PRED_OBS_RANGE_9_MAMMOTHS_V_2_2025-12-25_00-36-33/PPO_PredPreyGrass_618a3_00000_0_2025-12-25_00-36-33/checkpoint_000049/eval_checkpoint_000049_2025-12-25_11-31-33/agent_event_log_2025-12-25_11-31-33.json`

Method:
- Unique success = unique `(t, prey_id, predator_list)` in predator `eating_events`.
- Unique failure = unique `(t, prey_id, predator_list)` in predator `failed_eating_events`.
- Single = team size 1; Mutual = team size >1.
- Per-attempt metrics are computed by grouping all predator logs for the same attempt.

## Successful Captures

### Capture Counts

| Group | Unique captures | Share of captures |
| --- | ---: | ---: |
| Single (1 predator) | 7 | 1.83% |
| Mutual (>1 predator) | 376 | 98.17% |
| Total | 383 | 100.00% |

### Team Size Distribution

| Team size | Captures |
| --- | ---: |
| 1 | 7 |
| 2 | 136 |
| 3 | 137 |
| 4 | 67 |
| 5 | 26 |
| 6 | 7 |
| 7 | 1 |
| 8 | 1 |
| 9 | 1 |

### Prey Energy at Capture

| Group | Sum prey energy | Share of prey energy | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single | 58.90 | 1.41% | 8.41 | 8.68 | 7.10 | 9.16 |
| Mutual | 4110.10 | 98.59% | 10.93 | 9.90 | 3.88 | 17.82 |

### Predator Energy Gain (from captures)

| Group | Sum bite size | Mean bite per capture | Median bite per capture | Mean bite per predator |
| --- | ---: | ---: | ---: | ---: |
| Single | 58.90 | 8.41 | 8.68 | 8.41 |
| Mutual | 4110.10 | 10.93 | 9.90 | 3.58 |

### Capture Timing (step t)

| Group | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Single | 482.57 | 302.00 | 161.00 | 837.00 |
| Mutual | 474.51 | 477.00 | 1.00 | 999.00 |

## Failed Attempts

### Attempt Counts

| Group | Unique failed attempts | Share of failed attempts |
| --- | ---: | ---: |
| Single (1 predator) | 2578 | 67.91% |
| Mutual (>1 predator) | 1218 | 32.09% |
| Total | 3796 | 100.00% |

### Team Size Distribution

| Team size | Failed attempts |
| --- | ---: |
| 1 | 2578 |
| 2 | 914 |
| 3 | 231 |
| 4 | 47 |
| 5 | 19 |
| 6 | 5 |
| 7 | 1 |
| 8 | 1 |

### Prey Energy at Failed Attempt

| Group | Sum prey energy | Share of prey energy | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single | 30825.76 | 66.54% | 11.96 | 11.71 | 1.26 | 17.86 |
| Mutual | 15499.72 | 33.46% | 12.73 | 12.74 | 2.26 | 17.86 |

### Failed Attempt Timing (step t)

| Group | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Single | 446.79 | 452.00 | 0.00 | 999.00 |
| Mutual | 424.62 | 412.50 | 1.00 | 999.00 |

### Failed Attempt Energy Delta (sum of energy_after - energy_before across predators)

| Group | Sum delta | Mean per attempt | Median | Min | Max | Mean per predator |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Mutual | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Data Integrity Checks

- Missing predator logs for grouped captures: 0
- Missing predator logs for grouped failed attempts: 0
- Total prey energy == total bite size (successes): 4169.00 == 4169.00
- Total failed attempt delta: 0.00 (non-zero attempts: 0)
