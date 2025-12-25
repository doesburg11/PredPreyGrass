# Mammoths: Single vs Mutual Predator Captures

Source log:
- `/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/mammoths/ray_results/pred_decay_0_20/GRID_30_PRED_OBS_RANGE_9_MAMMOTHS_2025-12-23_23-09-36/PPO_PredPreyGrass_119c3_00000_0_2025-12-23_23-09-36/checkpoint_000000/eval_checkpoint_000000_2025-12-24_19-56-27/agent_event_log_2025-12-24_19-56-27.json`

Method:
- Unique success = unique `(t, prey_id, predator_list)` in predator `eating_events`.
- Unique failure = unique `(t, prey_id, predator_list)` in predator `failed_eating_events`.
- Single = team size 1; Mutual = team size >1.
- Per-attempt metrics are computed by grouping all predator logs for the same attempt.

## Successful Captures

### Capture Counts

| Group | Unique captures | Share of captures |
| --- | ---: | ---: |
| Single (1 predator) | 87 | 20.00% |
| Mutual (>1 predator) | 348 | 80.00% |
| Total | 435 | 100.00% |

### Team Size Distribution

| Team size | Captures |
| --- | ---: |
| 1 | 87 |
| 2 | 201 |
| 3 | 121 |
| 4 | 20 |
| 5 | 5 |
| 6 | 1 |

### Prey Energy at Capture

| Group | Sum prey energy | Share of prey energy | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single | 516.70 | 16.12% | 5.94 | 5.90 | 0.88 | 9.08 |
| Mutual | 2687.96 | 83.88% | 7.72 | 7.50 | 4.40 | 11.72 |

### Predator Energy Gain (from captures)

| Group | Sum bite size | Mean bite per capture | Median bite per capture | Mean bite per predator |
| --- | ---: | ---: | ---: | ---: |
| Single | 516.70 | 5.94 | 5.90 | 5.94 |
| Mutual | 2687.96 | 7.72 | 7.50 | 3.07 |

### Capture Timing (step t)

| Group | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Single | 404.80 | 346.00 | 21.00 | 916.00 |
| Mutual | 485.66 | 480.50 | 7.00 | 980.00 |

## Failed Attempts

### Attempt Counts

| Group | Unique failed attempts | Share of failed attempts |
| --- | ---: | ---: |
| Single (1 predator) | 1796 | 79.12% |
| Mutual (>1 predator) | 474 | 20.88% |
| Total | 2270 | 100.00% |

### Team Size Distribution

| Team size | Failed attempts |
| --- | ---: |
| 1 | 1796 |
| 2 | 408 |
| 3 | 58 |
| 4 | 7 |
| 5 | 1 |

### Prey Energy at Failed Attempt

| Group | Sum prey energy | Share of prey energy | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single | 14343.88 | 77.66% | 7.99 | 7.80 | 1.98 | 11.86 |
| Mutual | 4125.72 | 22.34% | 8.70 | 8.60 | 5.18 | 11.84 |

### Failed Attempt Timing (step t)

| Group | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Single | 470.48 | 450.00 | 0.00 | 977.00 |
| Mutual | 511.07 | 483.00 | 5.00 | 979.00 |

### Failed Attempt Energy Delta (sum of energy_after - energy_before across predators)

| Group | Sum delta | Mean per attempt | Median | Min | Max | Mean per predator |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Mutual | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Data Integrity Checks

- Missing predator logs for grouped captures: 0
- Missing predator logs for grouped failed attempts: 0
- Total prey energy == total bite size (successes): 3204.66 == 3204.66
- Total failed attempt delta: 0.00 (non-zero attempts: 0)
