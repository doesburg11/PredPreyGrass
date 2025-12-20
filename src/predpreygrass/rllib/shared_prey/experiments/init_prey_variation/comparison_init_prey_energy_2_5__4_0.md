# Shared prey eval comparison: init_prey=2.5/threshold=5.5 vs init_prey=3.0/threshold=6.0 vs init_prey=3.5/threshold=6.5 vs init_prey=4.0/threshold=7.0

## Conclusions
- Based on the outcomes: the defaults in config_env_shared_prey are changed to: 

    - "prey_creation_energy_threshold": 6.5,  # was 6.0
    - "initial_energy_prey": 3.5,  # was 3.0
- This is based on the cooperative attempt rate (9.4%) and cooperative succes rate (91.7%), in combination with a near 1000 (=max_steps) "epsisode_len_mean" at 1000 iterations. The latter can be verified in the Tensorboard charts. The two other higher cooperative attempt/succes rates (5.5/2.5, 7.0/4.0) have an "epsisode_len_mean" at only around 800 after 1000 iterations, which implies more failure to reach 1000 max_steps at each episode.


## Scope
- Log A (init_prey=2.5, prey_threshold=5.5): `src/predpreygrass/rllib/shared_prey/ray_results/pred_decay_0_20/GRID_30_PRED_OBS_RANGE_9_INITS_15_INIT_PREY_ENERGY_2_5_2025-12-19_23-59-23/PPO_PredPreyGrass_5c0be_00000_0_2025-12-19_23-59-23/checkpoint_000099/eval_checkpoint_000099_2025-12-20_10-17-26/agent_event_log_2025-12-20_10-17-26.json`
- Log B (init_prey=3.0, prey_threshold=6.0): `src/predpreygrass/rllib/shared_prey/ray_results/pred_decay_0_20/GRID_30_PRED_OBS_RANGE_9_INITS_15_INIT_PREY_ENERGY_3_0_2025-12-14_23-00-48/PPO_PredPreyGrass_595b9_00000_0_2025-12-14_23-00-48/checkpoint_000099/eval_checkpoint_000099_2025-12-15_23-08-59/agent_event_log_2025-12-15_23-08-59.json`
- Log C (init_prey=3.5, prey_threshold=6.5): `src/predpreygrass/rllib/shared_prey/ray_results/pred_decay_0_20/GRID_30_PRED_OBS_RANGE_9_INITS_15_INIT_PREY_ENERGY_3_5_2025-12-18_22-22-59/PPO_PredPreyGrass_ba9c8_00000_0_2025-12-18_22-22-59/checkpoint_000099/eval_checkpoint_000099_2025-12-19_12-35-07/agent_event_log_2025-12-19_12-35-07.json`
- Log D (init_prey=4.0, prey_threshold=7.0): `src/predpreygrass/rllib/shared_prey/ray_results/pred_decay_0_20/GRID_30_PRED_OBS_RANGE_9_INITS_15_INIT_PREY_ENERGY_4_0_2025-12-17_19-20-08/PPO_PredPreyGrass_0479f_00000_0_2025-12-17_19-20-08/checkpoint_000099/eval_checkpoint_000099_2025-12-19_15-11-35/agent_event_log_2025-12-19_15-11-35.json`
- Only config differences across runs: `initial_energy_prey` and `prey_creation_energy_threshold`

## Definitions
- Cooperative predation = predator event with `team_capture = true`
- Solo predation = predator event with `team_capture = false`
- Attempt = success + failed predator eating event
- `energy_resource` in predator events is prey energy at the encounter
- Prey eating events are grass bites (energy gain equals grass energy at that cell)
- Parent IDs are not recorded in the logs, so reproduction is summarized via `reproduction_events`

## Key outcomes
- Overall predator success rate peaks at 2.5/5.5 (65.7%), while 3.0/6.0-4.0/7.0 cluster around 45-48%.
- Coop attempt share is lowest at 3.0/6.0 (6.6%) and higher at 2.5/5.5 (11.4%) and 4.0/7.0 (10.7%); coop success is highest at 2.5/5.5 and 3.5/6.5 (92.4%/91.7%).
- Mean prey energy at successful capture declines as initial energy/threshold drop (4.0 to 2.5: 3.78 to 3.27).
- Prey lifetimes shorten sharply at 2.5/5.5 (median 20) with fewer grass-bite events and larger average bites (0.51).
- Predator starvation remains the dominant death cause across all runs.

## Predation and cooperation
### Event counts
| Metric | Log A (2.5/5.5) | Log B (3.0/6.0) | Log C (3.5/6.5) | Log D (4.0/7.0) |
| --- | --- | --- | --- | --- |
| Success events (total) | 761 | 513 | 464 | 583 |
| - cooperative | 122 | 52 | 88 | 96 |
| - solo | 639 | 461 | 376 | 487 |
| Failed events (total) | 398 | 549 | 559 | 632 |
| - cooperative | 10 | 18 | 8 | 34 |
| - solo | 388 | 531 | 551 | 598 |
| Attempts (total) | 1159 | 1062 | 1023 | 1215 |

### Rates
| Metric | Log A (2.5/5.5) | Log B (3.0/6.0) | Log C (3.5/6.5) | Log D (4.0/7.0) |
| --- | --- | --- | --- | --- |
| Coop attempt share | 11.4% | 6.6% | 9.4% | 10.7% |
| Coop success rate | 92.4% | 74.3% | 91.7% | 73.8% |
| Solo success rate | 62.2% | 46.5% | 40.6% | 44.9% |
| Overall success rate | 65.7% | 48.3% | 45.4% | 48.0% |
| Coop share of successes | 16.0% | 10.1% | 19.0% | 16.5% |

### Participation
| Metric | Log A (2.5/5.5) | Log B (3.0/6.0) | Log C (3.5/6.5) | Log D (4.0/7.0) |
| --- | --- | --- | --- | --- |
| Predators with any coop attempt | 76 / 206 | 42 / 159 | 56 / 149 | 75 / 195 |
| Predators with coop success | 74 / 206 | 35 / 159 | 52 / 149 | 61 / 195 |

### Helper counts
- Log A: 120 cooperative events involve 2 predators, 12 involve 3 predators.
- Log B: all cooperative events involve exactly 2 predators (70 events).
- Log C: all cooperative events involve exactly 2 predators (96 events).
- Log D: 124 cooperative events involve 2 predators, 6 involve 3 predators.

## Energy at capture (predator events)
| Metric | Log A (2.5/5.5) | Log B (3.0/6.0) | Log C (3.5/6.5) | Log D (4.0/7.0) |
| --- | --- | --- | --- | --- |
| Success mean (all) | 3.27 | 3.58 | 3.72 | 3.78 |
| Success median (all) | 3.19 | 3.52 | 3.45 | 3.45 |
| Success mean (coop) | 3.44 | 3.76 | 4.00 | 4.35 |
| Success mean (solo) | 3.24 | 3.56 | 3.65 | 3.67 |
| Failed mean (all) | 3.77 | 4.00 | 4.43 | 4.16 |
| Failed median (all) | 3.82 | 3.90 | 4.49 | 3.99 |
| Failed mean (coop) | 3.72 | 4.82 | 4.10 | 4.42 |
| Failed mean (solo) | 3.77 | 3.97 | 4.44 | 4.14 |

## Population and reproduction
| Metric | Log A (2.5/5.5) | Log B (3.0/6.0) | Log C (3.5/6.5) | Log D (4.0/7.0) |
| --- | --- | --- | --- | --- |
| Predator agents | 206 | 159 | 149 | 195 |
| Prey agents | 748 | 617 | 545 | 599 |
| Predator reproduction events | 191 | 144 | 134 | 180 |
| Prey reproduction events | 733 | 602 | 530 | 584 |

## Mortality and lifetime
| Metric | Log A (2.5/5.5) | Log B (3.0/6.0) | Log C (3.5/6.5) | Log D (4.0/7.0) |
| --- | --- | --- | --- | --- |
| Predator death causes | starved 193, time_limit 13 | starved 152, time_limit 7 | starved 142, time_limit 7 | starved 186, time_limit 9 |
| Prey death causes | eaten 698, starved 25, time_limit 25 | eaten 487, starved 83, time_limit 47 | eaten 420, starved 69, time_limit 56 | eaten 534, starved 32, time_limit 33 |
| Predator lifetime (mean / median steps) | 55.73 / 32 | 55.92 / 26 | 53.69 / 25 | 51.86 / 26 |
| Prey lifetime (mean / median steps) | 34.70 / 20 | 69.80 / 40 | 81.66 / 49 | 61.01 / 36 |

## Prey grass eating
| Metric | Log A (2.5/5.5) | Log B (3.0/6.0) | Log C (3.5/6.5) | Log D (4.0/7.0) |
| --- | --- | --- | --- | --- |
| Eating events | 7125 | 10463 | 11013 | 8898 |
| Bite mean | 0.51 | 0.38 | 0.36 | 0.44 |
| Bite median | 0.32 | 0.28 | 0.28 | 0.32 |

## Interpretation (data-driven)
- Lowering initial prey energy from 4.0 to 2.5 tracks with lower prey energy at capture and a sharp increase in predator success at the lowest setting.
- Coop dynamics are non-monotonic: 3.0/6.0 has the weakest coop participation and success, while 2.5/5.5 and 3.5/6.5 show strong coop success.
- Prey turnover accelerates at 2.5/5.5 (short lifetimes, fewer grass-bite events, larger bites), despite higher reproduction counts.
- Predator mortality remains dominated by starvation across all runs.
- These are single evaluation runs; replicate across seeds to confirm the pattern.
