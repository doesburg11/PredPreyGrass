# PPO vs APPO eval comparison (init_prey=2.5, prey_threshold=5.5)

## Scope
- PPO log: `src/predpreygrass/rllib/shared_prey/experiments/PPO_v_APPO/GRID_30_PRED_OBS_RANGE_9_INITS_15_INIT_PREY_ENERGY_2_5/PPO/PPO_PredPreyGrass_5c0be_00000_0_2025-12-19_23-59-23/checkpoint_000099/eval_checkpoint_000099_2025-12-20_10-17-26/agent_event_log_2025-12-20_10-17-26.json`
- APPO log: `src/predpreygrass/rllib/shared_prey/experiments/PPO_v_APPO/GRID_30_PRED_OBS_RANGE_9_INITS_15_INIT_PREY_ENERGY_2_5/APPO/APPO_PredPreyGrass/checkpoint_000099/eval_checkpoint_000099_2025-12-20_19-16-10/agent_event_log_2025-12-20_19-16-10.json`
- Same grid/obs settings (GRID_30, obs_range=9, inits=15); only algorithm differs.

## Definitions
- Cooperative predation = predator event with `team_capture = true`
- Solo predation = predator event with `team_capture = false`
- Attempt = success + failed predator eating event
- `energy_resource` in predator events is prey energy at the encounter
- Prey eating events are grass bites (energy gain equals grass energy at that cell)

## Key outcomes
- PPO achieves much higher overall success rate (65.7% vs 37.8%) driven by higher solo success (62.2% vs 31.7%).
- APPO has a higher coop share of successes (25.4% vs 16.0%) but lower coop success rate (86.1% vs 92.4%).
- APPO generates more attempts (1681 vs 1159) but far more failures (1046 vs 398).
- APPO shows longer predator/prey lifetimes (median predator 42.5 vs 32.0; median prey 38.5 vs 19.5).
- Prey grass eating is much higher in APPO (14611 vs 7125) but with smaller bites (mean 0.27 vs 0.51).

## Predation and cooperation
### Event counts
| Metric | PPO | APPO |
| --- | --- | --- |
| Success events (total) | 761 | 635 |
| - cooperative | 122 | 161 |
| - solo | 639 | 474 |
| Failed events (total) | 398 | 1046 |
| - cooperative | 10 | 26 |
| - solo | 388 | 1020 |
| Attempts (total) | 1159 | 1681 |

### Rates
| Metric | PPO | APPO |
| --- | --- | --- |
| Coop attempt share | 11.4% | 11.1% |
| Coop success rate | 92.4% | 86.1% |
| Solo success rate | 62.2% | 31.7% |
| Overall success rate | 65.7% | 37.8% |
| Coop share of successes | 16.0% | 25.4% |

### Participation
| Metric | PPO | APPO |
| --- | --- | --- |
| Predators with any coop attempt | 76 / 206 | 80 / 132 |
| Predators with coop success | 74 / 206 | 73 / 132 |

### Helper counts
- PPO: 2: 120, 3: 12
- APPO: 2: 178, 3: 9

## Energy at capture (predator events)
| Metric | PPO | APPO |
| --- | --- | --- |
| Success mean (all) | 3.27 | 3.35 |
| Success median (all) | 3.19 | 3.27 |
| Success mean (coop) | 3.44 | 3.66 |
| Success mean (solo) | 3.24 | 3.25 |
| Failed mean (all) | 3.77 | 3.95 |
| Failed median (all) | 3.82 | 4.11 |
| Failed mean (coop) | 3.72 | 4.38 |
| Failed mean (solo) | 3.77 | 3.94 |

## Population and reproduction
| Metric | PPO | APPO |
| --- | --- | --- |
| Predator agents | 206 | 132 |
| Prey agents | 748 | 618 |
| Predator reproduction events | 191 | 117 |
| Prey reproduction events | 733 | 603 |

## Mortality and lifetime
| Metric | PPO | APPO |
| --- | --- | --- |
| Predator death causes | starved 193, time_limit 13 | starved 123, time_limit 9 |
| Prey death causes | eaten 698, starved 25, time_limit 25 | eaten 553, starved 24, time_limit 41 |
| Predator lifetime (mean / median steps) | 55.73 / 32.0 | 69.89 / 42.5 |
| Prey lifetime (mean / median steps) | 34.70 / 19.5 | 66.80 / 38.5 |

## Prey grass eating
| Metric | PPO | APPO |
| --- | --- | --- |
| Eating events | 7125 | 14611 |
| Bite mean | 0.51 | 0.27 |
| Bite median | 0.32 | 0.16 |

## Interpretation (data-driven)
- PPO shows higher hunting efficiency with fewer failed attempts, while APPO produces more attempts but a lower overall success rate.
- APPO relies more on cooperative successes, but its solo success rate is substantially lower, which likely drives the overall gap.
- APPO sustains longer-lived populations (predators and prey) alongside higher grass-eating counts and smaller bites, suggesting slower prey turnover.
- PPO has more predators and prey overall but shorter prey lifetimes, indicating faster predation cycles at this setting.
- These are single evaluation runs; replicate across seeds to confirm the pattern.
