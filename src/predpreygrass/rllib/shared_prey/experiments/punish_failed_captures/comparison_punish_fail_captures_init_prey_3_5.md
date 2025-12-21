# Comparison: init_prey_energy=3.5 (baseline vs punish failed captures)

## Conclusion (single-eval snapshot)

Adding failed-capture penalties shifts outcomes toward **higher overall predator success** (+25% relative), driven primarily by a **stronger solo success rate** (+33.7% relative) even though **cooperative success drops** (-17.7% relative). The punish run shows **more prey and fewer predators**, with **shorter prey lifetimes** and **more grass-eating events**, indicating faster prey turnover despite lower median prey energy at capture. Cooperative attempts become more frequent but less reliable, while solo captures rise sharply. Treat these as directional signals only; multi-seed aggregation is needed to confirm the effect under stochasticity.

## Runs

- Baseline log: `src/predpreygrass/rllib/shared_prey/ray_results/pred_decay_0_20/GRID_30_PRED_OBS_RANGE_9_INITS_15_INIT_PREY_ENERGY_3_5_2025-12-18_22-22-59/PPO_PredPreyGrass_ba9c8_00000_0_2025-12-18_22-22-59/checkpoint_000099/eval_checkpoint_000099_2025-12-19_12-35-07/agent_event_log_2025-12-19_12-35-07.json`
- Punish-failed-captures log: `src/predpreygrass/rllib/shared_prey/ray_results/pred_decay_0_20/GRID_30_PRED_OBS_RANGE_9_INITS_15_INIT_PREY_ENERGY_3_5_PUNISH_FAIL_CAPTURES_2025-12-21_00-06-32/PPO_PredPreyGrass_86bdb_00000_0_2025-12-21_00-06-33/checkpoint_000099/eval_checkpoint_000099_2025-12-21_19-55-09/agent_event_log_2025-12-21_19-55-09.json`

## Definitions

- predator success = predator eating event (prey capture)
- predator failure = failed_eating_event (attack attempt that did not capture)
- attempt = success + failure
- cooperative event = team_capture = true
- solo event = team_capture = false
- prey energy at capture = energy_resource in predator eating events
- bite size = bite_size in predator eating events (per predator)
- failed attempt penalty = energy_before - energy_after in failed_eating_events (per predator)
- rates are shown as percentages; Δ is absolute; Δ% is relative change vs baseline

## Top-line comparison

| Metric | Baseline | Punish failed captures | Δ | Δ% |
| --- | --- | --- | --- | --- |
| Agents total | 694 | 831 | 137 | 19.7% |
| Predators total | 149 | 132 | -17 | -11.4% |
| Prey total | 545 | 699 | 154 | 28.3% |
| Predator median lifetime | 25 | 27.5 | 2.5 | 10.0% |
| Prey median lifetime | 49 | 34 | -15 | -30.6% |
| Predator attempts | 1023 | 1042 | 19 | 1.9% |
| Predator successes | 464 | 591 | 127 | 27.4% |
| Predator failures | 559 | 451 | -108 | -19.3% |
| Predator success rate | 45.4% | 56.7% | 11.4 pp | 25.0% |
| Predator coop success rate | 91.7% | 75.4% | -16.3 pp | -17.7% |
| Predator solo success rate | 40.6% | 54.2% | 13.7 pp | 33.7% |
| Coop attempts | 96 | 122 | 26 | 27.1% |
| Coop attempt share | 9.4% | 11.7% | 2.3 pp | 24.8% |
| Prey grass eats | 11013 | 12276 | 1263 | 11.5% |
| Median prey energy at capture | 3.45 | 3.17 | -0.28 | -8.1% |
| Median prey energy at coop capture | 4.14 | 3.43 | -0.70 | -17.0% |
| Median prey energy at solo capture | 3.45 | 3.11 | -0.34 | -9.9% |
| Median bite size (per predator) | 3.36 | 2.95 | -0.41 | -12.1% |
| Median bite size (coop) | 1.97 | 1.39 | -0.59 | -29.8% |
| Median bite size (solo) | 3.45 | 3.11 | -0.34 | -9.9% |
| Failed-attempt penalty total | 0.00 | 0.00 | 0.00 | n/a |
| Failed-attempt penalty median | 0.00 | 0.00 | 0.00 | n/a |
| Failed-attempt penalty mean | 0.00 | 0.00 | 0.00 | n/a |

## Outcome composition

### Predator successes and failures

| Outcome | Baseline | Punish failed captures | Δ | Δ% |
| --- | --- | --- | --- | --- |
| Successes (total) | 464 | 591 | 127 | 27.4% |
| - cooperative | 88 | 92 | 4 | 4.5% |
| - solo | 376 | 499 | 123 | 32.7% |
| Failures (total) | 559 | 451 | -108 | -19.3% |
| - cooperative | 8 | 30 | 22 | 275.0% |
| - solo | 551 | 421 | -130 | -23.6% |

### Predator attempt rates

| Rate | Baseline | Punish failed captures | Δ | Δ% |
| --- | --- | --- | --- | --- |
| Overall success rate | 45.4% | 56.7% | 11.4 pp | 25.0% |
| Coop success rate | 91.7% | 75.4% | -16.3 pp | -17.7% |
| Solo success rate | 40.6% | 54.2% | 13.7 pp | 33.7% |
| Coop attempt share | 9.4% | 11.7% | 2.3 pp | 24.8% |

## Lifetimes

| Metric | Baseline | Punish failed captures | Δ | Δ% |
| --- | --- | --- | --- | --- |
| Predator median lifetime | 25 | 27.5 | 2.5 | 10.0% |
| Predator mean lifetime | 53.7 | 68.0 | 14.3 | 26.6% |
| Prey median lifetime | 49 | 34 | -15 | -30.6% |
| Prey mean lifetime | 81.7 | 63.4 | -18.3 | -22.3% |

## Death causes

### Predators

- Baseline: {'starved': 142, 'time_limit': 7}
- Punish failed captures: {'starved': 126, 'time_limit': 6}

### Prey

- Baseline: {'eaten': 420, 'starved': 69, 'time_limit': 56}
- Punish failed captures: {'eaten': 544, 'starved': 100, 'time_limit': 55}

## Team size distribution (successful captures only)

Counts of predator_list sizes in successful captures; includes solo and cooperative events.

Baseline: {1: 376, 2: 88}


Punish failed captures: {2: 86, 1: 499, 3: 6}


## Notes and caveats

- Single-evaluation comparison; results are sensitive to seed and stochasticity.
- The punish run logs failed_eating_events with per-predator penalties; baseline may log failures without penalties.
- Penalty totals are summed per predator event (not per attack group).
