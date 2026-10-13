# base_environment_continuous: results

## Runs

| Run | Folder (`runs/base_environment_continuous_...`) | Start | Iterations | What | Status |
|---|---|---|---:|---|---|
| BE-01 | `2026-10-13_11-25-07_420466` | scratch | 3,000 | the milestone world (as AW-21 and AW-22 of `add_walls`) trained from scratch with this environment | completed |
| BE-02 | `2026-10-13_11-25-27_332055` | scratch | 3,000 | replicate of BE-01 (new training randomness) | completed |

Trained in the [PredPreyGrassContinuous](https://github.com/doesburg11/PredPreyGrassContinuous)
development repository; the run folders, launchers (`be01.py`, `be02.py`), evaluation
script (`evaluate_be.sh`, which refers to that repository's paths) and outputs are
in this module's `runs/` folder (gitignored). AW-21 and AW-22
are runs of that repository's `add_walls` module.

## 1. Validation against add_walls (runs BE-01 and BE-02)

**Question.** This environment reproduces `add_walls` exactly for the same
seeds and actions (SPEC.md, section 10). Does training with it also give the
same result as training with `add_walls`?

**Setup.** BE-01 and BE-02 train the milestone world (no walls, 75% of the
energy cost from movement, prey top speed 6.0, grass in 20 clusters of 5)
from scratch for 3,000 iterations, side by side on 14 simulation workers
each, exactly as `add_walls` runs AW-21 and AW-22 did. Same PPO settings;
training randomness is not seeded, so the runs can only agree
statistically.

**Learning curves**, per episode in blocks of 500 iterations (predator
births / prey births / episode length in decisions; births are the policy
returns divided by the +10 birth reward):

| Iterations | AW-21 | AW-22 | BE-01 | BE-02 |
|---|---|---|---|---|
| 1–500 | 7.7 / 56 / 152 | 10.8 / 102 / 283 | 10.0 / 88 / 242 | 9.5 / 82 / 228 |
| 501–1000 | 10.2 / 98 / 257 | 18.9 / 217 / 590 | 17.5 / 193 / 529 | 11.1 / 109 / 287 |
| 1001–1500 | 16.6 / 193 / 522 | 27.2 / 336 / 917 | 25.4 / 304 / 833 | 14.7 / 155 / 419 |
| 1501–2000 | 26.1 / 329 / 899 | 29.7 / 373 / 1,023 | 28.7 / 351 / 966 | 20.9 / 241 / 662 |
| 2001–2500 | 29.6 / 384 / 1,043 | 31.7 / 403 / 1,102 | 31.1 / 385 / 1,058 | 27.9 / 345 / 944 |
| 2501–3000 | 31.0 / 394 / 1,072 | 31.5 / 403 / 1,102 | 32.0 / 403 / 1,103 | 29.9 / 374 / 1,023 |

BE-01 learned about as fast as AW-22, BE-02 about as slowly as AW-21: the
same spread between replicates. All four ended near 30–32 predator births,
375–405 prey births and 1,020–1,100 decisions per episode.

**Final checks after 3,000 iterations** (50 episodes, seeds 0–49).
All four runs measured with the same script (`analysis/final_check.py`) in
this environment, with the same seeds and action sampling. Checkpoints of
`add_walls` runs play identically here, so this compares training, not
environments. (AW-21 and AW-22 score 33 and 38 here against 37 and 41 in
`add_walls`'s [RESULTS.md](https://github.com/doesburg11/PredPreyGrassContinuous/blob/master/add_walls/RESULTS.md),
because the actions are sampled differently.)

| | AW-21 | AW-22 | BE-01 | BE-02 |
|---|---:|---:|---:|---:|
| Coexisting | 33 | 38 | 35 | 37 |
| Prey extinct | 17 | 12 | 15 | 13 |
| Predators extinct | 0 | 0 | 0 | 0 |
| Catch risk (per prey per 1,000 steps) | 3.67 | 3.46 | 3.63 | 3.74 |
| Predators / prey alive on average | 11.6 / 16.1 | 11.6 / 16.3 | 11.4 / 15.8 | 11.7 / 14.9 |
| Predators stopped | 45% | 44% | 41% | 47% |
| Predators' time near grass | 39% | 40% | 39% | 38% |
| Prey's time near grass | 21% | 20% | 21% | 21% |
| Catches near grass | 44% | 45% | 46% | 46% |

"Near grass": within 64 units of an available patch.

**Tournaments** (checkpoints every 300 iterations, 10 episodes per matchup;
share of pairs against a fixed opponent in which the newer checkpoint does
better):

| Run | Newer prey harder to catch | Newer predators catch more |
|---|---:|---:|
| AW-21 | 89% | 64% |
| AW-22 | 79% | 71% |
| BE-01 | 86% | 76% |
| BE-02 | 85% | 65% |

Catch risk of each checkpoint against its own training partner fell from
6.2 and 6.9 at iteration 300 to about 3.4–3.8 from iteration 1,200 on (BE-01
and BE-02), while both species kept improving against fixed opponents: the
same Red Queen pattern as in AW-21 and AW-22.

**Wall time:** 9.1 and 8.9 hours for 3,000 iterations, against 9.9 and
10.5 hours for AW-21 and AW-22 (8–15% less; the learner takes most of the
time, and the environment itself plays about twice as fast).

**Conclusion:** trained with this environment, the milestone world reaches
the same state as with `add_walls`: 72 of 100 episodes coexisting (two
runs of 50) against 71 of 100, the same catch risk, populations and behaviour (predators
waiting near grass, about 45% of catches there), and the same arms race.
The environment can replace `add_walls` for this world.
