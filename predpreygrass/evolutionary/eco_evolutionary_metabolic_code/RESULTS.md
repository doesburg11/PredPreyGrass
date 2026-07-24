# Training Analysis — eco_evolutionary_metabolic_code

No training run has been launched yet. This file will be filled in after a
real run exists, following the same structure as `eco_evolutionary_investment/RESULTS.md`
(experiment setup, per-run summary, Darwinian/Baldwinian interaction analysis,
neutral-control comparison). See `README.md` for the trait design and
`predpreygrass/evolutionary/RESULTS.md`'s Trial 7 entry for the current status
of the decision to launch a pilot.

---

## 1. Experiment Setup

### Environment

| Parameter | Value |
|---|---|
| Grid | 25 × 25 |
| Max steps per episode | 1000 |
| Observation channels | 3 (predators, prey, grass) |
| Predator obs window | 7 × 7 |
| Prey obs window | 9 × 9 |
| Actions | 9 (3×3 Moore neighbourhood, stay included) |
| Movement energy cost | 0.0 (disabled) |
| Predator basal decay | 0.15 / step |
| Prey basal decay | 0.05 / step |
| Predator reproduction threshold | 12.0 energy |
| Prey reproduction threshold | 8.0 energy |
| Predator initial energy | 5.0 |
| Prey initial energy | 3.0 |
| Predator satiation cooldown | 8 steps |
| Max energy gain per prey (satiation ceiling) | 8.0 |
| Offspring investment fraction | 0.35 (fixed, non-heritable) |
| Grass patches | 100, max energy 2.0, regrowth 0.04/step |
| Initial population | 6 predators + 8 prey |
| Max population pool | 200 predators + 1000 prey |

### Genome (Darwinian layer)

| Parameter | Value |
|---|---|
| Heritable trait | `loci` — length-10 combinatorial locus code |
| Locus states | CORRECT / WRONG / PLASTIC (relative to an implicit fixed target) |
| Founder probabilities (both species) | correct 0.2 / wrong 0.3 / plastic 0.5 |
| Founder E[wrong loci] | 3.0 (needed for the primary drift-below-founder test) |
| Mutation rate | 0.05 per locus per reproduction |
| Solve bonus multiplier | 1.5x energy gain, from the step solved onward |

The locus code determines, via the per-individual lifetime-search mechanism
described in `README.md`, whether an agent achieves a metabolic-efficiency
bonus this life. It is inherited with uniform-resample mutation per locus and
is never directly observable by or accessible to the PPO policy. See
`README.md` for the full mechanism and the Hinton & Nowlan (1987) motivation.

### PPO configuration

Same hyperparameters as `eco_evolutionary_investment` (`config_ppo_gpu_eco_evolutionary.py`
/ `config_ppo_cpu_eco_evolutionary.py`, copied unchanged).

---

## 2. Smoke test

3-iteration CPU runs of both `tune_ppo_metabolic_code.py` and
`tune_ppo_metabolic_code_neutral_control.py` — see the commit/session notes for
the exact confirmation. Purpose: confirm no exceptions, and that
`live_haystack/{species}_mean_wrong_loci` starts near the founder expectation
(3.0) and `genome_neutral_drift_control` engages correctly in the control
variant.

## 3. Pilot 1

Not yet launched.

## 4. Replication (real vs. neutral control)

Not yet launched. Planned methodology (once a pilot looks sane): 3 real seeds
+ 3 neutral-control seeds, same Mann-Whitney U real-vs-control comparison used
for `eco_evolutionary_investment`'s R7, via `analyze_replication_seeds.py`.
