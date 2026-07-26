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

**2026-07-24**, GPU config, 3 iterations each: `tune_ppo_metabolic_code.py` and
`tune_ppo_metabolic_code_neutral_control.py`. Both completed cleanly, no
exceptions. `genome_neutral_drift_control: True` confirmed engaging correctly
in the control variant via `run_config.json`. `live_haystack/*_mean_wrong_loci`
started near the founder expectation (~2.6-2.7, vs. 3.0 expected), consistent
with a sane founder distribution. 32/32 unit tests pass.

## 3. Pilot vs. real run — decision

No separate throwaway pilot was run. Considered it, but concluded a dedicated
short pilot brings nothing a real run's early iterations don't also show —
`progress.csv` is inspectable at any time without stopping anything, so the
first ~250 iterations of an actual seeded real+control pair serve the same
sanity-check purpose while remaining usable data if they turn out healthy
(rather than being thrown away). Seed 42 (real + control) was launched
directly as the first pair of the eventual 3-seed replication, monitored
through its early iterations, and kept running once it looked sane.

## 4. Replication (real vs. neutral control) — complete

**Config:** identical real/control replication methodology as
`eco_evolutionary_investment`'s R7 — `tune_ppo_metabolic_code.py` /
`tune_ppo_metabolic_code_neutral_control.py`, 3 real seeds (42/43/44) + 3
neutral-control seeds (42/43/44), 1000 iterations each, sequential on one GPU
(unscaled base config: 25×25 grid, 6 predators + 8 prey initial).

**Launched/finished:** real 42 2026-07-24 18:17 → 23:53 (5.56h); control 42
23:53 → 2026-07-25 06:44 (6.81h); real 43 2026-07-25 10:01 → 15:17 (5.25h);
control 43 15:17 → 20:29 (5.16h); real 44 20:29 → 2026-07-26 02:12 (5.67h);
control 44 02:12 → 08:41 (6.46h). All 6 runs finished cleanly at 1000/1000
iterations, no crashes. Total wall-clock ~34.9h, sequential on one GPU,
spanning 2026-07-24 to 2026-07-26.

**Operational note — predator ID pool exhaustion:** seed 44's real run hit
the `n_possible_predators=200` capacity ceiling repeatedly (7/1000 iterations
nonzero on `predator_reproduction_blocked`, peak 460 blocked-reproduction
events in one iteration's rollout batch) — meaningfully more than seed 42 (1
nonzero iteration, peak 2) or seed 43 (never triggered). Not a crash — just
some late-episode predator reproduction silently capped in the affected
episodes. Notably, seed 44 is also the seed whose prey `mean_wrong_loci`
values are a clear outlier (see below) — both anomalies point at seed 44
specifically having produced atypical population dynamics, not a bug in the
mechanism.

**Result: null, and for the headline metric mildly reversed — in both species.**

| species | metric | real (n=3) | control (n=3) | U | p |
|---|---|---|---|---|---|
| predator | mean_wrong_loci | 3.1196 | 2.9224 | 7.0 | p(real<control)=0.900 |
| predator | fraction_solved | 0.0080 | 0.0070 | 6.0 | p(real>control)=0.350 |
| prey | mean_wrong_loci | 3.9164 | 3.2737 | 6.0 | p(real<control)=0.800 |
| prey | fraction_solved | 0.0087 | 0.0059 | 5.0 | p(real>control)=0.500 |

Per-seed data (Q1/Q5 quintile means):

| group | seed | species | metric | Q1 | Q5 |
|---|---|---|---|---|---|
| real | 42 | predator | mean_wrong_loci | 2.7686 | 2.6719 |
| real | 42 | predator | fraction_solved | 0.0028 | 0.0035 |
| real | 42 | prey | mean_wrong_loci | 3.5124 | 3.2180 |
| real | 42 | prey | fraction_solved | 0.0077 | 0.0148 |
| real | 43 | predator | mean_wrong_loci | 3.7940 | 3.9798 |
| real | 43 | predator | fraction_solved | 0.0021 | 0.0025 |
| real | 43 | prey | mean_wrong_loci | 3.0116 | 2.8867 |
| real | 43 | prey | fraction_solved | 0.0057 | 0.0114 |
| real | 44 | predator | mean_wrong_loci | 2.7423 | 2.7073 |
| real | 44 | predator | fraction_solved | 0.0084 | 0.0180 |
| real | 44 | prey | mean_wrong_loci | 5.5241 | 5.6445 |
| real | 44 | prey | fraction_solved | 0.0000 | 0.0000 |
| control | 42 | predator | mean_wrong_loci | 2.6000 | 2.6098 |
| control | 42 | predator | fraction_solved | 0.0032 | 0.0027 |
| control | 42 | prey | mean_wrong_loci | 2.8989 | 2.8908 |
| control | 42 | prey | fraction_solved | 0.0104 | 0.0130 |
| control | 43 | predator | mean_wrong_loci | 3.7698 | 3.9097 |
| control | 43 | predator | fraction_solved | 0.0016 | 0.0006 |
| control | 43 | prey | mean_wrong_loci | 2.7317 | 2.7826 |
| control | 43 | prey | fraction_solved | 0.0035 | 0.0036 |
| control | 44 | predator | mean_wrong_loci | 2.5491 | 2.2476 |
| control | 44 | predator | fraction_solved | 0.0109 | 0.0176 |
| control | 44 | prey | mean_wrong_loci | 5.4141 | 4.1478 |
| control | 44 | prey | fraction_solved | 0.0001 | 0.0009 |

**Predator:** real mean_wrong_loci (3.12) is *higher* than control (2.92) —
the opposite of the hypothesized direction (selection should push wrong-loci
count down). fraction_solved trends the right way (0.008 vs 0.007) but the
effect is tiny and p=0.35 is nowhere near significant at n=3.

**Prey:** same pattern — real (3.92) higher than control (3.27), wrong
direction on the headline metric. fraction_solved again trends the right way
(0.0087 vs 0.0059), not significant. Seed 44's prey values (5.52-5.64 real,
4.15-5.41 control) are far above every other seed's range (2.7-4.0) for
*both* groups — since both real and control show the same elevated pattern in
this one seed, it reads as an artifact of that seed's particular population
dynamics (small-population sampling noise in the founder draw, likely
compounded by the same seed's heavy predator-capacity blocking possibly
altering prey survival pressure), not a selection effect. It also has an
outsized influence on the n=3 aggregate — one atypical seed moves the group
mean a lot at this sample size.

**Interpretation:** this is the fourth trait design in this project
(`metabolic_rate`, `offspring_investment_fraction`, `cooperation_rate`, now
this combinatorial locus code) to fail the same test on its primary metric,
and unlike Trial 6's mixed (species-disagreeing) result, this one is
consistent across both species — both show real *higher* than control on
mean_wrong_loci, not lower. Despite being purpose-built to fix the two
theoretical gaps identified in `predpreygrass/evolutionary/RESULTS.md`'s
Hinton & Nowlan note (a true needle-in-a-haystack fitness landscape, and a
genuine per-individual lifetime search decoupled from PPO), it still doesn't
show selection beating neutral drift at this population/seed scale. The
secondary metric (fraction_solved) trends in the predicted direction for both
species but far too weakly to call anything at n=3.

**Verdict: null for criterion 3, and directionally reversed on the headline
metric for both species** — the cleanest (if most disappointing) null result
in the project's search so far, precisely because it agrees across species
rather than splitting the way Trial 6 did. Not proof the combinatorial-genome
approach as a category can't work here — the seed-44 outlier and the
predator-pool-capacity confound both suggest this specific run has more noise
than ideal, and the founder/mutation/bonus parameters were calculated
starting guesses, never tuned against real data before this replication.
See `predpreygrass/evolutionary/RESULTS.md`'s Trial 7 entry for the
cross-module framing and next-step decision.

**Status:** replication complete (2026-07-26). Decision on next step (more
seeds to average out seed 44's outlier influence, parameter retuning, raising
`n_possible_predators`, or stepping back to reconsider the search direction
entirely) not yet made.

---

*Addendum analysis date: 2026-07-26. Data source: 6 seeded runs (real seeds
42/43/44, control seeds 42/43/44) launched 2026-07-24 18:17, all finished
cleanly 2026-07-26 08:41 — see
`PPO_ECO_EVOLUTION_METABOLIC_CODE_SEED{42,43,44}_*` and
`PPO_ECO_EVOLUTION_METABOLIC_CODE_NEUTRAL_CONTROL_SEED{42,43,44}_*` under
`~/ray_results/`.*
