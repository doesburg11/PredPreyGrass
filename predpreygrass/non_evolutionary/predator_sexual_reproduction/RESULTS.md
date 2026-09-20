# Predator Sexual Reproduction — Training Results

Results from PPO training runs on the `predator_sexual_reproduction` environment. See
`README.md` for the mechanism design (sexed predators, mate-search + energy-threshold
reproduction, stochastic hunting, nuptial-gift provisioning, parental care) and the
human-cooperation framing behind it.

This is a running research log, not just a final write-up — it records the trial-and-error
trail (what was tried, why, what was found) so the search can be re-evaluated later without
reconstructing it from conversation history.

---

## Run names used in this log

- **REALISTIC** (Iteration 1): `PPO_PREDATOR_SEXUAL_REPRODUCTION_REALISTIC_SEED42`. Shipped-default
  hunting odds and a **sparse reward**: only `reproduction_reward_predator/prey` = 10.0 is
  nonzero. 100 iterations.
- **FORAGING** (Iteration 2): `PPO_PREDATOR_SEXUAL_REPRODUCTION_FORAGING_CHECK_SEED42`. Identical
  to REALISTIC, plus `reward_predator_catch_prey` = 1.0 and `reward_predator_gather_fruit` = 0.5
  for both sexes (a **foraging reward**). 300 iterations. A diagnostic, not a proposed reward.
- **POSITIVE_CONTROL** (Iteration 0): extreme hunting odds, sparse reward. Not in the chart.

All runs use seed 42.

---

## Iteration log

### Iteration 0 — Positive control, extreme hunting-risk asymmetry (complete)

**Config:** `PPO_PREDATOR_SEXUAL_REPRODUCTION_POSITIVE_CONTROL_SEED42`, seed 42, 100 iterations,
4 env runners. Hunting odds pushed to an extreme via the CLI overrides added for this purpose
(`tune_ppo_predator_sexual_reproduction.py`'s `--male-success-prob`/`--male-death-prob`/
`--female-success-prob`/`--female-death-prob`): male 95% success / 2% death, female 5% success /
90% death per hunting attempt. Everything else at shipped defaults (`mate_search_radius=3`,
`predator_birth_cost_share_female/_male=0.9/0.1`, `male_gift_donation_rate=0.3`,
`parent_offspring_share_rate=0.2`, initial population 6 male / 10 female predators / 8 prey).

**Why this module tests division of labor at all:** predators (not prey) were chosen as this
module's human analog specifically because humans hunt *and* gather while prey have no
equivalent role (see README.md's "Research motivation"). The male-hunts+gathers/
female-gathers-only split that predated this module's current design was replaced with a
symmetric hunting ability plus a hard-coded risk asymmetry (males safer, females riskier at
hunting), grounded in Trivers (1972) parental investment theory: the sex with more at stake in
reproduction (here, females via the 90/10 birth-cost split and the sole gestational-style
energy investment) should be more risk-averse. The design bet is that PPO training will *learn*
male/female behavioral specialization (males hunt more, females gather more) as an emergent
response to that risk asymmetry, rather than the specialization being hard-coded as a structural
incapability the way it was in an earlier design (see README.md's mechanism history).

**Purpose of this specific run:** before trusting a *null* result at the real, much milder risk
asymmetry (module defaults: male 90%/5%, female 20%/10%, see Iteration 1 below), first confirm
the mechanism is even *capable* of producing a visible specialization signal at all, at an odds
gradient sharp enough that if training can't produce a signal even here, it almost certainly
can't at the real, subtler odds either. This is the same "positive control before trusting a
null" logic used in `predpreygrass/evolutionary/eco_evolutionary_metabolic_rate/RESULTS.md`'s
Iteration 6 (sharpen the fitness gradient before concluding a trait lacks selective leverage).

**Result: the extreme killed the mechanism outright rather than stress-testing it.**
`hunting_success_rate_predator_male`/`_female` tracked the configured 95%/5% almost exactly
(0.956 / 0.047 at iteration 100) — the hunting-outcome code is wired correctly. But at a 90%
death chance per failed hunt, a female's expected survival is ~1 hunting attempt
(`1/0.9 ≈ 1.1`). With only 10 starting females, the female population went functionally extinct
in essentially **every single episode, from iteration 1 onward, and never recovered**:

| Iteration | Female hunting attempts | Female hunting deaths | Final # females | Female-extinct episode rate |
|---|---|---|---|---|
| 1 | 5.5 | 5.5 | 0.0 | 100% |
| 25 | 9.8 | 8.3 | 0.0 | 100% |
| 50 | 10.2 | 9.1 | 0.0 | 100% |
| 75 | 10.3 | 9.5 | 0.0 | 100% |
| 100 | 10.3 | 9.0 | 0.0 | 100% |

PPO never learned to keep females out of hunting despite the near-certain lethality — female
hunting attempts per episode actually *rose* slightly over training (5.5 → 10.3), not fell.

**Consequence: predator reproduction essentially never happens.** `births_predator_male` and
`births_predator_female` bounce around 0–0.3 per episode across training with no trend, landing
at exactly **0.0 at iteration 100**. Zero mate-gift events, zero parental-care events throughout
the run — both require a live mated pair, which this config almost never produces since females
die before pairing/reproduction can occur.

**Interpretation:** this specific odds combination is degenerate, not merely "extreme." A 90%
per-attempt death probability is closer to a guaranteed-death rule than a bad-expected-value bet
— it removes the female population before any specialization behavior, mating, or parental
investment has a chance to be observed, and gives PPO no time window in which avoiding hunting
would even matter (most females are already dead by the point avoidance could be learned). This
is **not evidence against the sexual-reproduction/specialization design** — it's evidence this
particular odds pairing is unsuitable for testing it. A milder female death probability (bad
expected value, survivable in the short-to-medium term) is needed to actually observe whether
PPO can learn male/female specialization while both sexes survive long enough to reproduce.

**Status:** complete, 100/100 iterations, no crashes. Raw data:
`~/simulation_results/ray_results/PPO_PREDATOR_SEXUAL_REPRODUCTION_POSITIVE_CONTROL_SEED42/`.

**Adjustment → Iteration 1 (launched):** rather than invent a new synthetic intermediate death
probability, use the module's own already-established realistic defaults directly — they're
already far milder than the positive control's 90% (10% female death per attempt, ~10-attempt
expected survival vs. ~1.1) and answer the module's actual research question rather than another
calibration step. No new number needed; `config_env.py`'s shipped values already are the
"survivable but bad-expected-value" config the positive control's null result called for.

### Iteration 1 — REALISTIC: shipped-default odds, sparse reward (complete)

**Config:** `PPO_PREDATOR_SEXUAL_REPRODUCTION_REALISTIC_SEED42`, seed 42, 100 iterations, no
hunting-probability overrides (male 90% success / 5% death, female 20% success / 10% death),
sparse reward (only `reproduction_reward_predator/prey` = 10.0 is nonzero), initial population
6 male / 10 female predators / 8 prey. Launched 2026-09-19 09:00 via `systemd-run --user`,
CPU-only (`--gpu-fraction 0`, to leave the GPU to another run), about 480 s/iteration.

**Training curves** (block means, per episode; see `results_figures/population_over_training.png`):

| Iterations | Episode length | Males alive at end | Females alive at end | Prey alive at end | Births M / F |
|---|---|---|---|---|---|
| 1-10 | 350 | 3.2 | 0.03 | 48.0 | 1.8 / 2.5 |
| 41-50 | 534 | 4.6 | 0.33 | 49.4 | 4.2 / 4.5 |
| 91-100 | 459 | 4.9 | 0.06 | 50.0 | 4.3 / 4.4 |

Improvement in the first 50 iterations, then a noisy plateau. Females are essentially extinct by
the end of every episode (0.06 alive), unlike males. Hunting attempts per episode are equal by
sex (about 85 each), so there is no division of labor in the counts.

**Checkpoint rollout** (`analyze_prey_approach_from_checkpoint.py`, checkpoints at iterations
10/50/100, 40 episodes each): approach bias toward prey and toward fruit is within +-0.01 cells
of a random mover for both sexes, P(step onto prey) is x0.96-1.04 of random, and predator policy
entropy is 1.95-2.12 nats against the 2.197 maximum for 9 actions. **The predator policies
learned essentially no steering in 100 iterations.** The prey policy did learn (entropy
1.87 -> 0.7-0.85). So the missing specialization is not a female-specific failure; predators with
a sparse, mate-dependent reward get almost no gradient. The metric was validated with synthetic
policies: always-approach scores +1.4 cells (x8.4), always-avoid scores -1.0 (x0).

### Iteration 2 — FORAGING: foraging-reward capability check (complete)

**Purpose:** can predators learn to steer at all in this environment if given a dense signal?
This is a diagnostic, not a proposed final reward: the project wants minimal shaping.

**Config:** `PPO_PREDATOR_SEXUAL_REPRODUCTION_FORAGING_CHECK_SEED42`, seed 42, **300 iterations**,
shipped-default hunting odds, plus `reward_predator_catch_prey = 1.0` and
`reward_predator_gather_fruit = 0.5` (new CLI flags `--reward-catch-prey`/`--reward-gather-fruit`),
paid to both sexes alike. GPU default setup. Ran 2026-09-19 23:45 to 2026-09-20 07:51 as unit
`psr-foraging-check` (about 8 hours; 30 s/iteration at first, about 115 s late, as episodes
lengthen). A first attempt with 100 iterations was stopped at iteration 26 and restarted with 300;
its partial output is kept as `..._ABORTED_PARTIAL` in `ray_results`.

**Training curves** (block means, per episode):

| Iterations | Episode length | Males alive at end | Females alive at end | Prey alive at end | Hunting attempts M / F | Births M / F |
|---|---|---|---|---|---|---|
| 1-10 | 323 | 5.0 | 0.0 | 42.1 | 41 / 42 | 2.5 / 2.0 |
| 91-100 | 829 | 8.9 | 1.0 | 41.5 | 208 / 179 | 16.7 / 16.7 |
| 191-200 | 839 | 13.9 | 1.7 | 34.6 | 270 / 187 | 25.3 / 19.9 |
| 291-300 | **1001** | 16.1 | 6.0 | 31.3 | 348 / 279 | 30.0 / 36.0 |

Episodes reach the 1000-step cap: the ecosystem became sustainable, with far more births than
in Iteration 1. Females stay the small, fragile population (6 against 16 males at the end).

**Rollout: approach toward the target** (P(step onto target when adjacent) as a multiple of a
random mover; 30 episodes per checkpoint; bootstrap CIs exclude 0 for every fruit value):

| | Iter 100 | Iter 150 | Iter 200 | Iter 300 |
|---|---|---|---|---|
| Male, fruit | x1.25 | x1.34 | x1.40 | **x1.54** |
| Female, fruit | x1.24 | x1.41 | x1.54 | **x1.65** |
| Male, prey | x1.00 | x1.03 | x1.06 | **x1.07** |
| Female, prey | x0.96 | x0.98 | x0.99 | **x0.99** |

**Findings**
1. Predators can learn to steer here. Fruit approach grew steadily and had not plateaued, in
   both sexes. Iteration 1 showed none. So the earlier null result was at least partly a
   reward-density problem.
2. Prey approach stays near random (about 2% of an always-approach policy). Males lean slightly
   toward prey (+0.027 cells) and females slightly away (-0.008): the direction the hypothesis
   predicts, but from a single seed and far too small to call specialization.
3. Hunting attempts by sex are still close (females about 80% of males). Success rates equal the
   configured odds (0.90 / 0.20), so they carry no information about skill.
4. **Likely reason there is no risk-driven specialization:** `penalty_predator_death_in_combat`
   is 0 and the catch reward is the same for both sexes, so the learner never feels the female's
   10% death risk. Only the diffuse loss of future reproduction reward penalizes dying.

**Caveats:** one seed; 30 rollout episodes per checkpoint; the foraging reward is shaping applied
symmetrically, so this run is a capability test, not evidence for the sparse-reward design.

---

## Next steps

1. **Run the combat-death penalty test** (`--penalty-combat-death` flag now exists; magnitude >= 0,
   stored as `penalty_predator_death_in_combat = -value`) on top of the foraging reward, and check whether female hunting drops
   relative to male. This is the direct test of the risk-asymmetry hypothesis. Reward shaping,
   so a decision for the researcher.
2. **More seeds** for the foraging configuration (seed 42 only so far) before trusting any
   sex-difference in prey approach.
3. **Fruit-only shaping** (no catch reward) to separate "learns to gather" from "learns to hunt".
4. **Female survival:** females are near-extinct in the sparse run and 6 vs 16 in the foraging
   run. Check whether birth cost, gift rate, or starting population is the limiting factor.
5. Longer runs if the fruit-approach trend continues past 300 iterations.
6. Unexplained: the early rise in births and episode length in Iteration 1 cannot come from
   predator learning; the prey policy changing behavior is the untested guess.
