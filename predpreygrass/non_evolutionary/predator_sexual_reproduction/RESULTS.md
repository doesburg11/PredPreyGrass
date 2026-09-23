# Predator Sexual Reproduction — Training Results

Results from PPO training runs on the `predator_sexual_reproduction` environment. See
`README.md` for the mechanism design (sexed predators, mate-search + energy-threshold
reproduction, stochastic hunting, nuptial-gift provisioning, parental care) and the
human-cooperation framing behind it.

This is a running research log, not just a final write-up — it records the trial-and-error
trail (what was tried, why, what was found) so the search can be re-evaluated later without
reconstructing it from conversation history.

---

## Narrative so far (start here)

The short version: the chain of failure → diagnosis → fix → next question, no data tables. Full
detail for each step is in the numbered Iteration log below; the shortest way in is to read this,
then jump to whichever iteration number you need.

1. **Nothing emerged.** Realistic hunting odds + a sparse reward (Iteration 1): predators barely
   learned to hunt at all, no sex difference visible.
2. **Diagnosed:** training itself was slow — learner-update-bound, not environment-bound
   (Iteration 5) — fixed with a larger PPO minibatch size (5.8x speedup). A prerequisite fix, not
   the main finding.
3. **Diagnosed:** with foraging rewards added, predators were "farming" fruit instead of hunting —
   a flat per-event reward paid the same whether a fruit patch was full or nearly empty, so there
   was no incentive to hunt over scavenging (Iteration 6).
4. **Fix:** switched to an energy-proportional reward (pay for energy actually gained, not a flat
   per-event bonus). Under this fix, a real division of labor appeared for the first time — males
   hunt more, females gather more — replicated across three seeds (Iteration 7).
5. **Asked why.** The built-in difference between the sexes is hunting odds (males succeed 90%,
   die 5%; females succeed 20%, die 10%). Ablations giving females the males' odds removed or
   reversed the split, pointing at success rate — not death risk — as the driver (Iterations 8-9).
6. **Confound found.** Those same ablations also collapsed the prey population every time (down to
   0.4-2.9 prey left alive) — so it was unclear whether the *odds change* or the *ecological
   collapse* removed the split. Not a settled answer.
7. **Fix:** built a new environment, `FixedPreyDensityEnv`, that tops the prey population back up
   to a floor whenever it drops, removing the collapse confound so the odds question can be tested
   on a stable ecology (Iteration 11).
8. **Bug found in that fix.** The per-episode budget of prey agent IDs (shared between normal
   births and floor top-ups) could itself run out under heavy hunting, silently disabling the
   floor. Fixed by raising the budget 25x (2,000 → 50,000), validated with a calibration run before
   committing further compute.
9. **Where we are now:** a 12-run factorial (2×2 over success rate × death risk, 3 seeds each,
   inside the fixed environment) has completed, without the collapse confound -- see Iteration 12,
   though it surfaced a new confound of its own (predator population size, not held fixed by the
   floor, grows sharply with hunting success).

---

## Run names used in this log

- **REALISTIC** (Iteration 1): `PPO_PREDATOR_SEXUAL_REPRODUCTION_REALISTIC_SEED42`. Shipped-default
  hunting odds and a **sparse reward**: only `reproduction_reward_predator/prey` = 10.0 is
  nonzero. 100 iterations.
- **FORAGING** (Iteration 2): `PPO_PREDATOR_SEXUAL_REPRODUCTION_FORAGING_CHECK_SEED42`. Identical
  to REALISTIC, plus `reward_predator_catch_prey` = 1.0 and `reward_predator_gather_fruit` = 0.5
  for both sexes (a **foraging reward**). 300 iterations. A diagnostic, not a proposed reward.
- **FORAGING_PENALTY** (Iteration 3): `PPO_PREDATOR_SEXUAL_REPRODUCTION_FORAGING_PENALTY_SEED42`.
  Identical to FORAGING, plus `penalty_predator_death_in_combat` = -1.0 (a predator that dies in a
  failed hunt receives -1.0; applies to both sexes). 300 iterations.
- **FORAGING_PENALTY02** (Iteration 4): `PPO_PREDATOR_SEXUAL_REPRODUCTION_FORAGING_PENALTY02_SEED42`.
  Identical to FORAGING_PENALTY but with `penalty_predator_death_in_combat` = -0.2. 300 iterations.
- **AB_MB1024** and **AB_EP10** (Iteration 5): 50-iteration speed variants of the FORAGING
  configuration (`..._AB_MB1024_SEED42`: `--minibatch-size 1024`; `..._AB_EP10_SEED42`:
  `--num-epochs 10`). All other runs use PPO `minibatch_size=128`, `num_epochs=30`.
- **REF** (Iteration 7): `PPO_PREDATOR_SEXUAL_REPRODUCTION_REF_PEN02_MB1024_SEED42`. The FORAGING_PENALTY02
  configuration (flat rewards catch 1.0 / fruit 0.5, penalty 0.2) at `--minibatch-size 1024`.
- **PROP k=0.2** (Iteration 7): `PPO_PREDATOR_SEXUAL_REPRODUCTION_PROP_REWARD_MB1024_SEED42`. Energy-proportional
  reward `reward_predator_per_energy` = 0.2, flat rewards 0, penalty 0.2, minibatch 1024.
- **PROP k=0.5** (Iteration 7): `PPO_PREDATOR_SEXUAL_REPRODUCTION_PROP_K05_MB1024_SEED42`, `..._SEED43`, `..._SEED44`:
  as PROP k=0.2 with `reward_predator_per_energy` = 0.5 (three seeds).
- **NOPEN** (Iteration 9): `PPO_PREDATOR_SEXUAL_REPRODUCTION_NOPEN_K05_MB1024_SEED42/43`: as PROP k=0.5 with `--penalty-combat-death 0.0`.
- **ABL_FSUCC40/60** (Iteration 9): `..._ABL_FSUCC40_K05_MB1024_SEED42`, `..._ABL_FSUCC60_K05_MB1024_SEED42`: as ABL_SUCCESS_ONLY but women's success is
  0.40 / 0.60 instead of 0.90 (death kept at their own 0.10). Response-surface points between the baseline and ABL_SUCCESS_ONLY.
- **ABL_SUCCESS_ONLY / ABL_DEATH_ONLY**, seeds 43 and 44 (Iteration 9): replications of the Iteration 8 single-factor ablations.
- **ABL_FDEATH20 / ABL_FDEATH30** (Iteration 10): `..._ABL_FDEATH20_K05_MB1024_SEED42/43`, `..._ABL_FDEATH30_K05_MB1024_SEED42/43`:
  as ABL_DEATH_ONLY but women's death chance is 20% / 30% instead of 5% (success held at their own 20%).
- **FixedPreyDensityEnv / SUCCESS_ONLY_MATCHED / EQUAL_ODDS_MATCHED** (Iteration 11): a new `PredPreyGrass` subclass
  (`fixed_prey_density_env.py`) that replenishes prey to a floor (20) after every step, to test the odds ablations without
  ecological collapse. Trained via a new script, `tune_ppo_fixed_prey_density.py`. `PPO_FIXED_PREY_DENSITY_SUCCESS_ONLY_SEED42`
  (women 90%/10%) and `..._EQUAL_ODDS_SEED42` (women 90%/5%), k=0.5, penalty 0.2, minibatch 1024, 300 iterations, seed 42.
- **ABL_EQUAL_ODDS** (Iteration 8): `PPO_PREDATOR_SEXUAL_REPRODUCTION_ABL_EQUAL_ODDS_K05_MB1024_SEED42`, `..._SEED43`: PROP k=0.5 with women
  given the men's hunting odds (success 0.90, death 0.05). **ABL_SUCCESS_ONLY** (`--female-success-prob 0.90 --female-death-prob 0.10`)
  and **ABL_DEATH_ONLY** (`--female-success-prob 0.20 --female-death-prob 0.05`), seed 42: single-factor variants.
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

**Evaluation episode** (`results_figures/evaluation_population_foraging_iter300_seed42.png`, final
checkpoint, seed 42, argmax actions): the episode runs the full 1000 steps with all three
populations alive. Prey oscillate 24-47, females peak at about 19 near step 350 and fall to 3 by
the end, males climb from 6 to about 17-21. One episode, illustrative only.

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

### Iteration 3 — FORAGING_PENALTY: combat-death penalty test (complete)

**Purpose:** test the likely explanation from Iteration 2: with no penalty for dying in a failed
hunt, the learner never feels the female's 10% death risk, so risk-driven division of labor has no
gradient. If that is right, adding a penalty should make females (riskier) hunt less than males.

**Config:** `PPO_PREDATOR_SEXUAL_REPRODUCTION_FORAGING_PENALTY_SEED42`, seed 42, 300 iterations,
identical to FORAGING (catch +1.0, fruit +0.5) plus `--penalty-combat-death 1.0`
(`penalty_predator_death_in_combat = -1.0`, both sexes). Ran 2026-09-20 09:58 to 19:06 as unit
`psr-foraging-penalty` (about 9 hours, 82-115 s/iteration).

**Training curves** (block means, per episode; FORAGING in brackets, see the chart):

| Iterations | Episode length | Males / females alive at end | Prey alive at end | Hunting attempts M / F | Births M / F |
|---|---|---|---|---|---|
| 91-100 | 707 (829) | 7.0 / 0.5 (8.9 / 1.0) | 44.5 (41.5) | 183 / 135 (208 / 179) | 12.4 / 11.8 (16.7 / 16.7) |
| 191-200 | 545 (839) | 4.1 / 0.3 (13.9 / 1.7) | 49.6 (34.6) | 108 / 74 (270 / 187) | 4.9 / 5.3 (25.3 / 19.9) |
| 291-300 | **607 (1001)** | 3.8 / 0.35 (16.1 / 6.0) | 49.8 (31.3) | 109 / 78 (348 / 279) | 3.0 / 3.6 (30.0 / 36.0) |

Unlike FORAGING, which improved steadily, this run peaked around iteration 70 (episode length
about 800) and then declined: the predator population shrank, births fell about 10x against
FORAGING, and prey rose to the 50 cap.

**Rollout: approach toward the target** (30 episodes per checkpoint; FORAGING at iteration 300 in
brackets):

| | Iter 100 | Iter 150 | Iter 200 | Iter 250 | Iter 300 |
|---|---|---|---|---|---|
| Female, prey (cells) | -0.010 | -0.025 | -0.039 | -0.044 | **-0.056** (-0.008) |
| Male, prey (cells) | +0.005 | +0.014 | +0.018 | +0.015 | +0.013 (+0.027) |
| Male, P(step onto prey) vs random | x0.96 | x0.99 | x0.95 | x0.92 | **x0.87** (x1.07) |
| Female, P(step onto prey) vs random | x0.94 | x0.96 | x0.95 | x0.97 | x0.93 (x0.99) |
| Female, fruit | x1.28 | x1.36 | x1.46 | x1.53 | x1.59 (x1.65) |
| Male, fruit | x1.12 | x1.21 | x1.30 | x1.39 | x1.52 (x1.54) |

**Findings**
1. **The penalty did not create a sex-specific division of labor.** Females do lean away from prey
   at a distance about 7x more than in FORAGING (-0.056 against -0.008 cells), the predicted
   direction. But males also became less willing to step onto an adjacent prey (x0.87 against
   x1.07 in FORAGING), because the penalty applies to both sexes and males can die too (5% per
   attempt).
2. **The penalty suppressed hunting in both sexes.** The female-to-male attempt ratio is unchanged
   (0.72 against 0.80), while attempts fell about 70% in both. Fewer predators alive also lowers the
   counts mechanically, so "hunt less" and "fewer hunters" cannot be fully separated.
3. **The ecosystem got worse.** Episodes end at 607 steps instead of hitting the 1000 cap, and
   females are nearly extinct at the end (0.35 alive).
4. **Fruit learning is unaffected** (x1.59 / x1.52 against x1.65 / x1.54), and by iteration 300 the
   male-female gap in fruit approach has closed.

**Caveats:** one seed per configuration, so the FORAGING baseline may itself be a lucky seed
(earlier trials in this project showed founder-effect luck). A penalty of 1.0 equals the catch
reward and may simply be too large; a penalty is not sex-specific in this implementation.

### Iteration 4 — FORAGING_PENALTY02: a smaller combat-death penalty (complete)

**Purpose:** Iteration 3 showed a penalty of 1.0 suppresses hunting in both sexes and collapses
the predator population. Test whether a much weaker penalty (0.2) keeps the female avoidance
without the collapse.

**Config:** `PPO_PREDATOR_SEXUAL_REPRODUCTION_FORAGING_PENALTY02_SEED42`, seed 42, 300 iterations,
identical to FORAGING plus `--penalty-combat-death 0.2` (`penalty_predator_death_in_combat = -0.2`,
both sexes). Ran 2026-09-20 20:31 to 2026-09-21 04:17 as unit `psr-foraging-penalty02` (about 7.8
hours, 82-115 s/iteration). Minibatch 128, 30 epochs, like every run up to this point.

**Training curves** (block means, per episode; FORAGING and FORAGING_PENALTY in brackets):

| Iterations | Episode length | Males / females alive at end | Prey alive at end | Hunting attempts M / F | Births M / F |
|---|---|---|---|---|---|
| 91-100 | 822 (829, 707) | 9.8 / 1.6 (8.9 / 1.0, 7.0 / 0.5) | 41.6 | 210 / 161 | 16.2 / 14.9 |
| 191-200 | 973 (839, 545) | 17.8 / 3.5 (13.9 / 1.7, 4.1 / 0.3) | 31.4 | 369 / 237 | 35.6 / 28.3 |
| 291-300 | **990** (1001, 607) | 19.2 / 4.9 (16.1 / 6.0, 3.8 / 0.35) | 28.3 | 366 / 278 | 36.9 / 39.2 |

The ecosystem stays as healthy as plain FORAGING (episodes at the 1000-step cap, births and
populations at or above FORAGING). The 1.0 penalty had already declined by iteration 150.

**Rollout: approach toward the target** (30 episodes per checkpoint; FORAGING and FORAGING_PENALTY
at iteration 300 in brackets):

| | Iter 100 | Iter 200 | Iter 300 |
|---|---|---|---|
| Male, prey (cells) | +0.014 | +0.031 | **+0.040** (+0.027, +0.013) |
| Female, prey (cells) | -0.004 | -0.011 | **-0.026** (-0.008, -0.056) |
| Male, P(step onto prey) vs random | x1.00 | x1.06 | **x1.08** (x1.07, x0.87) |
| Female, P(step onto prey) vs random | x0.99 | x0.99 | **x0.96** (x0.99, x0.93) |
| Male, fruit | x1.24 | x1.39 | x1.52 (x1.54, x1.52) |
| Female, fruit | x1.23 | x1.61 | **x1.74** (x1.65, x1.59) |

**Findings**
1. **No collapse.** Unlike 1.0, the 0.2 penalty leaves the ecosystem intact.
2. **The predicted pattern appears.** Males approach prey and step onto it more than a random
   mover (x1.08), where the 1.0 run made them stop (x0.87). Females lean away from prey about
   three times as much as in FORAGING (-0.026 against -0.008 cells) and approach fruit more than
   males (x1.74 against x1.52), where in FORAGING the sexes were equal. The prey avoidance grew
   over training (-0.004, -0.011, -0.026 at iterations 100, 200, 300).
3. **The hunting-attempt ratio does not show it.** The female/male attempt ratio was 0.64-0.77
   across blocks (FORAGING 0.69-0.86), too noisy to use; the rollout approach measurement is the
   informative one.

**Update (Iteration 7):** the male-toward / female-away prey pattern below did NOT reproduce when the same rewards were
re-run at minibatch 1024 (REF: men x0.85 on prey). Treat it as unconfirmed; only the female-over-male fruit preference held.

**Caveats:** one seed per configuration (earlier trials in this project showed founder-effect
luck); the rollout intervals cover only variation between episodes of one trained policy, not
between training runs. The effect is small: the male-female gap in stepping onto adjacent prey is
about 12 percentage points (x1.08 against x0.96), against 8 in FORAGING, where a real division of
labor would be far larger. Only one penalty value between 0 and 1 has been tried.

### Iteration 5 — Why training is slow, and the effect of the minibatch size (complete)

Recorded in full because it changes what an experiment costs and may change what the policies learn.

**Observation (2026-09-20 22:42, during the penalty-0.2 run at iteration 73).** The run used about
1.2 CPU cores on average out of 32 (94.7% idle, load average 1.2-1.4). The GPU (RTX 5070 Ti) sat at
31% utilization with 10.3 of 16.3 GB in use. RAM use was 31 of 93 GB. Ray reserved 28 CPUs and the
whole GPU on paper, far more than the run used. The 8 environment runners were almost idle; one
learner process ran at about 98% of a single core.

**Two earlier statements of mine were wrong** and are corrected here: (1) that iterations are
dominated by CPU environment simulation (they are not), and (2) that two runs would compete for the
same cores (they would not; only Ray's reservation and, as found below, GPU memory limit running
jobs in parallel).

**Search.**
- Read the PPO settings in the tune script: `train_batch_size_per_learner=1024` (in environment
  steps), `minibatch_size=128`, `num_epochs=30`.
- Tried `py-spy` on the running learner: refused (`Permission Denied`, `ptrace_scope=1`, would need
  root). Not pursued; no system security settings were changed.
- Read the per-phase timers already in the run's `progress.csv` and TensorBoard events:

| Iteration | Total | Environment sampling | Learner update |
|---|---|---|---|
| ~10 | 31.5 s | 1.1 s | 30.4 s |
| ~37 | 50.6 s | 1.4 s | 49.2 s |
| ~73 | 68.2 s | 1.6 s | 66.5 s |

  About 97% of every iteration is the learner update.
- The logged training counters show about 1.26 million samples trained per policy per iteration,
  identical for all three policies, with `module_train_batch_size_mean` = 128: about 9,800
  minibatches per policy, about 29,000 gradient steps per iteration, about 2.3 ms per step.
- Read the installed RLlib source (ray 2.58.0, `ray/rllib/utils/minibatch_utils.py`,
  `MiniBatchCyclicIterator`).

**Explanation.**
1. The cost is the number of tiny gradient steps, not the mathematics: 128-sample steps on a small
   network are limited by single-threaded Python and kernel-launch overhead, which is why one CPU
   core is saturated while the GPU idles.
2. The multi-agent iterator takes 128 samples from every policy in lockstep and runs until every
   policy has covered its data `num_epochs` times. The policy with the most samples (prey, with the
   most agent-steps per environment step) therefore sets the total number of steps. Policies with
   fewer samples (the two predator policies) are cycled repeatedly and effectively receive many
   more than 30 epochs per iteration, roughly 30 times the ratio of prey to predator samples.
3. Possible science side effect (hypothesis, not proven): that extra reuse of the small predator
   datasets may destabilize predator learning. Iteration 5's A/B result is consistent with it but
   does not prove it.
4. **GPU memory** limits parallelism: the learner grows from about 4 GB to about 10 GB of 16 GB, so
   only one GPU run fits at a time; a second could trigger an out-of-memory error in the running
   job. CPU-only learners (`--gpu-fraction 0`) are safe but were about 5 times slower in Iteration 1.

**A/B test.** Two 50-iteration variants of the FORAGING configuration (catch +1.0, fruit +0.5, no
penalty), seed 42, run one after the other on the GPU after the penalty-0.2 run finished, using the
new flags added to the tune script (defaults unchanged, 128 and 30):

```
python -m predpreygrass.non_evolutionary.predator_sexual_reproduction.tune_ppo_predator_sexual_reproduction \
    --seed 42 --max-iters 50 --reward-catch-prey 1.0 --reward-gather-fruit 0.5 \
    --name PPO_PREDATOR_SEXUAL_REPRODUCTION_AB_MB1024_SEED42 --minibatch-size 1024
# and, for the second variant: --name ..._AB_EP10_SEED42 --num-epochs 10
```

The baseline for comparison is the first 50 iterations of the existing FORAGING run
(minibatch 128, 30 epochs), not a re-run.

| | Baseline (128, 30 ep) | **MB1024** (1024, 30 ep) | EP10 (128, 10 ep) |
|---|---|---|---|
| Time for 50 iterations | 84.7 min | **14.7 min** | 30.3 min |
| Seconds per iteration | 41.3 | 8.1 | 15.6 |
| Learner update per iteration | 40.0 s | 6.9 s | 14.4 s |
| Speedup | 1x | **5.8x** | 2.8x |
| Samples trained per iteration | 4.08M | 3.45M | 1.15M |
| Iters 41-50: episode length | 657 | **992** | 811 |
| Births male / female | 8.8 / 9.4 | 27.3 / 29.8 | 17.6 / 19.3 |
| Males / females alive at end | 6.2 / 0.5 | 13.0 / 5.5 | 11.3 / 2.1 |
| Hunting attempts M / F | 161 / 122 | 265 / 262 | 183 / 213 |

**Rollout at iteration 50** (P(step onto target) vs a random mover, 30 episodes):

| | Baseline | MB1024 | EP10 |
|---|---|---|---|
| Male, fruit | x1.18 | **x1.34** | x1.18 |
| Female, fruit | x1.12 | **x1.47** | x1.27 |
| Male, prey | x0.96 | x1.05 | x0.93 |
| Female, prey | x0.95 | x1.00 | x0.87 |

**Findings**
1. `minibatch_size=1024` makes training **5.8x faster** (a 300-iteration run of about 1.5 hours
   instead of 8-9), matching the 5-8x estimate from the diagnosis.
2. Both variants also reached a healthier ecosystem than the baseline by iteration 50; MB1024 is
   ahead of EP10 on every measure. At iteration 50 MB1024 shows roughly the fruit approach the
   baseline reached at iteration 150 (male x1.34, female x1.41 there).
3. Without any penalty, MB1024 shows females approaching fruit more than males (+0.102 against
   +0.055 cells), a possible sex difference worth checking with more seeds.

**Caveats:** one seed each, so the speedup is solid but the learning advantage could partly be seed
luck; the hypothesis that excess predator epochs hurt learning is untested; the rollouts cover 50
iterations only, not whether the final behavior at 300 iterations differs. Any change of
`minibatch_size`/`num_epochs` changes the dynamics, so runs with different settings are not
directly comparable; every run in Iterations 0-4 used 128 / 30.

**Decision status:** the flags `--minibatch-size` / `--num-epochs` exist with the old defaults
(128 / 30) so all documented runs stay reproducible. Recommended, not yet adopted: default
`--minibatch-size 1024` for new runs, and re-run the key configurations (FORAGING, both penalties)
on it for comparability.

### Iteration 6 — Where does a predator's energy come from? (measurement)

**Question:** how much energy does a man (or woman) take in over a lifetime from fruit and from prey, and do gifts
matter? Asked because males approach fruit as strongly as females do (Iterations 2-4) and fruit looked "too easy".

**Method:** `analyze_energy_sources.py` rolls out a checkpoint (20 episodes, stochastic actions, on CPU) using a thin
subclass of the environment that only adds bookkeeping; the environment file is not touched. Per predator life it records
the lifetime and the gross energy taken in from fruit, from prey (the prey's full energy at the moment of a catch), and
the transfers in and out (mate gift, parental care), measured from the change in every predator's energy around
`_apply_male_gift` and `_share_energy_with_offspring`. Validated against the environment's own counters: prey catches
14/1, 29/5, 90/16 (male/female) match `hunting_successes` exactly, no hunt is left unmatched, and the care and gift
totals equal `parental_care_energy_total` / `mate_gift_energy_total` (9.597 and 2.806 in one test episode). A first
version counted only own foraging and was replaced after it was pointed out that gifts received were missing.

**Results, per life, all lives, final checkpoint (men | women):**

| Run | Men: life | fruit energy (fruits) | prey energy (catches) | prey share | net intake /100 steps | Women: life | fruit (fruits) | prey (catches) | prey share |
|---|---|---|---|---|---|---|---|---|---|
| Random policy | 236 | 35.9 (38.6) | 25.2 (5.7) | 41.3% | 25.8 | 148 | 22.1 (24.0) | 3.9 (0.9) | 14.8% |
| FORAGING (MB128) | 385 | 62.3 (121) | 40.9 (8.7) | 39.6% | 26.7 | 192 | 33.7 (59) | 5.5 (1.2) | 14.1% |
| FORAGING_PENALTY02 (MB128) | 389 | 60.6 (117) | 40.3 (8.5) | 39.9% | 25.8 | 179 | 31.5 (55) | 5.4 (1.1) | 14.5% |
| REF (MB1024) | 382 | 64.2 (159) | 38.5 (8.8) | 37.4% | 26.7 | 197 | 34.5 (69) | 5.3 (1.2) | 13.2% |

**Findings**
1. **Training did not change what men eat.** Their prey share (about 37-40%) equals a random walker's (41%); they only
   live longer and take in more in total.
2. **Gifts are energetically negligible.** A woman receives about 0.3-0.5 energy from her mate and 0.5-0.8 from parents per
   life (about 3% of her own intake); a man gives his mate about 0.4-0.5. The mate-gift mechanism has almost no effect
   at these settings.
3. **A fruit is nearly empty when eaten.** Men eat 121-159 fruits per life at only 0.4-0.5 energy each (full is 2.0), while a
   prey catch gives about 4.7. They re-visit fruit long before it regrows (0.04 energy per step).
4. **The flat fruit reward is being farmed.** The reward is a fixed +0.5 per fruit and +1.0 per catch, independent of the
   energy gained, so per unit of energy fruit pays about 4.6 times more than prey (0.5/0.51 against 1.0/4.7). A man collects
   about 61-80 reward per life from fruit, against about 9 from prey and about 8 from reproduction (0.8 births x 10).

### Iteration 7 — Energy-proportional reward (k = 0.2, k = 0.5) and a like-for-like reference

**Change:** new setting `reward_predator_per_energy` (k), default 0 = off. A forage pays reward = k x (gross energy
gained) for fruit and prey alike, added to the flat per-event rewards; here the flat rewards are set to 0. Motivation:
Iteration 6 finding 4. Implemented in `predpreygrass_rllib_env.py` (`_resolve_hunting_attempt` and the two fruit
branches) with the tune-script flag `--reward-per-energy`; 15 new tests (47 in total), Codex-reviewed. Codex found one
real, older bug that the change would have made worse: a prey that starves in the same step can still be caught by a
predator processed before it, giving the predator zero or negative energy; the gain is now clamped at 0, with a test
that fails without the clamp. Codex also asked for tests on failed hunts, `cumulative_rewards` and female fruit, added.

**Runs** (all seed 42, minibatch 1024, 300 iterations, penalty 0.2, 60-85 minutes each): REF (flat rewards, same as
FORAGING_PENALTY02 but at 1024, so the comparison is like for like), PROP k=0.2, PROP k=0.5. PROP k=0.5 was then
replicated with seeds 43 and 44 (about 60-85 minutes each, 2026-09-21 16:55-18:49). Chart:
`results_figures/population_over_training_mb1024.png`.

**Ecology, iterations 291-300 (block means, per episode):**

| | Episode length | Hunting attempts M / F | F/M ratio | Births M / F | Males / females alive | Prey alive |
|---|---|---|---|---|---|---|
| REF (flat) | 1001 | 342 / 239 | 0.70 | 31 / 32 | 17.2 / 6.4 | 32.0 |
| PROP k=0.2 | **640** | 154 / 115 | 0.75 | 8 / 8 | 5.1 / **0.6** | 47.6 |
| PROP k=0.5 | 926 | 473 / 191 | **0.40** | 37 / 39 | 20.1 / 5.4 | 19.2 |

REF stays healthy all run, so PROP k=0.2's decline (from about iteration 110) is caused by the reward, not by the faster
minibatch or seed luck. Likely reason: at k=0.2 a fruit pays only about 0.1 reward, which removes most of women's incentive
to gather; their population collapsed (0.3-0.6 alive) and births with it. At k=0.5 the collapse does not occur
(8.7 women alive at iteration 200, 5.4 at 300), though women drift down late in the run, and episodes end slightly
below the cap after iteration 245 (prey are depleted to about 19).

**Rollout: P(step onto an adjacent target) as a multiple of a random mover** (30 episodes; approach bias in cells in
brackets), iterations 100 / 200 / 300:

| | REF | PROP k=0.2 | PROP k=0.5 |
|---|---|---|---|
| **Male, prey** | x0.92 / x0.90 / x0.85 (+0.018 / +0.005 / -0.010) | x1.15 / x1.09 / x1.18 (+0.073 / +0.088 / +0.104) | x1.24 / x1.32 / **x1.44** (+0.201 / +0.275 / **+0.286**) |
| Female, prey | x1.04 / x0.97 / x0.91 | x0.98 / x0.95 / x1.00 | x1.14 / x1.08 / x0.99 |
| Male, fruit | x1.42 / x1.54 / x1.61 | x1.24 / x1.31 / x1.42 | x1.27 / x1.47 / x1.54 |
| **Female, fruit** | x1.61 / x1.75 / x1.78 | x1.21 / x1.18 / x1.12 | x1.72 / x1.87 / **x1.91** |

**Energy per life, final checkpoint (all lives):**

| | Men: life | fruit energy (fruits) | prey energy (catches) | prey share | net /100 steps | Women: life | fruit (fruits) | prey (catches) | prey share |
|---|---|---|---|---|---|---|---|---|---|
| REF | 382 | 64.2 (159) | 38.5 (8.8) | 37.4% | 26.7 | 197 | 34.5 (69) | 5.3 (1.2) | 13.2% |
| PROP k=0.2 | 307 | 82.5 (84) | 49.7 (11.1) | 37.6% | 42.9 | 166 | 25.0 (34) | 6.5 (1.4) | 20.5% |
| PROP k=0.5 | 363 | 55.9 (98) | 49.4 (10.3) | **46.9%** | 28.8 | 164 | 30.0 (57) | 3.7 (0.8) | **11.1%** |

**Replication of k = 0.5 (seeds 42, 43, 44; iteration 300, 30 rollout episodes, 20 energy episodes):**

| | Seed 42 | Seed 43 | Seed 44 | Flat-reward runs (FORAGING, PENALTY02, REF) |
|---|---|---|---|---|
| Male, prey: approach bias (cells) | +0.286 | +0.281 | +0.263 | -0.010 to +0.040 |
| **Male, prey: P(step onto adjacent) vs random** | x1.44 | x1.49 | x1.32 | x0.85 to x1.08 |
| Female, prey: P(step onto adjacent) | x0.99 | x1.02 | x1.07 | x0.91 to x0.99 |
| Male, fruit | x1.54 | x1.52 | x1.56 | x1.54 to x1.61 |
| **Female, fruit** | x1.91 | x1.86 | x1.83 | x1.65 to x1.78 |
| **Men: prey share of own energy intake** | 46.9% | 47.8% | 49.6% | 37.4% to 39.9% (random walker 41.3%) |
| Men: prey caught / fruits eaten per life | 10.3 / 98 | 10.3 / 99 | 10.6 / 96 | 8.5-8.8 / 117-159 |
| **Women: prey share of own energy intake** | 11.1% | 12.1% | 11.6% | 13.2% to 14.5% (random walker 14.8%) |
| Women: prey caught per life | 0.8 | 0.9 | 1.0 | 1.1 to 1.2 |
| Ecology, iters 291-300: episode length | 926 | 1001 | 1001 | 990 to 1001 |
| Ecology: males / females alive at end | 20.1 / 5.4 | 18.6 / 6.5 | 19.8 / 8.5 | 16.1-19.2 / 4.9-6.4 |
| Ecology: hunting attempts F/M ratio | 0.40 | 0.41 | 0.53 | 0.70 to 0.80 |

The three seeds agree closely, and none of the flat-reward runs overlaps them on the male prey measures or on the men's
prey share. The lowest male prey step-onto ratio at k = 0.5 (x1.32) is above the highest flat-reward value (x1.08); the men's
prey share (46.9-49.6%) is above the highest flat-reward value (39.9%) and above a random walker (41.3%). The spread between
the three seeds (0.023 cells in approach bias, +0.263 to +0.286) is about half the 0.050-cell difference between two flat-reward runs of
the same configuration (FORAGING_PENALTY02 at minibatch 128: +0.040; REF at minibatch 1024: -0.010).

**Findings**
1. **Flat rewards pay repeated eating of nearly empty fruit, and the proportional reward stops that.** In REF a fruit yields 0.40 energy; at k=0.2
   it yields 0.99 and net intake per step rises 60%. That flat rewards cause the missing male prey preference through this farming is a
   plausible explanation, not an isolated one: k=0.2 and k=0.5 pay the same per unit of energy for fruit and prey and differ only in reward scale,
   so the different outcomes at 0.2 and 0.5 cannot be attributed to closing the loophole alone.
2. **k = 0.2 makes men better foragers but starves the women** (population collapse above).
3. **k = 0.5 produces the sex pattern that was hypothesized, without collapse, and it replicates on three seeds:** men approach prey much more strongly
   (+0.286 cells, x1.44 stepping onto adjacent prey, against x0.85 in REF and -0.010 to +0.040 cells in the flat-reward runs),
   draw 46.9% of their energy from prey (above a random walker's 41%, the first time trained men do), while women hunt
   less than a random walker (11.1% prey share, 0.8 catches per life) and approach fruit the most of any run (x1.91).
4. **Earlier penalty finding not reproduced.** The FORAGING_PENALTY02 result (Iteration 4: men approach prey x1.08, women
   avoid x0.96) did not reproduce at minibatch 1024 with the same rewards (REF: men x0.85). The two runs differ in minibatch
   (128 against 1024), so the 0.05-cell difference is partly a systematic setting effect and not a seed-variance estimate; either way
   Iteration 4's male/female prey pattern should be treated as unconfirmed. The reproduced part is that women approach fruit somewhat more than men (x1.78 against x1.61 in REF).

**Caveats:** k = 0.5 has three seeds, but REF and PROP k=0.2 have one each and the flat-reward comparison rests on three runs of
related configurations; k=0.5 was chosen after seeing k=0.2 fail (two values tried), and the combat-death penalty (0.2) was not
re-tuned; prey are depleted to about 18-21 at k=0.5, which may limit predators in longer runs; the effect sizes are modest in absolute
terms (men step onto adjacent prey about 1.4x as often as a random mover, not 5x); the energy-source runs cover 20 episodes of the final
checkpoint only; hunting-attempt counts are noisy and were not used as evidence.

### Iteration 8 — Ablation: equal hunting odds for both sexes

**Purpose:** test whether the k = 0.5 division of labor (Iteration 7) depends on the built-in difference in hunting odds (men
90% success / 5% death, women 20% / 10%).

**Config:** as PROP k=0.5 (k = 0.5, flat rewards 0, penalty 0.2, minibatch 1024, 300 iterations) but with
`--female-success-prob 0.90 --female-death-prob 0.05`, so women have the men's odds. Seeds 42 and 43
(2026-09-21 19:30-21:10, about 50 minutes each).

**Results, iteration 300 (unequal k = 0.5 baseline in the last column):**

| | Equal odds, seed 42 | Equal odds, seed 43 | Unequal odds (three seeds) |
|---|---|---|---|
| Male, prey: approach bias (cells) | +0.254 | +0.397 | +0.26 to +0.29 |
| **Female, prey: approach bias (cells)** | **+0.406** | **+0.346** | +0.02 to +0.04 |
| **Men: prey share of own intake** | 38.0% | 33.7% | 46.9% to 49.6% |
| **Women: prey share of own intake** | 36.6% | 38.3% | 11.1% to 12.1% |
| Women / men hunting attempts (iters 291-300) | 2.33 | 1.41 | 0.40 to 0.53 |
| Prey alive at end / episode length (iters 291-300) | 1.6 / 268 | 0.0 / 175 | 18-21 / 926-1001 |
| Men / women life (steps) | 94 / 136 | 107 / 114 | about 370 / 165-200 |

**Findings**
1. **The division of labor disappears when the odds are equal**, on both seeds: women hunt as much as men (they approach prey as hard and
   make more attempts per episode) and both sexes take about the same share (34-38%) of their energy from prey.
2. **The ecosystem collapses**: with both sexes hunting at 90% success, prey are hunted to extinction within 175-270 steps and the
   episode ends. Behaviour figures for these runs therefore come from short, prey-poor lives.
3. **Working hypothesis: the hunting odds are necessary but not sufficient for the split.** (Not isolated; see the caveats.)
   - *Necessary (hypothesis):* removing the difference in odds removes the split (finding 1), but the equal-odds runs also collapse the prey, so
     necessity is not demonstrated under matched ecological conditions; the single-factor runs below suggest the difference in success rate
     is the part that matters.
   - *Not sufficient:* the same unequal odds (men 90/5, women 20/10) were in force in every earlier run, yet the split appeared only under
     the energy-proportional reward with k = 0.5, not with flat per-event rewards (Iterations 2-4, 7 REF) and not at k = 0.2, where women lost
     their incentive to gather and died out. A learner follows the payoff it is shown: with flat rewards a man was paid about 60-80 reward per
     life for eating nearly empty fruit against about 9 for prey (Iteration 6), so the real advantage of hunting never reached what he learned.
   Both a difference between the sexes and a reward that lets it be felt are needed.
4. **The remaining asymmetry did not produce a split on its own**: women still pay 90% of the birth cost and the gifts still run
   male-to-female, and with equal hunting odds there was no division of labor.

**Single-factor ablations (seed 42, run afterwards to separate success from death chance):** ABL_SUCCESS_ONLY gives women the men's
success (90%) but keeps their own death chance (10%); ABL_DEATH_ONLY gives them the men's death chance (5%) but keeps their own
success (20%). Both at k = 0.5, minibatch 1024, 300 iterations, 2026-09-21 21:11-23:02.

| Women's odds (success / death) | Men: prey share | **Women: prey share** | Women: prey approach (cells) | Women: P(step onto prey) | Women / men attempts | Prey alive | Episode length |
|---|---|---|---|---|---|---|---|
| **20% / 10%** (baseline, 3 seeds) | 46.9-49.6% | **11.1-12.1%** | +0.02 to +0.04 | x1.02-1.07 | 0.40-0.53 | 18-21 | 926-1001 |
| **20% / 5%** (death-only) | 44.9% | **15.3%** | +0.051 | x1.31 | 0.62 | 19.0 | 1001 |
| **90% / 10%** (success-only) | 33.4% | **47.9%** | +0.344 | x1.32 | 0.73 | 2.9 | 393 |
| **90% / 5%** (equal, seeds 42 / 43) | 38.0% / 33.7% | **36.6% / 38.3%** | +0.406 / +0.346 | x1.90 / x0.91 | 2.33 / 1.41 | 1.6 / 0.0 | 268 / 175 |

Men in the single-factor runs: prey approach +0.276 (death-only) and +0.358 (success-only) cells, life 381 and 294 steps; women live 199 and 132 steps.

**Findings from the single-factor runs**
1. **Women's success rate appears to be the more influential factor.** Giving women only the men's success (90%), with their 10% death chance kept, removes
   the split and even reverses it: women take 47.9% of their energy from prey against 33.4% for men, and approach prey as hard as men.
2. **The death chance alone does not remove the split.** With only the men's 5% death chance (success still 20%) men take 44.9% and women
   15.3% from prey, women still catch about a ninth of what men do (49 against 453 prey per episode) and the ecosystem stays healthy. There is a
   small shift toward hunting (women's prey share 11-12% to 15.3%, stepping onto prey x1.02-1.07 to x1.31, fruit approach x1.83-1.91 to x1.58),
   visible outside the three-seed baseline range but from one seed.
3. **The split looks more productivity-driven than risk-driven, on this evidence.** The module was designed around a "risk-driven" split (women
   riskier and less successful); these runs suggest the difference in success rate matters more than the 5-point difference in death chance
   tested, but they do not rule out a contribution of risk (one seed each; risk may also act through rare catastrophic outcomes). A rough estimate
   made earlier, that the higher death chance was probably the bigger reason women avoid hunting, was not supported: with that death chance
   unchanged, women hunt heavily once their success is high. The ablations are also not perfectly single-factor: success, death and failure share
   one random draw, so 90% success also removes the 70% harmless-failure outcome and changing the death chance changes the failure share.
4. **Consistent with learners following hunting productivity.** When women are as successful as men they hunt as much or more, which reads
   as adaptation to sex-specific hunting return rather than a fixed sex role (other explanations, such as differences in visited states, birth
   costs or prey depletion, are not excluded).

**Caveats:** two seeds for the equal-odds run and one seed for each single-factor run; the success-only and equal-odds runs are in a collapsed
regime (prey nearly extinct, short lives), so their behaviour numbers come from prey-poor worlds; the small death-only shift could be seed noise;
only the k = 0.5 reward has been tested.

### Iteration 9 — Second-opinion follow-up: no-penalty replication, shared-state analysis, response surface, and replicated ablations

A single overnight queue (`psr-queue34`, 2026-09-22 01:17-06:44) answering the gaps Codex flagged (Iteration 8's "Second opinion"):
whether the explicit combat-death penalty is needed, a policy-only comparison that does not depend on each policy's self-created
states, whether the success-rate effect is a threshold or a gradient, and replication of the single-factor ablations.

#### A. Does the explicit combat-death penalty matter? (NOPEN, two seeds)

k = 0.5, flat rewards 0, minibatch 1024, 300 iterations, `penalty_predator_death_in_combat = 0` (was -0.2). Iteration 300, against
the three penalty-0.2 seeds:

| | NOPEN s42 | NOPEN s43 | Penalty 0.2 (3 seeds) |
|---|---|---|---|
| Male, prey approach (cells) | +0.325 | +0.295 | +0.263 to +0.286 |
| Male, P(step onto adjacent prey) | x1.73 | x1.77 | x1.32 to x1.49 |
| Female, prey approach (cells) | +0.026 | +0.018 | +0.017 to +0.039 |
| Men: prey share of own energy | 49.9% | 47.0% | 46.9% to 49.6% |
| Women: prey share of own energy | 12.3% | 12.8% | 11.1% to 12.1% |
| Ecosystem (length, prey alive) | 1001, 23.8 | 1001, 22.7 | 926-1001, 18-21 |

The split is unchanged, if anything slightly stronger on the male side, without the explicit penalty. Deaths-in-combat per episode are
similar with and without it (about 20-29), so agents did not become reckless. **The explicit -0.2 penalty appears unnecessary for the
split at k = 0.5** (two seeds, one setting; it does not show that death risk itself is irrelevant, since dying still ends a predator's
future reward with or without the explicit term).

#### B. Shared-state analysis: does the difference survive when both policies see the same observations?

`analyze_shared_states.py` (new): a fixed bank of 14,876 predator observations from uniform-random-policy episodes (independent of any
trained policy); each checkpoint's male and female modules are queried on the same bank without stepping the environment, removing the
confound that rollouts measure a policy partly on the states it creates for itself. Prey approach-bias, male minus female, checkpoint 29:

| Run | Men | Women | **Men - women** |
|---|---|---|---|
| k=0.5, penalty 0.2 (seeds 42/43/44) | +0.230/+0.237/+0.189 | +0.009/+0.038/+0.043 | **+0.221/+0.199/+0.146** |
| k=0.5, no penalty (seeds 42/43) | +0.209/+0.228 | +0.030/+0.040 | **+0.178/+0.189** |
| REF (flat rewards) | +0.019 | -0.001 | +0.020 |
| PROP k=0.2 | +0.105 | +0.012 | +0.093 |
| FORAGING_PENALTY02 (minibatch 128) | +0.042 | -0.062 | +0.104 |
| ABL_DEATH_ONLY (women 20%/5%) | +0.200 | +0.040 | +0.160 |
| ABL_SUCCESS_ONLY (women 90%/10%) | +0.289 | +0.252 | +0.038 |
| ABL_EQUAL_ODDS, seed 42 / 43 | +0.241/+0.306 | +0.278/+0.211 | -0.037 / **+0.095** |

Confirms the rollout-based findings on a policy-only comparison: every k = 0.5 run (with or without the penalty) shows a difference of
+0.15 to +0.22 with intervals well above 0; REF shows about +0.02; success-only nearly erases it; death-only leaves it; k=0.2 is
intermediate. One nuance: ABL_EQUAL_ODDS seed 43 still shows a +0.095 preference difference even though the two sexes' *energy shares*
came out similar (Iteration 8) -- equal outcomes can hide a smaller remaining preference gap, because women there also succeed at 90%,
so a weaker preference still yields a lot of prey. Fruit, men minus women, is negative (women prefer fruit more) at k=0.5 with or
without the penalty (-0.05 to -0.14) and positive in REF and at k=0.2 (+0.06, +0.10), so the "women gather" side is also specific to
k=0.5. FORAGING_PENALTY02 (minibatch 128) shows a real +0.104 difference here, about half the k=0.5 size -- consistent with treating
its rollout result as a real but non-robust effect (the identical rewards at minibatch 1024, REF, show only +0.02).

#### C. Response surface: how does the split change with women's success rate?

Four points at k = 0.5, penalty 0.2, women's death chance held at 10% (their own), seed 42, 300 iterations: 20% (baseline), 40%, 60%,
and 90% (= ABL_SUCCESS_ONLY, death 10%).

| Women's success | Episode length | Women's catches/episode | Women/men attempts | Prey alive | Women: prey share of energy | Female prey approach (cells) |
|---|---|---|---|---|---|---|
| 20% (baseline) | 926 | 35.2 | 0.40 | 19.2 | 11.1% | +0.017 |
| **40%** | 913 | 76.6 | 0.43 | 21.4 | 26.4-27.2% | +0.106 |
| **60%** | **721** | **158.1** | **0.89** | **5.1** | 37.7-38.5% | +0.198 |
| 90% | 393 | 89.8 | 0.73 | 2.9 | 47.9% | +0.344 |

Catches roughly double from 20% to 40% while the attempt ratio and the ecosystem barely move; between 40% and 60% the ecosystem
collapses (episode length drops by 200 steps, prey fall from 21 to 5) and the attempt ratio jumps to near parity. **The response looks
like a fairly sharp threshold in the 40-60% band, not a smooth gradient** -- consistent with a state where a modest productivity edge is
absorbable by the ecosystem, but a large one triggers over-hunting and collapse. Only one seed per point.

#### D. Replicating the single-factor ablations (seeds 43, 44)

| | ABL_SUCCESS_ONLY (90%/10%) | | | ABL_DEATH_ONLY (20%/5%) | | |
|---|---|---|---|---|---|---|
| Seed | 42 | 43 | 44 | 42 | 43 | 44 |
| Women/men attempts | 0.73 | 0.92 | 0.80 | 0.62 | 0.78 | 0.80 |
| Prey alive (iters 291-300) | 2.9 | 0.4 | 0.5 | 19.0 | 22.8 | 22.8 |
| Episode length | 393 | 234 | 299 | 1001 | 1001 | 1001 |
| Men: prey share of energy | 33.4% | 34.0% | 33.4-33.5% | 44.9% | 52.4-52.7% | 50.7-50.8% |
| Women: prey share of energy | 47.9% | 40.1% | 41.0% | 15.3% | 14.7-14.9% | 15.2-15.4% |

**Both ablations replicate on three seeds.** ABL_SUCCESS_ONLY removes or reverses the split every time (prey collapse in all three:
0.4-2.9 alive), confirming it as the more influential single factor. ABL_DEATH_ONLY leaves the split intact and the ecosystem healthy in
all three, with a small, now-replicated shift toward more female hunting relative to the 20%/10% baseline (women's prey share 14.7-15.4%
against 11.1-12.1%; women/men attempts 0.62-0.80 against 0.40-0.53) -- consistent with Iteration 8's finding that the death-chance
contrast matters, just less than the success-rate contrast.

#### Updated statement

Under energy-proportional forage reward with k = 0.5, the male-prey / female-fruit differentiation replicates across three seeds, with
or without the explicit combat-death penalty, and survives a same-observation (shared-state) comparison that removes the
self-created-states confound. It responds to women's hunting success as a threshold around 40-60%, not a smooth gradient, and both
single-factor ablations (success, death chance) now have three-seed support: success is the more influential factor, and a lower death
chance alone gives a small, replicated, additional shift toward hunting. Not yet done: intermediate points on the death-chance axis;
seeds beyond three for any configuration; longer training; and a matched-ecology design that does not confound the ablation with prey
collapse (Codex's suggestion, still open).

**Caveats:** one seed at each response-surface point (40%, 60%); the ABL_EQUAL_ODDS/ABL_SUCCESS_ONLY regime remains prey-collapsed, so
those numbers describe short, prey-poor lives; the shared-state bank comes from uniform-random-policy episodes of one env configuration
and its intervals cover states within one checkpoint, not seeds.

### Iteration 10 — Death-chance response surface (5/10/20/30%)

**Purpose:** Iteration 9's response surface characterized women's *success* rate (20/40/60/90%, death held at 10%) and found a
sharp transition around 40-60%. This does the same for the *death-chance* axis, holding success at women's own 20%: 5% (=
ABL_DEATH_ONLY, three seeds, from Iteration 9), 10% (baseline, three seeds), and two new points, 20% and 30% (two seeds each,
seed 42 and 43), k=0.5, penalty 0.2, minibatch 1024, 300 iterations.

| Women's death chance | Episode length | Prey alive | Women/men attempts | Women: prey share of energy | Female prey approach (cells) |
|---|---|---|---|---|---|
| 5% (3 seeds) | 1001 | 19.0-22.8 | 0.62-0.80 | 14.7-15.4% | (not re-measured here; Iteration 9D) |
| 10% baseline (3 seeds) | 926-1001 | 18-21 | 0.40-0.53 | 11.1-12.1% | +0.017 to +0.039 |
| **20%** (2 seeds) | 901-975 | 27.5-30.2 | 0.33-0.37 | 11.1-12.4% | +0.003 to +0.008 |
| **30%** (2 seeds) | 757-818 | 24.9-31.7 | 0.23-0.29 | 9.2-9.4% | **-0.018 to -0.021** |

**Findings**
1. **The attempt ratio declines broadly across the range, though not every metric moves cleanly at every step.** Women/men
   attempts: 0.62-0.80 (5%) to 0.40-0.53 (10%) to 0.33-0.37 (20%) to 0.23-0.29 (30%) -- a consistent downward trend across all
   four points. Women's prey-energy share is lower at 30% (9.2-9.4%) than at 5% (14.7-15.4%), but the 10% and 20% points
   overlap almost completely (11.1-12.1% against 11.1-12.4%) -- a plateau in that particular metric, not a clean step down. So
   "broadly monotonic in attempt ratio, with a clearly lower endpoint at 30% in prey share" is the accurate summary, not a
   uniform monotonic decline in every measure.
2. **At 30% death chance, female prey approach turns negative in both runs tested** (-0.018, -0.021 cells; every lower
   death-chance point measured is positive or near zero). This is evidence of avoidance in both tested runs, not yet a
   seed-replicated effect with its own uncertainty estimate (no cross-seed interval is computed here, and the effect itself is
   small).
3. **Prey do not collapse anywhere on this axis** (18.6-31.7 alive throughout), unlike the success-rate axis above 40-60%, so
   this axis alone does not produce the ecological-collapse confound that motivated Iteration 11. Episode length does shorten
   somewhat at 30% (757-818, against about 1001 at 5-10%), so "prey do not collapse" is the precise claim; general ecosystem
   health (episode length, and by extension predator demography) is not perfectly flat across the range.

**Caveats:** two seeds at 20% and 30% (against three at 5% and 10%), and no cross-seed uncertainty interval for the female
prey-approach numbers at any point; death chance and success jointly determine the failure outcome's probability (100% -
success - death) in this 3-outcome draw, so raising death chance is really changing the death-versus-harmless-failure split
of the outcome, with success held fixed -- not a pure, fully isolated change in risk alone.

### Iteration 11 — Fixed prey-density experiment: does the split need the collapse?

*Wording tightened after a third Codex review, which caught a real factual error (see the end of this section).*

**Purpose:** every hunting-success ablation so far (Iteration 8's ABL_SUCCESS_ONLY/ABL_EQUAL_ODDS, Iteration 9's response
surface above 40-60%) removed the k=0.5 division of labor, but also collapsed the prey population (0.4-2.9 alive). Codex's
second opinion flagged this directly: it is unclear whether raising success removes the split, or whether the collapsing
ecology does. This experiment removes that confound by holding prey density near a floor.

**Mechanism** (`fixed_prey_density_env.py`, new): `FixedPreyDensityEnv(PredPreyGrass)` overrides `step()` so that, after every
step's normal processing (hunting, reproduction, deaths), if `current_num_prey` is below `prey_density_floor` (20), it spawns
enough replacement prey (default energy, at an empty cell, using the same spawn helper and `_next_prey_idx` ID pool as normal
prey reproduction) to bring it back to the floor. Not a birth: no reward, `episode_births` untouched. Deliberately narrower
than the abandoned prey/grass SUPPLY boost from Iteration 7's "matched-ecology attempt": that boosted the STARTING supply,
which gave predators an energy windfall and triggered a predator population boom that recrashed the boosted pool just as fast
or faster (two calibration runs, 3x and 5x boost, both collapsed to 0 prey within 10 iterations). Replacing losses one at a
time, instead of front-loading supply, avoids that windfall.

Trained via a new script, `tune_ppo_fixed_prey_density.py` (imports `EpisodeReturn`/`policy_mapping_fn` unchanged from the main
tune script; does not modify it). 12 tests in `tests/test_fixed_prey_density_env.py` cover replenishment, the floor as a
minimum not a cap, max-steps/predator-extinction pass-through, ID-pool exhaustion, and (added after a Codex review caught a
real bug) that a full extinction-then-replenish transition flips `terminations["__all__"]` back to False without
un-terminating the prey that actually died, and that replenishment refreshes every live agent's returned observation (the
base class generates final observations *before* this override runs, so without a refresh every other agent's observation
would be stale). All 57 tests (module total) pass; the observation-refresh test was verified to fail without the fix.

**A design limitation found in the process:** a smoke test (20 iterations, success-only odds) confirmed the mechanism works --
prey held at 20-24, full 1000-step episodes, predator populations growing gradually (14 to 27 males) rather than booming.
But over a full 300-iteration run, predator populations kept growing (43-46 males + 15-44 females by the end) and, at 90%
success, generate enormous numbers of successful catches per episode (up to about 1,900 for SUCCESS_ONLY_MATCHED, more for
EQUAL_ODDS_MATCHED where women hunt too). Every replenishment permanently consumes one ID from `n_possible_prey` (2000, a
per-episode budget shared with normal births), so under heavy enough hunting pressure the pool can be exhausted *within* a
single long episode, after which replenishment silently stops and prey collapse resumes:

| Run | Iters 1-50 (prey) | Iters 91-100 (prey) | Iters 191-300 (prey) |
|---|---|---|---|
| SUCCESS_ONLY_MATCHED (90%/10%) | ~20 (clean) | 20.1 (clean) | drops to 6.3 by iter 300 |
| EQUAL_ODDS_MATCHED (90%/5%) | 20.5-20.0 through iter 50 | 4.6 (already degraded) | 0.0 from iter ~190 |

EQUAL_ODDS_MATCHED broke down earlier (by iteration ~70) because women there also hunt productively (5% death, same as
men), roughly doubling the combined catch rate. **Not fixed here** (would need e.g. a larger pool, batched replenishment, or
capping predator reproduction); recorded as a real, open limitation of the mechanism, not of the underlying research question.

**The behavioral result, from the env-agnostic shared-state comparison** (unaffected by the pool exhaustion, since it never
steps the environment -- see Iteration 9B), on both the last genuinely clean checkpoint and all three requested checkpoints
(9/19/29 = iterations 100/200/300, seed 42 only):

| Checkpoint | Ecology at that point | Prey approach, men - women |
|---|---|---|
| SUCCESS_ONLY_MATCHED ckpt9 (iter 100) | clean | -0.001 [-0.013,+0.011] |
| SUCCESS_ONLY_MATCHED ckpt19 (iter 200) | clean | -0.016 [-0.029,-0.002] |
| SUCCESS_ONLY_MATCHED ckpt29 (iter 300) | degraded | -0.007 [-0.027,+0.012] |
| EQUAL_ODDS_MATCHED ckpt4 (iter 50) | clean | **-0.056 [-0.071,-0.042]** |
| EQUAL_ODDS_MATCHED ckpt9 (iter 100) | degraded | **-0.067 [-0.082,-0.054]** |
| EQUAL_ODDS_MATCHED ckpt19 (iter 200) | degraded | **-0.093 [-0.109,-0.075]** |
| EQUAL_ODDS_MATCHED ckpt29 (iter 300) | degraded | **-0.102 [-0.121,-0.081]** |

Compare against the same measure elsewhere: k=0.5 baseline +0.146 to +0.221 (the split); REF (flat rewards) +0.020 (no split);
the *unmatched* ABL_SUCCESS_ONLY +0.038 (a residual male lean survives); the *unmatched* ABL_EQUAL_ODDS -0.037/+0.095
(inconsistent between seeds).

**Findings**
1. **SUCCESS_ONLY_MATCHED: the large positive male lean disappears; two of the three checkpoints are consistent with zero, and
   the third (iteration 200) shows a small female lean whose interval excludes zero** (-0.016 [-0.029,-0.002]). None shows the
   +0.15 to +0.22 male lean of the unmatched k=0.5 baseline, and none matches the unmatched ABL_SUCCESS_ONLY's residual +0.038
   male lean either. So raising success alone, without an ecological collapse, at minimum removes the male-favoring split; a
   small female-favoring one at iteration 200 keeps this from being a clean "exactly zero" result.
2. **EQUAL_ODDS_MATCHED: the split does not just disappear, it reverses on every checkpoint tested, and the magnitude grows
   with more training** (-0.056 at iteration 50 to -0.102 at iteration 300). Women show more prey-directed behavior than men
   once the odds are literally identical, at every point measured.
3. **The behavioral read is not proven robust to the pool-exhaustion bug, and should not be described that way.** Within each
   run, the clean and degraded checkpoints agree in *sign* (SUCCESS_ONLY_MATCHED stays near zero; EQUAL_ODDS_MATCHED stays
   negative), but that is a weak test: the "clean" and "degraded" checkpoints also differ in training maturity (SUCCESS_ONLY
   compares iterations 100/200, clean, against 300, degraded; EQUAL_ODDS compares iteration 50, clean, against 100/200/300,
   all degraded), so any trend could reflect ordinary continued learning, the ecology problem, or both, with no same-iteration
   clean counterfactual to separate them. The EQUAL_ODDS_MATCHED effect nearly doubles in magnitude across checkpoints
   (-0.056 to -0.102), so even if the sign never flips, the *size* of the effect is plausibly still affected by whichever of
   these is responsible. The fair statement is that the sign does not visibly contradict the clean checkpoint -- not that the
   result is "robust to" the bug.
4. **This is more directly supportive of necessity than the unmatched ablations, but it does not settle the question.** In
   the unmatched ablations it was unclear whether the odds change or the ecological collapse removed the split; here, the
   clean checkpoints (iterations 100/200 for success-only, iteration 50 for equal-odds) show the same qualitative pattern
   without a collapse, which weakens the "it was just the collapse" hypothesis. But: each condition has only one training
   seed; the two conditions' clean checkpoints come from different training maturities; there is no run in
   `FixedPreyDensityEnv` at the *original* 20%/10% odds to confirm the split still appears at all under this environment
   before concluding that changing the odds removes it; and the reported intervals quantify variation across states in one
   fixed bank, not across training seeds. "Fixed prey-density" is also a more accurate name for the intervention than
   "matched ecology": it holds the prey count near a floor, but predator abundance, total catch throughput (up to about
   1,900 catches per episode), and the energy injected by each replacement (a full initial prey's worth) are not held fixed
   at all, and are known to differ a great deal between these two runs and from the earlier ablations.

**Bugs found and fixed along the way (unrelated to the environment):** `analyze_shared_states.py`,
`analyze_prey_approach_from_checkpoint.py`, and `analyze_energy_sources.py` all hardcoded the Tune trial-directory glob to
`PPO_PredPreyGrass_*/`, matching only the original tune script's registered env name. `tune_ppo_fixed_prey_density.py`
registers a differently-named env (`FixedPreyDensityPredPreyGrass`), so this glob matched nothing and raised `IndexError`.
Fixed by broadening to `PPO_*/` in all three (verified still uniquely matches the single trial directory in every run,
including the pre-existing ones).

**Caveats:** single seed (42) for both matched runs -- no cross-seed replication yet, unlike Iteration 9's three-seed
ablations; the two runs are not perfectly comparable to each other (their clean windows end at different iterations, 200 vs
50, due to the pool-exhaustion timing); no original-odds (20%/10%) control has been trained in `FixedPreyDensityEnv`, so the
floor mechanism's own effect on behavior (independent of the odds change) is not isolated; the mechanism's late-training
degradation is a design-breaking limitation (it silently stops enforcing the floor once the ID pool is exhausted, rather
than merely being inconvenient) that must be fixed (a larger pool, batched replenishment, or capped reproduction) before
this becomes the module's standard ablation method; `prey_density_floor` is validated only for being non-negative, not for
being achievable given the ID pool or grid capacity; only k=0.5 has been tested this way.

### Third opinion (Codex), 2026-09-22, and what changed

A follow-up review of Iterations 10-11 (read-only). It found one factual error and several overclaims, all corrected above:

1. **Factual error (high confidence):** the original wording called all three SUCCESS_ONLY_MATCHED checkpoints "statistically
   indistinguishable from zero," but the iteration-200 interval, as reported, excludes zero. Fixed to state the actual
   per-checkpoint result (finding 1 above).
2. **"Settles the concern" / "robust to the bug" were too strong** given a single seed per condition, unequal checkpoint
   maturity between the two conditions' clean windows, and no original-odds control run in the new environment. Reworded to
   "more directly supportive... does not settle" and to state plainly what the clean-vs-degraded comparison can and cannot
   show (findings 3-4 above).
3. **"Matched ecology" overstates what a prey-density floor holds fixed** (predator abundance, catch throughput, and the
   energy injected by each replacement all still vary); renamed to "fixed prey-density" throughout this section.
4. Iteration 10's wording was also softened: "monotonic" overstated the 10%/20% prey-share overlap (11.1-12.1% vs
   11.1-12.4%, not distinguishable); the attempt-ratio summary omitted the 5% point; "risk alone" is qualified as changing the
   death-versus-harmless-failure share of a 3-outcome draw, not a pure single-variable change; "the ecosystem stays healthy"
   is narrowed to "prey do not collapse" (episode length does shorten at 30%, 757-818 against about 1001 at lower death
   chances); and "clear two-seed-replicated evidence" of avoidance is now "evidence of avoidance in both tested runs" (a
   small effect, no cross-seed uncertainty estimate).
5. **Most valuable next experiment, per this review:** fix the pool exhaustion first, then run a full 2x2 odds factorial
   inside `FixedPreyDensityEnv` (20%/10%, 20%/5%, 90%/10%, 90%/5%), at least three seeds each, evaluated at the same
   predetermined checkpoints for every cell (including one verified-clean early checkpoint and the final one) -- this
   directly closes the missing-control and unequal-maturity gaps at once, rather than incrementally adding seeds to the
   currently-degrading implementation, which would just replicate a mixture of treatment effect and pool failure.

### Iteration 12 — The missing control: a full 2x2 odds factorial inside the fixed-density environment

**Purpose:** Iteration 11 ended with three gaps its own findings could not close: no original-odds (20%/10%) control
trained inside `FixedPreyDensityEnv`; only one seed per condition; and the pool-exhaustion bug meant the two conditions
tested only had a "clean" window at different, uncomparable training maturities. The third Codex review's top
recommendation (above) was to fix the exhaustion bug, then run the full 2x2 factorial at several seeds each, evaluated
at the same checkpoint for every cell. This iteration does that.

**Setup:** `n_possible_prey` raised 25x (2,000 -> 50,000) to fix the exhaustion bug, validated first via a dedicated
calibration run at the worst-case odds (90%/5%) -- confirmed clean for the full 300 iterations before committing
further compute (`final_num_prey` held at a mean of 20.0 over the last 10 logged iterations, versus the collapse the
old 2,000-ID pool showed by iteration 300 in Iteration 11). Four conditions, three seeds each (42/43/44), except
EQUALODDS seed 42, which reuses the calibration run itself rather than re-training it:

- **CONTROL** (20% success / 10% death) -- the missing original-odds control.
- **DEATHONLY** (20% success / 5% death) -- death risk alone equalized to the male's.
- **SUCCESSONLY** (90% success / 10% death) -- success rate alone equalized to the male's.
- **EQUALODDS** (90% success / 5% death) -- both equalized.

All twelve runs: k=0.5 energy-proportional reward, 0.2 combat-death penalty, minibatch 1024, 300 iterations,
`prey_density_floor`=20. Evaluated at checkpoint 29 (final) via two complementary measures -- not statistically
independent, since both evaluate the same trained policies, but different in what they capture: (a) the shared-state
policy comparison (both sex policies queried on an identical, fixed, random-policy-generated bank of 14,876 states,
unaffected by which states each policy visits on its own), and (b) completed-life energy-source accounting (where each
individual's own foraging energy actually came from, over 20 rollout episodes per run).

**First: did the pool-exhaustion fix generalize?** Yes, cleanly, in all twelve runs. `final_num_prey` (mean of the last
10 logged iterations) held at or near the floor in every single run -- 22.4-23.7 for CONTROL/DEATHONLY, 20.0-20.1 for
SUCCESSONLY/EQUALODDS -- and every run completed full 1001-step episodes throughout. No collapse and no sign of
exhaustion in any condition, not only the equal-odds worst case the calibration run tested alone.

**But predator population size is not held fixed, and it varies enormously across conditions** -- direct confirmation
of a caveat Iteration 11 could only state in principle. Total predators alive at the end (male+female, mean of last 10
iterations): CONTROL ~25-27, DEATHONLY ~27-29, SUCCESSONLY ~51-56, EQUALODDS ~82-92. Raising female hunting success
from 20% to 90% roughly doubles the predator population; equalizing death chance too very nearly doubles it again. The
floor keeps prey numbers stable, but it does so by feeding a steadily larger predator population as hunting becomes
more effective for both sexes (more successful hunting by anyone puts more energy into the shared pool both sexes'
reproduction draws on) -- SUCCESSONLY and EQUALODDS are not "the same ecology, different odds," they are objectively
more crowded ecologies. This means the experiment below establishes what happens to behavior when female hunting
success changes *and* the resulting endogenous ecology changes with it -- not that the odds asymmetry is necessary
under otherwise-matched ecological conditions, and not whether the behavioral change is a direct effect of the odds
themselves versus an indirect effect of the crowding, turnover, and competition that the odds change also causes.

**Shared-state comparison (approach-bias difference, male minus female; 95% bootstrap CI in brackets):**

| Condition | seed | prey approach diff(M-F) | fruit approach diff(M-F) |
|---|---|---|---|
| CONTROL | 42 | +0.150 [+0.135,+0.169] | -0.136 [-0.153,-0.120] |
| CONTROL | 43 | +0.144 [+0.128,+0.158] | -0.146 [-0.157,-0.135] |
| CONTROL | 44 | +0.221 [+0.208,+0.233] | -0.142 [-0.151,-0.133] |
| DEATHONLY | 42 | +0.115 [+0.103,+0.128] | -0.050 [-0.071,-0.024] |
| DEATHONLY | 43 | +0.124 [+0.103,+0.142] | -0.075 [-0.091,-0.062] |
| DEATHONLY | 44 | +0.115 [+0.100,+0.132] | -0.044 [-0.062,-0.029] |
| SUCCESSONLY | 42 | -0.005 [-0.022,+0.014] | +0.114 [+0.101,+0.128] |
| SUCCESSONLY | 43 | -0.017 [-0.032,+0.001] | +0.094 [+0.082,+0.106] |
| SUCCESSONLY | 44 | -0.016 [-0.034,+0.001] | +0.074 [+0.059,+0.087] |
| EQUALODDS | 42 | +0.113 [+0.091,+0.132] | -0.036 [-0.050,-0.021] |
| EQUALODDS | 43 | +0.008 [-0.005,+0.020] | +0.024 [+0.012,+0.036] |
| EQUALODDS | 44 | -0.035 [-0.049,-0.021] | +0.063 [+0.051,+0.074] |

(Adjacent step-onto-prey/fruit probabilities show the same qualitative pattern per condition; full detail in
`shared_states_factorial.log`, omitted here for space.)

**Energy-source comparison (prey share of own gross foraging energy, completed lives only):**

| Condition | seed | male prey share | female prey share |
|---|---|---|---|
| CONTROL | 42 / 43 / 44 | 52.6% / 52.1% / 51.9% | 11.4% / 11.4% / 13.0% |
| DEATHONLY | 42 / 43 / 44 | 51.0% / 52.7% / 50.4% | 13.7% / 14.9% / 14.8% |
| SUCCESSONLY | 42 / 43 / 44 | 27.8% / 29.9% / 28.0% | 40.5% / 42.2% / 40.7% |
| EQUALODDS | 43 / 44 (42 missing, see caveats) | 30.1% / 30.8% | 33.9% / 34.9% |

**What this shows, condition by condition:**

1. **CONTROL replicates the original division of labor cleanly, on both measures, all three seeds, in the stabilized
   environment.** This closes Iteration 11's biggest gap: the k=0.5 finding was never an artifact of measuring only in
   an environment where prey happened not to collapse -- it reproduces the same way when the ecology is actively held
   stable too.
2. **DEATHONLY: the split survives, somewhat attenuated, in these three seeds.** The prey approach-bias diff drops
   modestly from CONTROL's +0.144/+0.150/+0.221 to +0.115/+0.115/+0.124; the fruit approach-bias diff shrinks more
   noticeably, to roughly a third to half of CONTROL's magnitude. Female prey-energy share ticks up slightly (11-13%
   to 14-15%) but stays far below the male's ~51%. These are point-estimate comparisons, not a tested
   CONTROL-vs-DEATHONLY contrast (the reported CIs cover variation within one fixed state bank, not across the
   CONTROL/DEATHONLY conditions or across further seeds). The narrowest honest claim: reducing female death chance
   from 10% to 5%, success held at 20%, did not eliminate the split in these three seeds -- it does not establish that
   death risk in general is not a driver, only that this specific contrast is not sufficient on its own.
3. **SUCCESSONLY: the split neutralizes on the prey-approach dimension and reverses on the fruit dimension and on the
   realized energy share, consistently across all three seeds.** The prey approach-bias diff is close to zero and
   **all three CIs include zero** (-0.005 [-0.022,+0.014], -0.017 [-0.032,+0.001], -0.016 [-0.034,+0.001]); the fruit
   approach-bias diff flips sign, +0.074 to +0.114 (all three CIs exclude zero -- males now approach fruit *more* than
   females); and, descriptively, **female prey-energy share (40.5-42.2%) now exceeds the male's (27.8-29.9%) in all
   three seeds**, though no confidence interval is reported for this male-female energy-share difference (the
   approach-bias CIs above do not cover it). This is a more specific and more interesting pattern than "the split
   disappears" -- in the reported point estimates, it flips.
4. **EQUALODDS is the least clean cell, and is not presented as a clean result.** The shared-state prey-approach diff
   has no consistent sign across seeds (+0.113, +0.008, -0.035), nor does the fruit-approach diff (-0.036, +0.024,
   +0.063); seeds 42 and 44 in particular have CIs that individually exclude zero *in opposite directions*, which
   looks like genuine instability between training runs rather than sampling noise within one checkpoint's state bank
   (that bank is large, ~9,000-15,000 states, regardless of condition). The energy-source measure is more consistent
   in direction (female share 33.9-34.9% exceeds male's 30.1-30.8% in both available seeds) but on much smaller
   samples (49-106 completed lives per seed, against 500-900+ for CONTROL/DEATHONLY) -- there, shorter individual
   lifespans plausibly do explain noisier estimates. EQUALODDS's predator population is also 3-4x CONTROL's, so this
   condition's ecology is furthest from "same as CONTROL, only odds differ."

**Caveats:**
- Predator population size is not held fixed and grows substantially with hunting success (see above): these four
  conditions are not clean single-variable-changed comparisons of each other. SUCCESSONLY and especially EQUALODDS
  occur in more crowded ecologies than CONTROL/DEATHONLY, which is itself a plausible contributor to their noisier or
  reversed results, not only the odds change in isolation.
- EQUALODDS seed 42 (the reused calibration run) has no completed-life energy-source analysis -- it was not queued by
  the automatic pipeline (only the eleven newly trained runs were). A pipeline gap, not a missing result; can be
  backfilled by running `analyze_energy_sources.py` against that checkpoint directly if needed.
- Sample sizes for SUCCESSONLY/EQUALODDS completed lives are much smaller than for CONTROL/DEATHONLY (as low as 49
  lives for one EQUALODDS seed), reflecting shorter individual lifespans in the more crowded, higher-turnover
  ecologies -- per-seed point estimates for these two conditions are noisier.
- Single checkpoint (29, final) evaluated for all conditions in this iteration; no within-run training-trajectory
  comparison across conditions (available per-run at checkpoints 9/19/29 in `rollout_*.log` for anyone who wants it).
- Bootstrap CIs quantify variation across the roughly 9,000-15,000 states in one fixed bank (or the roughly 50-900
  completed lives in one rollout batch), not across training seeds; the "all three seeds agree in sign" pattern for
  CONTROL/DEATHONLY/SUCCESSONLY is the actual cross-seed evidence, not the individual CIs.

## Summary: is there a sex differentiation in foraging ("division of labor")?

**Working definition:** a difference between the sexes, emerging from training rather than hard-coded, in (a) what they do (approach toward
prey and fruit) and (b) where their gross own-forage energy comes from (prey share), with a viable population. It does not imply coordination.

**Evidence (Iteration 7, energy-proportional reward k = 0.5, seeds 42 / 43 / 44, iteration 300):** men step onto adjacent prey x1.44 / x1.49 /
x1.32 as often as a random mover (women x0.99 / x1.02 / x1.07; REF x0.85); men take 46.9 / 47.8 / 49.6% of their gross own-forage energy from prey
(random walker 41.3%; flat-reward runs 37.4-39.9%) and women 11.1 / 12.1 / 11.6% (random 14.8%; flat-reward runs 13.2-14.5%); women approach
fruit x1.91 / x1.86 / x1.83; the ecosystem is healthy in all three.

**Confirmed on a same-observation (shared-state) comparison and without the explicit death penalty (Iteration 9):** querying the male and
female policy on identical stored observations (removing the confound that rollouts measure a policy partly on states it creates for itself)
gives a prey approach-bias difference of +0.15 to +0.22 for every k = 0.5 run, with or without the -0.2 penalty, against about +0.02 for REF
(flat rewards). Dropping the penalty entirely (two seeds) leaves the split unchanged or slightly stronger and does not change deaths-in-combat
per episode, so the explicit penalty appears unnecessary for the split at this setting (it does not show death risk itself is irrelevant).

**Necessity, tested with a prey-density floor instead of the ecological collapse (Iteration 11, single seed) and then a full factorial
(Iteration 12, three seeds per cell):** a `FixedPreyDensityEnv` holds prey density near a floor by replacement spawns, so a high-success run
does not have to collapse the prey population. Iteration 11's single-seed runs first suggested that raising women's success alone removes the
split and that fully equalizing odds reverses it, but left the original-odds control untrained in this environment and could not separate
treatment effect from a pool-exhaustion bug. Iteration 12 fixed the bug (25x larger ID pool, validated by a calibration run) and ran all four
odds combinations at three seeds each: **CONTROL (20%/10%) replicates the original male-prey/female-fruit split cleanly on every seed** in the
stabilized environment, closing the missing-control gap. **DEATHONLY (20%/5%) leaves the split intact, somewhat attenuated, in these three
seeds** -- reducing death risk alone, over this specific 10%-to-5% contrast, was not sufficient to remove it (not a general claim that death
risk is irrelevant). **SUCCESSONLY (90%/10%) neutralizes the split on prey-approach (all three CIs include zero) and reverses it, in the
reported point estimates, on fruit-approach and on realized prey-energy share, in every seed** (women's prey-energy share, 40-42%, exceeds
men's, 28-30%, though no CI is reported for that difference). **EQUALODDS (90%/5%) is the least clean cell**: no consistent sign across seeds
on the shared-state measure -- two seeds even show CIs excluding zero in opposite directions, more consistent with genuine instability between
training runs than sampling noise -- though the energy-source measure leans the same reversed direction as SUCCESSONLY in both available
seeds. A confound that limits how any of this can be read as "just the odds": predator population size is not held fixed by the floor and
roughly triples from CONTROL (~26 total predators) to EQUALODDS (~85). This experiment establishes what happens to behavior when female hunting
success changes *and* the resulting endogenous ecology changes with it; it does not establish that the odds asymmetry is necessary under
otherwise-matched ecological conditions, nor separate a direct effect of the odds from an indirect effect of the crowding and competition the
odds change also causes. See Iteration 12 for the full tables.

**The death-chance axis, extended (Iteration 10):** at fixed 20% success, the women/men attempt ratio declines broadly from 5% to 30% death
chance (0.62-0.80 to 0.40-0.53 to 0.33-0.37 to 0.23-0.29); women's prey-energy share is lower at 30% (9.2-9.4%) than at 5% (14.7-15.4%), though
the 10% and 20% points overlap almost completely, so it is not a clean step down at every point. At 30% death chance, women's prey approach
turns negative in both runs tested (-0.018, -0.021) -- evidence of avoidance, not yet a seed-replicated effect with its own uncertainty
estimate. Prey do not collapse anywhere on this axis, unlike the success-rate axis; episode length does shorten somewhat at 30%.

**Reward design.** The same unequal odds gave no split under flat per-event rewards or at k = 0.2 (women starved). Flat rewards strongly reward
repeated consumption of depleted fruit (about 60-80 reward per life against about 9 for prey), a plausible reason for the missing male prey
preference; not isolated, because k = 0.2 and k = 0.5 differ only in reward scale.

**Limits:** partial, not exclusive (men still get about half their energy from fruit; a woman catches about 0.8-1.0 prey per life against about 10 for a
man) and no coordination is shown; most response-surface points (40%/60% success, and the original Iteration 11 single-seed matched-ecology runs,
now superseded by Iteration 12's three-seed factorial) have only one seed; minibatch 128 and 1024 runs are not like-for-like (a same-rewards
minibatch-128 run, FORAGING_PENALTY02, shows about half the shared-state effect size of the minibatch-1024 runs); only k = 0.5 is replicated at
three seeds and it was chosen after k = 0.2 failed; rollout metrics (outside the shared-state check) mix action preference with the states each
policy creates; energy figures are gross, aggregated over lives that include lives cut off at the end of the episode; the *unmatched* equal-odds/
success-only ablations (Iteration 8-9) still run in a collapsed, prey-poor regime -- Iteration 12's fixed-density factorial is the three-seed
alternative, but its own conditions differ substantially in predator population size from each other (see Iteration 12), so it trades the
collapse confound for a population-size confound rather than eliminating confounds altogether.

**Current best statement:** under energy-proportional forage reward with k = 0.5, the male-prey / female-fruit differentiation replicates across
three training seeds, with or without the explicit combat-death penalty, and survives a same-observation comparison that removes the
self-created-states confound. Men obtain roughly 47-50% of their gross own-forage energy from prey against roughly 11-13% for women. Flat-reward
runs do not show the same pattern and strongly reward repeated consumption of depleted fruit. A three-seed-per-cell factorial inside a
prey-density-floor environment (Iteration 12) now shows this is not an artifact of a collapsing ecology: the original odds (20%/10%)
reproduce the split cleanly under a stable prey population; equalizing death risk alone (10% to 5%, success held at 20%) leaves it intact,
attenuated, in these three seeds; equalizing success rate alone (20% to 90%, death held at 10%) neutralizes it on approach behavior (all
three CIs include zero) and reverses it, in the reported point estimates, on realized prey-energy share (women now derive 40-42% of their
foraging energy from prey against men's 28-30%); fully equalizing both odds gives the least consistent result across seeds, with two of
three seeds' CIs excluding zero in opposite directions. Changing female hunting success is associated with the largest change in this
factorial, but that same intervention also roughly doubles predator abundance (and equalizing both odds nearly triples it again), so its
direct effect cannot be cleanly separated here from the induced ecological change -- this is the most consistent descriptive result so far
on the necessity question, not a settled one. The death-chance axis alone, pushed to 30%, produces a smaller but real avoidance signal
without any ecological collapse (Iteration 10, two seeds).

### Second opinion (Codex), 2026-09-21, and what changed

An independent review of the conclusions and of `analyze_prey_approach_from_checkpoint.py` / `analyze_energy_sources.py` (read-only; findings
adopted here). Overall: the descriptive k = 0.5 result is reasonably strong; the causal wording was too strong.

Points accepted and applied to the wording above: (1) the fruit-farming explanation is plausible but not isolated (k = 0.2 vs 0.5 differ in reward scale);
(2) three seeds are few and the reported intervals cover episodes of one checkpoint, not training seeds; (3) "division of labor" suggests coordination,
so "partial sex differentiation" is used for the claim; (4) equal odds change the ecological regime (prey extinction), so necessity is not shown under
matched conditions; (5) "productivity-driven, not risk-driven" was too categorical (one seed, one 5-point death contrast, and the ablations are not
perfectly single-factor because success, death and failure share one draw); (6) treating the minibatch-128 versus 1024 differences as "run-to-run noise"
was wrong, since part is a systematic setting effect; (7) rollouts measure the policy on states it creates itself, and "step onto" is the probability of an
action whose nominal destination is the target's cell; (8) prey share is a ratio of means over lives that include censored lives, gross and aggregated;
(9) the explicit death penalty remains a confound.

A real bug found: the `censored` flag in `analyze_energy_sources.py` was wrong for lives still alive at the step cap (all agents are truncated at the
cap and were filtered out before the flag was computed). It is never used in the reported means, so no reported number changes; fixed.

Codex's suggested experiments, adopted as next steps: shared-state policy comparison (no training), energy per 100 agent-steps and completed-life
results, intermediate success rates for both sexes, replication of the ablations, and a reward-scale control. I judged its 8-10 seeds per cell too
expensive (days of compute at about an hour per run) and would use 2-3 seeds per key cell.

---

## Next steps

1. ~~Fix the replenishment pool exhaustion in `FixedPreyDensityEnv`, then run the full odds factorial inside it~~ **Done
   (Iteration 12):** `n_possible_prey` raised 25x, validated by a calibration run, then all four cells (20%/10%, 20%/5%,
   90%/10%, 90%/5%) trained at three seeds each, checkpoint 29. See Iteration 12 for results. Two threads it opened,
   still unresolved:
   - **The population-boom confound.** Predator population size is not held fixed by the prey-density floor and
     roughly triples from CONTROL to EQUALODDS, so the higher-success cells are measured in more crowded ecologies, not
     just different-odds versions of the same one. A cleaner test would also cap or otherwise control predator
     population size (or population density per unit of prey/fruit) across conditions -- candidate fixes are the same
     ones already listed for the ID-pool problem (a reproduction cap being the most direct).
   - **Backfill EQUALODDS seed 42's energy-source analysis** (missing because it reused the calibration run, which the
     automatic pipeline didn't queue for post-hoc analysis) by running `analyze_energy_sources.py` against that
     checkpoint directly.
2. **Death-chance response surface with more seeds** at 20% and 30% (currently two each), and consider intermediate points
   between 10% and 20%, given the trend has turned out to be consistent rather than negligible.
3. **Other k** (0.3, 0.7) and re-tuning the combat-death penalty on top of the energy-proportional reward (now looks droppable, see
   Iteration 9); a reward-scale control (for example flat rewards calibrated to the same expected total forage reward as k = 0.5) to
   separate closing the fruit loophole from raising the reward scale.
4. **More seeds where only one or two exist:** REF, PROP k=0.2, the response-surface points (40%, 60%).
5. **Decide the default minibatch.** Recommended, not yet adopted: `--minibatch-size 1024` (5.8x faster); the defaults stay at
   128 / 30 so all earlier runs remain reproducible. Runs since Iteration 7 pass the flag explicitly.
6. **Fruit-only shaping** (no catch reward) to separate learning to gather from learning to hunt.
7. **Longer training and multiple late checkpoints**, not only iteration 300, to check the effect is stable rather than a snapshot.
8. Unexplained: the early rise in births and episode length in Iteration 1 cannot come from predator learning; the prey
   policy changing behavior is the untested guess.
