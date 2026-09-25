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
10. **Population control, twice.** A reproduction-blocking cap (Iteration 13) and exact density that never blocks reproduction (Iteration 15)
    were tried to remove the population-size confound. Iteration 15 first appeared to show that the cap was the outlier; that was wrong (see 12).
11. **Coordination, first pass.** Female approach behavior is associated with the recorded mate's status (Iteration 14), with no support for
    provisioning as the explanation and no consistent payoff of pair differentiation (Iteration 16). Association only; no signaling channel exists.
12. **A methodological error found and fixed (Iteration 16).** The rollout-based analyses had been run in the plain base environment, not the
    prey-floor / density-target environment the runs were trained in, which distorted the energy-share and coordination numbers of Iterations
    12-15. Re-run in the trained environment: fixing population by either method brings the sexes to about parity in the high-success cells, while
    the uncapped runs keep a female lead, and a one-seed sweep suggests the gap rises with population. Shared-state results were never affected.

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
- **Iterations 12, 13, 15 run families** (all `PPO_FIXED_PREY_DENSITY_<CELL>_...` in `~/simulation_results/ray_results/`, cells CONTROL 20%/10%,
  DEATHONLY 20%/5%, SUCCESSONLY 90%/10%, EQUALODDS 90%/5% for the women's odds; k=0.5, penalty 0.2, minibatch 1024, 300 iterations,
  prey floor 20, 50,000 prey IDs, seeds 42/43/44): `..._<CELL>_SEED<n>` (Iteration 12; EQUALODDS seed 42 is `..._CALIB3_EQUALODDS_SEED42`),
  `..._<CELL>_CAP26_SEED<n>` (Iteration 13, `predator_population_cap=26`), `..._<CELL>_DENS26_SEED<n>` (Iteration 15, `predator_density_target=26`).
- **ABL_EQUAL_ODDS** (Iteration 8): `PPO_PREDATOR_SEXUAL_REPRODUCTION_ABL_EQUAL_ODDS_K05_MB1024_SEED42`, `..._SEED43`: PROP k=0.5 with women
  given the men's hunting odds (success 0.90, death 0.05). **ABL_SUCCESS_ONLY** (`--female-success-prob 0.90 --female-death-prob 0.10`)
  and **ABL_DEATH_ONLY** (`--female-success-prob 0.20 --female-death-prob 0.05`), seed 42: single-factor variants.
- **POSITIVE_CONTROL** (Iteration 0): extreme hunting odds, sparse reward. Not in the chart.

Unless a run family above says otherwise (seeds 43/44 exist for several), runs use seed 42.

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

**Correction (added in Iteration 16): the rollout-based energy-share and coordination figures in this iteration were computed in the plain base environment, not the prey-floor / density-target env the runs were trained in. See Iteration 16 for the corrected numbers; conclusions that depended on them are superseded there. Shared-state (approach-bias) tables are unaffected.**

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

### Iteration 13 — Controlling for population size: does the reversal survive?

**Correction (added in Iteration 16): the rollout-based energy-share and coordination figures in this iteration were computed in the plain base environment, not the prey-floor / density-target env the runs were trained in. See Iteration 16 for the corrected numbers; conclusions that depended on them are superseded there. Shared-state (approach-bias) tables are unaffected.**

**Purpose:** Iteration 12 found that raising female hunting success also roughly triples predator population, so it
was unclear whether the behavioral reversal it reported was a direct effect of the odds or an indirect effect of the
population/crowding change the odds also cause. A new `predator_population_cap` config option (added this iteration;
see the module code and its own tests) blocks predator reproduction once the population is already at or above a
ceiling, mirroring `FixedPreyDensityEnv`'s prey floor in the opposite direction. This iteration re-runs all four odds
conditions, three seeds each, with the cap set to 26 (close to CONTROL's natural level), so the population comparison
across conditions is far closer to matched than in Iteration 12.

A single-seed pilot (CONTROL and EQUALODDS at cap 26, seed 42) was run first and validated against a gate check
before committing to the full 12-run set, since the cap mechanism had never been exercised in a real training loop
before -- this project's own history (the ID-pool exhaustion bug, and the population-boom confound itself) is why
that step wasn't skipped: both were caught only by actually training, not by code review.

**Did the cap work?** Yes, closely, though "matched" overstates it -- "closer to a common ceiling" is the accurate
description. Final total predator population (male + female, mean of the last 10 logged iterations): CONTROL ~21-22,
DEATHONLY ~25-26, SUCCESSONLY ~26.0, EQUALODDS ~26.0 (exactly, all three seeds). CONTROL sits noticeably under the
cap (roughly 15-19% lower) because its natural uncapped population (~25-27) is already close to it and the cap has
little left to constrain; the other three conditions, whose uncapped populations ranged from ~26 to ~92, are now
held tightly at the same ceiling. This is a capped comparison, not a matched one -- CONTROL's lower population is a
real, not merely cosmetic, residual asymmetry -- but the 21-to-26 range here is far narrower than Iteration 12's
roughly 26-to-92 range.

**Shared-state comparison (approach-bias difference, male minus female; capped vs. the Iteration 12 uncapped runs):**

| Condition | seed | prey, capped | prey, uncapped | fruit, capped | fruit, uncapped |
|---|---|---|---|---|---|
| CONTROL | 42 | +0.190 [+.176,+.204] | +0.150 [+.135,+.165] | -0.099 [-.115,-.085] | -0.136 [-.156,-.119] |
| CONTROL | 43 | +0.132 [+.114,+.149] | +0.144 [+.129,+.158] | -0.077 [-.090,-.067] | -0.146 [-.156,-.135] |
| CONTROL | 44 | +0.099 [+.085,+.115] | +0.221 [+.208,+.233] | -0.063 [-.072,-.054] | -0.142 [-.153,-.133] |
| DEATHONLY | 42 | +0.107 [+.091,+.124] | +0.115 [+.101,+.129] | -0.097 [-.108,-.087] | -0.050 [-.071,-.028] |
| DEATHONLY | 43 | +0.104 [+.087,+.123] | +0.124 [+.104,+.144] | -0.159 [-.171,-.150] | -0.075 [-.091,-.060] |
| DEATHONLY | 44 | +0.093 [+.074,+.112] | +0.115 [+.098,+.132] | -0.037 [-.057,-.020] | -0.044 [-.060,-.029] |
| SUCCESSONLY | 42 | +0.044 [+.033,+.054] | -0.005 [-.024,+.013] | -0.004 [-.014,+.006] | +0.114 [+.099,+.126] |
| SUCCESSONLY | 43 | +0.053 [+.038,+.070] | -0.017 [-.033,+.001] | +0.020 [+.008,+.032] | +0.094 [+.083,+.108] |
| SUCCESSONLY | 44 | +0.010 [-.005,+.026] | -0.016 [-.033,+.001] | +0.029 [+.017,+.041] | +0.074 [+.061,+.086] |
| EQUALODDS | 42 | -0.004 [-.017,+.011] | +0.113 [+.094,+.131] | -0.054 [-.065,-.041] | -0.036 [-.050,-.020] |
| EQUALODDS | 43 | +0.027 [+.013,+.043] | +0.008 [-.004,+.019] | -0.055 [-.065,-.046] | +0.024 [+.010,+.039] |
| EQUALODDS | 44 | +0.008 [-.002,+.018] | -0.035 [-.050,-.020] | +0.000 [-.010,+.010] | +0.063 [+.053,+.074] |

**Energy-source comparison (prey share of own gross foraging energy, completed lives; capped vs. uncapped):**

| Condition | seed | male, capped | male, uncapped | female, capped | female, uncapped |
|---|---|---|---|---|---|
| CONTROL | 42 / 43 / 44 | 48.4% / 44.0% / 46.0% | 52.6% / 52.1% / 51.9% | 10.0% / 13.4% / 13.3% | 11.4% / 11.4% / 13.0% |
| DEATHONLY | 42 / 43 / 44 | 54.1% / 52.3% / 50.9% | 51.0% / 52.7% / 50.4% | 11.6% / 12.8% / 13.7% | 13.7% / 14.9% / 14.8% |
| SUCCESSONLY | 42 / 43 / 44 | 38.1% / 40.8% / 41.3% | 27.8% / 29.9% / 28.0% | 39.5% / 39.5% / 37.9% | 40.5% / 42.2% / 40.7% |
| EQUALODDS | 42 / 43 / 44 | 43.4% / 43.1% / 40.5% | 30.1% / 30.8% (2 seeds) | 37.1% / 38.4% / 39.8% | 33.9% / 34.9% (2 seeds) |

**What this shows:**

1. **CONTROL and DEATHONLY keep the same qualitative pattern, though some seed-level effect sizes shift
   considerably.** Every seed still shows a clear male-leaning prey split and a clear female-leaning fruit split
   under the cap -- capping does not flip or erase either condition's pattern, a useful sanity check on the mechanism
   itself. But calling this "essentially unchanged" would overstate it: CONTROL seed 44's prey diff drops from +0.221
   uncapped to +0.099 capped (roughly halved), and DEATHONLY seed 43's fruit diff more than doubles in magnitude
   (-0.075 to -0.159). The *direction* is stable; the *size* moves around seed to seed more than "essentially
   unchanged" suggests.
2. **SUCCESSONLY changes substantially, but the reversal is attenuated rather than fully gone.** Uncapped, each
   seed's prey-approach CI included zero (Iteration 12), and the energy share reversed outright (female 40-42%
   against male 28-30%). Capped, the prey-approach split shrinks a lot from CONTROL's level but stays **positive** on
   two of three seeds (+0.044, +0.053; the third, +0.010, is near zero), and the energy share **moves close to
   parity** rather than reversing (male 38-41%, female 38-40% -- seed-wise differences of only -1.4, +1.3, and +3.4
   percentage points, against CONTROL's roughly 33-42-point gaps). The fruit-approach reversal seen uncapped (+0.074
   to +0.114, all excluding zero) shrinks a great deal but does not fully disappear: capped fruit-approach diffs are
   -0.004, +0.020, +0.029 -- one near zero, but **two of three CIs still exclude zero on the positive (reversed)
   side**. So the clearest, most consistent part of this result is the energy-share and prey-approach convergence
   toward parity; the fruit-approach reversal is weakened, not eliminated.
3. **EQUALODDS's cross-seed inconsistency is much smaller capped than uncapped.** Uncapped, this was Iteration 12's
   least clean cell: prey-approach diffs of +0.113, +0.008, -0.035 -- no consistent sign, two seeds' CIs excluding
   zero in opposite directions. Capped, all three seeds land small (-0.004, +0.027, +0.008), with only one CI
   excluding zero (barely, on the positive side) -- much less scattered than uncapped, though still not a clean
   single sign on every CI. The energy share moves to near-parity in every seed (male 40.5-43.4%, female
   37.1-39.8%), each seed retaining a small (0.7-6.3 percentage-point) male lean rather than a scattered mix of
   directions.

**Interpretation.** The reversal Iteration 12 reported does not reproduce under this cap: at a population closer to a
common ceiling, SUCCESSONLY and EQUALODDS move much closer to parity instead of flipping past it, and CONTROL and
DEATHONLY's original pattern survives. **This is consistent with population growth or crowding having contributed to
the uncapped reversal, but it does not establish population size as the cause.** The cap changes population size and
introduces its own reproductive-selection mechanism (which pairs get the limited remaining reproduction slots) at the
same time, so this experiment cannot isolate one from the other -- "the reversal was a population-size artifact"
overstates what a single confounded intervention can show. The defensible statement: the reversal was not robust to
capping population growth; "removing the odds asymmetry reverses the division of labor" should be retired in favor of
"removing the odds asymmetry closes most of the gap toward parity, and does not reliably reverse it under this
intervention." This is a real, three-seed-replicated finding that the reversal fails to reproduce under capping -- not
a three-seed-replicated identification of population size as its cause.

**Update (added later; the interpretation above is left as originally written).** An intermediate update here, based on
Iteration 15, said the cap's "close to parity" result was specific to the reproduction-blocking cap. That was an artifact
of the rollout-environment mismatch described in Iteration 16 and is withdrawn. With the corrected analysis, the cap
(-0.8 SUCCESSONLY, -2.9 EQUALODDS) and exact density at the same target (+3.3, -0.7) are both close to parity, and only
the uncapped runs keep a clear female lead (+10.2, +4.5). The hedged reading in this iteration's Interpretation, that the
result is consistent with population growth contributing but does not prove it, stands.

**A data-provenance note, not a result:** the "uncapped" column above was regenerated by re-running
`analyze_shared_states.py` against the same Iteration 12 checkpoints as part of this iteration's capped-vs-uncapped
comparison, rather than copied from Iteration 12's own log. Point estimates match exactly, but CI endpoints differ by
a thousandth or two in places (e.g. CONTROL seed 42's prey CI is [+.135,+.169] in Iteration 12's table and
[+.135,+.165] here) -- bootstrap-resampling noise from a fresh 2,000-replicate draw against the same underlying
policy, not a data or measurement discrepancy. Worth a single canonical set of numbers if this table is ever
regenerated again, so a reader doesn't have to work that out themselves.

**Caveats:**
- The cap constrains growth; it does not force an exact population match. CONTROL's population (~21-22) sits below
  the other three conditions' ~25.5-26 (roughly 15-19% lower), a residual asymmetry, though far smaller than
  Iteration 12's 26-to-92 range. "Closer to a common ceiling" is accurate; "matched population" is not.
- Only one cap value (26) was tested; whether these results hold at other population sizes (e.g. much smaller or
  much larger shared ceilings) is untested.
- `predator_population_cap` changes which individuals get to reproduce when the population is near the ceiling
  (whichever pairs are processed first in a step, an explicitly documented but arbitrary tie-break -- see the code),
  a selection mechanism uncapped runs don't have. This is itself a new mechanism, not a neutral measurement
  intervention, and could in principle contribute to the behavioral differences reported here alongside the
  population-size effect it's meant to isolate.
- As in Iteration 12, these are two complementary but not statistically independent measures (shared-state and
  energy-source) on the same trained policies, and the shared-state CIs cover variation within one fixed state bank,
  not a formal capped-vs-uncapped or cross-seed significance test; the "all three seeds agree" pattern is the actual
  cross-seed evidence.

### Iteration 14 — Coordination or parallel specialization? A mate-proximity contingency test

**Correction (added in Iteration 16): the rollout-based energy-share and coordination figures in this iteration were computed in the plain base environment, not the prey-floor / density-target env the runs were trained in. See Iteration 16 for the corrected numbers; conclusions that depended on them are superseded there. Shared-state (approach-bias) tables are unaffected.**

**Purpose:** every result so far (Iterations 7-13) only shows an *average* difference between the sexes -- males hunt
more than females across the whole population. None of it says whether any individual predator's behavior is
associated with its own specific partner's real-time proximity, or whether the split is two fixed per-sex
tendencies running in parallel, indifferent to the partner's state. This iteration (`analyze_mate_contingency.py`,
new this iteration) asks that narrower question directly: is a predator's prey/fruit approach behavior associated
with its recorded mate's real-time proximity? (Proximity/status only -- this cannot and does not test whether
behavior responds to what the partner is actively *doing*, e.g. mid-hunt; see the caveats below.)

**Method.** `env.agent_mate` records the reproductive-partner bond formed the first time two specific predators
successfully reproduce (serial monogamy: a later re-mating overwrites it). At every step, a live predator with a
recorded mate is bucketed by that mate's status: **near** (mate alive, within Chebyshev distance 3), **far** (mate
alive, farther away), **dead** (mate recorded but no longer alive), or **abandoned** (has reproduced before but has
no current mate entry because its former partner re-mated elsewhere -- distinguished from **virgin**, never
reproduced at all, which is excluded from the contrast). far + dead + abandoned are combined into **away**, the
primary contrast against **near**, tested with a paired per-episode bootstrap of the near-minus-away difference in
approach_bias (the same metric `analyze_prey_approach_from_checkpoint.py` uses). Run against all three CONTROL
seeds (the best-established, most consistently replicated condition), checkpoint 29, 30 episodes each.

**Results (near-minus-away approach-bias difference, 95% bootstrap CI, "both" = episodes containing both near and
away decisions):**

| Sex | Target | seed 42 | seed 43 | seed 44 |
|---|---|---|---|---|
| male | prey | -0.000 [-.010,+.010] (30/30) | +0.018 [+.008,+.028] (30/30) | +0.019 [+.008,+.029] (30/30) |
| male | fruit | -0.016 [-.023,-.009] (30/30) | -0.004 [-.010,+.001] (30/30) | +0.000 [-.007,+.009] (30/30) |
| female | prey | -0.010 [-.019,-.002] (30/30) | -0.017 [-.024,-.008] (30/30) | -0.014 [-.025,-.004] (30/30) |
| female | fruit | -0.055 [-.062,-.048] (30/30) | -0.073 [-.080,-.066] (30/30) | -0.081 [-.090,-.074] (30/30) |

For context: 42.6-43.5% of all prey-decisions across the three seeds came from agents that had never reproduced
(bucket "virgin", excluded from the contrast above); 18.5-19.8% came from agents whose former mate had since re-mated
elsewhere (bucket "abandoned", included in "away" -- a Codex review caught an early version of this script
mislabeling these as "never-mated" and silently excluding them, discussed below).

**What this shows:**

1. **Females show a small contingency effect on both targets, statistically detectable in these evaluation episodes
   on every seed.** Every one of the six female cells above has a negative near-minus-away difference with a CI
   excluding zero: females approach both prey and fruit *less* when their recorded mate is nearby than when he is
   far, dead, or reassigned elsewhere. The fruit effect (-0.055 to -0.081) is much larger than the prey effect
   (-0.010 to -0.017) but both point the same direction on every seed. ("Statistically detectable" rather than
   "real": these CIs quantify evaluation-episode variability for one fixed trained checkpoint per seed, not
   uncertainty across training runs -- three seeds are three separate replications, not a formal population-level
   estimate.)
2. **Males show a pattern that is weaker overall and less consistent, but not uniformly weaker cell-for-cell.**
   Prey-approach is slightly *higher* near the mate on two of three seeds (+0.018, +0.019, both CIs excluding zero,
   comparable in magnitude to the corresponding female prey effects) and flat on the third (-0.000, CI includes
   zero) -- the opposite sign from the female prey effect on those two seeds, and not replicated on all three.
   Fruit-approach is close to zero on two seeds (CIs include zero) but seed 42's -0.016 CI excludes zero and is
   comparable in magnitude to the prey effects -- "close to zero on all three" would overstate the consistency here.
3. **A plausible, testable mechanism, not yet checked:** this module already has a unidirectional male-to-female
   energy-provisioning mechanic (`_apply_male_gift`) -- on a successful hunt, a male donates a share of the energy
   gained to his recorded mate if she is within `predator_gift_range`. If that donation correlates with the same
   "near" bucket used here, that would be *consistent with* (not proof of) a straightforward economic explanation for
   the female effect -- a provisioned female needing to forage less urgently -- that would not require anything
   resembling "coordination" in a richer sense. A bare correlation wouldn't isolate this from other things that also
   correlate with time spent near a mate (survival duration, energy, location, the mate's own hunting success), so
   it would need to be time-aligned or otherwise conditioned to really test the mechanism. `analyze_energy_sources.py`
   already logs per-life energy "received: from mate," making a first pass cheap, but not done in this iteration.

**A real bug found and fixed before trusting this result (Codex review):** the first version of `mate_bucket()`
lumped two very different populations into one "none" bucket -- agents that had never reproduced (true "virgins")
and agents that had reproduced before but whose partner had since re-mated with someone else (correctly "abandoned,"
not "never-mated"). In the initial smoke test this silently excluded about 20% of all decisions from the near-vs-away
contrast and mislabeled them. Fixed by splitting into distinct "virgin" and "abandoned" buckets, with "abandoned"
now correctly counted in "away" (no partner currently present, same as far/dead). The numbers reported above are
post-fix.

**What this does and does not establish.** A near-vs-away difference here is a marginal, observational association
between approach behavior and a specific partner's recorded proximity/status -- not proof that the policy
"recognizes" that individual as its mate (this environment's observations expose nearby predator occupancy and
energy, but nothing that identifies *which* nearby predator is the recorded mate specifically), not evidence of
communication (there is no signaling channel in this environment's action/observation space at all), and **not by
itself evidence that behavior is caused by or contingent on the partner's state**: a fixed policy reacting only to
ordinary state it already observes (own energy, local predator density, location, target configuration) could
produce the same association purely because mate proximity happens to correlate with those variables, without the
policy responding to the partner as such at all. The honest description of what was found: **approach behavior is
associated with the recorded mate's proximity/status, detectably for females on both targets and more weakly and
less consistently for males, especially on prey** -- inconsistent with the simplest description of the split as an
unconditional, partner-indifferent per-sex average, but not itself a demonstration of partner-contingent behavior,
mate recognition, causality, or coordination in the fuller sense.

**Caveats:**
- Only CONTROL (original odds) has been tested this way; whether the same pattern holds under DEATHONLY/SUCCESSONLY/
  EQUALODDS, or under the capped-population runs (Iteration 13), is untested.
- The "away" bucket pools far, dead, and abandoned together -- a heterogeneous population, not "distant but otherwise
  identical to near." The result may partly reflect differences associated with bereavement, partner reassignment,
  or the individual's history rather than proximity alone; it does not isolate proximity among currently-bonded,
  living pairs specifically.
- The bootstrap resamples whole episodes (an appropriate cluster-bootstrap treatment of the within-episode
  dependence of steps, if episodes are independent and reasonably representative sampling clusters), estimating a
  decision-weighted pooled ratio difference, not an equally-weighted per-episode mean. With only 30 episodes per
  seed and a simple percentile interval, coverage is approximate rather than exact, and none of the 12 reported
  sex/target/seed contrasts are adjusted for multiplicity -- CI exclusion here should be read as suggestive, not as
  a formal hypothesis test. "30/30 episodes have both" reports cluster availability, not a literal effective-sample-
  size calculation; episodes with very few qualifying decisions contribute less information than this count implies.
- The provisioning-mechanism hypothesis above is a plausible explanation, not a tested one in this iteration.

**Follow-up (added after the density-target runs finished): a dedicated compensation contrast and a pair-fitness
test, same three CONTROL seeds, checkpoint 29, 30 episodes each.**

*(b) Compensation: near vs. mate-dead only.* `analyze_mate_contingency.py` was extended with a near-minus-dead
contrast, because the "away" bucket above pools far, dead, and abandoned. Near-minus-dead approach-bias difference,
95% bootstrap CI:

| Sex | Target | seed 42 | seed 43 | seed 44 |
|---|---|---|---|---|
| female | fruit | -0.047 [-.058,-.036] | -0.074 [-.084,-.063] | -0.090 [-.103,-.079] |
| female | prey | -0.008 [-.022,+.006] | -0.011 [-.028,+.006] | -0.013 [-.028,+.001] |
| male | prey | +0.012 [+.001,+.024] | +0.031 [+.018,+.044] | +0.032 [+.018,+.046] |
| male | fruit | -0.019 [-.028,-.010] | -0.001 [-.009,+.007] | +0.005 [-.004,+.013] |

Females approach fruit more when their recorded mate is specifically dead than when he is nearby, on all three seeds,
at about the same size as the pooled near-away effect (-0.055, -0.073, -0.081). The female prey difference is
negative in the same direction on every seed but its CIs include zero (the dead bucket is small, so these are wide).
Males approach prey somewhat more when their mate is nearby than when she is dead (all three positive, all three CIs
excluding zero, seed 42 only narrowly). This is the same kind of marginal observational association as the near-away result, with the same
caveats. A dead mate can no longer provision, so it is what the untested provisioning hypothesis predicts, but it does
not distinguish that from other explanations. The one-step lag in how a just-dead mate is bucketed (see the script's
docstring) applies here too.

*(c) Pair fitness (`analyze_pair_fitness.py`, new).* Among (male, female) pairs that reproduced at least once in an
episode (1,777 / 1,671 / 1,726 pairs on the three seeds), does a larger difference between the two parents' own
prey-share go with more offspring from that pair? Episode-cluster bootstrap, tie-aware Spearman:

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| high-minus-low mean offspring (median split) | -0.017 [-.078,+.039] | -0.022 [-.090,+.050] | -0.007 [-.069,+.052] |
| Spearman rho | -0.019 [-.061,+.028] | -0.019 [-.055,+.019] | +0.006 [-.038,+.049] |

Every interval includes zero on every seed: no detectable association between pair differentiation and offspring
count. Scope limits: pairs are defined by having reproduced, so this cannot speak to whether differentiation affects
the chance of reproducing at all; differentiation uses each parent's whole-episode foraging, not only the period before
reproduction; it is association, not causation. Two bugs in the first draft (pair-level bootstrap ignoring episode
clustering; tie-incorrect Spearman) were caught by a Codex review and fixed before these numbers were produced. An
8-episode smoke test of the pre-fix statistics had shown a borderline negative signal; at full scale it disappears,
a reminder not to read smoke tests.

### Iteration 15 — Exact population density without blocking reproduction

**Correction (added in Iteration 16): the rollout-based energy-share and coordination figures in this iteration were computed in the plain base environment, not the prey-floor / density-target env the runs were trained in. See Iteration 16 for the corrected numbers; conclusions that depended on them are superseded there. Shared-state (approach-bias) tables are unaffected.**

**Purpose:** Iteration 13's own review flagged that `predator_population_cap` changes population size and, at the
same time, which pairs get to reproduce near the ceiling (an arbitrary agent-ID tie-break), so it could not isolate
population size. `FixedPredatorDensityEnv` (`fixed_predator_density_env.py`, new) removes that specific confound:
reproduction is never blocked (cost, reward and mate bond are unchanged), and after each step the population is pushed
back to a target by culling predators chosen uniformly at random across both sexes (overflow) or spawning random-sex
replacements with default energy and no parents (shortfall). It is not confound-free: it adds an exogenous,
policy-independent random death (and, when population is below target, free replacement individuals). A Codex review of
the new class found two real bugs, both fixed before use (the class would silently still block reproduction if
`predator_population_cap` were also set, now rejected; replenishment could abort on one coin-flip into an exhausted
ID pool while the other sex had room).

**Setup:** the same 2x2 odds factorial as Iterations 12-13 (CONTROL, DEATHONLY, SUCCESSONLY, EQUALODDS; k=0.5, penalty
0.2, minibatch 1024, 300 iterations, prey floor 20, 50,000 prey IDs), three seeds per cell, `predator_density_target=26`
(the same ceiling as Iteration 13), evaluated at checkpoint 29. A single-seed pilot (CONTROL and EQUALODDS, seed 42)
passed a gate check before the other ten runs were launched.

**Did the mechanism work?** Yes. Total predators at the end were 26.00 in all 12 runs, all episodes ran the full 1001
steps, prey stayed at 21-25. Population is now equal across all four conditions, including CONTROL, which the cap left
at ~21-22. (Completed lives in the two high-success cells are shorter, roughly 126-162 steps against 200-238 for
CONTROL/DEATHONLY, partly because culled predators count as completed lives.)

**Shared-state approach-bias difference (male minus female, 95% bootstrap CI):**

| Condition | seed | prey | fruit |
|---|---|---|---|
| CONTROL | 42 | +0.163 [+.150,+.176] | -0.076 [-.093,-.058] |
| CONTROL | 43 | +0.128 [+.110,+.143] | -0.186 [-.207,-.167] |
| CONTROL | 44 | +0.149 [+.124,+.179] | -0.072 [-.092,-.056] |
| DEATHONLY | 42 | +0.152 [+.137,+.169] | -0.071 [-.094,-.048] |
| DEATHONLY | 43 | +0.116 [+.100,+.132] | -0.167 [-.180,-.155] |
| DEATHONLY | 44 | +0.126 [+.111,+.140] | -0.182 [-.201,-.165] |
| SUCCESSONLY | 42 | +0.077 [+.066,+.087] | -0.027 [-.041,-.013] |
| SUCCESSONLY | 43 | +0.003 [-.013,+.020] | -0.033 [-.047,-.017] |
| SUCCESSONLY | 44 | -0.008 [-.021,+.005] | -0.077 [-.095,-.062] |
| EQUALODDS | 42 | +0.055 [+.036,+.072] | -0.071 [-.082,-.060] |
| EQUALODDS | 43 | -0.032 [-.046,-.018] | -0.067 [-.079,-.054] |
| EQUALODDS | 44 | -0.048 [-.064,-.033] | -0.072 [-.081,-.063] |

**Energy-source comparison (prey share of own gross foraging energy, completed lives):**

| Condition | male, seeds 42/43/44 | female, seeds 42/43/44 |
|---|---|---|
| CONTROL | 50.5% / 54.9% / 53.3% | 11.4% / 11.8% / 11.4% |
| DEATHONLY | 53.6% / 54.6% / 55.7% | 14.3% / 12.8% / 13.1% |
| SUCCESSONLY | 34.7% / 35.3% / 35.4% | 44.2% / 42.6% / 41.0% |
| EQUALODDS | 36.5% / 37.0% / 37.1% | 40.4% / 40.0% / 37.3% |

**The three designs side by side.** Female-minus-male prey-share gap in the high-success cells, in percentage points
(positive = females take the larger prey share):

| Cell | Uncapped (Iteration 12) | Cap, reproduction blocked (Iteration 13) | Density target (this iteration) |
|---|---|---|---|
| SUCCESSONLY | +12.3 to +12.7 | -3.4 to +1.4 | +5.6 to +9.5 |
| EQUALODDS | +3.8, +4.1 (two seeds only) | -6.3 to -0.7 | +0.2 to +3.9 |

**What this shows (SUPERSEDED for the energy-share findings in items 3 and 4 by Iteration 16; those numbers came from the wrong
environment, and the claim that the cap was the odd one out is withdrawn. Items 1-2 and the approach-bias parts survive):**

1. **CONTROL and DEATHONLY are the same in all three designs:** a clear male-leaning prey split and female-leaning fruit
   split on every seed (male ~50-56% of energy from prey against ~11-15% for females).
2. **In SUCCESSONLY and EQUALODDS the split is much smaller than in CONTROL/DEATHONLY on every measure, in every
   design.** This attenuation is the robust result.
3. **Whether the split overshoots into a female lead in realized prey-energy share depends on the design.** It does
   under the density target (SUCCESSONLY +5.6 to +9.5, all three seeds; EQUALODDS +0.2 to +3.9), about 40% smaller than
   uncapped in each cell (a comparison of mean gaps from rounded point estimates; uncapped EQUALODDS has only two
   seeds, and no interval supports the difference); it does not under the cap. The density-target result resembles the uncapped one in sign, and the
   reproduction-blocking cap is the odd one out among the tested designs. That is compatible with the confound named
   in Iteration 13's caveats but does not demonstrate it: random culling, replacement agents, different life
   lengths, and ordinary training variation also differ between the designs. This is a
   correction to how Iteration 13 read its own result: the "reversal does not reproduce" finding was specific to the
   cap, not a general consequence of holding population near a ceiling.
4. **The measures do not all agree.** The fruit-approach difference stays in the CONTROL direction (female > male)
   under the density target, attenuated (SUCCESSONLY -0.03 to -0.08; EQUALODDS -0.07 on all three seeds, unusually
   consistent). It does not reverse, unlike the uncapped run (+0.07 to +0.11). The prey-approach difference is small
   and mixed in sign for both high-success cells (one seed with a male lean, others near zero or slightly female).
   So the overshoot is in realized energy share, not in approach behavior toward fruit.

**Interpretation (superseded by Iteration 16: the energy-share numbers this rests on came from the wrong environment; the
attenuation finding survives, the design-dependence conclusion does not).** Raising female hunting success attenuates the male-prey/female-fruit split a great deal in every
design tried. How far the energy share overshoots into a female lead depends on how population is controlled: about
+12.6 points uncapped, about -1 point under the reproduction-blocking cap, about +7.5 points under exact density. Each
population-control mechanism is itself an intervention, so this does not identify one "true" size. Two things follow.
The reversal in direction is not simply a population-size artifact, since it survives when population is held fixed
without blocking reproduction. Its size is sensitive to how population is controlled, so a single number should not be
quoted. Necessity of the odds asymmetry is still not settled by this, only narrowed.

**Caveats:**
- One target value (26) has been tested. The density target also actively adds individuals in CONTROL and DEATHONLY
  (whose natural population is below 26), a different treatment from the cap, which never adds anyone.
- The random cull shortens lives and changes the completed-life sample in an exogenous way. It is uniform across
  living predators, so it is not correlated with foraging, but energy-share figures here are computed over lives that
  are partly ended by the cull.
- The shared-state CIs cover variation over one fixed state bank per checkpoint, not training-seed variation; "all
  seeds agree in sign" is the cross-seed evidence. Energy-share differences have no CI reported. The EQUALODDS
  uncapped column has two seeds, not three. Comparisons across designs use point estimates only; no formal test of
  design-by-condition interaction was run.
- The design comparison is between separately trained runs with different intervention mechanisms; differences could
  partly reflect run-to-run training variation, which three seeds per cell only partly averages out.

### Iteration 16 — Re-analysis in the environment each run was trained in; provisioning test; coordination across all designs

**What happened.** While extending the coordination tests, I found that every rollout-based analysis script (energy
sources, mate contingency, pair fitness, prey approach) built the plain base `PredPreyGrass` from a run's saved config
and silently ignored the prey floor and the predator density target the run was trained with. Measured on SUCCESSONLY
(uncapped): in the base env 2 of 3 episodes collapsed to zero prey after about 110-140 steps; in the env it was trained
in all three ran the full 1000 steps with prey held at 20 and 52-62 predators. On the density-target runs the population
was no longer pinned at the target (30-31 predators, prey 11-14). So Iterations 12-15's rollout-based **energy-share
tables** and Iteration 14's **coordination tests** were computed in a different ecology than the one the policies
learned in. Not affected: the shared-state comparisons (they query policies on one fixed base-env state bank shared by
all runs, deliberately), Iterations 0-11 (base-env runs, or shared-state results only). Fix: `analysis_env.py` builds the
trained env class from `run_config.json`; the energy instrumentation became a mixin that sits on any of them; 88 tests
pass, including forced hunt, fruit, gift and care transfers through each generated class. A Codex review of the fix found
no functional bug (it asked for stronger tests, the interactive evaluator fix and documentation of the bank's env, all done).
Everything below was re-run with the fixed scripts (checkpoint 29; 20 episodes for energy, 30 for the coordination tests).

**Corrected energy shares (prey share of own gross foraging energy, completed lives, seeds 42/43/44):**

| Design | Cell | male | female | female minus male, mean (points) |
|---|---|---|---|---|
| uncapped | CONTROL | 54.7 / 52.9 / 53.4 | 12.5 / 12.2 / 13.3 | -41.0 |
| uncapped | DEATHONLY | 52.3 / 54.7 / 53.0 | 14.0 / 15.7 / 15.2 | -38.4 |
| uncapped | SUCCESSONLY | 58.6 / 58.7 / 62.6 | 71.2 / 70.0 / 69.3 | **+10.2** |
| uncapped | EQUALODDS | 73.2 / 70.2 / 71.1 | 76.1 / 75.7 / 76.2 | **+4.5** |
| cap 26 | CONTROL | 49.4 / 45.4 / 46.5 | 12.1 / 12.7 / 13.1 | -34.5 |
| cap 26 | DEATHONLY | 54.8 / 52.1 / 50.6 | 12.3 / 12.8 / 13.6 | -39.6 |
| cap 26 | SUCCESSONLY | 41.4 / 42.4 / 44.9 | 42.0 / 43.8 / 40.6 | **-0.8** |
| cap 26 | EQUALODDS | 48.2 / 44.8 / 44.8 | 41.7 / 42.9 / 44.5 | **-2.9** |
| density 26 | CONTROL | 51.9 / 55.9 / 55.7 | 11.8 / 11.3 / 11.4 | -43.0 |
| density 26 | DEATHONLY | 55.2 / 56.4 / 57.6 | 13.7 / 12.6 / 12.9 | -43.3 |
| density 26 | SUCCESSONLY | 41.1 / 43.0 / 42.2 | 45.6 / 45.7 / 45.0 | **+3.3** |
| density 26 | EQUALODDS | 42.8 / 44.2 / 41.9 | 43.3 / 41.5 / 42.0 | **-0.7** |

Per-seed gaps for the four cells that matter: uncapped SUCCESSONLY +12.6/+11.3/+6.7, EQUALODDS +2.9/+5.5/+5.1; cap
SUCCESSONLY +0.6/+1.4/-4.3, EQUALODDS -6.5/-1.9/-0.3; density SUCCESSONLY +4.5/+2.7/+2.8, EQUALODDS +0.5/-2.7/+0.1. Levels
differ from the earlier tables (in the trained env prey are always plentiful, so both sexes take more of their energy
from prey in the uncapped high-success cells). Completed lives are shorter under density (about 80-150 steps against
130-300 elsewhere), partly because culled predators count as completed lives. The EQUALODDS uncapped seed 42
(the calibration run, previously missing its energy analysis) is now included, which completes that old backfill item.

**How this changes the earlier conclusions.**
- **Iteration 15's headline is withdrawn.** It said the reproduction-blocking cap was the outlier and that exact density
  brought the female lead back (about +7.5 points in SUCCESSONLY). With the trained env, density-26 gives +3.3 and
  -0.7, close to the cap's -0.8 and -2.9. Both fixed-population designs are near parity; only the uncapped runs,
  where population booms (about 50 in SUCCESSONLY, about 85 in EQUALODDS), keep a clear female lead (+10.2, +4.5).
- **Iteration 13's original, hedged reading is closer to right:** holding population fixed removes most of the female
  lead in realized prey-energy share. It is still not proof that population size is the cause, because both fixed-
  population designs are interventions (the cap blocks reproduction; the density target culls at random and adds
  replacements) and they differ by a few points from each other.
- **Preliminary dose-response (one seed at 13 and 52, seed 42; density-target design, target 26 seed 42 for comparison).**
  SUCCESSONLY female-minus-male gap: target 13 **-4.8**, target 26 **+4.5**, target 52 **+9.2**, uncapped **+12.6**
  (population about 52-56). EQUALODDS: 26 **+0.5**, 52 **+1.3**, uncapped **+2.9**. CONTROL stays strongly male-leaning
  at every population (-34 at 13, -40 at 26, -53 at 52, -42 uncapped). The gap rises with population in this one seed.
  Caveats: target 13 failed its pilot gate because about 15% of its episodes end early when a sex goes extinct under the
  random cull (episode length 846-949 against 1001); target-52 seeds 43/44 and target-13 seeds 43/44 are still running.
  This is a single-seed trend, not an established relationship.

**Provisioning test (`analyze_provisioning.py`, new; CONTROL uncapped and density-26, SUCCESSONLY density-26, three seeds
each).** If the female mate-proximity association came from being fed by a nearby mate, then near-mate females who had not
recently received a gift should look like females with no mate nearby, and the association should shrink once the
female's own energy is held fixed. Neither happens. Fruit approach, near-minus-away: crude -0.058/-0.072/-0.078
(uncapped CONTROL), energy-adjusted -0.059/-0.073/-0.081, near-but-no-recent-gift -0.055/-0.071/-0.075; recently gifted
females drop at most slightly more (gift-recent minus near-no-gift between -0.022 and +0.007 across the nine runs; 3 of 9
intervals exclude zero). The same holds for prey approach and in the other two sets of runs; own energy differs little across groups within a run
(group means about 7-11). So this test does **not** support gifts or own energy as the explanation. It does not exclude other
things that track mate proximity, and it cannot say what the policy responds to.

**Coordination tests across all 36 cell-runs (four cells x three designs x three seeds, 30 episodes each).**
- *Female fruit approach is lower with a living mate nearby than away* in **every one of the 36 runs** (all CIs exclude
  zero). Size depends on the cell: -0.045 to -0.08 in CONTROL and DEATHONLY, where females rarely hunt, in every design;
  -0.03 to -0.05 in the capped and density-target high-success cells; only about -0.005 to -0.013 in uncapped
  SUCCESSONLY/EQUALODDS. Near-versus-dead-mate contrasts are about the same size as near-versus-away, as in Iteration 14.
- *Female prey approach* is negative in most runs; largest under density-26 high-success (-0.06 to -0.095), mixed in
  capped CONTROL/DEATHONLY (-0.02 to +0.015).
- *Male prey approach* has no stable sign: near zero or slightly positive with a mate near in uncapped runs, negative in the
  capped and density-target high-success cells (-0.02 to -0.045).
- *Pair fitness* (differentiation between the two parents' prey shares vs offspring count, pairs that reproduced at least
  once): the earlier "clean null" no longer holds uniformly, but effects are small (|Spearman rho| <= 0.09) and the sign
  depends on the design. Slightly positive in the uncapped high-success cells (+0.02 to +0.048, most CIs exclude zero),
  slightly negative under density-26 (CONTROL -0.05 to -0.09 on all three seeds; EQUALODDS -0.06 to -0.09 on all
  three), mostly inside the intervals for the cap. Nothing supports differentiation paying off consistently. Differentiation
  is confounded with lifespan and with how the population is controlled (culls cut lives short), so I would not interpret
  the sign.

**Caveats.** Association only throughout; no signaling channel exists in this environment. Intervals are episode-level
bootstraps for one checkpoint per seed (not training-seed uncertainty), and none of the roughly 300 contrasts above
is adjusted for multiplicity. Only checkpoint 29 was analysed. The cross-design gap comparisons use point estimates with no
interval. The population-target sweep is incomplete (see above), and the target-13 pilot's early-ending episodes make its
seed-42 result less comparable.

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

**Necessity, tested with a prey-density floor instead of the ecological collapse (Iteration 11, single seed), then a full factorial
(Iteration 12, three seeds per cell, uncapped population), then the same factorial with population size also capped (Iteration 13, three seeds
per cell):** a `FixedPreyDensityEnv` holds prey density near a floor by replacement spawns, so a high-success run does not have to collapse the
prey population. Iteration 12 found that CONTROL replicates the original split cleanly, DEATHONLY leaves it intact and attenuated, SUCCESSONLY
neutralizes it on prey-approach and *reverses* it on fruit-approach and energy share, and EQUALODDS gives no consistent sign across seeds -- but
also found predator population size roughly triples between CONTROL (~26) and EQUALODDS (~85), an uncontrolled confound. **Iteration 13 added a
`predator_population_cap` (mirroring the prey floor in the opposite direction) and re-ran all four conditions with population held closer to a
shared ceiling (26; CONTROL sits noticeably lower, ~21-22, since its natural population needs little constraining).** CONTROL and DEATHONLY keep
the same qualitative pattern under the cap, though some seed-level effect sizes shift considerably (e.g. CONTROL seed 44's prey diff roughly
halves). SUCCESSONLY and EQUALODDS change more: **the split narrows a great deal and moves close to parity, rather than the outright reversal
Iteration 12 reported** -- SUCCESSONLY's energy share becomes close to equal between the sexes (38-41% vs 38-40%) instead of flipping (28-30% vs
40-42% uncapped), though its prey-approach diff, while much smaller than CONTROL's, stays slightly positive on two of three seeds, and its
fruit-approach reversal shrinks but does not fully disappear (two of three seeds still show a small positive/reversed CI). EQUALODDS's earlier
cross-seed inconsistency is much smaller capped than uncapped, though not perfectly resolved (one of three seeds' CIs still excludes zero).
**The corrected reading: the reversal Iteration 12 reported does not reproduce once population growth is capped -- consistent with population
size or crowding having contributed to it, but not proof that population size specifically was the cause**, since the cap simultaneously
introduces its own reproductive-selection mechanism (who gets the limited remaining reproduction slots) that this design cannot separate from
the population-size effect it targets. The defensible statement: removing the odds asymmetry closes most of the gap toward parity under this
intervention; it does not reliably reverse it. Necessity remains unsettled: the cap constrains growth rather than forcing an exact population
match (CONTROL's capped population sits meaningfully below the other three), and a cleaner test would need to separate the cap's own selection
effect from the population-size effect it is meant to isolate. See Iterations 12-13 for the full tables.

**Iteration 16 corrects Iterations 12-15's energy-share numbers (rollouts had been run in the wrong environment) and, with it, Iteration
15's reading.** In the trained environment, CONTROL and DEATHONLY keep the male-prey/female-fruit split in every design (female minus male prey share
-34 to -44 points). In SUCCESSONLY and EQUALODDS the split shrinks a great deal, to a small female lead uncapped (+10.2 and +4.5 points) and to about
parity when population is held at 26 by the reproduction-blocking cap (-0.8, -2.9) or by exact density (+3.3, -0.7). A single-seed sweep of the
density target suggests the gap rises with population (SUCCESSONLY -4.8 at 13, +4.5 at 26, +9.2 at 52, +12.6 uncapped), which fits population size
or crowding contributing to the female lead, but this is one seed at 13 and 52 with the other seeds still running, and both fixed-population designs
are interventions themselves, so population size is not established as the cause. Necessity of the odds asymmetry is narrowed, not settled. See
Iteration 16 (Iteration 15's claim that the cap was the outlier is withdrawn).

**The death-chance axis, extended (Iteration 10):** at fixed 20% success, the women/men attempt ratio declines broadly from 5% to 30% death
chance (0.62-0.80 to 0.40-0.53 to 0.33-0.37 to 0.23-0.29); women's prey-energy share is lower at 30% (9.2-9.4%) than at 5% (14.7-15.4%), though
the 10% and 20% points overlap almost completely, so it is not a clean step down at every point. At 30% death chance, women's prey approach
turns negative in both runs tested (-0.018, -0.021) -- evidence of avoidance, not yet a seed-replicated effect with its own uncertainty
estimate. Prey do not collapse anywhere on this axis, unlike the success-rate axis; episode length does shorten somewhat at 30%.

**Reward design.** The same unequal odds gave no split under flat per-event rewards or at k = 0.2 (women starved). Flat rewards strongly reward
repeated consumption of depleted fruit (about 60-80 reward per life against about 9 for prey), a plausible reason for the missing male prey
preference; not isolated, because k = 0.2 and k = 0.5 differ only in reward scale.

**Limits:** partial, not exclusive (men still get about half their energy from fruit; a woman catches about 0.8-1.0 prey per life against about 10 for a
man) and no causal coordination or task allocation is established; most response-surface points (40%/60% success, and the original Iteration 11 single-seed matched-ecology runs,
now superseded by Iteration 12's three-seed factorial) have only one seed; minibatch 128 and 1024 runs are not like-for-like (a same-rewards
minibatch-128 run, FORAGING_PENALTY02, shows about half the shared-state effect size of the minibatch-1024 runs); only k = 0.5 is replicated at
three seeds and it was chosen after k = 0.2 failed; rollout metrics (outside the shared-state check) mix action preference with the states each
policy creates; energy figures are gross, aggregated over lives that include lives cut off at the end of the episode; the *unmatched* equal-odds/
success-only ablations (Iteration 8-9) still run in a collapsed, prey-poor regime -- Iteration 12's fixed-density factorial is the three-seed
alternative, but its own conditions differed substantially in predator population size from each other (see Iteration 12), a confound Iteration
13 addressed with a population cap; that cap changed the finding (the reversal did not reproduce at closer-to-matched population, see Iteration
13) but introduced its own reproductive-selection confound in the process, so the module has moved from one confound to another rather than to a
confound-free design.

**Current best statement:** under energy-proportional forage reward with k = 0.5, the male-prey / female-fruit differentiation replicates across
three training seeds, with or without the explicit combat-death penalty, survives a same-observation comparison that removes the self-created-states
confound, and persists (female minus male prey share -34 to -44 points) in every design tried (uncapped, capped, exact density) when female
hunting odds stay at their original values, and when only death risk is equalized (10% to 5%). Equalizing success rate (SUCCESSONLY), and success
together with death risk (EQUALODDS), shrinks the split a great deal on every measure in every design, to a female lead of +10 and +4.5 points in the
uncapped runs and to about parity (-3 to +3 points) when population is held at 26 by either method. A one-seed density-target sweep suggests the gap
rises with population (-4.8 at 13, +4.5 at 26, +9.2 at 52, +12.6 uncapped for SUCCESSONLY); that is consistent with population size or crowding
contributing, not proof, since each control is itself an intervention. Success rate, not death risk, is the dominant lever on how much of the split
remains. Necessity of the odds asymmetry is narrowed, not settled. The death-chance axis alone, pushed to 30%, produces a smaller but real avoidance
signal without any ecological collapse (Iteration 10, two seeds). Note the rollout-based energy-share numbers of Iterations 12-15 were computed in the
wrong environment and are superseded by Iteration 16; the shared-state approach-bias tables are unaffected.

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

## Summary: is foraging behavior coordinated, or parallel individual specialization?

**Working definition:** whether a predator's own approach behavior is *associated with* its specific recorded mate's real-time
proximity/status, as opposed to being a fixed per-sex tendency indifferent to the partner's state. This is a different question from
"is there a sex differentiation" above -- it asks what kind of thing the differentiation is, not why it exists. It is a marginal,
observational test, not a causal or interventional one.

**Evidence (Iteration 14 as first run; corrected and extended in Iteration 16 -- the first run used the wrong environment):** in the trained
environment, across all 36 cell-runs (four cells x uncapped/capped/density-target x three seeds, checkpoint 29), females approach fruit less when
their recorded mate is alive and nearby than when he is far, dead or reassigned, and every one of the 36 intervals excludes zero. The size depends
on the cell: about -0.045 to -0.08 in CONTROL and DEATHONLY, -0.03 to -0.05 in the high-success cells under a fixed population, and only -0.005 to
-0.013 in the uncapped high-success cells. Near-versus-dead contrasts are about the same size. Female prey approach is negative in most runs (up to
-0.095 under density-26 high-success). Male prey approach has no stable sign.

**Current best statement:** the division of labor cannot be fully described as an unconditional, partner-indifferent per-sex average -- female
foraging behavior is statistically associated with a specific partner's recorded proximity/status in every cell and design tested. This is a
marginal observational association, not proof of mate recognition, causal partner-responsiveness, or communication (this environment has no signaling
channel), and it is confoundable by anything else correlated with proximity (local density, location, survival/reassignment history). **A direct
test of the provisioning explanation did not support it:** near-mate females who had not recently received a gift look like females with the
mate nearby, and adjusting for the female's own energy leaves the association unchanged (nine runs). So the test does not support gifts or own energy as the
explanation (it does not rule them out); what does account for it is not known. Pair differentiation shows no consistent relationship with offspring count: small associations (|Spearman rho| <= 0.09) whose sign
depends on how population is controlled (slightly positive uncapped, slightly negative under density-26), confounded with lifespan. Nothing here shows
that partner-associated behavior, or being more differentiated as a pair, pays off in offspring.

---

## Next steps

1. ~~Fix the replenishment pool exhaustion, run the full odds factorial, control for the population-boom confound~~ **Done as far as
   the design allows (Iterations 12, 13, 15, 16).** Population-matched follow-ups were run two ways; neither isolates population size,
   since each is an intervention. Threads still open:
   - **Finish the population-target sweep** (running): target 52 seeds 43/44 (all three cells) and target 13 seeds 43/44 (CONTROL,
     SUCCESSONLY; queued after the sweep). The one-seed dose-response (gap rises with population) needs three seeds per target before it
     is quoted. Target 13 loses about 15% of episodes early to sex extinction under the random cull, which limits its comparability.
   - **The density target and the cap are both interventions** (random death plus free replacements; blocked reproduction). No design tried
     equalizes population without either, so this may be a limit of the approach.
   - ~~Backfill EQUALODDS seed 42's energy-source analysis~~ **Done (Iteration 16).**
2. ~~Test the provisioning-mechanism hypothesis~~ **Done (Iteration 16): not supported** (gift timing and own energy do not account for the
   female mate-proximity association, nine runs). Open: what does account for it (local density, the mate's own foraging state, location, or
   the policy responding to a partner it cannot identify); a time-aligned or interventional test would be needed.
3. ~~Extend the mate-contingency and pair-fitness tests beyond CONTROL~~ **Done (Iteration 16), all four cells and three designs.** Open: the
   pair-fitness sign depends on the design and is confounded with lifespan; a cleaner fitness measure (e.g. offspring survival to reproduction) is untested.
4. **Death-chance response surface with more seeds** at 20% and 30% (currently two each), and consider intermediate points
   between 10% and 20%, given the trend has turned out to be consistent rather than negligible.
5. **Other k** (0.3, 0.7) and re-tuning the combat-death penalty on top of the energy-proportional reward (now looks droppable, see
   Iteration 9); a reward-scale control (for example flat rewards calibrated to the same expected total forage reward as k = 0.5) to
   separate closing the fruit loophole from raising the reward scale.
6. **More seeds where only one or two exist:** REF, PROP k=0.2, the response-surface points (40%, 60%).
7. **Decide the default minibatch.** Recommended, not yet adopted: `--minibatch-size 1024` (5.8x faster); the defaults stay at
   128 / 30 so all earlier runs remain reproducible. Runs since Iteration 7 pass the flag explicitly.
8. **Fruit-only shaping** (no catch reward) to separate learning to gather from learning to hunt.
9. **Longer training and multiple late checkpoints**, not only iteration 300, to check the effect is stable rather than a snapshot.
10. Unexplained: the early rise in births and episode length in Iteration 1 cannot come from predator learning; the prey
    policy changing behavior is the untested guess.
