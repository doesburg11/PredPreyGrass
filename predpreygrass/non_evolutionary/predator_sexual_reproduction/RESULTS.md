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
- **ABL_EQUAL_ODDS** (Iteration 8): `PPO_PREDATOR_SEXUAL_REPRODUCTION_ABL_EQUAL_ODDS_K05_MB1024_SEED42`, `..._SEED43`: PROP k=0.5 with women
  given the men's hunting odds (success 0.90, death 0.05). **ABL_SUCCESS_ONLY** (`--female-success-prob 0.90 --female-death-prob 0.10`)
  and **ABL_DEATH_ONLY** (`--female-success-prob 0.20 --female-death-prob 0.05`), seed 42: single-factor variants (running).
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
1. **The reward-farming loophole is real and the proportional reward closes it.** In REF a fruit yields 0.40 energy; at k=0.2 it
   yields 0.99 and net intake per step rises 60%.
2. **k = 0.2 makes men better foragers but starves the women** (population collapse above).
3. **k = 0.5 produces the sex pattern that was hypothesized, without collapse, and it replicates on three seeds:** men approach prey much more strongly
   (+0.286 cells, x1.44 stepping onto adjacent prey, against x0.85 in REF and a run-to-run noise of about +-0.04 cells),
   draw 46.9% of their energy from prey (above a random walker's 41%, the first time trained men do), while women hunt
   less than a random walker (11.1% prey share, 0.8 catches per life) and approach fruit the most of any run (x1.91).
4. **Earlier penalty finding not reproduced.** The FORAGING_PENALTY02 result (Iteration 4: men approach prey x1.08, women
   avoid x0.96) did not reproduce at minibatch 1024 with the same rewards (REF: men x0.85). Differences of about +-0.04
   cells between runs of one configuration are within run-to-run noise, so Iteration 4's male/female prey pattern should be
   treated as unconfirmed. The reproduced part is that women approach fruit somewhat more than men (x1.78 against x1.61 in REF).

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
3. **The hunting odds are necessary but not sufficient for the division of labor.**
   - *Necessary:* removing the difference in odds removes the split (finding 1).
   - *Not sufficient:* the same unequal odds (men 90/5, women 20/10) were in force in every earlier run, yet the split appeared only under
     the energy-proportional reward with k = 0.5, not with flat per-event rewards (Iterations 2-4, 7 REF) and not at k = 0.2, where women lost
     their incentive to gather and died out. A learner follows the payoff it is shown: with flat rewards a man was paid about 60-80 reward per
     life for eating nearly empty fruit against about 9 for prey (Iteration 6), so the real advantage of hunting never reached what he learned.
   Both a difference between the sexes and a reward that lets it be felt are needed.
4. **The remaining asymmetry did not produce a split on its own**: women still pay 90% of the birth cost and the gifts still run
   male-to-female, and with equal hunting odds there was no division of labor.

**Caveats:** two seeds; the ablation changes success (20% to 90%) and death chance (10% to 5%) together, so it does not say which
matters (single-factor runs ABL_SUCCESS_ONLY and ABL_DEATH_ONLY are running); a rough calculation suggests the death chance may
matter a lot, because dying ends a woman's whole future reward, not just costing the explicit -0.2, but this is an estimate, not a measurement.

## Summary: is there a division of labor?

**Working definition:** a difference between the sexes, emerging from training rather than hard-coded, in (a) what they do
(approach toward prey and fruit) and (b) where their energy comes from (prey share of intake), with a viable population.

**Evidence for (Iteration 7, energy-proportional reward k = 0.5, seeds 42 / 43 / 44, iteration 300):** men step onto adjacent prey
x1.44 / x1.49 / x1.32 as often as a random mover (women x0.99 / x1.02 / x1.07; flat-reward runs x0.85-1.08); men take 46.9 / 47.8 /
49.6% of their energy from prey (random walker 41.3%; flat-reward runs 37.4-39.9%) and women 11.1 / 12.1 / 11.6% (random 14.8%; flat-reward
runs 13.2-14.5%); women approach fruit x1.91 / x1.86 / x1.83; the ecosystem is healthy in all three.

**The hunting odds are necessary but not sufficient.**
- *Necessary* (Iteration 8): with women given the men's odds the split disappears on two seeds (prey share 34-38% for men and 37-38% for
  women; women hunt as much as men) and the prey are hunted to extinction.
- *Not sufficient* (Iterations 2-7): the same unequal odds gave no split under flat per-event rewards or under k = 0.2 (women starved);
  only the energy-proportional reward at k = 0.5 let the difference in odds show through, because a learner follows the payoff it is shown
  (the flat reward paid about 60-80 per life for nearly empty fruit against about 9 for prey).
- Both are required: a difference between the sexes that makes hunting pay for one and not the other, and a reward that lets it be felt.

**Limits:** (1) a shift in tendency, not exclusive specialization (men still get about half their energy from fruit; a woman catches
about 0.8-1.0 prey per life against about 10 for a man); (2) the equal-odds ablation changes success and death chance together, so it does
not say which one matters (single-factor runs are running); (3) only k = 0.5 is replicated, it was chosen after k = 0.2 failed, and the
absolute effect is modest; (4) the birth-cost asymmetry alone produced no split.

**Verdict:** a division of labor emerges reproducibly under one specific combination: unequal hunting odds AND an energy-proportional
reward with k = 0.5. It is not a general property of this environment, and which component of the odds (success or death chance) drives
it is not yet known.

---

## Next steps

1. **Other k** (0.3, 0.7) and re-tuning the combat-death penalty (0.2 now) on top of the energy-proportional reward; and whether
   prey depletion (about 18-21 alive) limits predators over longer runs.
2. **Female death chance** (currently 10% per hunting attempt, male 5%; `--female-death-prob`): try 30% on top of k = 0.5, watching
   the first 20 iterations for female extinction (at minibatch 1024 that takes a few minutes).
3. **Finish the single-factor ablations** (women get only the men's success, or only the men's death chance; running) to separate the two
   parts of the hunting-odds difference. Also replicate the baselines: REF and PROP k=0.2 have one seed each; more seeds would tighten the comparison.
4. **Decide the default minibatch.** Recommended, not yet adopted: `--minibatch-size 1024` (5.8x faster); the defaults stay at
   128 / 30 so all earlier runs remain reproducible. Runs since Iteration 7 pass the flag explicitly.
5. **Fruit-only shaping** (no catch reward) to separate learning to gather from learning to hunt.
6. Unexplained: the early rise in births and episode length in Iteration 1 cannot come from predator learning; the prey
   policy changing behavior is the untested guess.
