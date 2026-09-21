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

---

## Next steps

1. **More seeds, cheaply.** With `--minibatch-size 1024` a 300-iteration run takes about 1.5 hours.
   Run several seeds each of plain FORAGING and of the 0.2 penalty to learn how much of the
   FORAGING versus penalty difference is seed luck, and whether the sex pattern in Iteration 4 is
   real. Decide first whether 1024 becomes the default; if so, re-run the key configurations on it
   for a like-for-like comparison.
2. **Test the epoch hypothesis** (excess predator epochs hurt learning): compare 128 and 1024 over
   300 iterations and several seeds, not just 50 iterations of one seed.
3. **Other penalty values** (0.1, 0.4) around the 0.2 that worked, and a sex-specific penalty
   (females only) as a less clean alternative.
4. **Fruit-only shaping** (no catch reward) to separate learning to gather from learning to hunt.
5. **Female survival:** females stay a small population (about 5 against 19 males at iteration
   300 in Iteration 4); check birth cost, gift rate and starting population.
6. Unexplained: the early rise in births and episode length in Iteration 1 cannot come from
   predator learning; the prey policy changing behavior is the untested guess.
