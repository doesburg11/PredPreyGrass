# eco_evolutionary_deer_stag_hunt: results

## 1. Nature, nurture and both (pre-registered 2026-11-08)

### 1.1 Question and arms

The question is DESIGN.md's: in deer_stag_hunt's world, against frozen deer, do
hunters with heritable innate attraction genes plus shared PPO learning (arm B)
catch more adult deer, and more per attempt, than hunters with shared learning only
(L)? And more than hunters with the genes only (E)?

- **Genes:** four per hunter, `a_hunter`, `a_young`, `a_adult` and `neutral`.
  - They start at N(0, 0.1).
  - Each birth adds a mutation of N(0, 0.05) per gene, clipped to [-1, 1].
  - Each environment carries its own lineage; founders are the survivors of the
    previous episode.
  - The genes are in the hunters' observation in every arm.
- **E:** the genes' bias only, β₀ = 3.4417, with stops masked out and the cruise
  move when nothing is in view. No learning (`evolve.py`).
- **L:** PPO, genes not expressed (β = 0).
- **B:** PPO plus the bias inside the module's forward passes (β₀).
- **Deer:** frozen. Run k of every arm uses deer_stag_hunt run ((k − 1) mod 3) + 1's
  final deer policy.
- **Fixed in advance:** β₀ comes from DESIGN.md's rule, not from any result, and no
  setting changes after the pilot.

### 1.2 Pilot (before the main runs)

- **Runs:** pilot_L and pilot_B (run k = 1, 500 iterations) and pilot_E, matched to
  them on physics steps at 250 and 500 (`launchers/run_pilot.sh`).
- **Pilot passes if:**
  1. nothing crashes;
  2. at iteration 500, hunters are alive at the end of more than 25 of 50
     evaluated episodes in both L and B.

  Log-probabilities matching `net + bias` is checked by `tests/test_rl_module.py`
  (exploration and training passes give one distribution, ratio 1).
- **If the pilot fails:** the stage stops. Nothing is adjusted and re-run.
- **Pilot runs are not main runs:** their results are reported but enter no test.

### 1.3 Main runs

- **Runs:** E_k, L_k and B_k for k = 1..5 (`launchers/run_main.sh`). Seed k for every
  arm's run k.
- **L and B:** 1000 iterations, 14 env runners.
- **E:** 14 environments, matched to the median total physics steps of the ten L and B
  runs at iterations 250, 500, 750 and 1000.
- **E's budget:** each of E's 14 environments gets an exact 1/14 share of the
  matched total. A share is reached after a whole decision, so it can overrun by at
  most 7 physics steps. E refuses a number of environments different from the L and
  B runs'.
- **Recorded for every checkpoint:** physics steps, episodes, hunter births and
  generations (`budget.json`).
- **Seeds:** E runs seed every episode from seed k, so they are reproducible. L and B
  runs are not: RLlib seeds only their network initialisation and first episodes.
  Their runs differ by chance, as in deer_stag_hunt.

### 1.4 Measures and tests, set before the main runs

`launchers/evaluate_run.py` evaluates each run at 250, 500, 750 and 1000: evaluate.py
with 50 episodes, coop_fair.py with 20. Every episode starts from the checkpoint's
pooled gene pool.

**Measures:**

- **A:** adults caught per episode.
- **S:** adults caught ÷ (adults caught + failed adult attempts). For the run value
  it is pooled over all 150 episodes of the three checkpoints; zero if there were
  no attempts.
- **C (supporting):** adult approaches with a partner near ÷ chance.
- **Run value:** for A, C and coexistence, the mean over checkpoints 500, 750 and
  1000; for S, the pooled ratio above. This is the estimand of sustained
  late-training performance.

**Tests** (`launchers/verdict.py`; one-sided exact Mann-Whitney on run values, 5
against 5, p < 0.05):

1. B > L on A and on S. Both must hold.
2. B > E on A and on S. Both must hold.
3. In at least 4 of 5 B runs, the episodes with hunters and deer both alive at the end,
   averaged over the three checkpoints, number at least 45 of 50.

**Verdict:** if all three tests are met, the result is: "in this world, against fixed
deer, an innate attraction bias that evolves plus shared PPO learning catches more
adults, and more per attempt, than learning alone or the bias alone."

**Also reported:**

- each run's values, and differences in arm means with bootstrap 95% intervals over
  runs;
- C for every arm, and whether B's C exceeds 1 in all its runs;
- coexistence for every arm;
- gene means of the final gene pool;
- selection differentials, with L's unexpressed genes as the drift reference. A
  selection differential is computed per episode, over every hunter of the episode
  (founders and all their descendants together, as in continuous_traits); the run
  value is its mean over the run's finished episodes;
- checkpoints 250 and 500 on their own.

**What outcomes would mean, and the limits:** DESIGN.md, set in advance. Learning is
one shared policy per species, not individual lifetime learning, so a positive result
is not a classic Baldwin effect.

**Hard stop:** after the main runs, the experiment stops whatever the outcome.

### 1.5 Pilot result (2026-11-08): failed, the stage stops

The pilot ran from 06:53 to 09:28 without a crash. It fails its second criterion:
hunters are alive at the end of none of the 50 evaluated episodes, in every arm.

| Arm | Iteration | Coexisting (of 50) | Median hunter extinction (physics step) | A (adults per episode) | S | Hunter births in training (all episodes) |
| --- | --- | --- | --- | --- | --- | --- |
| L | 250 | 0 | 1351 | 0.34 | 0.013 | 27 in 5441 |
| L | 500 | 0 | 1285 | 0.14 | 0.005 | 78 in 10484 |
| B | 250 | 0 | 1487 | 0.14 | 0.004 | 50 in 4848 |
| B | 500 | 0 | 1425 | 0.14 | 0.005 | 94 in 9610 |
| E | 250 | 0 | 909 | 0.12 | 0.007 | 6 in 8027 |
| E | 500 | 0 | 826 | 0.16 | 0.012 | 14 in 16073 |

Results files: `results/pilot_<arm>_iter<250|500>_{eval,coop}.json`.

- **For comparison:** in deer_stag_hunt, all three runs coexisted in 50 of 50 episodes
  at iterations 250 and 500.
- **Learning signal:** the hunters' reward comes from births. Their mean episode return
  was above zero in 66 of L's 500 iterations and 85 of B's, never above 1.8. In
  deer_stag_hunt run_1 it was 230 by iteration 101.
- **Gene selection:** with under one hunter birth per hundred episodes, there was
  almost no selection. Gene means stayed within 0.1 of zero. E's lineages reached
  generation 2 at most.

**Likely cause (not tested):** the frozen deer. In deer_stag_hunt, hunters and deer
started naive together. The deer's returns there were about 90 at iteration 1, so
early hunters could catch and breed, and that gave PPO a reward to learn from. Here
the deer are deer_stag_hunt's final, trained deer from the first episode (deer
return about 970 at iteration 1). Naive hunters starve before they breed, so they
get almost no reward. Arm L, which has the bias switched off and a world shown equal
to deer_stag_hunt's (tests/test_equivalence.py), fails in the same way. That points
at the design choice of frozen trained deer, not at the gene code. This is an
inference from the comparison; no run has isolated it.

**Per the pre-registration (§1.2), nothing is adjusted and re-run. The main runs are
not launched.**

## 2. Hunters and deer starting together: deer replayed (pre-registered 2026-11-10, before any code)

A new stage, prompted by the user's decision after section 1's stop, not a rerun of
section 1. Design and reasons: DESIGN_2.md (approved 2026-11-10 with every
recommended option). Everything not named here is as in section 1 and DESIGN.md.

### 2.1 World, deer and arms

- **World:** deer_stag_hunt's touch rule (`deer_catch_rule: "touch"`), instant grass.
- **Genes, β₀ = 3.4417, arms E, L and B:** as in section 1.1.
- **Deer, replayed:** run k of every arm follows the deer of deer_stag_hunt
  touch_((k − 1) mod 3) + 1 in training order. Training iteration t collects its
  samples with the deer of checkpoint D(t): iter_000010 for t = 1-19;
  iter_(10 · floor(t / 10)) for t = 20-999; the final checkpoint for t = 1000. The
  deer never learn; they change before sampling iterations 20, 30, ..., 990 and 1000,
  in the learner and every env runner at once.
- **Arm E:** each of its 14 environments switches to D(t) when its own physics steps
  pass 1/14 of the median total physics steps of the L and B runs of the same schedule
  at iteration t (saved table). E's results are **reported, not tested**.

### 2.2 Pilot (before the main runs)

- **Runs:** pilot_L1, pilot_L2, pilot_L3 (L on touch_1, touch_2, touch_3 deer;
  seeds 1-3), pilot_B1 (B on touch_1, seed 1), 500 iterations, two at a time; pilot_E1
  (E on touch_1, matched to pilot_L1 and pilot_B1) on the CPU. Pilot runs enter no
  test.
- **Pilot passes if,** in every one of the four L and B pilot runs:
  1. nothing crashes;
  2. at iterations 250 and 500, hunters and deer are both alive at the end of at least
     40 of 50 evaluated episodes;
  3. at iteration 500, hunter births per evaluated episode are at least 5
     (deer_stag_hunt touch_1-3 at iteration 500: 18-22).
- E is reported only.
- **If the pilot fails:** this stage stops. Nothing is tuned or re-run: no change to
  β, the schedule, the schedule-to-run allocation, D(t), the thresholds or any world
  setting, and no run is excluded.

### 2.3 Main runs

- **Runs:** L_k, B_k (k = 1-6, 1000 iterations, 14 env runners, seed k, deer
  touch_((k − 1) mod 3) + 1), and E_k matched to L_k's and B_k's schedule (median of
  the four L and B runs sharing that schedule).
- A run that crashes is rerun from scratch with the same seed, and the crash recorded;
  no other rerun.

### 2.4 Measures and tests, set before the main runs

- **Evaluation:** each checkpoint saved after iteration t is evaluated against D(t),
  50 episodes, seeds 0-49 in every arm (evaluate.py; coop_fair.py 20 episodes).
- **Measures:** A (adults caught per episode), S (adults caught ÷ (caught + failed
  adult attempts), pooled over a run's 150 episodes), C (supporting): section 1.4.
  Run value: mean over checkpoints 500, 750 and 1000 (S pooled). Estimand:
  performance against the schedule.
- **Test 1, B beats L on A and on S:** per run k, B_k − L_k; one-sided sign-flip
  permutation test on the mean difference over the 6 pairs (all 64 sign
  arrangements, p = share with a mean at least the observed), p < 0.05 for A and for
  S. Both must hold.
- **Test 2, B sustains coexistence:** in at least 5 of 6 B runs, hunters and deer both
  alive at the end of at least 45 of 50 episodes, averaged over the three
  checkpoints.
- **Verdict:** if both tests are met: "in this world, against a replayed schedule of
  deer, an innate attraction bias that evolves plus shared PPO learning catches more
  adults, and more per attempt, than learning alone." Conditional on the three deer
  schedules.
- **Also reported:** every L and B final checkpoint against each of the three final
  deer (touch_1-3, iteration 1000), 50 episodes, seeds 0-49; B − L per schedule; E on
  every measure; per-run values, bootstrap intervals, C, gene means, selection
  differentials against L's unexpressed genes, checkpoints 250 and 500 (section 1.4).
- **Hard stop:** after the main runs, the stage stops whatever the outcome.

**Clarification (2026-11-10, after Codex's code review, before any run of this
section):** in section 2.1, E's switch "at iteration t" means the moment L and B
switch to D(t): before sampling iteration t, so at their median total physics
steps at the *end of iteration t − 1* (from each run's `iterations.jsonl`). This
is the reading the design states (deer change "before sampling" iteration t);
the wording above could also be read as the total at the end of iteration t,
one iteration later. E's checkpoints at marks 250, 500, 750 and 1000 then hold
D(mark), checked when each is saved. No other change.

### 2.5 Pilot result (2026-11-10): failed, the stage stops

The pilot ran from 04:11 to 06:50 without a crash. It fails its second criterion in
all four L and B runs: hunters and deer coexist in fewer than 40 of 50 evaluated
episodes. The third criterion is met everywhere (hunter births 19-22 per episode at
iteration 500, bar 5). This time the hunters do not starve; the deer are **eaten out**:
every non-coexisting episode ends with the deer extinct and hunters alive.

| Run | Iteration | Coexisting (of 50) | Median deer extinction (physics step) | A (adults per episode) | S | Hunter births per episode |
| --- | --- | --- | --- | --- | --- | --- |
| pilot_L1 | 250 | 18 | 2475 | 68.6 | 0.131 | 17.0 |
| pilot_L1 | 500 | 22 | 4139 | 84.2 | 0.139 | 19.0 |
| pilot_L2 | 250 | 30 | 3394 | 87.1 | 0.115 | 20.9 |
| pilot_L2 | 500 | 27 | 3493 | 83.9 | 0.131 | 19.3 |
| pilot_L3 | 250 | 32 | 3853 | 87.7 | 0.130 | 19.0 |
| pilot_L3 | 500 | 38 | 3373 | 94.5 | 0.131 | 20.4 |
| pilot_B1 | 250 | 15 | 2970 | 61.1 | 0.127 | 15.3 |
| pilot_B1 | 500 | 33 | 3630 | 85.0 | 0.137 | 21.5 |
| pilot_E1 (reported only) | 250 | 50 | - | 5.9 | 0.022 | 16.5 |
| pilot_E1 (reported only) | 500 | 49 (1 hunters extinct) | - | 1.4 | 0.008 | 12.8 |

Results files: `results/pilot_<L1|L2|L3|B1|E1>_iter<250|500>_{eval,coop}.json`,
`results/pilot2_check.txt`.

- **For comparison:** deer_stag_hunt touch_1-3 at iteration 500, against the same
  deer: 50 of 50 coexisting, A 25.7 / 28.4 / 13.9, S 0.026 / 0.028 / 0.017, hunter
  births 18-22. The L and B hunters here catch 3-6 times as many adults per episode,
  at about 5 times the success per attempt, against the very deer checkpoints the
  source hunters met.
- **The world and PPO settings** of pilot_L1 equal those of touch_1 apart from the
  genes (unexpressed in L), the seed and the run length (checked from the saved
  settings).
- **The failure the design named:** DESIGN_2.md said that if the hunters here learned
  faster than deer_stag_hunt's, the replayed deer "may be too weak and be eaten out".
  That is what happened.

**Likely cause (not tested):** the deer do not respond to these hunters. In
deer_stag_hunt each deer checkpoint was the deer's answer to the hunters of that
moment, and it kept changing as they did. Here each checkpoint stays put for ten
iterations against hunters it never trained against, and the hunters appear to find
ways of catching adults that those deer never had to answer. This is the familiar
weakness of training against frozen opponents. It is an inference from the
comparison above; no run has isolated it, and a hidden difference between the two
modules' training paths has not been ruled out beyond the settings check.

**E** coexists (49-50 of 50) but catches few adults, as its fixed cruise rule does
not learn; it is reported only.

**Per the pre-registration (§2.2), nothing is tuned or re-run. The main runs are not
launched.** Both deer designs tried so far fail in opposite directions: frozen
trained deer starve naive hunters (§1.5); replayed deer that cannot respond are eaten
out (§2.5). Deer that learn alongside the hunters in every arm (DESIGN_2.md's first
alternative, real coevolution but harder to attribute) would be a new design and a
new user decision.
