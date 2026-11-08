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
