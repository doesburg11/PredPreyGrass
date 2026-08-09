# Training Analysis — eco_evolutionary_erl_baldwin

**Status: implemented, tested, one short smoke run complete. No conclusion possible yet --
the smoke run is far too short to say anything about genetic assimilation.** See `README.md`
for the Ackley & Littman (1991) precedent this module replicates (with documented
adaptations) and why it's a structurally different test than Trials 1-9 or the positive
control.

---

## 1. What's been done so far

- **Implementation.** `genome.py` (per-agent genome, mutation, crossover), `networks.py`
  (evaluation/action network forward pass, local REINFORCE-style reinforcement update),
  `world.py` (self-contained predator-prey-grass simulator, no RLlib/PPO/Ray),
  `metrics.py` (functional-constraint tracker + CSV logger), `run_erl_simulation.py`
  (CLI entry point).
- **Unit tests.** 12/12 passing, including the critical correctness property flagged during
  design: `test_offspring_genome_does_not_inherit_parents_learned_weights` directly asserts
  that an agent's learned, post-lifetime action-network weights are never copied into
  offspring -- only the untouched genome record is (Darwinian, not Lamarckian; see README
  for why this matters).
- **Smoke run** (2026-08-09, seed=41, `--steps 20000 --log-every 1000
  --constraint-window 2000`): ran 9,830 steps at ~241 steps/sec (40.7s wall time) before
  predator extinction. No crashes; population dynamics show real predator-prey oscillation
  (predator count: 75→86→42→87→15→34→46→19→61 across the run; prey: 202→28→44→161→26→28→17→76→9
  at the same checkpoints) rather than a degenerate/frozen population. Extinction this early
  is expected and consistent with Ackley & Littman's own observation that "most initial agent
  populations die out quite quickly" -- not evidence of a bug.
- Genome-level stats (`{species}_eval_weight_absmean`, `_action_weight_absmean`) stayed
  roughly flat over the run (predator eval ~0.40-0.43, action ~0.38-0.40; prey eval
  ~0.34-0.38, action ~0.39-0.40) -- expected at this timescale; the paper's own clearest
  genetic-assimilation results took ~3 million steps to appear.
- Functional-constraint rates (`{species}_eval_site_change_rate` /
  `_action_site_change_rate`) were logged successfully and are in a sane range
  (predator: eval ~0.004-0.007, action ~0.005-0.007, no separation yet; prey: eval rate
  dropped from ~0.05 to ~0.007 over the run, action stayed ~0.007-0.009) -- too early and
  too few reproduction events (population maxed around 87) to read a genetic-assimilation
  signature into this; needs a much longer run.

## 2. Not yet done

- **A real, long run.** The paper's clearest genetic-assimilation evidence took ~3 million
  steps within a population that survived to that point; their long-term case study ran to
  ~9 million steps. At ~240 steps/sec single-threaded, 1 million steps is roughly 70 minutes
  -- feasible, but not yet attempted here. Also worth first tuning initial survival odds
  (learning rates, founder weight std, energy economics) given most short runs are expected
  to go extinct early, per the paper's own finding.
- **Multiple seeds.** One smoke run says nothing about whether *any* seed survives long
  enough to show assimilation; the paper's own comparative study ran 100 random initial
  populations and found only ~18% reached even 10,000 steps.
- **Reading the functional-constraint signature properly** once a long run exists: does
  `action_site_change_rate` drop below `eval_site_change_rate` over generations (the direct
  Baldwin-Effect signature), and does that happen for food-seeking behavior specifically
  (the paper's clearest case) as opposed to danger-avoidance (where they found "shielding"
  instead)?
- **No comparison against evolution-alone / learning-alone / no-adaptation controls yet**
  -- the paper's actual headline result was the *comparison* between ERL and these three
  degraded variants (100 seeds each), not ERL survival in isolation. Not built yet.

## 3. Next step

Tune initial survival odds with a few short (~50k-step) exploratory runs across seeds to
find parameters where populations reliably survive past ~100k steps, then launch a longer
run (multiple seeds, ~1M+ steps) and read the functional-constraint trajectories.
