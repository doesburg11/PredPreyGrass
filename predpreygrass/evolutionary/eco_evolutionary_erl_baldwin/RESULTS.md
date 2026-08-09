# Training Analysis — eco_evolutionary_erl_baldwin

**Status: implemented, tested, two performance fixes applied, 15-seed survival screen done,
a long run (seed 1, up to 1M steps) in progress.** See `README.md` for the Ackley & Littman
(1991) precedent this module replicates (with documented adaptations) and why it's a
structurally different test than Trials 1-9 or the positive control.

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

## 2. Performance fixes (found while screening seeds)

The smoke run's ~240 steps/sec turned out not to hold once population grew past ~100
agents. Profiling a slow seed found two O(agents²)-per-step hotspots, both fixed
2026-08-09:
- `_observe` rebuilt the full prey/predator position sets from scratch for *every
  individual agent's* observation each step. Fixed: built once per step instead (a
  documented simplification -- agents now sense positions as of the start of the step,
  not a live view updated by earlier-acting agents within the same step; eating/death
  still use fully live, current positions regardless).
- `_try_eat` linearly scanned the *entire* agent list for every predator, every step, to
  check for a co-located prey. Fixed: an O(1) per-step `row,col -> prey` dict, kept in
  sync as prey move or get eaten during the step.
- Profiling also found `rng.choice`'s generic validation overhead dominated
  `sample_action` at this (tiny, 5-action) scale. Replaced with a direct
  cumulative-probability draw.
- Net effect: ~2.3x speedup on a representative seed (232 vs. ~100 steps/sec at a
  ~230-agent population size). All 12 unit tests still pass unchanged.

## 3. Survival screen (15 seeds, up to 100k steps each, 2026-08-09)

Following the paper's own comparative-study spirit (run many random initial
populations, most die quickly, a minority survive far longer) rather than hand-tuning
parameters to force artificial stability:

| seed | extinction step | which species died |
|---|---|---|
| 3, 9 | 104, 91 | predator (near-instant) |
| 7, 11 | 722, 389 | prey / predator |
| 4, 5, 6, 13 | 2134, 1750, 2369, 1659 | mixed |
| 8, 12, 14, 15 | 2466, 4711, 3462, 3633 | mixed |
| 2, 10 | 12877, 12682 | predator (prey overran) |
| **1** | **not yet extinct** | still running |

Median extinction ~2,400 steps; two seeds (2, 10) reached ~12,800; **seed 1 is a clear
outlier**, surviving past the 90s screening cutoff twice. This distribution (most die
young, a minority survive far longer) is qualitatively the same shape the paper itself
reports (only ~18% of *their* 100 random populations reached even 10,000 steps) --
read as consistent with the mechanism working as expected, not as evidence of a bug
worth re-tuning away.

## 4. Long run in progress

Seed 1, launched 2026-08-09, `--steps 1000000 --log-every 2000 --constraint-window 10000`,
background process, output at `~/erl_results/ERL_BALDWIN_long_seed1/`. Not yet analyzed.

## 5. Still not done

- Reading the functional-constraint signature once the long run has enough data: does
  `action_site_change_rate` drop below `eval_site_change_rate` over generations (the
  direct Baldwin-Effect signature)?
- Multiple long-running seeds, not just seed 1 -- one survivor isn't enough to trust a
  pattern as real rather than this-seed-specific.
- No comparison against evolution-alone / learning-alone / no-adaptation controls yet --
  the paper's actual headline result was the *comparison* between ERL and these three
  degraded variants (100 seeds each), not ERL survival in isolation. Not built yet.
