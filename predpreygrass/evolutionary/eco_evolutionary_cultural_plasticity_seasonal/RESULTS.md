# Training Analysis — eco_evolutionary_cultural_plasticity_seasonal

**Status: smoke run, an extra short run, and a single-seed pilot have
completed; the first real replication seed (42) is in progress
(833/1000 iterations as of 2026-08-08 20:21). No results have been
analyzed yet — `analyze_replication_seeds.py` has not been run.** See
`README.md` for the target-dialect mechanism (a seasonal square-wave
cycle decides which dialect currently earns the coordination bonus,
instead of the local majority) and why it's a better-targeted test of
Rogers' Paradox than Trial 8 (`eco_evolutionary_cultural_plasticity`),
whose own `RESULTS.md` (§3-4) documents the flat, null result this
module exists to route around.

---

## 1. What's been done so far

- **Implementation.** Cloned from `eco_evolutionary_cultural_plasticity`
  (Trial 8); added `_current_target_dialect()` and changed
  `_dialect_match_bonus` to compare against it instead of
  `_local_majority_dialect`. Everything else (genome, cultural-learning
  update rule, satiation/sustainability mechanics, neutral-drift-control
  scaffold) is unchanged — see README's "What's new here vs. Trial 8".
- **Unit tests.** 29 tests passing: the 27 ported from Trial 8 (genome
  sampling/mutation, cultural-learning update rule, neutral-drift-control
  template selection, RLlib multi-agent contract tests, metrics builders),
  plus two new tests for `_current_target_dialect()`'s phase-cycling, plus
  the two Trial-8 coordination-bonus tests rewritten to encode the semantic
  change (matching the local majority but not the target grants no bonus,
  and vice versa) — the direct regression guard for this module's one
  behavioral change.

- **Training runs launched.** Four runs so far, none analyzed yet:
  - Smoke run (unseeded), 2026-08-07 17:15 — 20/20 iterations, completed.
  - Single-seed run (seed 1), 2026-08-07 17:30 — 300/300 iterations,
    completed.
  - Single-seed run (seed 2), 2026-08-07 19:40 — 150/150 iterations,
    completed.
  - Real replication seed 42, 2026-08-08 15:08 — **in progress**
    (833/1000 iterations as of 2026-08-08 20:21).

## 2. Not yet done

- **Confirm the phase-flip signature.** The smoke run's `dialect_match_rate`
  metrics went `nan` in `progress.csv` after iteration 3 for both species —
  not yet root-caused (could be a metrics-collection artifact rather than a
  real problem). Should be checked before trusting this run as validation
  that the mechanism is biting.
- **Finish real replication.** Seed 42 in progress; seeds 43/44 not yet
  started.
- **Neutral-control replication** (3 seeds) — not yet started.
- **Real vs. neutral-control comparison** on `plasticity` drift (criterion 3,
  via `analyze_replication_seeds.py`) — the actual Darwin-signal test this
  module exists to run; blocked on the above.
- **Parameter tuning.** `dialect_season_length_steps=25` is a first guess
  (Trial 8 episodes ran ~180-260 steps, so this gives ~2-2.5 full tours
  through all `n_dialects` per episode), not validated against real
  training data.

## 3. Next step

Let seed 42 finish, root-cause the `dialect_match_rate` `nan`s from the
smoke run, then launch seeds 43/44 (real) and the 3 neutral-control seeds
before running `analyze_replication_seeds.py`.
