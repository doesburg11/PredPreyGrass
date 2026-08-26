# Training Analysis — eco_evolutionary_cultural_plasticity_seasonal

**Status (corrected 2026-08-26 — this section was stale since 2026-08-08):**
smoke run, an extra short run, a single-seed pilot, and real replication
seed 42 all completed (seed 42 finished its full 1000/1000 iterations on
2026-08-08 — the "833/1000, in progress" note below was never updated after
that). Seeds 43/44 and the 3 neutral-control seeds were never launched; no
Mann-Whitney real-vs-control comparison exists for this module. Seed 42's
own `plasticity` trajectory was independently analyzed on 2026-08-26 via
Hunt (2006) model-fitting (see §4 below) — **flat, matching Trial 8's null
result.** Also found and fixed 2026-08-26: both training-launch scripts
(`tune_ppo_cultural_plasticity_seasonal.py` and its `_neutral_control`
counterpart) were writing experiment directories under the *plain* Trial 8
module's name (missing `_SEASONAL`), not this module's own — seed 42's data
above lives under that misnamed directory as a result; the launch scripts
and `analyze_replication_seeds.py`/`resume_training_...py` are fixed, but
the existing seed-42 directory itself was left as-is (not renamed). See
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
  - Real replication seed 42, 2026-08-08 15:08 — completed, full
    1000/1000 iterations (logged under the misnamed directory noted
    above: `PPO_ECO_EVOLUTION_CULTURAL_PLASTICITY_SEED42_2026-08-08_15-08-21`).

## 2. Not yet done

- **Confirm the phase-flip signature.** The smoke run's `dialect_match_rate`
  metrics went `nan` in `progress.csv` after iteration 3 for both species —
  not yet root-caused (could be a metrics-collection artifact rather than a
  real problem). Should be checked before trusting this run as validation
  that the mechanism is biting.
- **Seeds 43/44 and the 3 neutral-control seeds were never launched** — no
  Mann-Whitney real-vs-control comparison exists for this module (see §4
  for what the single completed seed shows via a method that doesn't need
  one).
- **Parameter tuning.** `dialect_season_length_steps=25` is a first guess
  (Trial 8 episodes ran ~180-260 steps, so this gives ~2-2.5 full tours
  through all `n_dialects` per episode), not validated against real
  training data.

## 3. Next step

If revisited: root-cause the `dialect_match_rate` `nan`s from the smoke run,
then launch seeds 43/44 (real) and the 3 neutral-control seeds (now that the
launch-script naming bug is fixed) before running `analyze_replication_seeds.py`
for a proper Mann-Whitney comparison.

## 4. Addendum (2026-08-26): Hunt (2006) model-fit on the one completed seed

No real-vs-control comparison is possible (§2), but seed 42's own
1000-generation `plasticity_mean` trajectory can still be checked directly
via `predpreygrass/evolutionary/model_selection.py`'s AICc model selection
(Stasis/URW/GRW), which needs only one run, not a control group. **URW wins
for both species**, `mstep` near zero and negatively signed for both
(predator -0.00003, prey -0.00001) — **flat, consistent with and
corroborating Trial 8's null result** (`eco_evolutionary_cultural_plasticity/RESULTS.md`
§3-4), the exact pattern this module's seasonal mechanism was designed to
test whether it could break out of. One seed only — not a substitute for
the real replication in §3, but the first actual analysis this module's
data has received.
