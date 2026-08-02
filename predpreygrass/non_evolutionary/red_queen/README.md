# Red Queen

Tests the **Red Queen Hypothesis** — that in a coevolutionary arms race, an
agent has to keep adapting just to maintain its *relative* fitness against
its opponent, not to improve some absolute score ("running to stay in
place"). Built on the same `type_1`/`type_2` structure as `walls_occlusion`
(no walls here), with no genetic mutation — this is `non_evolutionary/`, so
any adaptation is pure RL policy learning, not trait inheritance.

## Two things this module supports

1. **Competing prey types under one shared predator.** `type_2_predator` is
   disabled by default (`n_possible_type_2_predators=0`) so there's a single
   predator population, but two prey types coexist with a real trait
   difference: `type_1_action_range=3` (slower, tighter movement) vs.
   `type_2_action_range=5` (faster, wider movement). Both are predated on by
   the same shared predator policy, so training can show which prey "design"
   wins under identical predation pressure.

2. **The freeze/unfreeze evaluation harness** — the actual hypothesis test.
   Takes two checkpoints from one co-training run (an "early" and a "late"
   stage) and runs 4 matchups by freezing each side's policy at a chosen
   checkpoint:
   - `frozen_prey`: predator=late, prey=early (mismatched, predator-advantaged)
   - `frozen_predator`: predator=early, prey=late (mismatched, prey-advantaged)
   - `static_early`: both at the early checkpoint (matched control)
   - `static_late`: both at the late checkpoint (matched control)

   If the Red Queen dynamic holds, the mismatched pairs should show one side
   clearly dominating, while the matched pairs stay comparably balanced
   regardless of which absolute training stage they're matched at — because
   what determines outcome is *relative* training stage, not raw iteration
   count.

## Evaluation scripts

- **`evaluate_red_queen_freeze_type_1_only.py`** (original): runs each of the
  4 conditions exactly once (`seed=42`, one episode) against a single
  hardcoded checkpoint pair. This is enough to sanity-check the mechanism
  works, but not enough to actually support a claim about Red Queen dynamics
  — one single-episode point estimate per condition can't distinguish a real
  signal from that run's noise, and it only ever checks one arbitrarily
  chosen (early, late) split. Its hardcoded checkpoint path
  (`~/Dropbox/02_marl_results/predpreygrass_results/ray_results/PPO_2025-07-27_23-54-21`)
  no longer exists on disk.

- **`evaluate_red_queen_freeze_multi_seed.py`** (added to address the above):
  same 4-condition structure, but every condition is run across multiple
  seeds (mean ± std reported, not a single number), and every consecutive
  pair from a *list* of checkpoint iterations is evaluated — not just one
  hardcoded split — so you can see whether the pattern holds consistently
  across training. Also caches loaded `RLModule`s across seeds/conditions
  that reuse the same checkpoint, and writes full per-seed + aggregated
  results to JSON for later analysis.

  ```bash
  python -m predpreygrass.non_evolutionary.red_queen.evaluate_red_queen_freeze_multi_seed \
      --base-path /path/to/ray_results/PPO_RUN_NAME \
      --checkpoint-iters 300 600 1000 \
      --seeds 0 1 2 3 4 \
      --max-steps 1000 \
      --out red_queen_results.json
  ```

  Checkpoint directory naming defaults to `checkpoint_iter_{N}` (matching the
  original script's example run); pass
  `--checkpoint-dir-template "checkpoint_{iter:06d}"` if your run uses
  RLlib Tune's standard zero-padded naming instead.

  Verified end-to-end against a throwaway 3-iteration CPU-only training run
  (not committed) — loads checkpoints correctly, aggregates across seeds,
  and produces the expected JSON structure. Not verified against a real,
  fully-trained run, since no red_queen training run currently exists on
  disk (see below).

## What's still missing

There is currently **no training script** (`tune_ppo.py`) for this module,
and no existing red_queen checkpoints anywhere on disk — the run the original
evaluation script pointed at is gone. To actually produce a real result with
either evaluation script, a training run needs to exist first (checkpointed
at several iterations spanning the training curve, e.g. every 100-300
iterations). That's a real, multi-hour GPU commitment and wasn't done as
part of this fix, since another training job (`eco_evolutionary_nuptial_gift`
replication) was already using the GPU at the time.
