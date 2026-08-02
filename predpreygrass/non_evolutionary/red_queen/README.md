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
  original script's example run); `tune_ppo_red_queen.py` (below) produces
  RLlib Tune's standard zero-padded naming instead, so pass
  `--checkpoint-dir-template "checkpoint_{iter:06d}"` when pointing this
  script at a run it produced.

  Verified end-to-end against a throwaway 3-iteration CPU-only training run
  (not committed) — loads checkpoints correctly, aggregates across seeds,
  and produces the expected JSON structure.

## RLlib-compliance fix

This environment previously had the same two RLlib-compliance bugs found and
fixed elsewhere in this repo: a dying agent's terminal transition
(`terminated=True`, final reward, final observation) was silently dropped
before reaching RLlib, and newborn agent IDs were recycled within an episode
in a way that could conflate two unrelated individuals' trajectories into
one. Both are now fixed the same way as in `base_environment` and
`walls_occlusion` — terminating agents stay listed through the step they die
in, and newborn IDs (per predator/prey type, including mutation-driven type
switches) are assigned from monotonically increasing, never-reused counters.
`n_possible_type_1_predators`/`_2_predators`/`_1_prey`/`_2_prey` raised
accordingly across all 6 config presets that set them. Verified: RLlib
pre-check passes on `config_env_eval.py` and `config_env_train.py`, zero ID
reuse across 3 seeds with 110 real deaths tracked correctly.

**Unrelated bug also fixed**: `_handle_prey_reproduction`'s mutation check
read `self.rng.random() < self.mutation_rate_predator` — prey type-switching
was governed by the predator mutation-rate config value, not the prey one.
Corrected to `self.mutation_rate_prey`. Not an RLlib-compliance issue, but
found while reading this code for the fix above, and it directly affects the
"competing prey types" experiment's mutation dynamics.

## Training script

**`tune_ppo_red_queen.py`** (added — this module previously had none). Trains
with `config_env_eval.py` specifically (not `config_env_train.py`): that
config disables `type_2` entirely, so exactly two policies get created —
`type_1_predator` and `type_1_prey` — matching what both evaluation scripts
above expect. Follows the same structure as `walls_occlusion`'s and
`base_environment`'s training scripts (own `utils/networks.py` for
auto-sized conv nets per obs-window size, own `utils/episode_return_callback.py`
for per-group reward/timing logging), auto-selects
`config/config_ppo_gpu_default.py` (0.5 GPU request) or `config/config_ppo_cpu.py`
based on CPU count, and checkpoints every 10 iterations (100 kept) under
`~/ray_results/PPO_RED_QUEEN_<timestamp>/`.

```bash
python -m predpreygrass.non_evolutionary.red_queen.tune_ppo_red_queen
```

Verified end-to-end (not just written): ran a throwaway CPU-only,
1-iteration version of this exact training logic (single env runner, forced
`config_ppo_cpu.py`) and confirmed it produces exactly the two expected
policies and a valid checkpoint. Deliberately did not run the real
`config_ppo_gpu_default.py` path or a multi-iteration run, since
`eco_evolutionary_nuptial_gift`'s replication run was already using the GPU
at the time — confirmed unaffected throughout (same PID, same GPU memory
footprint, before and after). A first real attempt at concurrent env runners
(2 runners) hit a transient Ray worker crash unrelated to resource
contention (system had ample free RAM/CPU); a single-runner retry completed
cleanly, so this is flagged as a possible flake to watch for on a real run,
not a confirmed bug.

## What's still needed for a real result

No red_queen checkpoints exist on disk yet — the run the original evaluation
script pointed at is gone, and the new training script hasn't been run for
real. To actually produce a Red Queen result: run `tune_ppo_red_queen.py` to
completion (real GPU time, currently blocked by the concurrent
`eco_evolutionary_nuptial_gift` job), then run
`evaluate_red_queen_freeze_multi_seed.py` against several checkpoint
iterations spanning the training curve (e.g. every 100-300 iterations) with
`--checkpoint-dir-template "checkpoint_{iter:06d}"`.
