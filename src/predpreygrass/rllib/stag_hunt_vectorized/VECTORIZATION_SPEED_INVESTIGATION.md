# Stag Hunt Vectorization Investigation

## Goal
Speed up the RLlib `PredPreyGrass` environment by vectorizing hot paths (especially observation building),
then verify the impact on training throughput.

## Key environment changes (vectorized path)
- Vectorized observation batching and reused it in `reset()` and `step()`.
- Vectorized time-step updates and agent movements while preserving the sequential occupancy rules.
- Cached position maps for O(1) lookup:
  - `predator_positions_by_xy`
  - `grass_positions_by_xy`
- Avoided rebuilding action/observation spaces on every reset (cached once).
- Optimized neighborhood checks (`_predators_in_moore_neighborhood`) with cached offsets + dict lookups.
- Added or fixed bookkeeping for position maps on spawn, move, and removal.

## Benchmark tooling
Script: `src/predpreygrass/rllib/stag_hunt_vectorized/tests/benchmark_observations.py`

Useful flags:
- `--active-multiplier`, `--active-type-1-predator`, `--active-type-1-prey`, `--active-type-2-prey`
- `--obs-range`, `--predator-obs-range`, `--prey-obs-range`
- `--grid-size`
- `--bench-steps`, `--bench-steps-noobs`
- `--profile-steps`, `--profile-sort`, `--profile-top`

## Micro-bench results (observation batch vs loop)
All results are from local runs and may vary. These were recorded after the vectorized observation path
was introduced and iterated on.

Default config (30 agents, obs range 9):
- batch: ~0.07 ms
- loop:  ~0.31 ms
- speedup: ~4.2x

Double active agents (60 agents):
- batch: ~0.10 ms
- loop:  ~0.61 ms
- speedup: ~6.1x

Obs range 13 (30 agents):
- batch: ~0.09 ms
- loop:  ~0.31 ms
- speedup: ~3.3x

Explicit 90 agents (30/30/30):
- batch: ~0.13 ms
- loop:  ~0.91 ms
- speedup: ~6.9x

Larger grid (40) + larger predator range (13):
- batch: ~0.08 ms
- loop:  ~0.32 ms
- speedup: ~3.9x

## Step benchmarks (end-to-end env step)
`--bench-steps 200` (30 agents):
- early in the work: ~2.33 ms/step (p50 ~2.67 ms)
- after vectorization/profiling fixes: ~0.45–0.60 ms/step (p50 ~0.36–0.54 ms)

`--bench-steps-noobs 200`:
- ~2.18 ms/step (no-observation variant) before deeper step optimizations

Net: step time improved by roughly ~4–5x from the initial post-obs-vectorization baseline.

## Profiling highlights and fixes
Profiler runs (`--profile-steps`) showed early time sinks in:
- Observation space building and Gym space validation.
- Per-reset space rebuilds.
- Python-level loops in movement and neighborhood checks.

Fixes:
- Cached spaces to avoid per-reset rebuilds.
- Rewrote observation building to operate on batched positions.
- Vectorized movement update and neighborhood checks.

## Training throughput checks (RLlib/Tune metrics)
Metrics used:
- `ray/tune/env_runners/throughput_since_last_reduce`
- `ray/tune/env_runners/num_env_steps_sampled`
- `ray/tune/timers/env_runner_sampling_timer`
- `ray/tune/timing/iter_minutes`
- `ray/tune/timing/avg_minutes_per_iter`
- `ray/tune/timing/total_hours_elapsed`

Findings so far:
- `env_runner_sampling_timer` is consistently lower with the vectorized env.
- Overall throughput and `iter_minutes` do not move much unless learner cost is reduced.
  The learner/optimizer dominates once sampling is faster.

## PPO config tweaks tested
In `config_ppo_gpu_stag_hunt_vectorized.py`:
- `num_epochs`: 20 → 10
- `num_envs_per_env_runner`: 3 → 4

This reduced learner cost and increased sampling pressure; still only modest throughput gains unless
combined with the vectorized env.

## Isolation experiments (env vs PPO config)
Two helper scripts were added to isolate the effects:
- `tune_ppo_oldenv_newppo.py`: old env + new PPO config
- `tune_ppo_newenv_oldppo.py`: new env + old PPO config

Run both for ~150–200 iterations and compare the same window (e.g., iters 50–200) using the
metrics above to attribute the speedup between env vs PPO config.

## Other operational changes
- Evaluation was fully removed from `tune_ppo.py` to avoid eval overhead during speed testing.

## Summary
- Observation batching is ~3–7x faster than the looped version across tested scenarios.
- End-to-end `step()` improved by ~4–5x from the early post-obs-vectorization baseline.
- Training throughput improvement is gated by learner cost; reducing epochs or batch complexity
  is required to see wall-clock gains from the faster env.
