# ERL Flagship (Trial 13) — the proximate-vs-ultimate reward question, in the richer ecology

## Why this module exists

Trial 12 (`eco_evolutionary_erl_baldwin`) found, at n=30 confirmed, that evolution's
discovered reward function (`genome.eval_weights`) diverges from the fitness
criterion it's selected for (`offspring_count`): agents consistently over-weight
`health_norm`/`energy_norm` relative to their near-zero correlation with realized
reproduction. That result lives in a small, abstract World-AL-style ecology (a
custom pure-Python simulator, hand-designed 7-channel ray-cast observation).

This module asks the same question — does an evolved reward function diverge from
fitness? — inside the project's flagship module
(`predpreygrass/non_evolutionary/base_environment`) instead: a richer, 25×25
predator/prey/grass ecology with a spatial image observation, normally trained via
RLlib PPO against a fixed sparse reward. Only **prey** get an evolved reward genome
here; predators keep flagship's existing behavior (a frozen, pretrained PPO
checkpoint run in inference mode — see "Predator handling" below).

## Architecture

Ports `eco_evolutionary_erl_baldwin`'s genome/REINFORCE machinery
(`genome.py`, `networks.py`) onto flagship's actual grid/energy/reproduction
mechanics, rather than reimplementing them:

- **Evaluation network** (`genome.eval_weights`, `eval_bias`): fixed for the
  agent's entire life, evolved only via reproduction. A linear map from an
  8-channel feature vector (`features.py`) to a scalar "goodness" value.
- **Action network** (`genome.action_weights`, `action_bias`): the genome's
  *initial* weights only. A live copy (`driver.PreyGenomeState.action_weights`)
  is adjusted every step by the same REINFORCE-style update as Trial 12
  (`networks.reinforce_update`), using the agent's own eval-network output as
  its intrinsic reward — no external reward signal.
- **Reproduction**: strictly Darwinian — always copies from `genome`, never from
  the live, post-learning `action_weights` (see
  `tests/test_genome_inheritance.py::test_offspring_genome_does_not_inherit_parents_learned_weights`).

flagship's own env (`PredPreyGrass`) is reused directly as a plain Python
simulator (`env.step(action_dict)` in a hand-written loop, `driver.py`) — no
`ray.init()`/`PPOConfig`/`Tuner` for Trial 13 itself. See "RLlib or not" below.

## Deviations from Trial 12 — read before trusting a result as directly comparable

1. **Asexual, mutation-only inheritance, not sexual crossover.** flagship's actual
   reproduction trigger (`predpreygrass_rllib_env.py:441-469`) is asexual/clonal —
   one parent's energy crosses a threshold, one child spawns adjacent. Trial 12's
   `_nearest_mate` + `genome.crossover` has no flagship equivalent and doesn't
   port; inventing new mate-search logic would mean reinventing flagship's
   reproduction trigger rather than hooking into it. `genome.mutate` (rate=0.05,
   std=0.05) is the only inheritance mechanism here.

2. **8 feature channels, not 7 — and two of Trial 12's channels are dropped.**
   `features.py`'s `[energy_norm, predator_dx, predator_dy, predator_proximity,
   food_dx, food_dy, food_proximity, local_grass_density]` echoes Trial 12's
   channel philosophy but isn't the same list: `in_tree` has no flagship
   equivalent (no shelter mechanic), and flagship prey have no `health` distinct
   from `energy` (no separate `health_norm`). This is a hand-reduction of the
   *same* raw window flagship's own CNN policy already receives (via
   `env._get_observation`, reused directly) — not additional information.

3. **Two required `config_env` overrides, both load-bearing, not cosmetic:**
   - `max_steps`: flagship's default (1000) truncates the "episode" and resets
     every agent to initial positions/counts. Overridden to an effectively
     unbounded value (10,000,000) so extinction is the only natural stopping
     condition, matching Trial 12's design.
   - `n_possible_prey`/`n_possible_predators`: flagship allocates agent IDs
     monotonically and **never recycles them on death**
     (`predpreygrass_rllib_env.py`'s `_next_prey_idx`/`_next_predator_idx`).
     Reproduction silently and permanently stops once cumulative births hit this
     cap (2000 by default) — invisible under flagship's own max_steps=1000
     (which resets the counter every episode), but a hard ceiling on TOTAL
     lifetime births once that reset is removed. Raised to 500,000 here.

4. **Predator handling: a frozen, pretrained PPO checkpoint, not a rule-based
   hunter.** Predators are inference-only — `predator_policy.py` loads an
   already-converged `predator_policy` RLModule checkpoint from a prior
   base_environment tournament run (`RLModule.from_checkpoint`, the same
   no-Ray-runtime pattern `master_tournament_matrix.py` already uses) and never
   receives gradient updates during a Trial 13 run. Chosen over hand-coding a
   rule-based hunter (Trial 12's `Carnivore`) because it reuses a known-good,
   already-calibrated adversary instead of open-ended new tuning with no
   reference point in this codebase.

5. **No RLlib training loop for Trial 13 itself ("RLlib or not").** Concurrent
   population is small; an 8×9 linear genome (~90 params) is the same scale
   network Trial 12 already runs fast in pure Python. RLlib's env-runner/learner
   abstractions assume one stable shared-weight policy across a whole collected
   batch before a synchronized gradient step — there's no clean way to express
   "spawn a new module with mutated weights mid-episode, for exactly one agent
   id" inside `MultiRLModuleSpec`/`policy_mapping_fn`. `RLModule.from_checkpoint`
   is used for exactly one thing: the frozen predator.

6. **Non-reproducible env-internal randomness.** flagship's own
   `_find_available_spawn_position` fallback path uses the bare global
   `np.random` module (not a seeded per-instance generator) — unlike this
   module's own `driver.rng`, which is fully seeded and checkpointed. A resumed
   run is not bit-for-bit reproducible against an uninterrupted one, even with
   identical `--seed`. Not fixed here — flagship's env is shared code, out of
   scope to modify for this module.

7. **Predator/prey founding-population overrides, a Stage-0 calibration fix, not
   a cosmetic default.** `initial_energy_predator=11.0` (was 5.0),
   `n_initial_active_prey=16` (was 8), `n_initial_active_predator=4` (was 6).
   See "Status" below and `config.py`'s own docstring for the full diagnosis —
   flagship's stock values left predators starving out within ~100-150 steps
   against genome-driven prey specifically (not against the checkpoint's own
   co-trained PPO prey, verified directly), because genome prey's untrained
   early foraging ramps population far slower than the PPO prey the predator
   was calibrated against, and predators only had a ~33-step energy runway to
   find a first meal.

## Status (2026-09-13): Stage 0 complete, predator-survival calibration resolved

**Stage 0 (smoke test) passed all its mechanics checks**, both via the unit test
suite (`tests/`, 23/23 passing) and real live runs: `env.step()` runs continuously
past the 1000-step boundary with no premature reset; reproduction/mutation/
lineage-logging/checkpoint-round-trip all work correctly on real data (verified
via `eval_checkpoint.py` against a saved checkpoint — distinct evolving lineages,
generations incrementing, genomes mutating parent-to-child as expected); the
`n_possible_prey` override is nowhere near exhausted at this scale.

**Predator-survival finding, diagnosed and fixed.** At flagship's stock values,
predators consistently starved out within ~100-150 steps against genome-driven
prey. Before adjusting anything, this was verified NOT to be a loading/inference
bug: the identical checkpoint, loaded via the identical code path, thrived
(6→13, then 6→19-21 under Trial 13's own config overrides) when pitted against
its own co-trained PPO `prey_policy` instead of genome-driven prey — ruling out a
broken predator and confirming flagship's real reproduction/energy mechanics work
as expected end-to-end.

Root cause: a predator only has `initial_energy_predator / energy_loss_per_step_predator`
≈ 33 steps of runway before starving with zero catches. flagship's PPO
`prey_policy` is a fully-trained forager that reproduces fast (8→32 prey by step
20), giving predators abundant targets almost immediately — the population
density the checkpoint's hunting behavior was implicitly calibrated against.
Genome-driven prey start as random, untrained linear policies with no learned
foraging yet, so their population ramps far slower early on; predators were
starving out before genome-prey density ever caught up.

Fix (now in `config.py`): raise `initial_energy_predator` to 11.0 — deliberately
kept strictly below `predator_creation_energy_threshold` (12.0) so predators
still must actually hunt to reproduce (15.0+ was tried and rejected: it let
founders reproduce for free at spawn with zero hunting, an artifact rather than a
fix) — combined with more founding prey (16) and fewer founding predators (4) for
better early encounter odds. Verified across 3 seeds at up to 800 steps: real,
sustained hunting/reproduction/predator-prey cycling for hundreds of steps (one
seed ran 700+ steps with predators cycling 1-12 and prey 12-46), a substantial
improvement over the ~150-step collapse. Eventual predator extinction in some
seeds at long horizons was still observed and is accepted as normal finite-
population stochastic dynamics (Trial 12's own conditions don't guarantee
eternal survival either) — not something further tuned away.

Ready for Stage 1 (single-seed pilot, 50,000-100,000 steps) — see the
Darwin/Baldwin Trial Log for status.

## Usage

```bash
# Stage 0: smoke test, mechanics only
python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.run_trial13_simulation \
    --steps 5000 --seed 1 --log-every 200

# Inspect a checkpoint standalone (no training loop)
python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.eval_checkpoint \
    <out_dir>/checkpoints/checkpoint_step_5000.pkl --steps 0

# Proximate-vs-ultimate reward analysis, once a real lineage_fitness.csv exists
python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.analyze_proximate_reward \
    <out_dir>/lineage_fitness.csv
```

Run the test suite: `pytest predpreygrass/evolutionary/eco_evolutionary_erl_flagship/tests/`
