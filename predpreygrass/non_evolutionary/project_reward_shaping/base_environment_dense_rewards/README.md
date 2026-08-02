# Predator-Prey-Grass: dense energy-delta reward

This module is a variant of [`base_environment`](../../base_environment) that
replaces the sparse, reproduction-only reward with a **dense, biologically
literal reward**: on every step, each agent's reward is exactly its own net
energy delta for that step (`energy_after - energy_before`), which folds in
metabolic decay, movement cost, eating (grass or prey), and reproduction cost
to the parent. There are no hand-designed reward-shaping constants — reward
*is* the organism's physiological energy balance, not an artificial event
pulse.

Everything else (grid/observation settings, energy thresholds, spawning
rules, satiation-free catch mechanics) is identical to `base_environment`, so
this module is meant to be trained head-to-head against
[`base_environment_sparse_rewards`](../base_environment_sparse_rewards) —
the sparse-reward sibling that carries the *same* RLlib-compliance fixes
(see below) applied to the original `base_environment` logic, so the only
intended difference between the two is reward density itself. (Plain
`base_environment` is kept untouched as the original historical reference
and is **not** the fair comparison partner for this module — it still has
the bugs described below.)

## Reward mechanics

- **Surviving / reproducing agents**: reward = energy at end of step minus
  energy at start of step. A parent that reproduces this step automatically
  has the offspring's initial energy debited from its own balance, so the
  reproduction cost is captured for free — no separate reproduction bonus is
  needed or applied.
- **Prey caught by a predator**: its own reward is `0 - energy_before` (its
  energy account goes to zero on death); the energy itself transfers to the
  predator, which shows up as part of the predator's own net delta that step.
- **Agents that starve** (energy depleted): reward = the (negative) energy
  value at removal minus energy at the start of the step.
- **Newborns**: reward 0 for the step they're spawned on (no `energy_before`
  to compare against — they didn't exist at the start of that step).

## RLlib-compliance fixes (also applied in `base_environment_sparse_rewards`)

Two bugs present in the original `base_environment` are fixed here:

1. **Termination-reporting timing**: `base_environment`'s output filter used
   `self.agents` *after* dying agents were already removed from it, so a
   dying agent's `terminated=True`, final reward, and final observation were
   silently dropped before ever reaching RLlib. Fixed by deferring removal
   from `self.agents` to the start of the *next* step, so a terminating
   agent stays listed through the step in which it dies (matching what
   RLlib's env-checker expects), and by not returning entries for agents
   that were only ever declared in `self.possible_agents` on truncation.
2. **Agent-ID reuse within an episode**: `base_environment` recycles freed
   ID slots (`predator_0`..`49`, `prey_0`..`49`) for newborns, which
   collides with RLlib's `MultiAgentEpisode` design (one continuous
   trajectory per agent-ID string per episode) — once fix #1 correctly
   reports a termination, RLlib hard-errors the moment a reused ID produces
   more data. Fixed by assigning newborn IDs from a monotonically
   increasing per-species counter that never repeats within an episode,
   backed by a much larger ID pool (`n_possible_predators`/`n_possible_prey`
   raised from 50 to 2000 — cheap for RLlib, which only uses this list to
   build a per-episode space dict once per reset).

Without fix #1, fix #2 silently conflates unrelated individuals' trajectories
into one fabricated continuous episode object instead of crashing — which is
what let `base_environment` run without erroring despite having both bugs.
Measured ID-reuse rate under `base_environment`'s default config: ~75% of all
births reuse a retired ID within the same episode, so this was not a rare
edge case.

## Training

Run [`tune_ppo_base_environment_dense_rewards.py`](./tune_ppo_base_environment_dense_rewards.py)
and compare results against
[`base_environment_sparse_rewards`](../base_environment_sparse_rewards)'s
equivalent tune script, with matching resource configuration so wall-clock
differences don't confound the comparison.
