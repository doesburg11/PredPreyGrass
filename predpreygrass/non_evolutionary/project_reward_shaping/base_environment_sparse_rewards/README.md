# Predator-Prey-Grass: sparse reward (RLlib-compliance-fixed)

This module is a copy of [`base_environment`](../../base_environment) with its
reward logic **completely untouched** (sparse, reproduction-only: the only
nonzero reward anywhere is a flat `+10` bonus at the instant of successful
reproduction) plus two RLlib-compliance bug fixes applied. It exists purely
to be a fair comparison partner for
[`base_environment_dense_rewards`](../base_environment_dense_rewards) —
without these fixes, comparing the two reward schemes would be confounded by
whether agent identities are tracked correctly, not just by reward density.

`base_environment` itself is left completely alone as the original,
untouched historical reference — it is **not** the comparison partner for
`base_environment_dense_rewards`; this module is.

## The two fixes (identical to those in `base_environment_dense_rewards`)

1. **Termination-reporting timing**: `base_environment`'s output filter used
   `self.agents` *after* dying agents were already removed from it, so a
   dying agent's `terminated=True`, final reward, and final observation were
   silently dropped before ever reaching RLlib. Fixed by deferring removal
   from `self.agents` to the start of the *next* step, so a terminating
   agent stays listed through the step in which it dies.
2. **Agent-ID reuse within an episode**: `base_environment` recycles freed
   ID slots (`predator_0`..`49`, `prey_0`..`49`) for newborns, which
   collides with RLlib's `MultiAgentEpisode` design (one continuous
   trajectory per agent-ID string per episode) — once fix #1 correctly
   reports a termination, RLlib hard-errors the moment a reused ID produces
   more data. Fixed by assigning newborn IDs from a monotonically
   increasing per-species counter that never repeats within an episode,
   backed by a much larger ID pool (`n_possible_predators`/`n_possible_prey`
   raised from 50 to 2000).

Without fix #1, fix #2 silently conflates unrelated individuals' trajectories
into one fabricated continuous episode object instead of crashing — which is
what let `base_environment` run without erroring despite having both bugs.
Measured ID-reuse rate under this config: ~75% of all births reuse a retired
ID within the same episode, so this was not a rare edge case.

No reward values, thresholds, or mechanics were changed — every number in
`config_env.py` besides the ID pool size matches `base_environment` exactly.

## Training

Run [`tune_ppo_base_environment_sparse_rewards.py`](./tune_ppo_base_environment_sparse_rewards.py)
and compare results against
[`base_environment_dense_rewards`](../base_environment_dense_rewards)'s
equivalent tune script, with matching resource configuration so wall-clock
differences don't confound the comparison.
