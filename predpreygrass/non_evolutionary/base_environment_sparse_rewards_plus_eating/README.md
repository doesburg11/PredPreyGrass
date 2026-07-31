# Predator-Prey-Grass: sparse rewards + eating bonus

This module tests a more targeted follow-up question than
[`base_environment_dense_rewards_additive`](../base_environment_dense_rewards_additive)
did. That comparison found dense reward (even with the reproduction bonus
restored) underperformed sparse reward, likely because the continuous
per-step energy-delta signal adds noise into the same reward channel as the
reproduction event, diluting the one signal that matters most. This module
asks the more surgical version of the original question: does rewarding the
*eating* event specifically help, **without** reintroducing that continuous
noise source at all?

Same clean, discrete, event-based reward style as
[`base_environment_sparse_rewards`](../base_environment_sparse_rewards) — no
per-step energy delta, no decay/movement terms in the reward at all — just
one more event type rewarded (eating) in addition to reproduction.

## Reward mechanics

- Reproduction: `+10` to the parent (`reproduction_reward_predator`/`_prey`), same as
  `base_environment_sparse_rewards`.
- Eating: `+1` to a predator that catches prey (`reward_predator_catch_prey`);
  `+0.1` to a prey that eats grass (`reward_prey_eat_grass`).
- Everything else (`reward_predator_step`, `reward_prey_step`,
  `penalty_prey_caught`) stays at `0.0`, matching the sparse baseline.

### Why the eating reward is asymmetric (`+1` predator / `+0.1` prey), not flat `+1`/`+1`

Measured directly by running `base_environment_sparse_rewards`'s final
trained checkpoint (3 seeds, full 1000-step episodes): predators catch prey
roughly **4.4 times per reproduction**, but prey eat grass roughly **60.5
times per reproduction** — grass regrows slowly and gives a small amount per
visit, so prey need many more, smaller meals to reach their reproduction
threshold than predators need catches.

A flat `+1` for both would make prey's *total* eating reward per
reproduction cycle (`60.5 × 1 = 60.5`) six times larger than the
reproduction reward itself (`10.0`) — eating would dominate prey's incentive
structure rather than staying a secondary signal, undermining the same
"keep reproduction primary" property this whole comparison is trying to
preserve. `+1` predator / `+0.1` prey keeps each species' total eating
reward per cycle in a similar, clearly-subordinate range relative to its own
reproduction reward (predator: `4.4 × 1 = 4.4`; prey: `60.5 × 0.1 ≈ 6.05` —
both well under `10.0`).

## Training

Run [`tune_ppo_base_environment_sparse_rewards_plus_eating.py`](./tune_ppo_base_environment_sparse_rewards_plus_eating.py)
and compare against
[`base_environment_sparse_rewards`](../base_environment_sparse_rewards) (no
eating reward) and
[`base_environment_dense_rewards_additive`](../base_environment_dense_rewards_additive)
(continuous signal instead of a clean eating event), with matching resource
configuration.
