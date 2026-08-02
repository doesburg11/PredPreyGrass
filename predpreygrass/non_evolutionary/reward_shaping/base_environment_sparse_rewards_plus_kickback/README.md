# Predator-Prey-Grass: sparse rewards + kick-back bonus

Same clean, event-based sparse reward as
[`base_environment_sparse_rewards`](../base_environment_sparse_rewards), plus
one more discrete event type: a **kick-back bonus** — a second `+10` reward
to a grandparent, paid every time its own child successfully reproduces
(i.e. every time a grandchild is born). No continuous energy-delta signal,
same as every other module in this `base_environment_*` family.

## Why this specific idea, and why now

This module exists because the mechanism it tests — reward a parent when its
child reproduces — already exists elsewhere in this repo, in
[`kick_back_rewards`](../kick_back_rewards)
(`_reward_parent_for_child_reproduction`), verified RLlib-compliant and bug
free. That module was already tested at `kin_kick_back_reward = 4.0`
(roughly 0.4x the `10.0` reproduction reward) and found no benefit. This
module tests the same mechanism at full `1:1` weight (`kickback_reward = 10.0`,
equal to the reproduction reward itself), reimplemented in this repo's
single-predator/single-prey-type `base_environment_*` family so it's
directly comparable to the other four runs already completed here
(`base_environment_sparse_rewards`, `base_environment_dense_rewards`,
`base_environment_dense_rewards_additive`,
`base_environment_sparse_rewards_plus_eating`) rather than
`kick_back_rewards`' more complex two-type structure.

## Reward mechanics

- Reproduction: `+10` to the parent, same as `base_environment_sparse_rewards`.
- Kick-back: `+10` to the **grandparent**, the moment its child reproduces —
  i.e. `agent_parent[reproducing_agent]`, looked up at the instant
  `reproducing_agent` itself produces a new offspring. Fires once per
  grandchild (not capped at one per lineage — a grandparent with three
  grandchildren born across the episode collects the bonus three times).
  Only paid if the grandparent is still alive at that moment: RLlib cannot
  deliver a new reward to an agent that has already been marked
  `terminated=True`, so a grandparent that dies before its child reproduces
  simply forfeits that credit. This is the same constraint that applies to
  `kick_back_rewards`' equivalent mechanism.
- Initial (non-reproduced) agents have no recorded parent, so they can never
  trigger a kickback for anyone.
- Everything else (`reward_predator_step`, `reward_prey_step`,
  `reward_predator_catch_prey`, `reward_prey_eat_grass`,
  `penalty_prey_caught`) stays at `0.0`, matching the sparse baseline.

## Training

Run [`tune_ppo_base_environment_sparse_rewards_plus_kickback.py`](./tune_ppo_base_environment_sparse_rewards_plus_kickback.py)
and compare against
[`base_environment_sparse_rewards`](../base_environment_sparse_rewards) (no
kickback) with matching resource configuration. Worth also revisiting
[`kick_back_rewards`](../kick_back_rewards)'s own historical results at
magnitude `4.0` if this run shows a different outcome at `10.0`.

## Result

Trained the full 1000 iterations (14.44h, completed 2026-08-01). Final
checkpoint evaluated the same way as every other module in this family (3
seeds, full 1000-step episodes, deterministic actions, births counted via
each species' monotonic newborn-ID counter):

**117.0 predator / 562.3 prey births — 86% / 96% of the sparse baseline**,
zero extinctions across all 3 seeds, every episode ran the full 1000 steps.

This is the best-recovering of the four reward-shaping variants tested in
this family on *both* axes (ahead of `base_environment_sparse_rewards_plus_eating`'s
82%/94%), despite adding a secondary reward at a much larger magnitude —
but still short of pure sparse. The full ranking, the mechanistic
explanation for both of those findings (why kickback beats eating, and why
it still trails sparse), and citations live in
[`../README.md`](../README.md), sections 4 through 9.
