# Predator-Prey-Grass: dense reward + reproduction bonus (additive)

This module exists to answer one specific question raised by comparing
[`base_environment_dense_rewards`](../base_environment_dense_rewards)
against
[`base_environment_sparse_rewards`](../base_environment_sparse_rewards):
once both were trained to completion (1000 iterations each, identical
resource config, both with the same RLlib-compliance fixes), sparse showed
roughly **2x** the reproduction rate of dense for both species, a more
balanced final predator:prey ratio, and zero extinction events across
tested seeds versus one extinction event out of three tested seeds for
dense.

The likely explanation: `base_environment_dense_rewards` uses "pure
replacement" reward -- the dense per-step energy delta *replaces* the
sparse reward entirely, so reproduction is a pure energy **cost** to the
parent with no compensating signal, unlike the sparse variant's explicit
`+10` bonus. This module tests that explanation directly by using an
**additive** reward: the same dense per-step energy delta, PLUS the flat
`+10` reproduction bonus layered on top for the parent, on the step it
reproduces. If this closes the gap with `base_environment_sparse_rewards`,
the loss of the reproduction incentive (not reward density itself) was the
cause. If it doesn't, that points more toward reward density itself, or
something else, being the driver.

## Reward mechanics

Identical to `base_environment_dense_rewards` (energy at end of step minus
energy at start of step, folding in decay/move/eat/reproduction-cost) with
one addition: a parent that reproduces this step also receives
`reproduction_reward_predator` / `reproduction_reward_prey` (default `10.0`
each, matching the sparse variant's bonus) added on top of its own net
energy delta for that step. The newborn still gets reward `0` on its spawn
step. Nothing else about the reward, environment mechanics, or the
RLlib-compliance fixes (deferred termination timing, never-reused agent
IDs) differs from `base_environment_dense_rewards`.

## Training

Run [`tune_ppo_base_environment_dense_rewards_additive.py`](./tune_ppo_base_environment_dense_rewards_additive.py)
and compare against both
[`base_environment_sparse_rewards`](../base_environment_sparse_rewards) and
[`base_environment_dense_rewards`](../base_environment_dense_rewards), with
matching resource configuration.
