# Direct Reciprocity

This module tests cooperation without coordination under necessity.

## Core idea

- Every prey is solo-catchable by a single predator.
- Predators still learn only from reproduction rewards.
- A predator gets a second action component: `share_food`.
- After a successful solo capture, the predator may voluntarily transfer a fixed fraction of the prey energy to a nearby predator.
- Sharing is immediately costly to the sharer and individually unnecessary.

## Reciprocity mechanism

- Predators maintain private, partner-specific trust values.
- If predator `A` shares with predator `B`, then `B`'s trust in `A` increases.
- If `A` has a sharing opportunity and refuses, nearby predators decrease their trust in `A`.
- Predator observations can include a trust channel showing how much the focal predator trusts nearby predators.

This is intended to isolate direct reciprocity:

`costly help now -> greater probability of costly help returned later`

## Important difference from `shared_prey`

- `shared_prey`: multiple predators can pool energy to catch prey.
- `direct_reciprocity`: capture is individual; cooperation happens after capture through voluntary energy sharing.

So the environment no longer studies "cooperate because prey is too strong", but "cooperate because help may be returned later".

## Main files

- `predpreygrass_rllib_env.py`: environment implementation
- `config/config_env_direct_reciprocity.py`: default environment config
- `tune_ppo.py`: PPO training entrypoint
- `random_policy.py`: rollout/debug script

## Key config knobs

- `share_fraction`
- `trust_enabled`
- `include_trust_channel`
- `trust_positive_delta`
- `trust_negative_delta`
- `trust_decay`

## Minimal hypothesis

If direct reciprocity is doing real work, predators should share more often with partners that shared in the past than with partners that previously refused.
