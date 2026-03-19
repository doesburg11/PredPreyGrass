# Direct Reciprocity

This module tests cooperation without coordination under necessity.

## Core idea

- Every prey is solo-catchable by a single predator.
- Predators still learn only from reproduction rewards.
- A predator gets a second action component: `share_food`.
- After a successful solo capture, the predator may voluntarily transfer a fixed fraction of the prey energy to a nearby predator.
- Sharing is immediately costly to the sharer and individually unnecessary.
- Predator reproduction can be delayed by a configurable cooldown, allowing temporary energy surplus to accumulate between births.

## Reciprocity mechanism

- Predators maintain private, partner-specific trust values.
- When predator `A` has a sharing opportunity, it first selects a single target predator `B`.
- If `A` shares with `B`, then only `B`'s trust in `A` increases.
- If `A` refuses, then only that same selected target `B` decreases trust in `A`.
- Predator observations can include a trust channel showing how much the focal predator trusts nearby predators.

This keeps observability symmetric and dyadic:

- sharing is a private `A -> B` interaction
- refusal is also a private `A -> B` interaction
- uninvolved nearby predators do not update trust from that event

### Trust range and decay

- Trust is bounded to the interval `[0.0, 1.0]`.
- `0.5` is the neutral default.
- Values above `0.5` mean positive prior experience.
- Values below `0.5` mean negative prior experience or distrust.

The current implementation uses symmetric decay toward neutral:

- if trust is above `0.5`, decay lowers it toward `0.5`
- if trust is below `0.5`, decay raises it toward `0.5`

So old positive history is gradually forgotten, but old negative history is also gradually forgiven.

This means distrust is not permanent by default. If stronger memory is desired later, decay can be removed or made asymmetric.

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
- `predator_reproduction_cooldown_steps`
- `trust_enabled`
- `include_trust_channel`
- `trust_positive_delta`
- `trust_negative_delta`
- `trust_decay`

## Predator Reproduction Cooldown

- `predator_reproduction_cooldown_steps` blocks predator reproduction for a fixed number of steps after each successful birth.
- The default direct reciprocity config sets this to `10`.
- This creates a window where energy above the reproduction threshold cannot be converted into immediate offspring, making voluntary transfers less structurally dominated by instant reproduction.

## Reciprocity Metrics

Training now logs direct-reciprocity metrics via the episode callback:

- `share_events_total`
- `share_opportunities_total`
- `share_refusals_total`
- `share_decision_rate`
- `share_rate_when_prior_helper_available`
- `share_rate_when_no_prior_helper_available`
- `reciprocal_share_rate`
- `reciprocal_dyads`

Interpretation:

- If reciprocity is emerging, `share_rate_when_prior_helper_available` should exceed `share_rate_when_no_prior_helper_available`.
- `reciprocal_share_rate` measures how often a share goes to a predator that previously shared with the donor earlier in the episode.
- `reciprocal_dyads` counts predator pairs with two-way sharing in the same episode.

## Minimal hypothesis

If direct reciprocity is doing real work, predators should share more often with partners that shared in the past than with partners that previously refused.
