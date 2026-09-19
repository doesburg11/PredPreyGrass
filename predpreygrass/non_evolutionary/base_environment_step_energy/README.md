# Predator-Prey-Grass base environment: homeostatic + move step energy

This module is a **follow-up to plain `base_environment`**, where no move cost exists at all: `base_environment` charges `energy_loss_per_step_predator`/`_prey` every step, unconditionally, regardless of the action taken -- an agent that picks noop pays exactly the same tax as one that moves, and `_get_movement_energy_cost` sits in `base_environment` as a hook for a movement-based cost but is stubbed to always return `0` (dead code there). This module implements that hook for real, so that movement itself carries a cost `base_environment` never had.

**See [`RESULTS.md`](./RESULTS.md) for the full investigation log** -- the reasoning, a failed first design, and the sequence of runs that led to the current defaults below. This file just describes the current structure and how to run it.

## Purpose

The per-step tax is split into two independent, additive costs instead of `base_environment`'s single flat one:

- **`homeostatic_energy_cost_per_step_predator`/`_prey`** -- charged every step, regardless of action. Models basal metabolic upkeep (breathing, thermoregulation, cellular maintenance): every real organism burns energy continuously just to stay alive, even at complete rest, so this can never be zero (see `RESULTS.md` section 6 for what went wrong when an earlier design set it to zero -- predators could ambush prey for free, indefinitely, which no real animal can do).
- **`move_energy_cost_per_step_predator`/`_prey`** -- charged *on top* of the homeostatic cost, only on steps where the agent's action isn't noop. Models the additional metabolic cost of locomotion.

So resting costs `homeostatic`, moving costs `homeostatic + move` -- always more expensive than resting, never free, and never equal to `base_environment`'s single number since that number no longer exists as one quantity. Current defaults (`config_env.py`; see `RESULTS.md` sections 10-12 and 19):

| | resting (homeostatic only) | moving (homeostatic + move) | `base_environment`'s flat tax (for reference) |
|---|---|---|---|
| predator | 0.10 | 0.18 | 0.15 |
| prey | 0.035 | 0.07 | 0.05 |

**Why this matters**: charging more for movement than for standing still introduces a real explore/exploit tradeoff -- an agent can choose to wait cheaply versus actively forage/hunt at a cost -- which is a more biologically realistic energy economy than `base_environment`'s flat, action-independent tax, and a stronger test of whether the sparse reproduction-only reward is still enough to sustain the ecosystem under that economy.

## Design choice: action-based, not displacement-based

The move cost is charged based on the **action chosen**, not on whether the agent's position actually changed. An agent that picks a directional action but is blocked (target cell occupied, so it stays in place -- see `_get_move`) still pays the move cost. This models the cost of *attempting* to move (effort expended) rather than the cost of *displacement*.

## Current baseline (inherited from `base_environment`, unchanged)

- Predators, prey, and grass are randomly placed in a gridworld at reset.
- Predators and prey are learning agents with separate RLlib policies.
- Grass is a non-learning environment resource.
- Agents observe only a local window around their position.
- Prey gain energy by eating grass; predators gain energy by catching prey.
- Predators and prey reproduce asexually once their energy crosses the configured threshold.
- New agents spawn near their parent.
- Rewards are sparse by default: reproduction is rewarded, while eating, catching, step, and death rewards can be configured in [`config_env.py`](./config_env.py).
- Training uses [`tune_ppo_base_environment_step_energy.py`](./tune_ppo_base_environment_step_energy.py).
- Interactive evaluation uses [`evaluate_ppo_from_checkpoint_debug.py`](./evaluate_ppo_from_checkpoint_debug.py).

## Status

**Partly validated at training scale (updated 2026-09-19).** Getting here took two wrong turns worth knowing about before trusting any future change to this module -- both documented in full in `RESULTS.md`:

1. An earlier design (a `move_fraction` parametrization, since removed from the code) let noop be completely free. It looked sustainable in a 100-iteration probe, then a full 500-iteration run showed it causes an accelerating predator-favoring collapse (prey population more than halved, 29% of episodes ending in prey extinction by the end) -- because a predator could ambush for zero energy cost indefinitely, which no real organism can do. Fixed by making the homeostatic cost structurally non-zero (`RESULTS.md` §5-9).
2. The first version of the current additive model (predator move cost 0.10, same as homeostatic) avoided that collapse but settled at a predator population well below `base_environment`'s own equilibrium (`RESULTS.md` §10).

The current defaults (predator move cost eased to 0.08) fixed that: a full 500-iteration run held a stable, low-extinction equilibrium (~12-14 predators, ~24-29 prey, 0% extinction, full-length episodes) for the final 400 iterations, tracking `base_environment`'s own equilibrium (~18-19 / ~19-25) closely and without drift (`RESULTS.md` §11-12).

**Replicated across six seeds (300-iteration runs, `RESULTS.md` §19):** it is sustainable in most seeds but not all. Compared with `base_environment` (~19 predators and 0% predator extinction in every seed), the defaults give fewer predators in all six seeds (mean ~10.6, range 3.8-14.1) and more prey (mean ~32 vs ~19). Predator extinction is 0-4% in five seeds, but in seed 45 the predator population never established (~4 predators and 34-48% extinction across all 300 iterations). So expect roughly one failed run in six until this is understood.

**Open**: why seed 45 failed and whether a slightly easier predator cost, longer training or more seeds change the failure rate; how long the stability of the other seeds holds beyond 500 iterations; and what causes the extra predator clustering (`RESULTS.md` §16-18).

**To run**: `python -m predpreygrass.non_evolutionary.base_environment_step_energy.tune_ppo_base_environment_step_energy --seed 42 --max-iters 500`, using `config_env.py`'s shipped defaults, or override any of the four cost parameters directly:

```
--homeostatic-cost-predator FLOAT
--homeostatic-cost-prey FLOAT
--move-cost-predator FLOAT
--move-cost-prey FLOAT
```

Compare against a same-seed `base_environment` run the way `RESULTS.md` documents doing (episode-count-weighted pooling, not per-iteration means; `PPO_BASE_ENVIRONMENT_SEED42_2026-09-05_18-55-45` is an existing same-seed 1000-iteration control already on disk).
