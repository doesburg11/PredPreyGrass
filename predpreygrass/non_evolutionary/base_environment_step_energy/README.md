# Predator-Prey-Grass base environment: homeostatic + move step energy

This environment starts as a copy of [`base_environment`](../base_environment). The behavioral change is how the per-step energy tax is charged. **See [`RESULTS.md`](./RESULTS.md) for the full investigation log** -- the reasoning, the sweep results, a failed first design, and why the model below looks the way it does. This file just describes the current structure and how to run it.

## Purpose

`base_environment` charges `energy_loss_per_step_predator`/`_prey` every step, unconditionally, regardless of the action taken -- an agent that picks noop pays exactly the same tax as one that moves. `_get_movement_energy_cost` exists in `base_environment` as a hook for a movement-based cost but is stubbed to always return `0` -- dead code there.

This module implements that hook for real, as two independent, additive costs:

- **`homeostatic_energy_cost_per_step_predator`/`_prey`** -- charged every step, regardless of action. This models basal metabolic upkeep (breathing, thermoregulation, cellular maintenance): every real organism burns energy continuously just to stay alive, even at complete rest, so this can never be zero (see `RESULTS.md` section 6 for what went wrong when an earlier design set it to zero).
- **`move_energy_cost_per_step_predator`/`_prey`** -- charged *on top* of the homeostatic cost, only on steps where the agent's action isn't noop. Models the additional metabolic cost of locomotion.

So resting costs `homeostatic`, moving costs `homeostatic + move` -- always more expensive than resting, never free. Current defaults (`config_env.py`) set resting somewhat below `base_environment`'s original flat tax and moving somewhat above it:

| | resting (homeostatic only) | moving (homeostatic + move) | `base_environment`'s flat tax |
|---|---|---|---|
| predator | 0.10 | 0.20 | 0.15 |
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

**Not yet validated at training scale.** A prior design (a single unconditional-vs-move split of one fixed budget, controlled by a now-removed `--move-fraction` flag, with the flat/homeostatic side driven all the way to zero) looked sustainable in a 100-iteration probe but was shown by a full 500-iteration run to produce an accelerating predator-favoring collapse (prey population more than halved, 29% of episodes ending in prey extinction by the final 100 iterations) -- see `RESULTS.md` for the full story, including why 100 iterations wasn't long enough to catch it. The current independent-additive-cost model (this README) and its defaults are a redesign in response to that finding, chosen for both empirical and biological reasons (homeostatic cost can no longer be zero), but haven't themselves been run at training scale yet.

**To run**: `python -m predpreygrass.non_evolutionary.base_environment_step_energy.tune_ppo_base_environment_step_energy --seed 42 --max-iters 500`, using `config_env.py`'s shipped defaults, or override any of the four cost parameters directly:

```
--homeostatic-cost-predator FLOAT
--homeostatic-cost-prey FLOAT
--move-cost-predator FLOAT
--move-cost-prey FLOAT
```

Compare against a same-seed `base_environment` run the way `RESULTS.md` documents doing (episode-count-weighted pooling, not per-iteration means; `PPO_BASE_ENVIRONMENT_SEED42_2026-09-05_18-55-45` is an existing same-seed 1000-iteration control already on disk).
