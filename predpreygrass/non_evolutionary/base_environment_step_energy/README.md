# Predator-Prey-Grass base environment: movement-conditional step energy

This environment starts as a copy of [`base_environment`](../base_environment). The only behavioral change is how the per-step energy tax is charged.

## Purpose

**What's different from `base_environment`**: in the baseline, `energy_loss_per_step_predator`/`_prey` is charged every step regardless of the action taken — an agent that picks the noop action (`action_to_move_tuple[4] == (0, 0)`) pays exactly the same energy tax as one that moves. `_get_movement_energy_cost` exists as a hook for a movement-based cost but is stubbed to always return `0` — it's dead code in `base_environment`.

This module actually implements that hook: choosing any of the 8 directional actions costs `energy_loss_per_move_predator`/`_prey`; choosing noop costs nothing extra. The existing per-step tax (`energy_loss_per_step_predator`/`_prey`) is kept — it still applies unconditionally, every step, whether or not the agent moved — but at half its `base_environment` default, since simply adding the new move cost on top of the old, unreduced flat cost would raise total drain and risk starving populations out before training converges.

**Net effect of the defaults in [`config_env.py`](./config_env.py)**: an agent that moves every single step pays the same total per-step energy cost as `base_environment` (0.15 for predators, 0.05 for prey — flat + move cost each contribute half). An agent that noops every step pays half that (0.075 / 0.025) — standing still is a real, cheaper option, not just behaviorally inert as in the baseline. This is a starting point for the investigation, not a tuned result — see "Status" below.

**Why this matters**: in `base_environment`, there is no energy cost tied to actually moving, so the only thing that makes an agent's action choice matter for survival is finding food/mates, not the act of searching itself. Charging energy for movement (and not for standing still) introduces a real explore/exploit tradeoff — an agent can choose to wait cheaply versus actively forage/hunt at a cost — which is a more biologically realistic energy economy and a stronger test of whether the sparse reproduction-only reward is still enough to sustain the ecosystem under a stricter energy budget.

## Design choice: action-based, not displacement-based

The move cost is charged based on the **action chosen**, not on whether the agent's position actually changed. An agent that picks a directional action but is blocked (target cell occupied, so it stays in place — see `_get_move`) still pays the move cost. This models the cost of *attempting* to move (effort expended) rather than the cost of *displacement* — consistent with how "noop costs no energy" was framed as an action-level distinction, not an outcome-level one.

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

Implemented and smoke-tested (2026-09-15): noop and directional-move energy deltas verified to match config exactly on a fresh reset (0.075 vs. 0.15 for predators at default config). **Not yet trained.** This module exists to investigate a specific question — does `base_environment` remain sustainable (both species surviving to the episode horizon, reproduction still occurring) once movement itself carries an energy cost, under the halved-flat/added-move-cost split described above — and that question is open until a real training run (and ideally a `base_environment` vs. this module comparison, same seed, same iteration count) is done.

**To run**: `python -m predpreygrass.non_evolutionary.base_environment_step_energy.tune_ppo_base_environment_step_energy --seed 42 --max-iters 1000`, then compare population/extinction/return curves against a same-seed `base_environment` run the way [`drive_conditioned_environment`](../drive_conditioned_environment/README.md#results-baseline-vs-drive-conditioned-single-seed-2026-09-06) compared itself to baseline (episode-count-weighted pooling, not per-iteration means).

**If it turns out too costly** (extinctions before the horizon, populations collapsing): the flat/move split ratio in `config_env.py` is the first thing to retune — e.g. keep more of the total budget in the unconditional flat cost and less in the move-conditional cost, which weakens the noop-vs-move distinction but preserves sustainability; or lower both proportionally. The 50/50 split used here is a reasonable starting guess, not a validated one.
