# PredPreyGrass – Walls & Occlusion Efficiency (Environment Spec)

This variant is an efficiency-focused rewrite of the Walls & Occlusion environment. It keeps the same modeling spirit (grid world with predators, prey, and grass) and adds clean line-of-sight (LOS) mechanics, optional observation masking, and a compact step loop suitable for multi-agent PPO search.

- Env file: `predpreygrass_rllib_env.py`
- Default config: `config/config_env_walls_occlusion.py`

Core goals:
- Make the environment faster and simpler to maintain.
- De-duplicate logic and centralize config-driven behavior.
- Keep it “cooperation-search-ready” for RLlib multi-agent PPO experiments.

## High-level model

- Grid world of size `grid_size × grid_size` with static walls, moving predators and prey, and stationary grass tiles.
- Agents consume energy every step; prey eat grass to gain energy; predators eat prey.
- Agents can reproduce when their energy crosses a type-specific threshold, creating a new agent (subject to available slots and spawn space).
- Episodes truncate at `max_steps` or terminate early if either predators or prey go extinct.

## Agent identity and types

- Learning agents are named: `type_{1|2}_{predator|prey}_{i}`.
- Non-learning resources: `grass_{i}`.
- Policy grouping is derived from the first three tokens: e.g. `type_1_predator` groups all `type_1_predator_*` agents.

## Observation space

Each learning agent receives a square, centered window. Shape depends on species:
- Predators: `(num_channels [+1 if visibility], predator_obs_range, predator_obs_range)`
- Prey: `(num_channels [+1 if visibility], prey_obs_range, prey_obs_range)`

Channels (fixed order):
1. Walls (binary 0/1)
2. Predators (energy value at cell or 0)
3. Prey (energy value at cell or 0)
4. Grass (energy value at cell or 0)

Optional visibility channel (appended as last channel when `include_visibility_channel=True`):
- Per-pixel LOS mask relative to the observing agent: `1.0` if the ray from agent to that cell crosses no wall (excluding endpoints), else `0.0`.

Masking option (`mask_observation_with_visibility=True`):
- Multiplies the dynamic channels (predators, prey, grass) by the LOS mask so entities behind walls appear as zeros even if they are inside the window.
- Works with or without the extra visibility channel.

## Action space and movement

- Discrete moves over a square action kernel per type: size `type_k_action_range × type_k_action_range` (k ∈ {1,2}).
- The kernel enumerates integer offsets `(dx, dy)` over `[-Δ, …, Δ] × [-Δ, …, Δ]`, where `Δ = (range-1)//2`.
- Moves are clipped to grid bounds.

Movement constraints:
- Entering a wall cell is blocked (`move_blocked_reason = "wall"`).
- Entering a cell already occupied by the same species is blocked (`"occupied"`). Co-location with the other species is allowed and triggers engagements.
- If `respect_los_for_movement=True`:
  - Diagonal “corner cutting” is blocked when either adjacent orthogonal neighbor is a wall (`"corner_cut"`).
  - A Bresenham LOS check blocks moves when any intervening wall lies strictly between start and end (`"los"`).

Infos per step include `los_rejected` (0/1) and `move_blocked_reason` when applicable.

## Energy model

- Per-step decay: `energy_loss_per_step_predator` and `energy_loss_per_step_prey`.
- Movement cost (configurable): `move_energy_cost_factor * distance * current_energy`.
- Eating gain with caps and efficiencies:
  - Predator eating prey: gain = `min(prey_energy, max_energy_gain_per_prey) * energy_transfer_efficiency`.
  - Prey eating grass: gain = `min(grass_energy, max_energy_gain_per_grass) * energy_transfer_efficiency`.
- Absolute caps: `max_energy_predator`, `max_energy_prey`, `max_energy_grass`.
- Grass regenerates each step by `energy_gain_per_step_grass` up to `max_energy_grass`.

## Engagements and rewards

- Predator on prey cell: prey is removed; predator gains energy; rewards assigned via `reward_predator_catch_prey` and `penalty_prey_caught`.
- Prey on grass cell: grass energy at that cell is set to 0; prey gains energy; reward via `reward_prey_eat_grass`.
- Step rewards: `reward_predator_step`, `reward_prey_step` (often set to 0 for search-friendliness).
- Reproduction rewards: `reproduction_reward_predator`, `reproduction_reward_prey`.

Reward config values can be scalars or dicts keyed by policy group (e.g. `{"type_1_predator": 0.0, ...}`), and are resolved per agent.

## Reproduction

- Thresholds: `predator_creation_energy_threshold`, `prey_creation_energy_threshold`.
- Cooldown: `reproduction_cooldown_steps`; Chance gates: `reproduction_chance_predator`, `reproduction_chance_prey`.
- Mutation: type flips with probability `mutation_rate_predator` / `mutation_rate_prey`.
- Slot availability: a child can be created only if a `type_k_*_{i}` ID is available (bounded by `n_possible_*`). If no slot is available, the parent still receives the reproduction reward.
- Energy transfer: child receives `initial_energy_{species} * reproduction_energy_efficiency`, and the parent pays `initial_energy_{species}`. Grid is updated for both.
- Spawn location: prefer an adjacent free cell; otherwise, any free cell; if none exists, no spawn occurs (reward is still granted if slot-limited).

## Episode flow and termination

Order per step:
1) Truncation check (`max_steps`).
2) Per-step energy decay and age update.
3) Grass regeneration.
4) Movement (with LOS/corner rules if enabled) and movement energy cost.
5) Engagements (eat prey/grass, rewards, removals).
6) Remove terminated agents and record stats.
7) Reproduction attempts and child insertion.
8) Build observations for survivors.

Termination and truncation:
- `terminations["__all__"] = (active_num_prey <= 0) or (active_num_predators <= 0)`.
- Truncation at `max_steps` returns zero rewards and marks all agents (including inactive ones) as truncated.

## Walls and LOS

- Placement modes:
  - `random`: choose `num_walls` unique cells.
  - `manual`: honor `manual_wall_positions` list of `(x, y)` cells (duplicates and OOB are ignored).
- The observation channel 0 is a binary wall mask. LOS masking uses Bresenham rays to check visibility per cell.

Note: The environment samples free positions for agents/grass from non-wall cells. In manual mode, ensure `num_walls` is consistent with the actual `manual_wall_positions` count; otherwise the free-cell sanity check may be optimistic. When in doubt, set `num_walls` to 0 for manual mode and control walls entirely via `manual_wall_positions`.

## Configuration reference (selected)

- Grid/obs: `grid_size`, `num_obs_channels` (default 4), `predator_obs_range`, `prey_obs_range`.
- Visibility: `mask_observation_with_visibility` (bool), `include_visibility_channel` (bool), `respect_los_for_movement` (bool).
- Energy: `initial_energy_predator`, `initial_energy_prey`, `energy_loss_per_step_predator`, `energy_loss_per_step_prey`, `move_energy_cost_factor`.
- Caps and efficiency: `max_energy_gain_per_grass`, `max_energy_gain_per_prey`, `max_energy_predator`, `max_energy_prey`, `max_energy_grass`, `energy_transfer_efficiency`, `reproduction_energy_efficiency`.
- Reproduction: thresholds, cooldowns, chances; `mutation_rate_predator`, `mutation_rate_prey`.
- Population limits and initial counts: `n_possible_*`, `n_initial_active_*` per type/species.
- Grass: `initial_num_grass`, `initial_energy_grass`, `energy_gain_per_step_grass`.
- Walls: `wall_placement_mode` ("random"|"manual"), `num_walls`, `manual_wall_positions`.
- Rewards: per-step, engagement, and reproduction (scalar or per-policy-group dicts).
- Debug/logging: `debug_mode`, `verbose_movement`, `verbose_decay`, `verbose_reproduction`, `verbose_engagement`.

See `config/config_env_walls_occlusion.py` for a ready-to-run setup featuring manual maze-like walls, LOS masking, and an appended visibility channel.

## Returned infos and stats

Per-step infos (per agent):
- `los_rejected`: 0/1 flag when a move was blocked by LOS; `move_blocked_reason`: one of `wall|occupied|corner_cut|los`.

Tracked statistics:
- `unique_agent_stats` per unique life (with births, distance, energy in/out, reward, death cause, etc.).
- `per_step_agent_data` summarizing energy deltas (decay, move, eat, reproduction) and age/offspring for all active agents.

Utility methods for analysis:
- `get_total_energy_by_type()`
- `get_total_offspring_by_type()`
- `get_total_energy_spent_by_type()`

## Quick usage (minimal)

Example environment construction (outside RLlib) for smoke-tests:

```python
from predpreygrass.rllib.walls_occlusion_efficiency.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.walls_occlusion_efficiency.config.config_env_walls_occlusion import config_env

env = PredPreyGrass(config_env)
obs, _ = env.reset(seed=123)
actions = {aid: env.action_spaces[aid].sample() for aid in obs.keys()}
obs, rew, term, trunc, info = env.step(actions)
```

## Gotchas and tips

- Observation shape changes when `include_visibility_channel=True`; ensure RLlib policies/modules are built using spaces from a sample env with your chosen config.
- If you set `type_2_action_range=0` you must also ensure there are no type-2 agents; otherwise the action space would be invalid.
- For manual walls, prefer leaving `num_walls=0` to avoid mismatches in free space checks.
- Movement cost can be disabled by setting `move_energy_cost_factor=0.0` (as in the default config).

## Status and next steps

This environment is intended to be “search-ready.” Follow-ups aimed at cooperation dynamics (e.g., partial consumption/carcass mechanics) can be layered on later without changing IDs or policy grouping.
