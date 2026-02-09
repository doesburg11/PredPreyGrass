## Sexual Reproduction Environment

This module extends the forward-view stag hunt ecology with **sexual reproduction for predators**, **age-based female fertility**, **male competition via mate choice**, and **energy provisioning**. Prey reproduction remains asexual and energy-threshold based.

**Quick Summary**
- Predators are split into male and female types (configurable).
- Defaults: male = `type_1_predator` (`predator_male_type = 1`), female = `type_2_predator` (`predator_female_type = 2`).
- Females are fertile only within an age window.
- Default fertility window: ages `5` through `50` (inclusive), set by `predator_fertility_age_min = 5` and `predator_fertility_age_max = 50`.
- A fertile female selects the **highest-energy male** within a mating radius.
- Both parents pay an energy cost of `predator_mating_parent_energy_share * child_initial_energy + predator_reproduction_cost_by_sex[parent_sex]`. Defaults: `predator_mating_parent_energy_share = 0.5`; `predator_energy_init_by_sex = {male: 120.0, female: 100.0}` so `child_initial_energy` is `120.0` for a male child or `100.0` for a female child; `predator_reproduction_cost_by_sex = {male: 40.0, female: 100.0}`. That yields default totals: male parent cost = `0.5 * child_initial_energy + 40.0`, female parent cost = `0.5 * child_initial_energy + 100.0`; a child spawns near the female.
- Adults can **transfer energy** to nearby **own offspring** (children only).

**Execution Order (per step)**
1. Energy decay + age increment. Params: `predator_idle_cost_by_sex = {male: 1.1, female: 1.0}`, `energy_loss_per_step_prey = {type_1_prey: 1.0, type_2_prey: 0.1}`, `predator_fertility_age_min = 5`, `predator_fertility_age_max = 50`, `predator_male_mating_age_min = 5`, `predator_child_age_max = 10`, `predator_adult_age = predator_child_age_max + 1` unless overridden, `verbose_decay = False`. `predator_idle_cost_by_sex` is required and must include `male` and `female`.
2. Grass regeneration. Params: `energy_gain_per_step_grass = 0.8`, `max_energy_grass = 30.0`.
3. Movement. Params: `type_1_action_range = 3`, `type_2_action_range = 3`, `grid_size = 30`, `respect_los_for_movement = False`, `wall_placement_mode = "manual"`, `num_walls = 0`, `manual_wall_positions = ()`, `predator_action_cost_by_sex = {move: {male: 2.2, female: 2.0}}`, `verbose_movement = False`.
4. Engagements (predation). Params: `team_capture_margin = 0.0`, `team_capture_equal_split = True`, `bite_size_prey = {type_1_prey: 30.0, type_2_prey: 3.0}`, `predator_action_cost_by_sex = {attack: {male: 8.0, female: 10.0}}`, `predator_kill_efficiency_by_sex = {male: 1.1, female: 1.0}`, `predator_failed_attack_damage_by_sex = {male: 15.0, female: 10.0}`, `verbose_engagement = False`.
5. Removals (death). Params: `death_penalty_predator = 0.0`, `death_penalty_type_1_prey = 0.0`, `death_penalty_type_2_prey = 0.0`.
6. Predator reproduction. Params: `predator_sexual_reproduction_enabled = True`, `predator_male_type = 1`, `predator_female_type = 2`, `predator_fertility_age_min = 5`, `predator_fertility_age_max = 50`, `predator_male_mating_age_min = 5`, `predator_mating_radius = 1`, `predator_mating_energy_threshold = 140.0`, `predator_male_single_mating_per_step = True`, `predator_offspring_type_prob = {type_1_predator: 0.5, type_2_predator: 0.5}`, `predator_mating_parent_energy_share = 0.5`, `predator_reproduction_cost_by_sex = {male: 40.0, female: 100.0}`, `initial_energy_predator = 110.0`, `predator_energy_init_by_sex = {male: 120.0, female: 100.0}`, `reproduction_reward_predator = {type_1_predator: 10.0, type_2_predator: 0.0}`, `n_possible_type_1_predators = 2000`, `n_possible_type_2_predators = 2000`, `verbose_reproduction = False`, and when sexual reproduction is disabled, `energy_treshold_creation_predator = 140.0` is used for asexual spawning.
7. Prey reproduction. Params: `energy_treshold_creation_prey = {type_1_prey: 220.0, type_2_prey: 70.0}`, `initial_energy_prey = {type_1_prey: 160.0, type_2_prey: 40.0}`, `reproduction_reward_prey = {type_1_prey: 10.0, type_2_prey: 10.0}`, `n_possible_type_1_prey = 1000`, `n_possible_type_2_prey = 2000`.
8. Provisioning transfers. Params: `provisioning_enabled = True`, `provisioning_radius = 1`, `provisioning_amount = 0.5`, `provisioning_cost_multiplier = 0.1`, `provisioning_min_donor_energy = 2.0`, `predator_child_age_max = 10`, `predator_adult_age = predator_child_age_max + 1` unless overridden.
9. Observations + output + episode termination. Params: `predator_obs_range = 9`, `prey_obs_range = 9`, `mask_observation_with_visibility = False`, `include_visibility_channel = False`, `strict_rllib_output = True`, `max_steps = 1000`, `debug_mode = False`. Observation channels are auto-computed from the fixed layout (walls, type_1_predators, type_2_predators, type_1_prey, type_2_prey, grass) = `6`, plus `+1` if `include_visibility_channel = True`.

**Initialization (on reset)**
Init-time parameters that define the starting state. Params: `seed = 41`, `n_initial_active_type_1_predator = 10`, `n_initial_active_type_2_predator = 10`, `n_initial_active_type_1_prey = 10`, `n_initial_active_type_2_prey = 10`, `initial_num_grass = 100`, `initial_energy_grass = 30.0`, `initial_energy_predator = 110.0`, `predator_energy_init_by_sex = {male: 120.0, female: 100.0}`, `initial_energy_prey = {type_1_prey: 160.0, type_2_prey: 40.0}`, `grid_size = 30`, `wall_placement_mode = "manual"`, `num_walls = 0`, `manual_wall_positions = ()`.

**Predator Sexual Reproduction Logic**
Sexual reproduction **replaces** the asexual predator reproduction when enabled.

**Female fertility**: Fertile if `predator_fertility_age_min <= age <= predator_fertility_age_max`. If `predator_fertility_age_max` is `None`, fertility has no upper bound.

**Mating eligibility**: Both parents must be above the energy gate. Males must be at least `predator_male_mating_age_min`. Candidates are all males within `predator_mating_radius` (Chebyshev distance). The **highest-energy male** is selected.

**Energy effects**: Child energy = `predator_energy_init_by_sex` for the child's sex (falls back to `initial_energy_predator`). Each parent pays `predator_mating_parent_energy_share * child_energy` plus `predator_reproduction_cost_by_sex[parent_sex]`. The energy gate is `max(predator_mating_energy_threshold, parent_cost)`.

**Offspring type**: Sampled using `predator_offspring_type_prob` (e.g., 0.5/0.5). If the chosen type has no available IDs, the other type is attempted.

**Placement**: Child spawns adjacent to the female if possible; otherwise any free cell.

**Rewards**: Both parents receive `reproduction_reward_predator` (type-specific). The child starts with reward `0`.

**Optional constraint**: `predator_male_single_mating_per_step` prevents a male from mating multiple times in the same step.

**Prey Reproduction**
- Unchanged from the base ecology.
- A prey reproduces when its energy exceeds the prey threshold.
- This is still asexual and energy-only.

**Provisioning Transfers (Protection/Energy Support)**
Adults can transfer energy to **nearby own offspring** (children only).

**Eligibility**: Adults are agents with `age >= predator_adult_age`. Children are agents with `age <= predator_child_age_max`. Recipient must be **the donor's own offspring**; no spouse/mate transfers.

**Transfer rules**: Donor must have energy >= `max(provisioning_min_donor_energy, donor_cost)`. Donor picks the **lowest-energy eligible** recipient in radius. Donor gives `provisioning_amount`. Donor pays `provisioning_amount * (1 + provisioning_cost_multiplier)`. Transfers are logged in per-step deltas as `energy_provision`.

**Key Configuration Knobs**
Defaults live in `src/predpreygrass/rllib/sexual_reproduction/config/config_env_sexual_reproduction.py`.

| Key | Purpose | Default |
| --- | --- | --- |
| `predator_sexual_reproduction_enabled` | Enable sexual reproduction for predators | `True` |
| `predator_male_type` | Predator type used as male | `1` |
| `predator_female_type` | Predator type used as female | `2` |
| `predator_fertility_age_min` | Minimum fertile age for females | `5` |
| `predator_fertility_age_max` | Maximum fertile age for females | `50` |
| `predator_male_mating_age_min` | Minimum age for males to mate | `5` |
| `predator_mating_radius` | Mate-search radius (Chebyshev) | `1` |
| `predator_mating_energy_threshold` | Min energy required to mate | `10.0` |
| `predator_male_single_mating_per_step` | One male per step | `True` |
| `predator_offspring_type_prob` | Offspring type distribution | `{type_1_predator: 0.5, type_2_predator: 0.5}` |
| `predator_mating_parent_energy_share` | Parent cost share of child energy | `0.5` |
| `provisioning_enabled` | Enable energy transfers | `True` |
| `provisioning_radius` | Transfer radius (Chebyshev) | `1` |
| `provisioning_amount` | Energy given to recipient | `0.5` |
| `provisioning_cost_multiplier` | Extra donor cost percentage | `0.1` |
| `provisioning_min_donor_energy` | Min energy to allow giving | `2.0` |
| `predator_child_age_max` | Max age for child status | `10` |

**Notes and Design Implications**
- If female predators are scarce or age-gated too tightly, reproduction will drop sharply.
- Increasing `predator_mating_radius` makes mate-finding easier but can reduce competition.
- Provisioning can stabilize offspring survival but may slow predator growth if cost is high.

**Default Config (Full)**
All parameter names and defaults as defined in `config_env_sexual_reproduction.py`:

```python
config_env = {
    "seed": 41,
    "max_steps": 1000,
    "strict_rllib_output": True,
    "grid_size": 30,
    "predator_obs_range": 9,
    "prey_obs_range": 9,
    "type_1_action_range": 3,
    "type_2_action_range": 3,
    "reproduction_reward_predator": {
        "type_1_predator": 10.0,
        "type_2_predator": 0.0,
    },
    "reproduction_reward_prey": {
        "type_1_prey": 10.0,
        "type_2_prey": 10.0,
    },
    "death_penalty_predator": 0.0,
    "death_penalty_type_1_prey": 0.0,
    "death_penalty_type_2_prey": 0.0,
    "predator_energy_init_by_sex": {
        "male": 120.0,
        "female": 100.0,
    },
    "predator_idle_cost_by_sex": {
        "male": 1.1,
        "female": 1.0,
    },
    "predator_action_cost_by_sex": {
        "attack": {
            "male": 8.0,
            "female": 10.0,
        },
        "move": {
            "male": 2.2,
            "female": 2.0,
        },
    },
    "predator_kill_efficiency_by_sex": {
        "male": 1.1,
        "female": 1.0,
    },
    "predator_failed_attack_damage_by_sex": {
        "male": 15.0,
        "female": 10.0,
    },
    "predator_reproduction_cost_by_sex": {
        "male": 40.0,
        "female": 100.0,
    },
    "energy_loss_per_step_prey": {
        "type_1_prey": 1.0,
        "type_2_prey": 0.1,
    },
    "energy_treshold_creation_predator": 140.0,
    "predator_sexual_reproduction_enabled": True,
    "predator_male_type": 1,
    "predator_female_type": 2,
    "predator_fertility_age_min": 5,
    "predator_fertility_age_max": 50,
    "predator_male_mating_age_min": 5,
    "predator_mating_radius": 1,
    "predator_mating_energy_threshold": 140.0,
    "predator_male_single_mating_per_step": True,
    "predator_offspring_type_prob": {
        "type_1_predator": 0.5,
        "type_2_predator": 0.5,
    },
    "predator_mating_parent_energy_share": 0.5,
    "provisioning_enabled": True,
    "provisioning_radius": 1,
    "provisioning_amount": 0.5,
    "provisioning_cost_multiplier": 0.1,
    "provisioning_min_donor_energy": 2.0,
    "predator_child_age_max": 10,
    "energy_treshold_creation_prey": {
        "type_1_prey": 220.0,
        "type_2_prey": 70.0,
    },
    "initial_energy_predator": 110.0,
    "initial_energy_prey": {
        "type_1_prey": 160.0,
        "type_2_prey": 40.0,
    },
    "bite_size_prey": {
        "type_1_prey": 30.0,
        "type_2_prey": 3.0,
    },
    "team_capture_margin": 0.0,
    "team_capture_equal_split": True,
    "max_energy_grass": 30.0,
    "n_possible_type_1_predators": 2000,
    "n_possible_type_2_predators": 2000,
    "n_possible_type_1_prey": 1000,
    "n_possible_type_2_prey": 2000,
    "n_initial_active_type_1_predator": 10,
    "n_initial_active_type_2_predator": 10,
    "n_initial_active_type_1_prey": 10,
    "n_initial_active_type_2_prey": 10,
    "initial_num_grass": 100,
    "initial_energy_grass": 30.0,
    "energy_gain_per_step_grass": 0.8,
    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_decay": False,
    "verbose_reproduction": False,
    "debug_mode": False,
    "mask_observation_with_visibility": False,
    "include_visibility_channel": False,
    "respect_los_for_movement": False,
    "wall_placement_mode": "manual",
    "num_walls": 0,
    "manual_wall_positions": (),
}
```
