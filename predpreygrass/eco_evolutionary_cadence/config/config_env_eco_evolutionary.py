config_env = {
    "seed": 41,
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    # Observation channels: predators, prey, grass. Grid edges are handled by
    # clipping the observation window and leaving out-of-grid cells at zero.
    "num_obs_channels": 3,
    "predator_obs_range": 7,
    "prey_obs_range": 9,
    # Action space: 3x3 Moore neighbourhood (8 directions + stay = 9 actions).
    # Speed controls movement frequency (cadence), not distance.
    "action_range": 3,
    # Rewards
    "reproduction_reward_predator": {
        "predator": 10.0,
    },
    "reproduction_reward_prey": {
        "prey": 10.0,
    },
    "lineage_reward_coeff": {
        "predator": 0.0,
        "prey": 0.0,
    },
    "max_agent_age": {
        # None ⇒ unlimited lifespan; set to an int to auto-terminate after that many steps
        "predator": None,
        "prey": 400,
    },
    "carcass_only_predator_age": {
        # Juvenile predators younger than this many steps may only bite carcasses (already-dead prey)
        # Set to None/negative to disable the restriction for a policy group.
        "predator": None,
    },
    # Energy settings
    "energy_loss_per_step_predator": 0.20, # basal metabolism
    "energy_loss_per_step_prey": 0.05, # basal metabolism
    "movement_energy_cost_per_cell_predator": 0.05,
    "movement_energy_cost_per_cell_prey": 0.02,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,
    # Heritable biological trait. Speed controls movement cadence (cooldown between moves).
    "genome_enabled": True,
    # Expose the agent's own normalised speed as a 4th observation channel.
    "include_speed_in_obs": True,
    # Expose a move_available flag (0/1) as an extra observation channel so the
    # policy knows whether its action this step will actually execute.
    "include_move_available_in_obs": True,
    "founder_genome": {
        "predator": {
            "speed_mean": 0.5,   # mid-range: founders start at cooldown ~5
            "speed_std": 0.1,
        },
        "prey": {
            "speed_mean": 0.5,
            "speed_std": 0.1,
        },
    },
    "genome_mutation": {
        "rate": 0.05,
        "std": 0.05,  # ~1 cooldown step per favourable mutation
    },
    "trait_bounds": {
        "speed": (0.0, 1.0),  # normalised: 0.0=slowest (cooldown=max), 1.0=fastest (cooldown=1)
    },
    # Cadence: slowest agent (speed_min) moves every max_cooldown steps.
    # Fastest agent (speed_max) moves every step (cooldown=1).
    "max_cooldown": 10,
    "movement_speed_cost_exponent": 2.0,
    # Energy intake caps
    "max_energy_gain_per_grass": float('inf'), # 1.5
    "max_energy_gain_per_prey": float('inf'),  # 2.5
    # Absolute energy caps
    "max_energy_grass": 2.0,
    # Learning agents
    "n_possible_predators": 400,
    "n_possible_prey": 1200,
    "n_initial_active_predators": 10,
    "n_initial_active_prey": 10,
    # Grass settings
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.04,
    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_decay": False,
    "verbose_reproduction": False,
    "debug_mode": False,
}
