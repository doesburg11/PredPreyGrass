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
    # Action space settings
    "action_range": 5,
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
    "max_fertility_age": {
        # None ⇒ unlimited fertility window; set to an int to cap fertile age in steps
        "predator": None,
        "prey": None,
    },
    "max_agent_age": {
        # None ⇒ unlimited lifespan; set to an int to auto-terminate after that many steps
        "predator": None,
        "prey": None,
    },
    "carcass_only_predator_age": {
        # Juvenile predators younger than this many steps may only bite carcasses (already-dead prey)
        # Set to None/negative to disable the restriction for a policy group.
        "predator": None,
    },
    # Energy settings
    "energy_loss_per_step_predator": 0.20, # 0.15
    "energy_loss_per_step_prey": 0.05,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,
    # Heritable biological traits. These affect bodies/life-history, not policy weights.
    "genome_enabled": True,
    "founder_genome": {
        "predator": {
            "speed_mean": 1.0,
            "speed_std": 0.0,
            "movement_cost_multiplier_mean": 1.0,
            "movement_cost_multiplier_std": 0.05,
            "reproduction_threshold_multiplier_mean": 1.0,
            "reproduction_threshold_multiplier_std": 0.05,
            "offspring_energy_fraction_mean": 0.5,
            "offspring_energy_fraction_std": 0.05,
        },
        "prey": {
            "speed_mean": 1.0,
            "speed_std": 0.0,
            "movement_cost_multiplier_mean": 1.0,
            "movement_cost_multiplier_std": 0.05,
            "reproduction_threshold_multiplier_mean": 1.0,
            "reproduction_threshold_multiplier_std": 0.05,
            "offspring_energy_fraction_mean": 0.5,
            "offspring_energy_fraction_std": 0.05,
        },
    },
    "genome_mutation": {
        "rate": 0.05,
        "std": 0.03,
    },
    "trait_bounds": {
        "speed": (0.5, 2.0),
        "movement_cost_multiplier": (0.5, 2.0),
        "reproduction_threshold_multiplier": (0.5, 2.0),
        "offspring_energy_fraction": (0.25, 0.9),
    },
    "speed_distance_threshold": 1.5,
    "slow_max_move_distance": 1,
    "fast_max_move_distance": 2,
    "slow_speed_cost_multiplier": 1.0,
    "fast_speed_cost_multiplier": 1.8,
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
    "energy_gain_per_step_grass": 0.08, # 0.04
    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_decay": False,
    "verbose_reproduction": False,
    "debug_mode": False,
}
