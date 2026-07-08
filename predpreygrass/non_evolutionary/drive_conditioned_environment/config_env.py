config_env = {
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    "num_obs_channels": 4,  # Border, Predator, Prey, Grass
    # Drive-conditioned observation settings.
    # These scalars are broadcast as constant channels over each local observation.
    "enable_drive_channels": True,
    "predator_drive_channels": [
        "hunger_pressure",
        "reproductive_readiness",
        "prey_opportunity",
    ],
    "prey_drive_channels": [
        "hunger_pressure",
        "reproductive_readiness",
        "predator_danger_pressure",
        "grass_opportunity",
    ],
    "predator_hunger_safe_energy": 5.0,
    "prey_hunger_safe_energy": 3.0,
    "prey_opportunity_normalizer": 9.0,
    "predator_danger_normalizer": 10.0,
    "grass_opportunity_normalizer": 10.0,
    "predator_obs_range": 7,
    "prey_obs_range": 9,
    # Rewards
    "reward_predator_catch_prey": 0.0,
    "reward_prey_eat_grass": 0.0,
    "reward_predator_step": 0.0,
    "reward_prey_step": 0.0,
    "penalty_prey_caught": 0.0,
    "reproduction_reward_predator": 10.0,
    "reproduction_reward_prey": 10.0,
    # Energy settings
    "energy_loss_per_step_predator": 0.15,
    "energy_loss_per_step_prey": 0.05,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    # Learning agents
    "n_possible_predators": 50,
    "n_possible_prey": 50,
    "n_initial_active_predator": 6,
    "n_initial_active_prey": 8,
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,
    # Grass settings
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.04,
    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_spawning": False,
}
