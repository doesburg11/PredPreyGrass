config_env_base = {
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    "num_obs_channels": 4,
    "predator_obs_range": 7,
    "prey_obs_range": 9,
    # Action space settings
    "type_1_action_range": 3,
    "type_2_action_range": 5,
    # Rewards
    "reward_predator_catch_prey": {
        "type_1_predator": 0.0,
        "type_2_predator": 0.0,
    },
    "reward_prey_eat_grass": {
        "type_1_prey": 0.0,
        "type_2_prey": 0.0,
    },
    "reward_predator_step": {
        "type_1_predator": 0.0,
        "type_2_predator": 0.0,
    },
    "reward_prey_step": {
        "type_1_prey": 0.0,
        "type_2_prey": 0.0,
    },
    "penalty_prey_caught": {
        "type_1_prey": 0.0,
        "type_2_prey": 0.0,
    },
    "reproduction_reward_predator": {
        "type_1_predator": 10.0,
        "type_2_predator": 10.0,
    },
    "reproduction_reward_prey": {
        "type_1_prey": 10.0,
        "type_2_prey": 10.0,
    },
    # Energy settings
    "energy_loss_per_step_predator": 0.06,
    "energy_loss_per_step_prey": 0.02,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "move_energy_cost_factor": 0.01,
    "initial_energy_predator": 6.0,
    "initial_energy_prey": 3.0,
    # Learning agents
    "n_possible_type_1_predators": 50,  # lowered from 50
    "n_possible_type_2_predators": 0,
    "n_possible_type_1_prey": 40,  # lowerd from 60
    "n_possible_type_2_prey": 40,
    "n_initial_active_type_1_predator": 12,  # 5
    "n_initial_active_type_2_predator": 0,  # 5
    "n_initial_active_type_1_prey": 10,  # 7
    "n_initial_active_type_2_prey": 10,  # 7
    # mutation settings
    "mutation_rate_predator": 0.05,
    "mutation_rate_prey": 0.05,
    # Grass settings
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.1,
    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_decay": False,
    "verbose_reproduction": False,
    "debug_mode": False,
    # Energy intake caps
    "max_energy_gain_per_grass": 1.5,
    "max_energy_gain_per_prey": 5.0,
    # Absolute energy caps
    "max_energy_predator": 20.0,
    "max_energy_prey": 14.0,
    "max_energy_grass": 2.0,
    "reproduction_cooldown_steps": 5,
    "reproduction_chance_predator": 0.95,
    "reproduction_chance_prey": 0.95,
    # Energy transfer and reproduction efficiency
    "energy_transfer_efficiency": 0.9,
    "reproduction_energy_efficiency": 0.9,
}
