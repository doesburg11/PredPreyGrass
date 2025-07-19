config_env = {
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    "num_obs_channels": 4,  # Border, Predator, Prey, Grass
    "predator_obs_range": 7,  # 7
    "prey_obs_range": 9,  # 9
    # Action space settings
    "speed_1_action_range": 3,
    "speed_2_action_range": 5,
    # Rewards
    "reward_predator_catch_prey": 0.0,
    "reward_prey_eat_grass": 0.0,
    "reward_predator_step": 0.0,
    "reward_prey_step": 0.0,
    "penalty_prey_caught": 0.0,
    "reproduction_reward_predator": 10.0,
    "reproduction_reward_prey": 10.0,
    # Energy settings
    "energy_loss_per_step_predator": 0.06,
    "energy_loss_per_step_prey": 0.02,  # 0.05
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "move_energy_cost_factor": 0.01,
    # Learning agents
    "n_possible_speed_1_predators": 30,  # lowered from 50
    "n_possible_speed_2_predators": 0,
    "n_possible_speed_1_prey": 40,  # lowerd from 60
    "n_possible_speed_2_prey": 0,
    "n_initial_active_speed_1_predator": 24,  # 5
    "n_initial_active_speed_1_prey": 20,  # 7
    "n_initial_active_speed_2_predator": 0,  # 5
    "n_initial_active_speed_2_prey": 0,  # 7
    "initial_energy_predator": 6.0,
    "initial_energy_prey": 3.0,
    # mutation settings
    "mutation_rate_predator": 0.0,
    "mutation_rate_prey": 0.0,
    # Grass settings
    "initial_num_grass": 100,  # 100
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.1,  # 0.04
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
    # Reproduction control
    "reproduction_cooldown_steps": 5,
    "reproduction_chance_predator": 0.95,
    "reproduction_chance_prey": 0.95,
    # Energy transfer and reproduction efficiency
    "energy_transfer_efficiency": 0.9,
    "reproduction_energy_efficiency": 0.9,
}
