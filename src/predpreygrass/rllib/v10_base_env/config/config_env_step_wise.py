config_env = {
    "max_steps": 1000,
    "seed": 42,
    # Grid and Observation Settings
    "grid_size": 10,
    "num_obs_channels": 5,  # Wall, Predator, Prey, Grass, Water
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
    "energy_loss_per_step_predator": 0.15,
    "energy_loss_per_step_prey": 0.05,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "move_energy_cost_factor": 0.0,  # energy cost = distance * factor * current_energy # 0.1
    # Water
    "initial_hydration_predator": 3.0,
    "initial_hydration_prey": 2.0,
    "dehydration_per_step_predator": 0.1,
    "dehydration_per_step_prey": 0.025,
    "max_hydration_predator": 4.0,
    "max_hydration_prey": 3.0,
    "n_steps_river_change": 50,  # number of steps before rivers change
    # Learning agents
    "n_possible_speed_1_predators": 30,  # 30
    "n_possible_speed_2_predators": 30,  # 30
    "n_possible_speed_1_prey": 40,  # 40
    "n_possible_speed_2_prey": 40,  # 40
    "n_initial_active_speed_1_predator": 5,  # 5
    "n_initial_active_speed_1_prey": 7,  # 7
    "n_initial_active_speed_2_predator": 5,  # 5
    "n_initial_active_speed_2_prey": 7,  # 7
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,
    # mutation settings
    "mutation_rate_predator": 0.05,  # mutation probability from speed_1 to speed_2
    "mutation_rate_prey": 0.05,  # and vice versa
    # Grass settings
    "initial_num_grass": 10,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.08,
    "verbose_engagement": True,
    "verbose_termination": False,
    "verbose_movement": False,
    "verbose_decay": False,
    "verbose_reproduction": True,
    "debug_mode": True,
}
