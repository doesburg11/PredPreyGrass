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
    "energy_loss_per_step_predator": 0.08,  # 0.15
    "energy_loss_per_step_prey": 0.02,  # 0.05
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "move_energy_cost_factor": 0.01,  # 0.0  # energy cost = distance * factor * current_energy
    # Learning agents
    "n_possible_speed_1_predators": 30,  # 30
    "n_possible_speed_2_predators": 30,  # 30
    "n_possible_speed_1_prey": 40,  # 40
    "n_possible_speed_2_prey": 40,  # 40
    "n_initial_active_speed_1_predator": 10,  # 5
    "n_initial_active_speed_1_prey": 10,  # 7
    "n_initial_active_speed_2_predator": 10,  # 5
    "n_initial_active_speed_2_prey": 10,  # 7
    "initial_energy_predator": 6.5,  # increased from 5.0 to allow for more steps
    "initial_energy_prey": 3.0,
    # mutation settings
    "mutation_rate_predator": 0.05,  # mutation probability from speed_1 to speed_2
    "mutation_rate_prey": 0.05,  # and vice versa
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
    "max_energy_gain_per_grass": 1.5,  # or any reasonable value < initial_energy_grass
    "max_energy_gain_per_prey": 3.5,  # < average prey energy
    # Absolute energy caps
    "max_energy_predator": 20.0,
    "max_energy_prey": 14.0,
    "max_energy_grass": 2.0,  
    # Reproduction control
    "reproduction_cooldown_steps": 5,
    "reproduction_chance_predator": 0.85,
    "reproduction_chance_prey": 0.95,
    # Energy transfer and reproduction efficiency
    "energy_transfer_efficiency": 0.9,  # e.g. 85% of energy is absorbed from food
    "reproduction_energy_efficiency": 0.9,  # e.g. only 85% of energy investment goes to child
}
