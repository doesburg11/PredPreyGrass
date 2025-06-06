config_env = {

    "max_steps": 1000,

    # Grid and Observation Settings
    "grid_size": 30,
    "num_obs_channels": 4,  # Border, Predator, Prey, Grass
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
    "n_possible_speed_1_predators": 15,
    "n_possible_speed_2_predators": 15,
    "n_possible_speed_1_prey": 21,
    "n_possible_speed_2_prey": 21,
    "n_initial_active_speed_1_predator": 5,
    "n_initial_active_speed_1_prey": 7,
    "n_initial_active_speed_2_predator": 5,
    "n_initial_active_speed_2_prey": 7,
    
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,

    # mutation settings
    "mutation_rate_predator": 0.05,
    "mutation_rate_prey": 0.05,
        
    # Grass settings
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.04,

    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_spawning": False,

}
