config_env = {
    "seed": 42,
    "max_steps": 100,
    # Grid and Observation Settings
    "grid_size": 10,
    "num_obs_channels": 4,
    "predator_obs_range": 7,
    "prey_obs_range": 9,
    # Action space settings
    "action_range": 3,
    # Rewards
    "reproduction_reward_predator": 10.0,
    "reproduction_reward_prey": 10.0,
    # Energy settings
    "energy_loss_per_step_predator": 0.15,
    "energy_loss_per_step_prey": 0.15,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "move_energy_cost_factor": 0.0,
    "move_energy_cost_predator": 0.0,
    "move_energy_cost_prey": 0.00,
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,
    # Learning agents
    "n_possible_predators": 50,
    "n_possible_prey": 50,
    "n_initial_active_predator": 3,
    "n_initial_active_prey": 3,
    # mutation settings
    "mutation_rate_predator": 0.0,
    "mutation_rate_prey": 0.0,
    # Grass settings
    "initial_num_grass": 10,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.04,
    "verbose_engagement": True,
    "verbose_movement": False,
    "verbose_decay": False,
    "verbose_reproduction": False,
    "debug_mode": False,
    # Energy intake caps
    "max_energy_gain_per_grass": float('inf'),
    "max_energy_gain_per_prey": float('inf'),
    # Absolute energy caps
    "max_energy_predator": float('inf'),
    "max_energy_prey": float('inf'),
    "max_energy_grass": 2.0,
    "reproduction_cooldown_steps": 0,
    "reproduction_chance_predator": 1.0,
    "reproduction_chance_prey": 1.0,
    # Energy transfer and reproduction efficiency
    "energy_transfer_efficiency": 1.0,
    "reproduction_energy_efficiency": 1.0,
}