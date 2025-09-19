config_env = {
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    "num_obs_channels": 4,
    "predator_obs_range": 9,
    "prey_obs_range": 9,
    # Action space settings
    "type_1_action_range": 3,
    "type_2_action_range": 0,
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
        "type_2_predator": 0.0,
    },
    "reproduction_reward_prey": {
        "type_1_prey": 10.0,
        "type_2_prey": 0.0,
    },
    # Energy settings
    "energy_loss_per_step_predator": 0.15,
    "energy_loss_per_step_prey": 0.05,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "move_energy_cost_factor": 0.0,
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,
    # Learning agents
    "n_possible_type_1_predators": 50,
    "n_possible_type_2_predators": 0,
    "n_possible_type_1_prey": 50,
    "n_possible_type_2_prey": 0,
    "n_initial_active_type_1_predator": 6,
    "n_initial_active_type_2_predator": 0,
    "n_initial_active_type_1_prey": 8,
    "n_initial_active_type_2_prey": 0,
    # mutation settings
    "mutation_rate_predator": 0.0,
    "mutation_rate_prey": 0.0,
    # Grass settings
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.04,
    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_decay": False,
    "verbose_reproduction": False,
    "debug_mode": False,
    # Visibility & LOS behavior
    # When True, dynamic observation channels (predators/prey/grass) are masked by
    # a per-agent line-of-sight (LOS) mask so entities behind walls are hidden from the agent.
    # This does NOT change the number of observation channels.
    "mask_observation_with_visibility": True,
    # Optionally append a visibility channel for debugging/learning signal purposes.
    # Keep this False if your trained checkpoints assume a fixed num_obs_channels.
    "include_visibility_channel": True,
    # Optionally restrict movement so that agents cannot move to cells without clear LOS.
    "respect_los_for_movement": True,
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
    # --- Wall placement ---
    #  Zig-zag walls 
    "wall_placement_mode": "manual",
    "manual_wall_positions": (
        [(x, 6 + (x % 2)) for x in range(6, 18)] +
        [(x, 17 - (x % 2)) for x in range(6, 18)]
    ),
}