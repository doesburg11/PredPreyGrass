config_env = {
    "seed": 41,
    # Training settings
    "max_steps": 150,
    "strict_rllib_output": True, # When True, only alive agent IDs are emitted each step.
    # Grid and Observation Settings
    "grid_size": 30, 
    "num_obs_channels": 5, # obsolete
    "predator_obs_range": 9,
    "prey_obs_range": 9,
    # Action space settings
    "type_1_action_range": 3,
    "type_2_action_range": 3,
    # Rewards
    "reproduction_reward_predator": {
        "type_1_predator": 10.0,
        "type_2_predator": 0.0,
    },
    "reproduction_reward_prey": {
        "type_1_prey": 10.0,
        "type_2_prey": 10.0,
    },
    "death_penalty_predator": 0.0,
    "death_penalty_type_1_prey": 0.0,
    "death_penalty_type_2_prey": 0.0,
    # Energy settings
    "energy_loss_per_step_predator": 0.08, # 0.05
    "energy_loss_per_step_prey": {
        "type_1_prey": 0.1,
        "type_2_prey": 0.02,
    },
    "energy_percentage_loss_per_failed_attacked_prey": 0.00, # 0.0
    "failed_attack_kills_predator": False,
    "energy_treshold_creation_predator": 10.0,
    "energy_treshold_creation_prey": {
        "type_1_prey": 18.0,
        "type_2_prey": 2.7,
    },
    "initial_energy_predator": 4.0,
    "initial_energy_prey": {
        "type_1_prey": 10.0,
        "type_2_prey": 1.5,
    },
    "bite_size_prey": {
        "type_1_prey": 3.0,
        "type_2_prey": 0.3,
    },
    # Cooperative capture predators
    "team_capture_margin": 0.0,  # optional safety margin; set >0 to demand extra energy
    "team_capture_equal_split": True,  # If False, split prey energy proportionally among helpers
    # Voluntary participation + free-riding
    "team_capture_join_cost": 0.2,  # fixed energy cost paid only by joining predators on success
    "team_capture_scavenger_fraction": 0.1,  # fraction of prey energy reserved for nearby non-joiners
    # Absolute energy caps
    "max_energy_grass": 3.0,
    # Learning agents
    "n_possible_type_1_predators": 2000,
    "n_possible_type_2_predators": 0,
    "n_possible_type_1_prey": 1000,
    "n_possible_type_2_prey": 2000,
    "n_initial_active_type_1_predator": 10,
    "n_initial_active_type_2_predator": 0,
    "n_initial_active_type_1_prey": 10,
    "n_initial_active_type_2_prey": 10,
    # Grass settings
    "initial_num_grass": 100, # 100
    "initial_energy_grass": 3.0,
    "energy_gain_per_step_grass": 0.08, # 0.08
    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_decay": False,
    "verbose_reproduction": False,
    "debug_mode": False,
    # Visibility & LOS behavior
    # When True, dynamic observation channels (predators/prey/grass) are masked by
    # a per-agent line-of-sight (LOS) mask so entities behind walls are hidden from the agent.
    # This does NOT change the number of observation channels.
    "mask_observation_with_visibility": False,
    # Optionally append a visibility channel for debugging/learning signal purposes.
    # Keep this False if your trained checkpoints assume a fixed num_obs_channels.
    "include_visibility_channel": False,
    # Optionally restrict movement so that agents cannot move to cells without clear LOS.
    "respect_los_for_movement": False,
    # --- Wall placement ---
    # Use manual wall layout: centered 12x12 square (side=12) with 2-cell opening on each side.
    # Grid size is 25 -> start = (25-12)//2 = 6, square spans 6..17 inclusive.
    # Openings located at the two middle coordinates of each side.
    "wall_placement_mode": "manual",
    "num_walls": 0,  # ignored when using manual placement
    "manual_wall_positions": (
    ),
}
