config_env = {
    "seed": 41,
    # Training settings
    "max_steps": 1000,
    "strict_rllib_output": True, # When True, only alive agent IDs are emitted each step.
    # Grid and Observation Settings
    "grid_size": 30,  # 25
    "num_obs_channels": 4,
    "predator_obs_range": 9,
    "prey_obs_range": 9,
    # Action space settings
    "type_1_action_range": 3,
    "type_2_action_range": 0,
    # Rewards
    "reproduction_reward_predator": {
        "type_1_predator": 10.0,
        "type_2_predator": 0.0,
    },
    "reproduction_reward_prey": {
        "type_1_prey": 10.0,
        "type_2_prey": 0.0,
    },
    # Energy settings
    "energy_loss_per_step_predator": 0.05, # 0.15
    "energy_loss_per_step_prey": 0.1,  # 0.05 
    "energy_percentage_loss_per_failed_attacked_prey": 0.0, # 0.1
    "predator_creation_energy_threshold": 10.0,
    "prey_creation_energy_threshold": 18,  # was 6.5
    "initial_energy_predator": 4.0,
    "initial_energy_prey": 10.0,  # was 3.5
    # Cooperative capture predators
    "team_capture_margin": 0.0,  # optional safety margin; set >0 to demand extra energy
    "team_capture_equal_split": True,  # If False, split prey energy proportionally among helpers
    # Absolute energy caps
    "max_energy_grass": 3.0,
    # Learning agents
    "n_possible_type_1_predators": 2000,
    "n_possible_type_2_predators": 0,
    "n_possible_type_1_prey": 1000,
    "n_possible_type_2_prey": 0,
    "n_initial_active_type_1_predator": 50,
    "n_initial_active_type_2_predator": 0,
    "n_initial_active_type_1_prey": 10,
    "n_initial_active_type_2_prey": 0,
    # Grass settings
    "initial_num_grass": 100,
    "initial_energy_grass": 4.0,
    "energy_gain_per_step_grass": 0.08, # 0.04
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
