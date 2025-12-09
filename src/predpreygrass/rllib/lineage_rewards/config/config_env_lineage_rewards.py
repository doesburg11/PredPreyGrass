config_env = {
    "seed": 41,
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    "num_obs_channels": 4,
    "predator_obs_range": 7,
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
    "lineage_reward_coeff": {
        "type_1_predator": 0.0,
        "type_2_predator": 0.0,
        "type_1_prey": 0.0,
        "type_2_prey": 0.0,
    },
    "max_fertility_age": {
        # None ⇒ unlimited fertility window; set to an int to cap fertile age in steps
        "type_1_predator": None,
        "type_2_predator": None,
        "type_1_prey": None,
        "type_2_prey": None,
    },
    "max_agent_age": {
        # None ⇒ unlimited lifespan; set to an int to auto-terminate after that many steps
        "type_1_predator": None,
        "type_2_predator": None,
        "type_1_prey": None,
        "type_2_prey": None,
    },
    "carcass_only_predator_age": {
        # Juvenile predators younger than this many steps may only bite carcasses (already-dead prey)
        # Set to None/negative to disable the restriction for a policy group.
        "type_1_predator": None,
        "type_2_predator": None,
    },
    # Energy settings
    "energy_loss_per_step_predator": 0.20, # 0.15
    "energy_loss_per_step_prey": 0.05,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,
    # Energy intake caps
    "max_energy_gain_per_grass": float('inf'), # 1.5
    "max_energy_gain_per_prey": float('inf'),  # 2.5
    # Absolute energy caps
    "max_energy_grass": 2.0,
    # Learning agents
    "n_possible_type_1_predators": 400,
    "n_possible_type_2_predators": 0,
    "n_possible_type_1_prey": 1200,
    "n_possible_type_2_prey": 0,
    "n_initial_active_type_1_predator": 10,
    "n_initial_active_type_2_predator": 0,
    "n_initial_active_type_1_prey": 10,
    "n_initial_active_type_2_prey": 0,
    # Grass settings
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
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
    "mask_observation_with_visibility": True,
    # Optionally append a visibility channel for debugging/learning signal purposes.
    # Keep this False if your trained checkpoints assume a fixed num_obs_channels.
    "include_visibility_channel": True,
    # Optionally restrict movement so that agents cannot move to cells without clear LOS.
    "respect_los_for_movement": True,
    # --- Wall placement ---
    # Use manual wall layout: centered 12x12 square (side=12) with 2-cell opening on each side.
    # Grid size is 25 -> start = (25-12)//2 = 6, square spans 6..17 inclusive.
    # Openings located at the two middle coordinates of each side.
    "wall_placement_mode": "manual",
    "num_walls": 0,  # ignored when using manual placement
    "manual_wall_positions": (
        [(x, 6) for x in range(6, 18) if x not in (9, 14)] +
        [(x, 17) for x in range(6, 18) if x not in (11, 16)] +
        [(6, y) for y in range(7, 17) if y not in (10, 15)] +
        [(17, y) for y in range(7, 17) if y not in (12, 16)]
    ),
}