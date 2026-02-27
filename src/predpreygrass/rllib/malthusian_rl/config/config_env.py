config_env = {
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    "num_obs_channels": 4,
    "predator_obs_range": 7,
    "prey_obs_range": 9,
    # Action space settings
    "type_1_action_range": 3,
    "type_2_action_range": 3,
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
    "energy_loss_per_step_predator": 0.15,
    "energy_loss_per_step_prey": 0.05,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "move_energy_cost_factor": 0.0,
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,
    # Learning agents
    "n_possible_type_1_predators": 25,
    "n_possible_type_2_predators": 25,
    "n_possible_type_1_prey": 25,
    "n_possible_type_2_prey": 25,
    "n_initial_active_type_1_predator": 3,
    "n_initial_active_type_2_predator": 3,
    "n_initial_active_type_1_prey": 4,
    "n_initial_active_type_2_prey": 4,
    # Guard against accidental permanent extinction at reset:
    # if a species has n_possible > 0, enforce at least min_initial_mass_per_species agents.
    "enforce_min_initial_mass_per_species": True,
    "min_initial_mass_per_species": 1,
    # Malthusian scaffold (episode-end phi -> mu update across islands).
    "enable_malthusian_update": True,
    "malthusian_eta": 0.2,
    "malthusian_mu_floor": 0.0,
    # Ecology-driven phi score components:
    # phi = w_offspring*offspring + w_survival*survival + w_foraging*times_ate
    #     + w_energy*relative_energy_delta + w_death*death_indicator + w_reward*cumulative_reward
    "malthusian_phi_weights": {
        "offspring": 2.0,
        "survival": 1.0,
        "foraging": 0.5,
        "energy": 0.25,
        "death": -1.0,
        "reward": 0.0,
    },
    # Optional absolute clip on per-agent phi contribution; use None to disable.
    "malthusian_phi_clip": None,
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
    # Hard-island default: full cross-wall split with no gates.
    # On a 25x25 grid this yields 4 disconnected islands of equal free area (12x12 each).
    "wall_placement_mode": "manual",
    "manual_wall_positions": (
        [(x, 12) for x in range(25)] +
        [(12, y) for y in range(25)]
    ),
}
