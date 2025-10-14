config_env = {
    # --- General settings ---
    "seed": 42,
    "max_steps": 1000,

    # --- Grid and Observation ---
    "grid_size": 25,
    "num_obs_channels": 4,
    "predator_obs_range": 7,
    "prey_obs_range": 9,

    # --- Action space ---
    "type_1_action_range": 3,
    "type_2_action_range": 0,

    # --- Rewards ---
    "reward_predator_catch_prey": {"type_1_predator": 0.0, "type_2_predator": 0.0},
    "reward_prey_eat_grass": {"type_1_prey": 0.0, "type_2_prey": 0.0},
    "reward_predator_step": {"type_1_predator": 0.0, "type_2_predator": 0.0},
    "reward_prey_step": {"type_1_prey": 0.0, "type_2_prey": 0.0},
    "penalty_prey_caught": {"type_1_prey": 0.0, "type_2_prey": 0.0},
    "reproduction_reward_predator": {"type_1_predator": 10.0, "type_2_predator": 0.0},
    "reproduction_reward_prey": {"type_1_prey": 10.0, "type_2_prey": 0.0},

    # --- Energy settings ---
    "energy_loss_per_step_predator": 0.15,
    "energy_loss_per_step_prey": 0.05,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "move_energy_cost_factor": 0.0,
    "kill_energy_cost": 0.0,  # Extra energy cost for killing prey (predator only)
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,
    "kill_prey_energy_cost": 0.0,  # Energy cost to predator for killing prey

    # --- Agent population ---
    "n_possible_type_1_predators": 50,
    "n_possible_type_2_predators": 0,
    "n_possible_type_1_prey": 50,
    "n_possible_type_2_prey": 0,
    "n_initial_active_type_1_predator": 10,
    "n_initial_active_type_2_predator": 0,
    "n_initial_active_type_1_prey": 10,
    "n_initial_active_type_2_prey": 0,

    # --- Mutation ---
    "mutation_rate_predator": 0.0,
    "mutation_rate_prey": 0.0,

    # --- Grass ---
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.04,

    # --- Verbosity & Debug ---
    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_decay": False,
    "verbose_reproduction": False,
    "debug_mode": False,

    # --- Visibility & LOS ---
    "mask_observation_with_visibility": True,  # mask obs channels by LOS
    "include_visibility_channel": True,        # append visibility channel
    "respect_los_for_movement": True,         # restrict movement by LOS

    # --- Intake caps (bite size per event) ---
    "max_eating_predator": 100.0,  # max energy predator can eat per event (from prey or carcass)
    "max_eating_prey": 100.0,      # max energy prey can eat per event (from grass)

    # --- Debug/compatibility flags ---
    "strict_no_carcass_channel": True,  # Set True for strict old-env observation compatibility

    # --- Absolute energy caps ---
    "max_energy_predator": float('inf'),
    "max_energy_prey": float('inf'),
    "max_energy_grass": 2.0,

    # --- Reproduction ---
    "reproduction_cooldown_steps": 0,
    "reproduction_chance_predator": 1.0,
    "reproduction_chance_prey": 1.0,

    # --- Efficiency ---
    "energy_transfer_efficiency": 1.0,
    "reproduction_energy_efficiency": 1.0,

    # --- Wall placement ---
    "wall_placement_mode": "manual",
    "num_walls": 0,  # ignored when using manual placement
    "manual_wall_positions": [
        *( [(x, 6) for x in range(6, 18) if x not in (9, 14)] ),
        *( [(x, 17) for x in range(6, 18) if x not in (11, 16)] ),
        *( [(6, y) for y in range(7, 17) if y not in (10, 15)] ),
        *( [(17, y) for y in range(7, 17) if y not in (12, 16)] ),
    ],
}