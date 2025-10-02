config_env = {
    "max_steps": 800,
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
    # Selfish Gene lineage reward (Tier-1): count of living offspring born within this window (steps)
    # Tune suggestion: 100–150; start with 150 for this setup
    "lineage_reward_window": 150,
    # Optional per-species lineage windows (fallback to lineage_reward_window if omitted)
    "lineage_reward_window_predator": 150,
    "lineage_reward_window_prey": 80,
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
    "n_possible_type_2_predators": 50,
    "n_possible_type_1_prey": 50,
    "n_possible_type_2_prey": 50,
    "n_initial_active_type_1_predator": 6,
    "n_initial_active_type_2_predator": 6,
    "n_initial_active_type_1_prey": 8,
    "n_initial_active_type_2_prey": 8,
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
    # Deterministic placement/PRNG
    "seed": 42,
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
    # Kin-density observation feature (observation-only; no reward change)
    # When enabled, append an extra channel with the normalized count of same-policy agents
    # in a radius around the focal agent (helps policies learn kin-supportive behaviors
    # without altering the reward function).
    "include_kin_density_channel": True,
    # If True, count only kin that are LOS-visible (blocked by walls) for kin-density.
    # Keeps the signal aligned with what the agent can plausibly perceive in mazes/corridors.
    "kin_density_los_aware": True,
    # Neighborhood radius R (Chebyshev as implemented) to count same-policy agents.
    # Guidance:
    # - Start with R = 2 for most runs (local, captures clusters without washing out).
    # - With LOS-aware enabled, keep R within observation windows; with predator_obs_range=7 and
    #   prey_obs_range=9, R up to 3 remains local, but R=2 works well for both species.
    # - Adjust: sparse populations/heavy occlusion → R=3; very dense populations → R=1–2.
    "kin_density_radius": 2,
    # Normalization cap: kin count is divided by this cap (clipped) to map into [0,1].
    # Heuristic: expected same-policy neighbors ≈ ρ_g × A_R, where A_R = (2R+1)^2 − 1 and
    # ρ_g is group density (agents of that policy / free cells). Set this cap ≈ 1.5–2× that
    # expectation to avoid early saturation while keeping the channel informative.
    "kin_density_norm_cap": 8,
    # Optional cooperation logging (post-hoc analysis, no reward shaping)
    # When enabled, the environment writes one JSON file per episode with minimal
    # per-step agent data used by the analysis script (AI/KPA metrics).
    "enable_coop_logging": False,
    # Directory is created if missing; relative paths are resolved from CWD.
    "coop_log_dir": "output/coop_logs",
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
    # Use manual wall layout: centered 12x12 square (side=12) with 2-cell opening on each side.
    # Grid size is 25 -> start = (25-12)//2 = 6, square spans 6..17 inclusive.
    # Openings located at the two middle coordinates of each side.
    # When switching to random walls, set `wall_placement_mode` to "random" and use `num_walls` below.
    "num_walls": 0,
    "wall_placement_mode": "manual",
    "manual_wall_positions": (
        [(x, 6) for x in range(6, 18) if x not in (9, 14)] +
        [(x, 17) for x in range(6, 18) if x not in (11, 16)] +
        [(6, y) for y in range(7, 17) if y not in (10, 15)] +
        [(17, y) for y in range(7, 17) if y not in (12, 16)]
    ),
}