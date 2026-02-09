config_env = {
    "seed": 41,  # RNG seed
    # Training settings
    "max_steps": 1000,  # Episode length cap (time limit)
    "strict_rllib_output": True,  # Emit only alive agent IDs each step
    # Grid and Observation Settings
    "grid_size": 30,  # Square grid side length (cells)
    "predator_obs_range": 9,  # Predator observation window size
    "prey_obs_range": 9,  # Prey observation window size
    # Action space settings
    "type_1_action_range": 3,  # Action range for type 1 agents
    "type_2_action_range": 3,  # Action range for type 2 agents
    # Rewards
    "reproduction_reward_predator": {  # Predator reproduction reward
        "type_1_predator": 10.0,  # Reward for male
        "type_2_predator": 10.0,  # Reward for female
    },
    "reproduction_reward_prey": {  # Prey reproduction reward
        "type_1_prey": 10.0,  # Reward for mammoth
        "type_2_prey": 10.0,  # Reward for rabbit
    },
    "death_penalty_predator": 0.0,  # Predator death penalty
    "death_penalty_type_1_prey": 0.0,  # Type 1 prey death penalty
    "death_penalty_type_2_prey": 0.0,  # Type 2 prey death penalty
    # Sex-based physical strength abstraction (energy economics)
    "predator_energy_init_by_sex": {  # Initial energy by sex
        "male": 120.0,  # Male initial energy
        "female": 100.0,  # Female initial energy
    },
    "predator_idle_cost_by_sex": {  # Idle metabolism by sex
        "male": 0.9,  # Male idle cost
        "female": 0.7,  # Female idle cost
    },
    "predator_action_cost_by_sex": {  # Action energy costs by sex
        "attack": {
            "male": 0.01,  # Male attack cost
            "female": 0.08,  # Female attack cost
        },
        "move": {
            "male": 0.0,  # Male move cost
            "female": 0.0,  # Female move cost
        },
    },
    "predator_kill_efficiency_by_sex": {  # Energy gain multiplier by sex
        "male": 1.1,  # Male kill efficiency
        "female": 1.0,  # Female kill efficiency
    },
    "predator_failed_attack_damage_by_sex": {  # Failed attack penalty by sex
        "male": 0.015,  # Male failed-attack damage
        "female": 0.01,  # Female failed-attack damage
    },
    "predator_reproduction_cost_by_sex": {  # Extra reproduction cost by sex
        "male": 0.0,  # Male reproduction cost
        "female": 0.0,  # Female reproduction cost
    },
    "energy_loss_per_step_prey": {  # Prey idle cost by type
        "type_1_prey": 0.5,  # Type 1 prey idle cost
        "type_2_prey": 0.05,  # Type 2 prey idle cost
    },
    "energy_treshold_creation_predator": 110.0,  # Asexual predator spawn threshold
    # Sexual reproduction (predators)
    "predator_sexual_reproduction_enabled": True,  # Enable sexual reproduction
    "predator_male_type": 1,  # Predator type used as male
    "predator_female_type": 2,  # Predator type used as female
    "predator_fertility_age_min": 5,  # Female minimum fertile age
    "predator_fertility_age_max": 1000,  # Female maximum fertile age
    "predator_male_mating_age_min": 5,  # Male minimum mating age
    "predator_mating_radius": 1,  # Mate search radius (Chebyshev)
    "predator_mating_energy_threshold": 140.0,  # Min energy to mate
    "predator_male_single_mating_per_step": True,  # One mating per male per step
    "predator_offspring_type_prob": {  # Offspring type distribution
        "type_1_predator": 0.5,  # Prob child is type 1
        "type_2_predator": 0.5,  # Prob child is type 2
    },
    "predator_mating_parent_energy_share": 0.5,  # Parent cost share of child energy
    # Provisioning (adjacent transfers)
    "provisioning_enabled": True,  # Enable energy transfers
    "provisioning_radius": 1,  # Transfer radius (Chebyshev)
    "provisioning_amount": 0.5,  # Energy given per transfer
    "provisioning_cost_multiplier": 0.1,  # Extra donor cost multiplier
    "provisioning_min_donor_energy": 2.0,  # Min donor energy to give
    "predator_child_age_max": 10,  # Max age for "child" status
    "energy_treshold_creation_prey": {  # Asexual prey spawn thresholds
        "type_1_prey": 220.0,  # Type 1 prey threshold
        "type_2_prey": 70.0,  # Type 2 prey threshold
    },
    "initial_energy_predator": 110.0,  # Fallback predator initial energy
    "initial_energy_prey": {  # Prey initial energy by type
        "type_1_prey": 160.0,  # Type 1 prey initial energy
        "type_2_prey": 40.0,  # Type 2 prey initial energy
    },
    "bite_size_prey": {  # Prey bite size by type
        "type_1_prey": 30.0,  # Type 1 prey bite size
        "type_2_prey": 3.0,  # Type 2 prey bite size
    },
    # Cooperative capture predators
    "team_capture_margin": 0.0,  # Extra energy margin required
    "team_capture_equal_split": True,  # Split prey energy equally among helpers
    # Absolute energy caps
    "max_energy_grass": 30.0,  # Max grass energy
    # Learning agents
    "n_possible_type_1_predators": 2000,  # ID pool size for type 1 predators
    "n_possible_type_2_predators": 2000,  # ID pool size for type 2 predators
    "n_possible_type_1_prey": 1000,  # ID pool size for type 1 prey
    "n_possible_type_2_prey": 2000,  # ID pool size for type 2 prey
    "n_initial_active_type_1_predator": 10,  # Initial type 1 predators
    "n_initial_active_type_2_predator": 10,  # Initial type 2 predators
    "n_initial_active_type_1_prey": 10,  # Initial type 1 prey
    "n_initial_active_type_2_prey": 10,  # Initial type 2 prey
    # Grass settings
    "initial_num_grass": 100,  # Initial grass patches
    "initial_energy_grass": 30.0,  # Initial grass energy
    "energy_gain_per_step_grass": 0.8,  # Grass regrowth per step
    "verbose_engagement": False,  # Verbose engagement logging
    "verbose_movement": False,  # Verbose movement logging
    "verbose_decay": False,  # Verbose decay logging
    "verbose_reproduction": False,  # Verbose reproduction logging
    "debug_mode": False,  # Debug mode flag
    # Visibility & LOS behavior
    "mask_observation_with_visibility": False,  # Mask obs by LOS
    "include_visibility_channel": False,  # Append visibility channel
    "respect_los_for_movement": False,  # Restrict movement by LOS
    # --- Wall placement ---
    "wall_placement_mode": "manual",  # Wall placement mode
    "num_walls": 0,  # Number of walls (ignored for manual)
    "manual_wall_positions": (),  # Manual wall coordinates
}
