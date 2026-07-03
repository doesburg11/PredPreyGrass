config_env = {
    "seed": 41,
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    "num_obs_channels": 3,
    "predator_obs_range": 7,
    "prey_obs_range": 9,
    # Action space settings: 3x3 Moore neighbourhood (8 directions + stay).
    "action_range": 3,
    # Rewards
    "reproduction_reward_predator": {
        "predator": 10.0,
    },
    "reproduction_reward_prey": {
        "prey": 10.0,
    },
    # Energy settings
    "basal_energy_cost_predator": 0.15,
    "basal_energy_cost_prey": 0.05,
    "movement_energy_cost_per_cell_predator": 0.0,
    "movement_energy_cost_per_cell_prey": 0.0,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,
    # Cooperation genome. Each step an agent donates cooperation_rate * (this
    # step's positive net energy gain) to same-species neighbors within
    # cooperation_range. Donation only happens on steps with a successful
    # catch/graze and only when an eligible neighbor is present (meal-sharing,
    # not a continuous tax on stock energy). See README_COOPERATION.md.
    "genome_enabled": True,
    "founder_genome": {
        "predator": {
            "cooperation_rate_mean": 0.0,
            "cooperation_rate_std": 0.05,
        },
        "prey": {
            "cooperation_rate_mean": 0.0,
            "cooperation_rate_std": 0.05,
        },
    },
    "genome_mutation": {
        "rate": 0.05,
        "std": 0.04,
    },
    "trait_bounds": {
        "cooperation_rate": (0.0, 1.0),
    },
    # Chebyshev-distance radius within which same-species neighbors are
    # eligible to receive a donation.
    "cooperation_range": 2,
    # Absolute energy caps
    "max_energy_grass": 2.0,
    # Learning agents
    "n_possible_predators": 500,
    "n_possible_prey": 1000,
    "n_initial_active_predators": 6,
    "n_initial_active_prey": 8,
    # Grass settings
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.04,
    "debug_mode": False,
}
