config_env = {
    "seed": 41,
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    # Observation channels: predators, prey, grass. Grid edges are handled by
    # clipping the observation window and leaving out-of-grid cells at zero.
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
    "energy_loss_per_step_predator": 0.15,
    "energy_loss_per_step_prey": 0.05,
    "movement_energy_cost_per_cell_predator": 0.0,
    "movement_energy_cost_per_cell_prey": 0.0,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "initial_energy_predator_at_reset": 5.0,
    "initial_energy_prey_at_reset": 3.0,
    # Individual-level throttles on predator hunting (satiation), ported from
    # eco_evolutionary_metabolic_rate Iteration 2 -- same starting values since
    # the base energy economy here (thresholds/initial energies above) matches
    # that module almost exactly. Regulates predator population growth through
    # each predator's own recent hunting success (a Holling-type handling-time
    # mechanism) rather than an artificial population-level rule. Needs a pilot
    # run to confirm these transfer cleanly before trusting them -- investment
    # changes the parent's post-reproduction energy balance in a way metabolic
    # rate doesn't, so the sustainability effect may differ. Steps after a
    # catch before the same predator can catch again ("digesting").
    "predator_satiation_cooldown": 8,
    # Per-catch energy cap ("satiation ceiling") -- a predator can't extract
    # more than this from a single kill regardless of the prey's own energy.
    "max_energy_gain_per_prey": 8.0,
    # Heritable biological trait. Investment affects offspring starting energy, not policy weights.
    "genome_enabled": True,
    "founder_genome": {
        "predator": {
            "offspring_investment_fraction_mean": 0.35,
            "offspring_investment_fraction_std": 0.08,
        },
        "prey": {
            "offspring_investment_fraction_mean": 0.35,
            "offspring_investment_fraction_std": 0.08,
        },
    },
    "genome_mutation": {
        "rate": 0.05,
        "std": 0.04,
    },
    "trait_bounds": {
        "offspring_investment_fraction": (0.10, 0.80),
    },
    # Absolute energy caps
    "max_energy_grass": 2.0,
    # Learning agents
    "n_possible_predators": 200,
    "n_possible_prey": 1000,
    "n_initial_active_predators": 6,
    "n_initial_active_prey": 8,
    # Grass settings
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.04,
    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_decay": False,
    "verbose_reproduction": False,
    "debug_mode": False,
}
