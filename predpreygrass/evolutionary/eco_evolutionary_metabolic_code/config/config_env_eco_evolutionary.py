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
    # Individual-level throttles on predator hunting (satiation), ported
    # unchanged from eco_evolutionary_investment/eco_evolutionary_metabolic_rate
    # -- an already-validated, orthogonal sustainability mechanism (criteria
    # 1/2), not something this trial (Trial 7) is re-testing. Regulates
    # predator population growth through each predator's own recent hunting
    # success (a Holling-type handling-time mechanism) rather than an
    # artificial population-level rule. Steps after a catch before the same
    # predator can catch again ("digesting").
    "predator_satiation_cooldown": 8,
    # Per-catch energy cap ("satiation ceiling") -- a predator can't extract
    # more than this from a single kill regardless of the prey's own energy.
    "max_energy_gain_per_prey": 8.0,
    # Offspring investment is a FIXED, non-heritable constant in this module
    # (see README.md "What's deliberately unchanged") -- the heritable trait
    # here is the combinatorial locus code below, not investment fraction.
    "offspring_investment_fraction": 0.35,
    # Heritable biological trait: a combinatorial "metabolic code" genome
    # (Hinton & Nowlan, 1987 needle-in-haystack design). Each locus is
    # CORRECT/WRONG/PLASTIC relative to an implicit fixed target; only a
    # zero-WRONG genotype can ever fully match, and PLASTIC loci are searched
    # fresh every step within an individual's own lifetime. See
    # utils/genome.py and README.md for the full mechanism.
    "genome_enabled": True,
    "haystack_num_loci": 10,
    "haystack_founder_probs": {
        "predator": {"correct": 0.2, "wrong": 0.3, "plastic": 0.5},
        "prey": {"correct": 0.2, "wrong": 0.3, "plastic": 0.5},
    },
    "haystack_mutation": {
        # Per-locus, per-reproduction-event probability of resampling that
        # locus uniformly among CORRECT/WRONG/PLASTIC.
        "rate": 0.05,
    },
    # Energy-gain multiplier applied from the step an agent fully solves its
    # genome onward (see predpreygrass_rllib_env.py's engagement handlers).
    "haystack_bonus_multiplier": 1.5,
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
