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
    # unchanged from eco_evolutionary_metabolic_rate/investment/metabolic_code
    # -- an already-validated, orthogonal sustainability mechanism (criteria
    # 1/2), not something this trial (Trial 8) is re-testing. Regulates
    # predator population growth through each predator's own recent hunting
    # success (a Holling-type handling-time mechanism) rather than an
    # artificial population-level rule. Steps after a catch before the same
    # predator can catch again ("digesting").
    "predator_satiation_cooldown": 8,
    # Per-catch energy cap ("satiation ceiling") -- a predator can't extract
    # more than this from a single kill regardless of the prey's own energy.
    "max_energy_gain_per_prey": 8.0,
    # Offspring investment is a FIXED, non-heritable constant in this module
    # (see README.md "What's deliberately unchanged") -- the heritable
    # channels here are plasticity/dialect (see below), not investment
    # fraction.
    "offspring_investment_fraction": 0.35,
    # --- Dual-inheritance genome: two heritable channels ---
    # `plasticity` (continuous, [0,1]): the Darwinian trait -- the per-check
    # probability an agent's live dialect updates toward the local majority.
    # `dialect` (categorical, one of n_dialects): the founder cultural bias a
    # newborn is seeded with; freely overridable within its own lifetime.
    "genome_enabled": True,
    "n_dialects": 4,
    "founder_genome": {
        "predator": {"plasticity_mean": 0.1, "plasticity_std": 0.05},
        "prey": {"plasticity_mean": 0.1, "plasticity_std": 0.05},
    },
    # Per-reproduction-event mutation of the continuous plasticity trait
    # (Gaussian perturbation, same convention as every prior module's scalar
    # trait mutation).
    "genome_mutation": {
        "rate": 0.05,
        "std": 0.03,
    },
    # Per-reproduction-event mutation of the categorical dialect trait
    # (uniform resample among all n_dialects, same convention as
    # eco_evolutionary_metabolic_code's per-locus mutation).
    "dialect_mutation": {
        "rate": 0.05,
    },
    "trait_bounds": {
        "plasticity": (0.0, 1.0),
    },
    # --- Cultural mechanics (not heritable; see predpreygrass_rllib_env.py) ---
    # Chebyshev-distance radius defining a same-species "local culture group"
    # for the social-learning check (_apply_cultural_learning).
    "culture_range": 3,
    # Energy-gain multiplier applied to a catch/graze event when the agent's
    # live dialect matches the environment's current target dialect (see
    # _current_target_dialect / dialect_season_length_steps below) -- NOT the
    # local majority. Local-majority copying (_apply_cultural_learning) is
    # still how an agent's live dialect updates; it's just no longer what
    # determines the bonus, so a flip in the target creates a real,
    # repeated fitness gap between fast- and slow-adapting agents.
    "coordination_bonus_multiplier": 1.5,
    # Steps between per-agent social-learning checks.
    "plasticity_check_interval": 5,
    # Steps per phase before _current_target_dialect() rotates to the next
    # dialect index (cycling through all n_dialects, then repeating).
    # Originally 25, raised to 100 after the pilot showed dialect_match_rate
    # sitting flat at the ~0.25 chance baseline the entire run: with
    # plasticity_check_interval=5 and founder plasticity mean 0.1, expected
    # adoption latency is ~1/0.1 * 5 = ~50 steps for an average agent to
    # catch up to a new target -- longer than the original 25-step phase, so
    # the target was flipping before the population could ever converge on
    # it. 100 gives ~2x that latency per phase, comfortable headroom for
    # slower-adapting individuals too.
    "dialect_season_length_steps": 100,
    # Expose own live dialect + local-majority-match flag as two extra
    # observation channels, so the shared PPO policy can condition
    # movement/hunting on cultural state (the "reverse leg").
    "include_culture_in_obs": True,
    # Absolute energy caps
    "max_energy_grass": 2.0,
    # Learning agents. IDs are never reused within an episode, so this must
    # cover cumulative births over a whole episode, not just concurrent
    # population size. Raised from 200 to 1000 (matching n_possible_prey)
    # after the pilot showed the pool exhausting ~4600 times across 300
    # iterations in long/high-reproduction episodes, which would silently
    # cap predator_births -- the primary metric this experiment cares about.
    "n_possible_predators": 1000,
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
