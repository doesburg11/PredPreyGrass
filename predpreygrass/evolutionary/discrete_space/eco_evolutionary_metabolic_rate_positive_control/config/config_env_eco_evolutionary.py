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
    # Soft carrying-capacity cap: predators skip reproduction once their count
    # reaches this multiple of the current prey count (energy-eligible parents
    # just wait rather than reproducing unconditionally). Effective at preventing
    # boom-bust crashes, but blocks indiscriminately (any eligible predator, not
    # just low-fitness ones), which dilutes genome-based selection pressure.
    # Disabled by default (None) in favor of the individual-level throttles below;
    # kept available for A/B comparison.
    "predator_reproduction_max_ratio": None,
    # Individual-level, biologically-motivated throttles on predator hunting,
    # in place of the population-level ratio cap above. These regulate predator
    # population growth through each predator's own energy/hunting history
    # (real starvation-driven scarcity) rather than an omniscient census rule.
    # Steps after a catch before the same predator can catch again ("digesting").
    "predator_satiation_cooldown": 8,
    # Per-catch energy cap ("satiation ceiling") — a predator can't extract more
    # than this from a single kill regardless of the prey's own energy level.
    "max_energy_gain_per_prey": 8.0,
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,
    # Metabolic rate genome. Gain scales as food * metabolic_rate**alpha (sub-linear,
    # digestive saturation); cost scales as base_cost * metabolic_rate (linear).
    # This creates a policy-dependent interior optimum — see README_METABOLIC_RATE.md.
    "genome_enabled": True,
    # Neutral-drift null model: when True, an offspring's genome template is a
    # uniformly random currently-alive same-species agent instead of whoever
    # actually reproduced -- severs genome from reproductive success while
    # leaving population/energy dynamics unchanged. Used only by the dedicated
    # neutral-control config/tune script; keep False for real experiment runs.
    "genome_neutral_drift_control": False,
    "founder_genome": {
        "predator": {
            "metabolic_rate_mean": 1.0,
            "metabolic_rate_std": 0.10,
        },
        "prey": {
            "metabolic_rate_mean": 1.0,
            "metabolic_rate_std": 0.10,
        },
    },
    # std lowered 0.04 -> 0.01 for Pilot 2 (Iteration 2): Pilot 1 (alpha=3.0, std=0.04)
    # showed a real but weak, plateauing signal (predator mr_repro_spearman consistently
    # positive but small; population mean settled ~1.05-1.06, far short of the 2.0 bound)
    # despite a 16x fitness gradient -- consistent with mutation re-randomizing the small
    # (~20-30 individual) population faster than selection can compound an advantage.
    # This change isolates that hypothesis: same overwhelming gradient, less mutation
    # noise fighting it. See RESULTS.md.
    "genome_mutation": {
        "rate": 0.05,
        "std": 0.01,
    },
    "trait_bounds": {
        "metabolic_rate": (0.5, 2.0),
    },
    # Exponent for energy gain scaling. Deliberately pushed super-linear (>1) here,
    # unlike the parent module's sub-linear 0.4-0.7 range. At alpha=3.0, energy gain
    # grows as metabolic_rate**3 while cost stays linear in metabolic_rate, so net
    # efficiency (gain/cost) scales as metabolic_rate**2 -- an overwhelming, monotonic
    # advantage for high metabolic_rate with no interior optimum and no dependence on
    # policy quality. This is a positive control, not a biologically motivated trait:
    # the point is to check whether the pipeline (genome inheritance + mutation +
    # differential reproduction + population-level metric aggregation) can detect ANY
    # selection signal at all, given an unmissably large fitness gradient, before
    # trusting null results from subtler trait designs. See RESULTS.md.
    "metabolic_rate_alpha": 3.0,
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
