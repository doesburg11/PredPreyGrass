"""Default simulation parameters for eco_evolutionary_erl_baldwin.

Deliberately reuses magnitudes close to the rest of this project's
predator-prey-grass modules (basal costs, energy thresholds, grid size)
rather than Ackley & Littman's original World AL constants, since the
point of this module is to test the ERL mechanism on top of ecology
already known to sustain in this codebase, not to replicate their exact
world. See README.md.
"""

config_erl = {
    "seed": 41,
    "grid_size": 25,
    "sense_range": 6,
    "n_initial_prey": 8,
    "n_initial_predators": 6,
    "founder_weight_std": 0.5,
    # Energy
    "initial_energy_prey": 3.0,
    "initial_energy_predator": 5.0,
    "basal_energy_cost_prey": 0.05,
    "basal_energy_cost_predator": 0.15,
    "movement_energy_cost": 0.02,
    "max_energy_grass": 2.0,
    "grass_regrow_rate": 0.04,
    "grass_visibility_threshold": 0.1,
    "max_energy_gain_per_bite": 1.0,
    "max_energy_gain_per_prey": 8.0,
    "max_energy_prey_for_norm": 8.0,
    "max_energy_predator_for_norm": 12.0,
    # Reproduction
    "reproduction_energy_threshold_prey": 8.0,
    "reproduction_energy_threshold_predator": 12.0,
    "reproduction_energy_cost_prey": 4.0,
    "reproduction_energy_cost_predator": 6.0,
    "mate_search_radius": 3,
    # Genome mutation
    "mutation_rate": 0.05,
    "mutation_std": 0.05,
    # Local reinforcement learning (within-lifetime, live action network only)
    "lr_positive": 0.05,
    "lr_negative": 0.02,
    # Population safety cap (prevents runaway population from making a
    # smoke-test run unboundedly slow; not present in Ackley & Littman)
    "max_population_per_species": 300,
}
