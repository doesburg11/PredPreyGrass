from .config_env_train import config_env

config_env = {
    **config_env,
    "n_possible_type_1_predators": 100,
    "n_possible_type_2_predators": 0,
    "n_possible_type_1_prey": 100,
    "n_possible_type_2_prey": 100,
    "n_initial_active_type_1_predator": 15,
    "n_initial_active_type_2_predator": 0,
    "n_initial_active_type_1_prey": 10,
    "n_initial_active_type_2_prey": 10,
    "mutation_rate_predator": 0.0,
    "mutation_rate_prey": 0.0,
    "move_energy_cost_factor": 0.00,  # Reduced to zero for testing
}
