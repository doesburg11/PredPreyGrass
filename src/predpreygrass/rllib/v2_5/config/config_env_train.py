from .config_env_base import config_env_base

config_env = {
    **config_env_base,
    "n_possible_speed_1_predators": 30,
    "n_possible_speed_2_predators": 0,
    "n_possible_speed_1_prey": 30,
    "n_possible_speed_2_prey": 30,
    "n_initial_active_speed_1_predator": 15,
    "n_initial_active_speed_2_predator": 0,
    "n_initial_active_speed_1_prey": 10,
    "n_initial_active_speed_2_prey": 10,
    "mutation_rate_predator": 0.0,
    "mutation_rate_prey": 0.0,
}
