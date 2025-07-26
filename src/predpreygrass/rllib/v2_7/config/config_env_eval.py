from .config_env_train import config_env

config_env = {
    **config_env,
    "grid_size": 25,
    "n_possible_type_1_predators": 30,
    "n_possible_type_2_predators": 30,
    "n_possible_type_1_prey": 40,
    "n_possible_type_2_prey": 40,
    "n_initial_active_type_1_predator": 10,
    "n_initial_active_type_2_predator": 10,
    "n_initial_active_type_1_prey": 10,
    "n_initial_active_type_2_prey": 10,
}
