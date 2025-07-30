from .config_env_train import config_env

config_env = {
    **config_env,
    "max_steps": 1000,
    "n_possible_type_1_predators": 50,
    "n_possible_type_2_predators": 0,
    "n_possible_type_1_prey": 50,
    "n_possible_type_2_prey": 0,
    "n_initial_active_type_1_predator": 20,
    "n_initial_active_type_2_predator": 0,
    "n_initial_active_type_1_prey": 20,
    "n_initial_active_type_2_prey": 0,
}
