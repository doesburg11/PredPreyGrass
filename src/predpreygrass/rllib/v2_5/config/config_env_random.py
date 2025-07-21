from .config_env_train import config_env

config_env = {
    **config_env,
    "energy_loss_per_step_predator": 0.15,
    "energy_loss_per_step_prey": 0.05,
    "move_energy_cost_factor": 0.0,
    "mutation_rate_predator": 0.05,
    "mutation_rate_prey": 0.05,
    "initial_num_grass": 50,
    "energy_gain_per_step_grass": 0.08,
    "verbose_engagement": True,
    "verbose_reproduction": True,
    "n_possible_speed_2_predators": 30,
    "n_possible_speed_2_prey": 40,
    "n_initial_active_speed_1_predator": 20,
    "n_initial_active_speed_1_prey": 20,
    "n_initial_active_speed_2_predator": 0,
    "n_initial_active_speed_2_prey": 0,
}
