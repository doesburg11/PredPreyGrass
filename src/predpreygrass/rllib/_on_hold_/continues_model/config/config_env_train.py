# v2_0 without mutation
from .config_env_base import config_env_base

config_env = {
    **config_env_base,
    # override base config with specific settings for this environment
    # Energy settings
    "energy_loss_per_step_predator": 0.15,
    "energy_loss_per_step_prey": 0.05,
    "move_energy_cost_factor": 0.0025,
    "initial_energy_predator": 5.0,
    # Learning agents
    "n_possible_type_1_predators": 30,
    "n_possible_type_2_predators": 30,
    "n_initial_active_type_1_predator": 10,
    "n_initial_active_type_2_predator": 10,
    # mutation settings
    "mutation_rate_predator": 0.05,  # No mutations for this environment
    "mutation_rate_prey": 0.05,  # No mutations for this environment
    # Grass settings
    "initial_num_grass": 50,
    "energy_gain_per_step_grass": 0.08,
    # Energy intake caps
    "max_energy_gain_per_grass": 100.0,
    "max_energy_gain_per_prey": 100,
    # Absolute energy caps
    "max_energy_predator": 100.0,
    "max_energy_prey": 100.0,
    "reproduction_cooldown_steps": 0,
    "reproduction_chance_predator": 1.0,
    "reproduction_chance_prey": 1.0,
    # Energy transfer and reproduction efficiency
    "energy_transfer_efficiency": 1.0,
    "reproduction_energy_efficiency": 1.0,
}
