config_env = {
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    "num_obs_channels": 4,  # Border, Predator, Prey, Grass
    "predator_obs_range": 7,
    "prey_obs_range": 9,
    # Rewards
    "reward_predator_catch_prey": 0.0,
    "reward_prey_eat_grass": 0.0,
    "reward_predator_step": 0.0,
    "reward_prey_step": 0.0,
    "penalty_prey_caught": 0.0,
    "reproduction_reward_predator": 10.0,
    "reproduction_reward_prey": 10.0,
    # Energy settings
    "energy_loss_per_step_predator": 0.15,
    "energy_loss_per_step_prey": 0.05,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    # Predators
    "n_possible_predators": 50,
    "n_initial_active_predator": 6,
    "initial_energy_predator": 5.0,
    # Cooperator prey (donate energy to adjacent prey each step)
    "n_possible_cooperator_prey": 50,
    "n_initial_active_cooperator_prey": 10,
    # Defector prey (keep all energy)
    "n_possible_defector_prey": 50,
    "n_initial_active_defector_prey": 10,
    # Shared prey energy settings
    "initial_energy_prey": 3.0,
    # Grass settings
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.04,
    # Network reciprocity cooperation parameters
    # Fraction of own energy donated to EACH adjacent prey per step.
    # Defection is individually rational: a cooperator surrounded only by
    # defectors loses energy; a defector surrounded by cooperators gains for free.
    # But a cooperator cluster nets positive because each member both donates
    # and receives — the benefit_multiplier makes the group gain > individual loss.
    "cooperation_cost": 0.05,
    # Factor applied to the donation before crediting the recipient.
    # Must be > 1 for cooperation to be socially beneficial.
    # Rule of thumb: benefit_multiplier > 1 / (avg cooperator neighbours + 1)
    # to ensure a dense cooperator cluster outcompetes isolated defectors.
    "cooperation_benefit_multiplier": 1.5,
    # Verbose
    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_spawning": False,
    "verbose_cooperation": False,
}
