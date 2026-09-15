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
    # base_environment applies energy_loss_per_step_* unconditionally every
    # step, regardless of the action taken (including noop). This module adds
    # a second, movement-conditional cost on top of that: energy_loss_per_move_*
    # is only charged when the agent's action is not noop. To keep total
    # per-step drain from simply stacking on top of the base_environment
    # figures (which would starve agents faster and risk collapsing the
    # population before training converges), the flat per-step figures below
    # are halved from base_environment's 0.15/0.05, with the other half moved
    # into the move cost. Net effect: an agent that moves every step pays the
    # same total as base_environment; an agent that noops pays half -- so
    # "standing still" is a real, cheaper option rather than a free lunch on
    # top of the old cost.
    "energy_loss_per_step_predator": 0.075,
    "energy_loss_per_step_prey": 0.025,
    "energy_loss_per_move_predator": 0.075,
    "energy_loss_per_move_prey": 0.025,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    # Learning agents. IDs are never reused within an episode (RLlib requires
    # each agent-ID string to map to exactly one trajectory per episode), so
    # these must comfortably cover cumulative births over a whole episode, not
    # just concurrent population size.
    "n_possible_predators": 2000,
    "n_possible_prey": 2000,
    "n_initial_active_predator": 6,
    "n_initial_active_prey": 8,
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,
    # Grass settings
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.04,
    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_spawning": False,
}
