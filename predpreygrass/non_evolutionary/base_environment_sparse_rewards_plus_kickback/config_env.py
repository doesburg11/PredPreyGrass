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
    # Kick-back bonus: a second +10 to a grandparent, every time its own
    # child reproduces (i.e. every time a grandchild is born). Fires once per
    # grandchild (repeatable, not capped), only if the grandparent is still
    # alive to collect it. Equal weight to the primary reproduction reward,
    # by design -- this repo's kick_back_rewards module already tested a
    # weaker version of this same mechanism (magnitude 4.0, ~0.4x the
    # reproduction reward) and found it made no difference; this tests
    # whether a full 1:1 weighting changes that.
    "kickback_reward_predator": 10.0,
    "kickback_reward_prey": 10.0,
    # Energy settings
    "energy_loss_per_step_predator": 0.15,
    "energy_loss_per_step_prey": 0.05,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    # Learning agents. IDs are never reused within an episode (RLlib requires
    # each agent-ID string to map to exactly one trajectory per episode), so
    # these must comfortably cover cumulative births over a whole episode, not
    # just concurrent population size. Matches the value used in
    # base_environment_dense_rewards for a fair, resource-comparable setup.
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
