config_env = {
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    "num_obs_channels": 4,  # Border, Predator, Prey, Grass
    "predator_obs_range": 7,
    "prey_obs_range": 9,
    # Rewards: same clean, event-based sparse-reward design as
    # base_environment_sparse_rewards (no continuous energy-delta signal),
    # PLUS a flat reward for the eating event itself, not just reproduction.
    # Magnitudes are asymmetric by design, not a flat +1/+1: measured under
    # the trained sparse policy, predators eat ~4.4 times per reproduction
    # but prey eat ~60.5 times per reproduction (grass regrows slowly and
    # gives small amounts per visit, so prey need many more small meals to
    # reach their reproduction threshold than predators do catches). A flat
    # +1 for both would make prey's total eating reward per reproduction
    # cycle (~60.5) six times *larger* than the reproduction reward itself
    # (10.0), swamping the primary incentive. +1 predator / +0.1 prey keeps
    # each species' total eating reward per cycle in a similar, clearly
    # secondary range relative to its own reproduction reward
    # (predator: 4.4*1=4.4, prey: 60.5*0.1=6.05 -- both well below 10.0).
    "reward_predator_catch_prey": 1.0,
    "reward_prey_eat_grass": 0.1,
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
