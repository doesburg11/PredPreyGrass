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
    # step, regardless of the action taken (including noop). This module
    # replaces that with a purely movement-conditional cost: the flat,
    # always-on tax is zeroed out, and the entire base_environment total
    # (0.15 predator / 0.05 prey) is charged only via energy_loss_per_move_*,
    # i.e. only on steps where the agent's action is not noop -- standing
    # still costs nothing at all. Validated sustainable (0% extinction, full
    # 1000-step episodes, ongoing reproduction by iteration ~30-40) via a
    # single-seed, 100-iteration move-fraction sweep on 2026-09-16 -- see
    # README.md's "Results" section, including caveats and the 0.5/0.75
    # split points also tested. Reproduce or resweep with
    # tune_ppo_base_environment_step_energy.py's --move-fraction flag.
    "energy_loss_per_step_predator": 0.0,
    "energy_loss_per_step_prey": 0.0,
    "energy_loss_per_move_predator": 0.15,
    "energy_loss_per_move_prey": 0.05,
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
