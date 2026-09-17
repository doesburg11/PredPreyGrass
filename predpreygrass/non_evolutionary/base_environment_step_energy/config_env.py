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
    # base_environment charges energy_loss_per_step_* unconditionally every
    # step, regardless of the action taken (including noop) -- 0.15 predator
    # / 0.05 prey. This module splits that single figure into two
    # independent, additive costs instead of one flat tax:
    #   homeostatic_energy_cost_per_step_* -- always charged, every step,
    #     regardless of action (breathing/thermoregulation/upkeep; the
    #     reason noop can never be free -- see RESULTS.md section 6 for why
    #     a zero-cost noop turned out to be a real problem, not just an
    #     experimental simplification).
    #   move_energy_cost_per_step_* -- charged ON TOP of the homeostatic
    #     cost, only on steps where the agent's action is not noop.
    # These are independent values, not a split of a fixed shared total (an
    # earlier energy_loss_per_step_*/energy_loss_per_move_* design coupled
    # them via a --move-fraction split so their sum was pinned to
    # base_environment's original figure -- see RESULTS.md for why that
    # both mismodeled real metabolism and empirically failed at 500
    # iterations). Defaults here are deliberately set so resting costs
    # somewhat less than base_environment's original flat tax, and moving
    # costs somewhat more: predator 0.10 resting / 0.20 moving (vs.
    # original 0.15 flat); prey 0.035 resting / 0.07 moving (vs. original
    # 0.05 flat). Not yet validated at training scale -- see RESULTS.md.
    "homeostatic_energy_cost_per_step_predator": 0.10,
    "homeostatic_energy_cost_per_step_prey": 0.035,
    "move_energy_cost_per_step_predator": 0.10,
    "move_energy_cost_per_step_prey": 0.035,
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
