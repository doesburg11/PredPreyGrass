config_env = {
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    "num_obs_channels": 5,  # Border, Predator, Prey, Grass, Fruit
    "predator_obs_range": 7,
    "prey_obs_range": 9,
    # Rewards
    "reward_predator_catch_prey": 0.0,
    "reward_predator_gather_fruit": 0.0,
    "reward_prey_eat_grass": 0.0,
    "reward_predator_step": 0.0,
    "reward_prey_step": 0.0,
    "penalty_prey_caught": 0.0,
    # Reward when a predator dies from a failed hunting attempt (see the
    # Hunting section below). Kept as its own key, distinct from
    # penalty_prey_caught (which is the prey's reward on a successful catch),
    # so it's separately tunable.
    "penalty_predator_death_in_combat": 0.0,
    "reproduction_reward_predator": 10.0,
    "reproduction_reward_prey": 10.0,
    # Energy settings: same additive homeostatic (always charged) + move
    # (charged on top, only when the action isn't noop) split validated in
    # base_environment_step_energy -- see that module's config_env.py/RESULTS.md
    # for the full rationale. Applies uniformly to predator_male/predator_female;
    # only reproduction and foraging are sex-differentiated here.
    "homeostatic_energy_cost_per_step_predator": 0.10,
    "homeostatic_energy_cost_per_step_prey": 0.035,
    "move_energy_cost_per_step_predator": 0.08,
    "move_energy_cost_per_step_prey": 0.035,
    "predator_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    # Sexual reproduction: a predator_male and a predator_female must BOTH
    # independently clear predator_creation_energy_threshold AND be within
    # mate_search_radius (Chebyshev distance) of each other. This can never be
    # an exact-cell match (unlike predator-catches-prey or prey-eats-grass):
    # predator_male/predator_female share one grid layer (channel 1), so
    # movement collision already forbids two predators -- of either sex --
    # from ever occupying the same cell. Modeled on
    # eco_evolutionary_nuptial_gift's cooperation_range, which faces the
    # identical constraint for its male-female proximity check.
    "mate_search_radius": 3,
    # Birth cost is split asymmetrically between parents, modeled on parental
    # investment theory (Trivers, 1972): the female bears the larger share of
    # the shared reproductive cost because she is also the structurally
    # riskier forager here (see prey_vs_predator_female_* below -- her
    # hunting attempts have a much higher death probability and lower success
    # probability than the male's). Unlike an even 50/50 split, this
    # deliberately introduces a second, related asymmetry between the sexes
    # rather than isolating mate-finding as the sole new variable under test.
    # __init__ raises ValueError unless the two shares are non-negative and
    # sum to exactly 1.0 (an invalid split would silently create or destroy
    # energy at every birth).
    "predator_birth_cost_share_female": 0.9,
    "predator_birth_cost_share_male": 0.1,
    # Hunting is a 3-outcome stochastic contest for BOTH predator sexes (not a
    # deterministic, male-only catch): "success" (predator eats the prey,
    # exactly as a deterministic catch would have), "predator_dies" (the prey
    # is left completely unharmed; the predator itself is removed, using the
    # same bookkeeping as starvation), or "failure" (nothing happens -- same
    # as an ordinary non-engaged step). Failure probability is implied
    # (1 - success - death), not stored as its own key, to avoid a redundant
    # value that could silently drift out of sync with the other two.
    # predator_male is a low-risk/high-success hunter; predator_female is a
    # high-risk/low-success hunter -- this asymmetry is the mechanism meant to
    # produce emergent (not hardcoded) sex-based foraging specialization.
    # __init__ raises ValueError if success + death > 1.0 for either sex: an
    # invalid probability here is a correctness bug (it implies a negative
    # failure probability and silently wrong simulation dynamics), not just a
    # tuning choice like this module's otherwise-unvalidated population-size
    # settings, so it gets an explicit invariant check.
    "prey_vs_predator_male_success_prob": 0.90,
    "prey_vs_predator_male_death_prob": 0.05,
    "prey_vs_predator_female_success_prob": 0.20,
    "prey_vs_predator_female_death_prob": 0.10,
    #
    # Learning agents. IDs are never reused within an episode (RLlib requires
    # each agent-ID string to map to exactly one trajectory per episode), so
    # these must comfortably cover cumulative births over a whole episode, not
    # just concurrent population size.
    "n_possible_predator_male": 1000,
    "n_possible_predator_female": 1000,
    "n_possible_prey": 2000,
    "n_initial_active_predator_male": 3,
    "n_initial_active_predator_female": 3,
    "n_initial_active_prey": 8,
    "initial_energy_predator_male": 5.0,
    "initial_energy_predator_female": 5.0,
    "initial_energy_prey": 3.0,
    # Grass settings -- prey food only. Predators cannot eat grass in this module.
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.04,
    # Fruit settings -- predator food only (both sexes gather it; prey cannot
    # eat fruit). Same regrowth shape as grass, kept as a separate resource
    # pool/grid channel so predator and prey foraging never compete for the
    # same patches.
    "initial_num_fruit": 100,
    "initial_energy_fruit": 2.0,
    "energy_gain_per_step_fruit": 0.04,
    "verbose_engagement": False,
    "verbose_movement": False,
    "verbose_spawning": False,
}
