"""Default environment config for pack_hunt_opponent_shaping.

See the module README for the reasoning behind each of these, in particular
Section 5 (the three radii and why they're a starting hypothesis, not a
derived constant) and Section 6 (why the round timeout is load-bearing).
"""

config_env = {
    "seed": None,
    # Grid
    "grid_size": 10,
    # Fixed population -- no reproduction, no death, this many predators for
    # the whole run. Prey is scripted (not a learning agent), always exactly one.
    "n_predators": 3,
    # Episode is a fixed step budget, not a fixed round count -- rounds
    # (hunts) happen as many times as fit within it, respawning the prey each
    # time one ends. See README Section 7 on why a fixed *round* count would
    # invite backward-induction end-game defection; a fixed *step* budget only
    # weakly reproduces that effect at the very end of an episode.
    "max_episode_steps": 500,
    # A round (one hunt attempt) times out after this many steps without a
    # capture -- necessary so universal scrounging (zero engaged predators,
    # zero catch probability) can't stall the round forever. See README
    # Section 6.
    "round_timeout_steps": 15,
    # The three radii (Manhattan distance), see README Section 5.
    "capture_radius": 1,
    "engagement_radius": 2,
    "sharing_radius": 3,
    # Effort cost paid every step a predator is within engagement_radius of
    # the prey, regardless of round outcome.
    "engagement_cost": 0.05,
    # Per-engaged-predator contribution to that step's catch probability:
    # p_catch = 1 - (1 - catch_prob_per_engaged) ** (num engaged this step).
    # Only rolled on steps where at least one predator is within capture_radius.
    "catch_prob_per_engaged": 0.25,
    # Reward on a successful catch, split equally among every predator within
    # sharing_radius at that step -- engaged or not. See README Section 5.
    "capture_reward": 5.0,
    # Minimum Manhattan distance from every predator when the prey respawns,
    # so a new round doesn't start with an instant, free capture.
    "prey_respawn_min_distance": 3,
    # Discount factor for training (unused by random_policy.py itself, which
    # doesn't compute returns -- carried here for the training scripts this
    # config will also feed). See README Section 7 on why gamma < 1 matters
    # more here than as a generic RL hyperparameter.
    "gamma": 0.95,
}
