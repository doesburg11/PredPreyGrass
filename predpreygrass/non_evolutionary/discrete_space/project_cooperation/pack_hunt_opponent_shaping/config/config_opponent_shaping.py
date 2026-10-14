config_opponent_shaping = {
    "seed": None,
    "hidden_size": 32,
    # Rollout horizon for *training* -- deliberately shorter than the
    # viewer's max_episode_steps=500, so a batch of rollouts is cheap to
    # collect every iteration. Spans several hunting rounds (round_timeout
    # is 15 steps by default), which is what's needed for the last-round
    # memory signal to matter at all -- see the module README Sections 6-7
    # on why a single round per rollout wouldn't give reciprocity anything
    # to condition on.
    "horizon": 120,
    "batch_size": 64,
    "iterations": 200,
    "log_every": 5,
    # Step sizes, matching the paper's own delta/eta naming (Eq. 4.5-4.7):
    # delta scales the whole update (own-gradient + correction), eta
    # additionally scales just the opponent-shaping correction term.
    # UNTUNED: Foerster2018's own delta~0.3-0.5 is calibrated for a
    # 5-parameter sigmoid table with O(1) gradients; a several-hundred-
    # parameter neural net's unnormalized REINFORCE gradient, summed over a
    # 120-step horizon, is a different scale entirely. 0.01 is a
    # conservative starting guess (closer to a typical policy-gradient
    # learning rate than to the paper's own delta), not a derived value --
    # watch for exploding/oscillating rewards and lower it if so.
    "delta": 0.01,
    "eta": 1.0,
    # Training-loop discount, applied within reward-to-go/the correction
    # term over the *rollout* horizon above -- see module README Section 7
    # for why gamma < 1 (not a fixed known horizon) is what avoids
    # backward-induction end-game defection; this is that same gamma,
    # reused here rather than duplicated with config_env's copy.
    "gamma": 0.95,
}
