"""A simple rule-based predator: move toward the nearest visible prey, explore
randomly if none is visible. Replaces `predator_policy.FrozenPredatorPolicy` as
the default (see README.md's "Predator handling" section for the full history):
a predator that learned to hunt via PPO against a CNN-based prey_policy turned
out not to transfer to hunting genome-driven prey at all -- diagnosed directly
by measuring action-distribution entropy (genome-prey's random 8-feature linear
policy is close to uniform/max-entropy, ~1.92 of a 2.197 maximum, versus even
the earliest available real prey_policy checkpoint's ~1.76, itself already
noticeably more structured/predictable). A predator's learned pursuit strategy
is specifically calibrated to exploit STRUCTURE in movement; it has no grip on
genuinely unpredictable movement, regardless of how "trained" it otherwise is.

A rule-based hunter sidesteps the whole problem: "move toward the nearest
visible thing" needs no calibration against any particular prey behavior
distribution -- it works the same whether prey move randomly, cleverly, or
anything in between. This is the same design Trial 12's own Carnivore uses
(a hard-coded FSA, never affected by the adaptive side's strategy) -- see
eco_evolutionary_erl_baldwin/world.py's `_carnivore_fsa_action`.

`FrozenPredatorPolicy` (predator_policy.py) is kept in the module, not deleted
-- it's real, tested, working code (see tests/test_reproducibility.py), just no
longer the default choice, the same way Trial 12 keeps its own dead-end
strategies (C/ERLC/K/ERLK/S/ERLS) in the codebase rather than removing them.
"""

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import CH_PREY, nearest_offset


class RuleBasedPredatorPolicy:
    """`act(observation)` has the identical signature as
    `FrozenPredatorPolicy.act` -- a drop-in replacement, no driver.py changes
    needed."""

    def __init__(self, action_to_move_tuple: dict[int, tuple[int, int]], seed: int | None = None):
        self._move_to_action = {move: action for action, move in action_to_move_tuple.items()}
        self._explore_actions = [action for action, move in action_to_move_tuple.items() if move != (0, 0)]
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> int:
        """Single-agent action for one (4, obs_range, obs_range) observation."""
        half = observation.shape[1] // 2
        dr, dc, proximity = nearest_offset(observation[CH_PREY], half)
        if proximity == 0.0 and dr == 0.0 and dc == 0.0:
            return int(self.rng.choice(self._explore_actions))
        step = (int(np.sign(dr)), int(np.sign(dc)))
        return self._move_to_action[step]
