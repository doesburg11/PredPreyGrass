"""A single, shared predator policy -- one set of action_weights/action_bias
updated from every living predator's own experience each step (pooled, not per
agent), rather than N independently-learning predators. See this module's
README.md ("Predator strategy") for the full history: replaces both the
frozen-PPO-checkpoint predator (learned against a CNN-based prey_policy's
action-distribution statistics, which never transferred to genome-driven prey's
much higher-entropy movement -- confirmed by direct measurement, not guessed)
and the purely-reactive rule-based hunter (competent but never improves, so
either predator or prey goes extinct depending on seed, since it can't adapt
its behavior to the actual population dynamics it's facing) with a predator
that learns online, against the real prey it's actually up against in this run.

Unlike prey's evolved `eval_weights` (an intrinsically DISCOVERED reward --
that's the scientific point of Trial 13's prey side, see genome.py), the
predator's reward here is directly hand-specified: net energy change over the
step (positive on a successful catch -- a caught prey's energy transfers
directly to the predator, predpreygrass_rllib_env.py:342 -- negative otherwise
from the ambient per-step energy cost). Predators aren't a subject of the
reward-divergence question this trial tests; there's no reason to make their
reward a mystery for evolution to work out, only a real, adapting threat.

Sharing weights across all predators (rather than each getting its own, as
prey do) is deliberate: every predator's step contributes training signal to
the same small policy, so it converges from pooled experience across the whole
population rather than each individual having to rediscover effective hunting
independently -- both cheaper and far more sample-efficient, and it's exactly
how flagship's own `predator_policy` is centralized too (one shared network
across all predator agents), just updated via REINFORCE instead of PPO.
"""

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.networks import (
    action_probs,
    reinforce_update,
    sample_action,
)


class CentralizedPredatorPolicy:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        rng: np.random.Generator,
        init_std: float = 0.5,
        lr_positive: float = 0.05,
        lr_negative: float = 0.02,
    ):
        self.action_weights = rng.normal(0.0, init_std, size=(obs_dim, n_actions))
        self.action_bias = rng.normal(0.0, init_std, size=n_actions)
        self.lr_positive = lr_positive
        self.lr_negative = lr_negative

    def act(self, features: np.ndarray, rng: np.random.Generator) -> int:
        probs = action_probs(features, self.action_weights, self.action_bias)
        return sample_action(probs, rng)

    def update(self, features: np.ndarray, action: int, reinforcement: float) -> None:
        """In-place update of the ONE shared policy -- called once per living
        predator per step (see driver.py's _select_predator_action), so every
        predator's experience updates the same weights."""
        reinforce_update(
            self.action_weights, self.action_bias, features, action, reinforcement,
            self.lr_positive, self.lr_negative,
        )

    def action_weight_absmean(self) -> float:
        return float(np.abs(self.action_weights).mean())
