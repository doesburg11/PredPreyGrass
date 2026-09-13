"""Per-prey genome for Trial 13 (ERL genome architecture ported onto the flagship
PredPreyGrass ecology). Structurally identical to
eco_evolutionary_erl_baldwin/genome.py, minus the kinship_sensitivity/
alarm_call_propensity fields (no flagship equivalent -- those are Trial-12-specific
cooperation/communication extensions, see this module's README.md).

  - eval_weights / eval_bias: the evaluation network. Fixed for the agent's entire
    life -- a genetically specified "sense of goodness" of the current situation.
    Never touched by learning.
  - action_weights / action_bias: the action network's INITIAL weights only. A live
    copy of these is made at birth and adjusted during the agent's life by
    reinforcement learning (see networks.py) -- but the genome record itself is
    never modified by that learning.

Reproduction always copies from the genome record (this module), never from an
agent's live, post-learning action network -- Darwinian, not Lamarckian, exactly as
in Trial 12. See this module's README.md for why Trial 13 uses mutation-only
(asexual) inheritance, unlike Trial 12's sexual crossover.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Genome:
    eval_weights: np.ndarray  # shape (obs_dim,)
    eval_bias: float
    action_weights: np.ndarray  # shape (obs_dim, n_actions)
    action_bias: np.ndarray  # shape (n_actions,)

    def copy(self) -> "Genome":
        return Genome(
            eval_weights=self.eval_weights.copy(),
            eval_bias=self.eval_bias,
            action_weights=self.action_weights.copy(),
            action_bias=self.action_bias.copy(),
        )

    def flatten(self) -> np.ndarray:
        """All genome sites as one flat vector."""
        return np.concatenate(
            [self.eval_weights, [self.eval_bias], self.action_weights.ravel(), self.action_bias]
        )


def founder_genome(
    obs_dim: int,
    n_actions: int,
    rng: np.random.Generator,
    init_std: float = 0.5,
    fixed_eval_weights: np.ndarray | None = None,
) -> Genome:
    """`fixed_eval_weights`, if given (shape (obs_dim,)), replaces the random init
    for every founder's `eval_weights` instead of a random one. Everything else
    (eval_bias, action_weights, ...) is still randomly initialized as usual."""
    return Genome(
        eval_weights=np.array(fixed_eval_weights, dtype=float) if fixed_eval_weights is not None
        else rng.normal(0.0, init_std, size=obs_dim),
        eval_bias=float(rng.normal(0.0, init_std)),
        action_weights=rng.normal(0.0, init_std, size=(obs_dim, n_actions)),
        action_bias=rng.normal(0.0, init_std, size=n_actions),
    )


def mutate(genome: Genome, rng: np.random.Generator, rate: float, std: float) -> Genome:
    """Return a mutated copy of `genome`; each site independently mutated with probability `rate`."""
    child = genome.copy()

    mask = rng.random(child.eval_weights.shape) < rate
    child.eval_weights[mask] += rng.normal(0.0, std, size=int(mask.sum()))

    if rng.random() < rate:
        child.eval_bias += float(rng.normal(0.0, std))

    mask = rng.random(child.action_weights.shape) < rate
    child.action_weights[mask] += rng.normal(0.0, std, size=int(mask.sum()))

    mask = rng.random(child.action_bias.shape) < rate
    child.action_bias[mask] += rng.normal(0.0, std, size=int(mask.sum()))

    return child
