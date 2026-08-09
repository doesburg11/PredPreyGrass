"""Per-agent genome for ERL agents (Ackley & Littman 1991 style).

Each agent's genome directly encodes the weights of two single-layer
networks:
  - eval_weights / eval_bias: the evaluation network. Fixed for the
    agent's entire life -- a genetically specified "sense of goodness"
    of the current situation. Never touched by learning.
  - action_weights / action_bias: the action network's INITIAL weights
    only. A live copy of these is made at birth and adjusted during the
    agent's life by reinforcement learning (see networks.py) -- but the
    genome record itself is never modified by that learning.

This separation is the whole point: reproduction always copies from the
genome record (this module), never from an agent's live, post-learning
action network. Whatever an agent learned during its life is discarded
at reproduction; only the pre-learning genome (plus mutation/crossover)
is passed to offspring. That is what makes this Darwinian rather than
Lamarckian -- the architecture has no channel for learned/acquired
changes to reach the next generation, not merely a convention against
it. See README.md.
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
        """All genome sites as one flat vector, for functional-constraint tracking."""
        return np.concatenate(
            [self.eval_weights, [self.eval_bias], self.action_weights.ravel(), self.action_bias]
        )


def founder_genome(obs_dim: int, n_actions: int, rng: np.random.Generator, init_std: float = 0.5) -> Genome:
    return Genome(
        eval_weights=rng.normal(0.0, init_std, size=obs_dim),
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


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    """Uniform crossover per-site between two parent genomes (Ackley & Littman's B2)."""

    def mix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        mask = rng.random(x.shape) < 0.5
        out = x.copy()
        out[mask] = y[mask]
        return out

    return Genome(
        eval_weights=mix(a.eval_weights, b.eval_weights),
        eval_bias=a.eval_bias if rng.random() < 0.5 else b.eval_bias,
        action_weights=mix(a.action_weights, b.action_weights),
        action_bias=mix(a.action_bias, b.action_bias),
    )
