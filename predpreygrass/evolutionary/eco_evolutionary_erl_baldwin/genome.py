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
    # --- kin-selection condition (K / ERLK) -- NEW, unused by ERL/E/L/F/B/C/ERLC ---
    # An evolvable "nepotism" trait: how strongly this agent discounts
    # aggression toward genetically similar agents. Passed through
    # sigmoid(...) before use (see world.py), so any real value is valid --
    # very negative means "never discount," very positive means "discount
    # up to the configured cap." See world.py's kin-selection docstring.
    kinship_sensitivity: float = 0.0

    def copy(self) -> "Genome":
        return Genome(
            eval_weights=self.eval_weights.copy(),
            eval_bias=self.eval_bias,
            action_weights=self.action_weights.copy(),
            action_bias=self.action_bias.copy(),
            kinship_sensitivity=self.kinship_sensitivity,
        )

    def flatten(self) -> np.ndarray:
        """All genome sites as one flat vector, for functional-constraint tracking.

        Deliberately EXCLUDES `kinship_sensitivity` -- adding a site here
        would change the vector length the validated
        FunctionalConstraintTracker's eval/action split is computed against
        (see metrics.py), and that split is what the module's headline
        genetic-assimilation result depends on. Kin selection's own
        selection signature is tracked separately (see world.py's
        genome_stats -- `kinship_sensitivity_mean`), not folded into this
        method, so ERL/E/L/F/B/C/ERLC's constraint tracking is completely
        unaffected regardless of whether kin selection is in use.
        """
        return np.concatenate(
            [self.eval_weights, [self.eval_bias], self.action_weights.ravel(), self.action_bias]
        )


def founder_genome(obs_dim: int, n_actions: int, rng: np.random.Generator, init_std: float = 0.5) -> Genome:
    return Genome(
        eval_weights=rng.normal(0.0, init_std, size=obs_dim),
        eval_bias=float(rng.normal(0.0, init_std)),
        action_weights=rng.normal(0.0, init_std, size=(obs_dim, n_actions)),
        action_bias=rng.normal(0.0, init_std, size=n_actions),
        kinship_sensitivity=float(rng.normal(0.0, init_std)),
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

    if rng.random() < rate:
        child.kinship_sensitivity += float(rng.normal(0.0, std))

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
        kinship_sensitivity=a.kinship_sensitivity if rng.random() < 0.5 else b.kinship_sensitivity,
    )


def genome_similarity(a: Genome, b: Genome, scale: float) -> float:
    """RBF-kernel proxy for genetic relatedness: exp(-euclidean_distance / scale)
    over the BEHAVIORAL genes only (eval_weights, eval_bias, action_weights,
    action_bias) -- deliberately excludes `kinship_sensitivity` itself, so an
    agent's own evolved nepotism level doesn't inflate its measured
    relatedness to others. Returns a value in (0, 1]; 1.0 for identical
    genomes, approaching 0 for very different ones.

    This is a proxy, not literal genealogical relatedness (no parent/lineage
    bookkeeping) -- in a population that mates locally (see
    `mate_search_radius`) and reproduces with crossover+mutation, genome
    similarity and true kinship are correlated (kin share recent common
    ancestry, hence similar weights) but not identical; see world.py's
    kin-selection docstring for why this simplification was chosen.
    """
    va = np.concatenate([a.eval_weights, [a.eval_bias], a.action_weights.ravel(), a.action_bias])
    vb = np.concatenate([b.eval_weights, [b.eval_bias], b.action_weights.ravel(), b.action_bias])
    dist = float(np.linalg.norm(va - vb))
    return float(np.exp(-dist / scale))
