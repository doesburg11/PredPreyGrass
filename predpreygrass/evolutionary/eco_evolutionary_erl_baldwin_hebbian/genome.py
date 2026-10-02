"""Per-agent genome for ERL-Hebbian agents.

Fork of `eco_evolutionary_erl_baldwin` that replaces ERL's live-weight
reward-modulated policy-gradient update (Ackley & Littman 1991) with a
decaying, genome-scaled eligibility trace carrying the SAME action-credited
gradient direction, instead of applying it directly and unboundedly to live
weights -- see networks.py's module docstring for the revision history (an
earlier, plain-correlation reward-modulated Hebbian version per Miconi et
al. was tried first and diagnosed as having a real credit-assignment gap,
not a tuning problem) and README.md's "Diagnostic history" for the
controlled comparison that found it.

Each agent's genome directly encodes the weights of two single-layer
networks, plus a new plasticity-coefficient matrix for the second one:
  - eval_weights / eval_bias: the evaluation network. Fixed for the
    agent's entire life -- a genetically specified "sense of goodness"
    of the current situation. Never touched by learning. Unchanged from
    ERL Baldwin.
  - action_weights / action_bias: the action network's BASE weights.
    Unlike ERL Baldwin, these are never modified in place during life --
    they stay exactly as inherited, for the agent's entire life (same
    "fixed, inherited" role eval_weights already had).
  - action_alpha_weights / action_alpha_bias: NEW. Per-connection
    plasticity coefficients, same shape as action_weights/action_bias.
    Genetically inherited and mutated like any other site. At runtime
    (see networks.py's `effective_action_weights`) the network actually
    used to pick actions is `action_weights + action_alpha_weights *
    hebb_trace` -- base plus genome-scaled plasticity. An agent born with
    alpha == 0 everywhere behaves exactly like strategy "E" (evolution,
    no learning): the trace can move, but has no effect.

The live, within-lifetime-only state is now a Hebbian trace (`Agent.hebb_trace`
/ `hebb_bias_trace` in world.py), not a learned copy of the weights
themselves -- see networks.py. Reproduction always copies from the genome
record (this module: action_weights AND action_alpha_weights), never from an
agent's live trace. The trace is discarded at reproduction; only the genome
(plus mutation/crossover) is passed to offspring -- Darwinian, not
Lamarckian, same architectural guarantee as ERL Baldwin, just enforced via a
different mechanism (trace is separate per-life state, not a divergent copy
of a weight matrix). See README.md.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Genome:
    eval_weights: np.ndarray  # shape (obs_dim,)
    eval_bias: float
    action_weights: np.ndarray  # shape (obs_dim, n_actions) -- BASE weights, fixed for life
    action_bias: np.ndarray  # shape (n_actions,) -- BASE bias, fixed for life
    # --- Hebbian plasticity coefficients -- NEW, this module's whole point ---
    # Same shape as action_weights/action_bias. Scale how much the per-life
    # Hebbian trace (Agent.hebb_trace, world.py) can shift the effective
    # action network away from the base weights above. See networks.py's
    # `effective_action_weights` and this module's docstring.
    action_alpha_weights: np.ndarray  # shape (obs_dim, n_actions)
    action_alpha_bias: np.ndarray  # shape (n_actions,)
    # --- kin-selection condition (K / ERLK) -- NEW, unused by ERL/E/L/F/B/C/ERLC ---
    # An evolvable "nepotism" trait: how strongly this agent discounts
    # aggression toward genetically similar agents. Passed through
    # sigmoid(...) before use (see world.py), so any real value is valid --
    # very negative means "never discount," very positive means "discount
    # up to the configured cap." See world.py's kin-selection docstring.
    kinship_sensitivity: float = 0.0
    # --- communication condition (S / ERLS) -- NEW, unused by other strategies ---
    # Evolvable propensity to emit an alarm call when a carnivore is
    # detected nearby (sigmoid-transformed before use, see world.py). This
    # is deliberately evolvable rather than hard-coded: whether a costly
    # signal is worth emitting is itself the scientific question, per
    # Ackley & Littman's own 1994 follow-up ("Altruism in the Evolution of
    # Communication") -- see world.py's module docstring.
    alarm_call_propensity: float = 0.0

    def copy(self) -> "Genome":
        return Genome(
            eval_weights=self.eval_weights.copy(),
            eval_bias=self.eval_bias,
            action_weights=self.action_weights.copy(),
            action_bias=self.action_bias.copy(),
            action_alpha_weights=self.action_alpha_weights.copy(),
            action_alpha_bias=self.action_alpha_bias.copy(),
            kinship_sensitivity=self.kinship_sensitivity,
            alarm_call_propensity=self.alarm_call_propensity,
        )

    def flatten(self) -> np.ndarray:
        """All genome sites as one flat vector, for functional-constraint tracking.

        Includes `action_alpha_weights`/`action_alpha_bias` as part of the
        "action" portion (concatenated right after action_weights/action_bias):
        the plasticity coefficients are now as much a part of the action
        network's genome as the base weights are -- a site that mutates
        freely vs. one selection has constrained is exactly the signal this
        tracker exists to detect, and alpha is where that signal would show
        up if learning is being assimilated here. `FunctionalConstraintTracker`
        (metrics.py) computes `action_dim` from `obs_dim`/`n_actions` and must
        stay in sync with this layout.

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
            [
                self.eval_weights,
                [self.eval_bias],
                self.action_weights.ravel(),
                self.action_bias,
                self.action_alpha_weights.ravel(),
                self.action_alpha_bias,
            ]
        )


def founder_genome(
    obs_dim: int,
    n_actions: int,
    rng: np.random.Generator,
    init_std: float = 0.5,
    fixed_eval_weights: np.ndarray | None = None,
) -> Genome:
    """`fixed_eval_weights`, if given (shape (obs_dim,)), replaces the random
    init for every founder's `eval_weights` -- e.g. an empirically-evolved
    weight vector from `analyze_proximate_reward.py`, to test a specific
    discovered reward function directly rather than a random one. Everything
    else (eval_bias, action_weights, ...) is still randomly initialized as
    usual. Pair with `strategy="L"` (genome cloned exactly, no mutation) to
    keep the reward fixed across generations while each agent still learns
    its own action network via RL within its lifetime.

    `action_alpha_weights`/`action_alpha_bias` are initialized at the same
    scale (`init_std`) as every other weight site -- not a value taken from
    Miconi et al., who train plasticity coefficients by gradient descent
    rather than initializing and evolving them from scratch. Flagged here as
    a free hyperparameter choice, same spirit as this module's `lr_positive`/
    `lr_negative` before it."""
    return Genome(
        eval_weights=np.array(fixed_eval_weights, dtype=float) if fixed_eval_weights is not None
        else rng.normal(0.0, init_std, size=obs_dim),
        eval_bias=float(rng.normal(0.0, init_std)),
        action_weights=rng.normal(0.0, init_std, size=(obs_dim, n_actions)),
        action_bias=rng.normal(0.0, init_std, size=n_actions),
        action_alpha_weights=rng.normal(0.0, init_std, size=(obs_dim, n_actions)),
        action_alpha_bias=rng.normal(0.0, init_std, size=n_actions),
        kinship_sensitivity=float(rng.normal(0.0, init_std)),
        alarm_call_propensity=float(rng.normal(0.0, init_std)),
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

    mask = rng.random(child.action_alpha_weights.shape) < rate
    child.action_alpha_weights[mask] += rng.normal(0.0, std, size=int(mask.sum()))

    mask = rng.random(child.action_alpha_bias.shape) < rate
    child.action_alpha_bias[mask] += rng.normal(0.0, std, size=int(mask.sum()))

    if rng.random() < rate:
        child.kinship_sensitivity += float(rng.normal(0.0, std))

    if rng.random() < rate:
        child.alarm_call_propensity += float(rng.normal(0.0, std))

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
        action_alpha_weights=mix(a.action_alpha_weights, b.action_alpha_weights),
        action_alpha_bias=mix(a.action_alpha_bias, b.action_alpha_bias),
        kinship_sensitivity=a.kinship_sensitivity if rng.random() < 0.5 else b.kinship_sensitivity,
        alarm_call_propensity=a.alarm_call_propensity if rng.random() < 0.5 else b.alarm_call_propensity,
    )


def genome_similarity(a: Genome, b: Genome, scale: float) -> float:
    """RBF-kernel proxy for genetic relatedness: exp(-euclidean_distance / scale)
    over the BEHAVIORAL genes only (eval_weights, eval_bias, action_weights,
    action_bias, action_alpha_weights, action_alpha_bias) -- deliberately
    excludes `kinship_sensitivity` itself, so an agent's own evolved nepotism
    level doesn't inflate its measured relatedness to others. Returns a value
    in (0, 1]; 1.0 for identical genomes, approaching 0 for very different
    ones.

    This is a proxy, not literal genealogical relatedness (no parent/lineage
    bookkeeping) -- in a population that mates locally (see
    `mate_search_radius`) and reproduces with crossover+mutation, genome
    similarity and true kinship are correlated (kin share recent common
    ancestry, hence similar weights) but not identical; see world.py's
    kin-selection docstring for why this simplification was chosen.
    """
    va = np.concatenate(
        [a.eval_weights, [a.eval_bias], a.action_weights.ravel(), a.action_bias,
         a.action_alpha_weights.ravel(), a.action_alpha_bias]
    )
    vb = np.concatenate(
        [b.eval_weights, [b.eval_bias], b.action_weights.ravel(), b.action_bias,
         b.action_alpha_weights.ravel(), b.action_alpha_bias]
    )
    dist = float(np.linalg.norm(va - vb))
    return float(np.exp(-dist / scale))
