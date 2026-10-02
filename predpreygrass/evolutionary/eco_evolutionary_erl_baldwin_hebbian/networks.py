"""Forward pass and action-credited eligibility-trace plasticity for
ERL-Hebbian agents.

Fork of `eco_evolutionary_erl_baldwin/networks.py`. The evaluation network is
still a fixed (genome-specified, never learned) linear map from observation
to a scalar "goodness" value -- unchanged. What's replaced is how the action
network adapts during life.

ERL Baldwin's `reinforce_update` directly modified the LIVE action network's
weights in place (a policy-gradient step). This module instead keeps the
action network's weights fixed for life (same as eval_weights) and adds a
separate per-life trace; the network actually used to act is a genome-scaled
combination of the two (`effective_action_weights`). Only the trace is ever
updated during life -- the base weights and the genome's plasticity
coefficients (`action_alpha_weights`/`action_alpha_bias`, genome.py) never
change outside of reproduction.

Reinforcement signal (unchanged from ERL Baldwin, Ackley & Littman 1991,
Section 2): R_t = E_t - E_{t-1}, the change in the agent's own innate
evaluation of its situation from one step to the next. No external reward
function.

**Revision history, and why the update rule changed mid-module:** the first
version of `hebbian_trace_update` correlated `obs` against the full
`prev_probs` output vector (plain reward-modulated Hebbian plasticity, per
Miconi et al.'s "Backpropamine," 2019) -- reinforcing whatever the policy
already tended to prefer across ALL actions, not specifically the action
that was taken. A controlled diagnostic (same founder genome, same
placement, same seed, a carnivore to evade, compared directly against ERL
Baldwin's `reinforce_update` from the identical starting point -- see
README.md's "Diagnostic history" section) showed that version essentially
never moved an agent's behavior: the chosen action stayed fixed for an
entire 3000-step lifetime regardless of a visible, approaching threat, while
ERL Baldwin's mechanism visibly adapted. Scaling the plasticity coefficients
up to 20x didn't close the gap. The diagnosed reason: untargeted correlation
against the whole probability vector carries no information about which
action was actually responsible for the outcome -- a real credit-assignment
gap, not a tuning gap.

The update below fixes that by crediting the SPECIFIC action taken, the same
quantity REINFORCE's own gradient uses
(`grad_logits = one_hot(action_taken) - probs`, see
`eco_evolutionary_erl_baldwin/networks.py::reinforce_update`), but routed
through a decaying per-life trace scaled by an evolvable genome coefficient
instead of directly and unboundedly modifying live weights. This is no
longer *plain* Hebbian plasticity (Miconi et al.'s original formulation has
no action-specific credit term at all) -- it's closer to what the
computational-neuroscience literature calls reward-modulated / three-factor
Hebbian learning, or equivalently REINFORCE with an eligibility trace
(Sutton & Barto): the trace IS essentially REINFORCE's per-step gradient
direction, accumulated with decay rather than applied directly. Still
meaningfully different from ERL Baldwin architecturally (bounded,
decaying, genome-scaled trace vs. an unbounded direct weight accumulator),
so it still answers the module's real question -- just not by testing
*unmodified* Miconi-style plasticity anymore. See README.md for the full
reasoning.
"""

import numpy as np


def routed_action_alpha(
    action_alpha_weights: np.ndarray,
    action_alpha_bias: np.ndarray,
    mode: str = "genomic",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the coefficients used to route plastic traces into behavior.

    ``uniform_positive`` is a magnitude-matched mechanistic control: each
    tensor is replaced by its own mean absolute genomic value.  It removes
    both random signs and per-site heterogeneity without modifying the genome.
    """
    if mode == "genomic":
        return action_alpha_weights, action_alpha_bias
    if mode == "uniform_positive":
        return (
            np.full_like(action_alpha_weights, np.mean(np.abs(action_alpha_weights))),
            np.full_like(action_alpha_bias, np.mean(np.abs(action_alpha_bias))),
        )
    raise ValueError(f"unknown action_alpha_mode {mode!r}")


def evaluate(obs: np.ndarray, eval_weights: np.ndarray, eval_bias: float) -> float:
    return float(np.dot(obs, eval_weights) + eval_bias)


def effective_action_weights(
    action_weights: np.ndarray,
    action_bias: np.ndarray,
    action_alpha_weights: np.ndarray,
    action_alpha_bias: np.ndarray,
    hebb_trace: np.ndarray,
    hebb_bias_trace: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The network actually used to act this step: genome base weights plus
    the genome's own plasticity coefficients scaling the agent's current
    Hebbian trace. Pure function -- never mutates its inputs, callers decide
    whether to cache the result. An agent whose `hebb_trace` is still all
    zero (newborn, or a non-learning strategy that never updates it) behaves
    identically to using `action_weights`/`action_bias` directly."""
    eff_weights = action_weights + action_alpha_weights * hebb_trace
    eff_bias = action_bias + action_alpha_bias * hebb_bias_trace
    return eff_weights, eff_bias


def action_probs(obs: np.ndarray, action_weights: np.ndarray, action_bias: np.ndarray) -> np.ndarray:
    logits = obs @ action_weights + action_bias
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def sample_action(probs: np.ndarray, rng: np.random.Generator) -> int:
    # Equivalent to rng.choice(len(probs), p=probs) but ~10x faster here --
    # rng.choice's generic-array-of-any-size validation/setup overhead
    # dominates for an action space this small (profiled: this was the
    # single largest tottime contributor in a multi-hundred-agent run).
    u = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if u < cumulative:
            return i
    return len(probs) - 1


def hebbian_trace_update(
    hebb_trace: np.ndarray,
    hebb_bias_trace: np.ndarray,
    obs: np.ndarray,
    prev_probs: np.ndarray,
    prev_action: int,
    reinforcement: float,
    eta: float,
    trace_clip: float,
) -> None:
    """In-place, action-credited eligibility-trace update of the LIVE trace
    only -- never the genome's base weights or plasticity coefficients.

    `obs` is the presynaptic input (the observation that produced
    `prev_probs`/`prev_action`). `credit = one_hot(prev_action) - prev_probs`
    is the SAME quantity ERL Baldwin's `reinforce_update` calls
    `grad_logits` -- it's positive for the action actually taken and
    negative for the others, scaled by how confident the policy already was.
    This is what makes the update action-specific rather than plain
    correlation against the whole output distribution (see this module's
    docstring for why the untargeted version was replaced). `reinforcement`
    is E_t - E_{t-1}, computed by the caller exactly as in ERL Baldwin.

    credit = one_hot(prev_action, n_actions) - prev_probs
    hebb_trace(t) = clip((1 - eta) * hebb_trace(t-1)
                          + eta * reinforcement * outer(obs, credit),
                          -trace_clip, trace_clip)

    Zero reinforcement leaves the trace exactly unchanged except for its own
    decay term -- there is no reinforcement-free branch (unlike ERL
    Baldwin's `reinforce_update`, which early-returns on exactly zero
    reinforcement): decay alone is a real, if small, effect here, since the
    trace is continuous state rather than a one-shot weight nudge.
    """
    credit = -prev_probs
    credit[prev_action] += 1.0

    hebb_trace *= 1.0 - eta
    hebb_trace += eta * reinforcement * np.outer(obs, credit)
    np.clip(hebb_trace, -trace_clip, trace_clip, out=hebb_trace)

    hebb_bias_trace *= 1.0 - eta
    hebb_bias_trace += eta * reinforcement * credit
    np.clip(hebb_bias_trace, -trace_clip, trace_clip, out=hebb_bias_trace)
