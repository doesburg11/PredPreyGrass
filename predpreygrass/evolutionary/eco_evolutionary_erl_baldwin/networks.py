"""Forward pass and local reinforcement learning for ERL agents.

The evaluation network is a fixed (genome-specified, never learned)
linear map from observation to a scalar "goodness" value. The action
network is a linear map from observation to action logits; it starts
each agent's life as a copy of the genome's action_weights/action_bias
and is adjusted every step by `reinforce_update`.

Reinforcement signal (Ackley & Littman 1991, Section 2): R_t = E_t -
E_{t-1}, the change in the agent's own innate evaluation of its
situation from one step to the next. No external reward function.

`reinforce_update` implements the same complementary-reinforcement
logic as the paper's CRBP (positive reinforcement makes the action just
taken more likely given that observation; negative reinforcement makes
it less likely) as a standard softmax policy-gradient step, rather than
replicating CRBP's original 2-bit stochastic-threshold backprop scheme
bit-for-bit. This is a documented simplification -- see README.md.
"""

import numpy as np


def evaluate(obs: np.ndarray, eval_weights: np.ndarray, eval_bias: float) -> float:
    return float(np.dot(obs, eval_weights) + eval_bias)


def action_probs(obs: np.ndarray, action_weights: np.ndarray, action_bias: np.ndarray) -> np.ndarray:
    logits = obs @ action_weights + action_bias
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def sample_action(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs))


def reinforce_update(
    action_weights: np.ndarray,
    action_bias: np.ndarray,
    obs: np.ndarray,
    action_taken: int,
    reinforcement: float,
    lr_positive: float,
    lr_negative: float,
) -> None:
    """In-place update of the LIVE action network only -- never the genome.

    Increases the probability of `action_taken` given `obs` when
    `reinforcement > 0`; decreases it when `reinforcement < 0`. Uses
    lr_positive vs. lr_negative (mirroring CRBP's eta_+/eta_-) so the two
    directions can be tuned independently.
    """
    if reinforcement == 0.0:
        return
    probs = action_probs(obs, action_weights, action_bias)
    grad_logits = -probs
    grad_logits[action_taken] += 1.0  # d log pi(a|obs) / d logits
    lr = lr_positive if reinforcement > 0 else lr_negative
    step = lr * reinforcement * grad_logits
    action_weights += np.outer(obs, step)
    action_bias += step
