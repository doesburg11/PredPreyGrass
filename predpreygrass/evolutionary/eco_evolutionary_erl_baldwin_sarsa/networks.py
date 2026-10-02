"""Linear softmax SARSA(lambda) for lifetime-only action-value learning."""

import numpy as np


def q_values(obs: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Return the four linear action values for one observation."""
    return obs @ weights + bias


def softmax_probs(values: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable Boltzmann behavior policy over action values."""
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("cannot sample from non-finite action values")
    scaled = values / temperature
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


def sample_action(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Sample without ``rng.choice``'s generic-array setup overhead."""
    u = rng.random()
    cumulative = 0.0
    for i, probability in enumerate(probs):
        cumulative += probability
        if u < cumulative:
            return i
    return len(probs) - 1


def sarsa_lambda_update(
    weights: np.ndarray,
    bias: np.ndarray,
    eligibility_weights: np.ndarray,
    eligibility_bias: np.ndarray,
    obs: np.ndarray,
    action: int,
    reward: float,
    next_obs: np.ndarray | None,
    next_action: int | None,
    *,
    alpha: float,
    gamma: float,
    trace_lambda: float,
    terminal: bool = False,
) -> float:
    """Apply one accumulating-trace, semi-gradient SARSA(lambda) update.

    ``next_action`` must be the action the behavior policy will actually
    execute.  At a terminal transition the bootstrap is zero and both next
    fields must be ``None``.  Returns the TD error for diagnostics.
    """
    if terminal:
        if next_obs is not None or next_action is not None:
            raise ValueError("terminal SARSA update cannot have a next state/action")
        next_q = 0.0
    else:
        if next_obs is None or next_action is None:
            raise ValueError("non-terminal SARSA update needs next_obs and next_action")
        next_q = float(q_values(next_obs, weights, bias)[next_action])

    current_q = float(q_values(obs, weights, bias)[action])
    td_error = reward + gamma * next_q - current_q
    if not np.isfinite(td_error):
        raise FloatingPointError("non-finite SARSA TD error")

    eligibility_weights *= gamma * trace_lambda
    eligibility_bias *= gamma * trace_lambda
    eligibility_weights[:, action] += obs
    eligibility_bias[action] += 1.0

    weights += alpha * td_error * eligibility_weights
    bias += alpha * td_error * eligibility_bias

    if terminal:
        eligibility_weights.fill(0.0)
        eligibility_bias.fill(0.0)
    return td_error
