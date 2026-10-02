"""Linear TD(0) actor-critic updates for lifetime-only learning."""

import numpy as np


def state_value(obs: np.ndarray, weights: np.ndarray, bias: float) -> float:
    return float(obs @ weights + bias)


def _bounded_scale(update_norm: float, maximum: float) -> float:
    if update_norm == 0.0 or update_norm <= maximum:
        return 1.0
    return maximum / update_norm


def actor_critic_update(
    actor_weights: np.ndarray,
    actor_bias: np.ndarray,
    critic_weights: np.ndarray,
    critic_bias: np.ndarray,
    obs: np.ndarray,
    probs: np.ndarray,
    action: int,
    reward: float,
    next_obs: np.ndarray | None,
    *,
    actor_alpha: float,
    critic_beta: float,
    gamma: float,
    actor_max_update_norm: float,
    critic_max_update_norm: float,
    terminal: bool = False,
) -> tuple[float, float, float]:
    """Apply one semi-gradient TD(0) critic and score-function actor update."""
    if terminal:
        if next_obs is not None:
            raise ValueError("terminal actor-critic update cannot have next_obs")
        next_value = 0.0
    else:
        if next_obs is None:
            raise ValueError("non-terminal actor-critic update needs next_obs")
        next_value = state_value(next_obs, critic_weights, float(critic_bias[0]))

    value = state_value(obs, critic_weights, float(critic_bias[0]))
    td_error = reward + gamma * next_value - value
    if not np.isfinite(td_error):
        raise FloatingPointError("non-finite actor-critic TD error")

    critic_delta_w = critic_beta * td_error * obs
    critic_delta_b = critic_beta * td_error
    critic_norm = float(np.sqrt(np.dot(critic_delta_w, critic_delta_w) + critic_delta_b**2))
    critic_scale = _bounded_scale(critic_norm, critic_max_update_norm)
    critic_weights += critic_scale * critic_delta_w
    critic_bias[0] += critic_scale * critic_delta_b

    credit = -probs.copy()
    credit[action] += 1.0
    actor_delta_w = actor_alpha * td_error * np.outer(obs, credit)
    actor_delta_b = actor_alpha * td_error * credit
    actor_norm = float(np.sqrt(np.sum(actor_delta_w**2) + np.sum(actor_delta_b**2)))
    actor_scale = _bounded_scale(actor_norm, actor_max_update_norm)
    actor_weights += actor_scale * actor_delta_w
    actor_bias += actor_scale * actor_delta_b
    return td_error, actor_norm * actor_scale, critic_norm * critic_scale
