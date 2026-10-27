"""Fixed, non-heritable social preference for frozen-policy evaluation only.

No rewards, genes, physics, or PPO training are modified.
"""
import numpy as np

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.environment import DIRECTIONS, SPEEDS, SELF_SIZE, SLOT_SIZE


def social_scores(observation, predator_slots):
    """Action alignment with nearest visible conspecific, zero if unseen.

    Existing observation: predator slots start at SELF_SIZE; each has
    [seen, dx, dy, distance, vx, vy]. Coordinates are world-relative.
    Stopped actions have no direction and therefore no social bias.
    """
    obs = np.asarray(observation, dtype=float)
    nearest = None
    for i in range(predator_slots):
        slot = obs[SELF_SIZE + i * SLOT_SIZE:SELF_SIZE + (i + 1) * SLOT_SIZE]
        if len(slot) != SLOT_SIZE or slot[0] <= 0.5:
            continue
        x, y = float(slot[1]), float(slot[2])
        dist = float(np.hypot(x, y))
        if dist > 1e-12 and (nearest is None or dist < nearest[0]):
            nearest = (dist, x, y)
    scores = np.zeros(DIRECTIONS * len(SPEEDS), dtype=float)
    if nearest is None:
        return scores
    dist, x, y = nearest
    for direction in range(DIRECTIONS):
        theta = 2 * np.pi * direction / DIRECTIONS
        alignment = (np.sin(theta) * x - np.cos(theta) * y) / dist
        for speed_idx, speed in enumerate(SPEEDS):
            scores[speed_idx * DIRECTIONS + direction] = alignment if speed > 0 else 0.0
    return scores


def biased_logits(logits, observation, preference, strength, predator_slots):
    """Add per-agent social action bias, without changing action space."""
    if not np.isfinite(preference) or not np.isfinite(strength) or not -1 <= preference <= 1 or strength < 0:
        raise ValueError("preference must be in [-1,1], strength nonnegative")
    values = np.asarray(logits, dtype=float)
    if values.shape != (DIRECTIONS * len(SPEEDS),):
        raise ValueError("expected 48 action logits")
    return values + strength * preference * social_scores(observation, predator_slots)


def probabilities(logits):
    z = np.asarray(logits, dtype=float)
    exp = np.exp(z - np.max(z))
    return exp / exp.sum()
