"""Hand-reduced predator observation, mirroring features.py's design for prey:
nearest-prey direction/proximity plus the predator's own energy state. Used by
centralized_predator.CentralizedPredatorPolicy -- see that module and this
module's README.md ("Predator strategy") for why predators now learn too (one
shared, centrally-updated policy), rather than the frozen-PPO-checkpoint or
purely-reactive rule-based approaches tried first.
"""

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import CH_PREY, nearest_offset

PREDATOR_FEATURE_NAMES = ["prey_dx", "prey_dy", "prey_proximity", "energy_norm"]


def extract_predator_features(env, agent_id: str) -> np.ndarray:
    """4-dim feature vector (PREDATOR_FEATURE_NAMES order) for one living predator."""
    obs = env._get_observation(agent_id)  # shape (4, obs_range, obs_range)
    half = obs.shape[1] // 2

    prey_dx, prey_dy, prey_proximity = nearest_offset(obs[CH_PREY], half)
    energy_norm = min(env.agent_energies[agent_id] / env.predator_creation_energy_threshold, 1.0)

    return np.array([prey_dx, prey_dy, prey_proximity, energy_norm], dtype=np.float64)
