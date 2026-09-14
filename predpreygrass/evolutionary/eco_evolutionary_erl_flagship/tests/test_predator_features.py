import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import CH_PREY
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.predator_features import (
    PREDATOR_FEATURE_NAMES,
    extract_predator_features,
)


class _FakeEnv:
    def __init__(self, obs, energy, threshold):
        self._obs = obs
        self.agent_energies = {"predator_0": energy}
        self.predator_creation_energy_threshold = threshold

    def _get_observation(self, agent_id):
        return self._obs


def _empty_obs(obs_range=7):
    return np.zeros((4, obs_range, obs_range))


def test_feature_vector_shape_and_names():
    env = _FakeEnv(_empty_obs(), energy=4.0, threshold=12.0)
    feat = extract_predator_features(env, "predator_0")
    assert feat.shape == (len(PREDATOR_FEATURE_NAMES),) == (4,)


def test_energy_norm_clipped_to_one():
    env = _FakeEnv(_empty_obs(), energy=20.0, threshold=12.0)
    feat = extract_predator_features(env, "predator_0")
    assert feat[PREDATOR_FEATURE_NAMES.index("energy_norm")] == 1.0


def test_no_prey_visible_gives_zero_offsets():
    env = _FakeEnv(_empty_obs(), energy=4.0, threshold=12.0)
    feat = extract_predator_features(env, "predator_0")
    for name in ["prey_dx", "prey_dy", "prey_proximity"]:
        assert feat[PREDATOR_FEATURE_NAMES.index(name)] == 0.0


def test_prey_detected_at_known_offset():
    obs = _empty_obs()
    half = 3  # obs_range=7 -> half = 7 // 2 = 3
    obs[CH_PREY, half + 1, half - 1] = 2.0
    env = _FakeEnv(obs, energy=4.0, threshold=12.0)
    feat = extract_predator_features(env, "predator_0")
    idx = PREDATOR_FEATURE_NAMES.index
    assert feat[idx("prey_dx")] == pytest.approx(1.0 / half)
    assert feat[idx("prey_dy")] == pytest.approx(-1.0 / half)
    assert feat[idx("prey_proximity")] == pytest.approx(1.0 - 1.0 / half)
