import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import (
    CH_BORDER,
    CH_GRASS,
    CH_PREDATOR,
    FEATURE_NAMES,
    extract_prey_features,
)


class _FakeEnv:
    """Exposes exactly what extract_prey_features needs: _get_observation,
    agent_energies, prey_creation_energy_threshold -- no real PredPreyGrass needed."""

    def __init__(self, obs, energy, threshold):
        self._obs = obs
        self.agent_energies = {"prey_0": energy}
        self.prey_creation_energy_threshold = threshold

    def _get_observation(self, agent_id):
        return self._obs


def _empty_obs(obs_range=9):
    return np.zeros((4, obs_range, obs_range))


def test_feature_vector_shape_and_names():
    env = _FakeEnv(_empty_obs(), energy=4.0, threshold=8.0)
    feat = extract_prey_features(env, "prey_0")
    assert feat.shape == (len(FEATURE_NAMES),) == (8,)


def test_energy_norm_clipped_to_one():
    env = _FakeEnv(_empty_obs(), energy=20.0, threshold=8.0)  # over threshold
    feat = extract_prey_features(env, "prey_0")
    assert feat[FEATURE_NAMES.index("energy_norm")] == 1.0


def test_energy_norm_below_threshold_not_clipped():
    env = _FakeEnv(_empty_obs(), energy=4.0, threshold=8.0)
    feat = extract_prey_features(env, "prey_0")
    assert feat[FEATURE_NAMES.index("energy_norm")] == pytest.approx(0.5)


def test_no_predator_or_food_visible_gives_zero_offsets():
    env = _FakeEnv(_empty_obs(), energy=4.0, threshold=8.0)
    feat = extract_prey_features(env, "prey_0")
    for name in ["predator_dx", "predator_dy", "predator_proximity", "food_dx", "food_dy", "food_proximity"]:
        assert feat[FEATURE_NAMES.index(name)] == 0.0


def test_predator_detected_at_known_offset():
    obs = _empty_obs()
    half = 4  # obs_range=9 -> half = 9 // 2 = 4
    obs[CH_PREDATOR, half + 2, half] = 5.0  # a predator 2 cells away on one axis
    env = _FakeEnv(obs, energy=4.0, threshold=8.0)
    feat = extract_prey_features(env, "prey_0")
    idx = FEATURE_NAMES.index
    assert feat[idx("predator_dx")] == pytest.approx(2.0 / half)
    assert feat[idx("predator_dy")] == pytest.approx(0.0)
    assert feat[idx("predator_proximity")] == pytest.approx(1.0 - 2.0 / half)


def test_food_detected_at_known_offset():
    obs = _empty_obs()
    half = 4
    obs[CH_GRASS, half, half - 1] = 2.0  # grass one cell away on the other axis
    env = _FakeEnv(obs, energy=4.0, threshold=8.0)
    feat = extract_prey_features(env, "prey_0")
    idx = FEATURE_NAMES.index
    assert feat[idx("food_dx")] == pytest.approx(0.0)
    assert feat[idx("food_dy")] == pytest.approx(-1.0 / half)
    assert feat[idx("food_proximity")] == pytest.approx(1.0 - 1.0 / half)


def test_nearest_of_multiple_predators_chosen():
    obs = _empty_obs()
    half = 4
    obs[CH_PREDATOR, half + 3, half] = 1.0  # farther
    obs[CH_PREDATOR, half + 1, half] = 1.0  # nearer
    env = _FakeEnv(obs, energy=4.0, threshold=8.0)
    feat = extract_prey_features(env, "prey_0")
    assert feat[FEATURE_NAMES.index("predator_dx")] == pytest.approx(1.0 / half)


def test_local_grass_density_counts_in_bounds_cells_only():
    obs = _empty_obs()
    obs[CH_BORDER, :, :5] = 1.0  # first 5 columns out-of-bounds
    obs[CH_GRASS, :, 5:] = 1.0  # every in-bounds cell has grass
    env = _FakeEnv(obs, energy=4.0, threshold=8.0)
    feat = extract_prey_features(env, "prey_0")
    assert feat[FEATURE_NAMES.index("local_grass_density")] == pytest.approx(1.0)


def test_local_grass_density_zero_when_no_grass():
    env = _FakeEnv(_empty_obs(), energy=4.0, threshold=8.0)
    feat = extract_prey_features(env, "prey_0")
    assert feat[FEATURE_NAMES.index("local_grass_density")] == 0.0
