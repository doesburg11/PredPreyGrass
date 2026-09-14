import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.centralized_predator import CentralizedPredatorPolicy


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_shapes(rng):
    policy = CentralizedPredatorPolicy(obs_dim=4, n_actions=9, rng=rng)
    assert policy.action_weights.shape == (4, 9)
    assert policy.action_bias.shape == (9,)


def test_act_returns_valid_action_index(rng):
    policy = CentralizedPredatorPolicy(obs_dim=4, n_actions=9, rng=rng)
    features = np.array([0.5, -0.2, 0.8, 0.3])
    action = policy.act(features, rng)
    assert 0 <= action < 9


def test_update_changes_shared_weights(rng):
    policy = CentralizedPredatorPolicy(obs_dim=4, n_actions=9, rng=rng)
    before = policy.action_weights.copy()
    policy.update(features=np.array([1.0, 0.0, 0.0, 0.0]), action=3, reinforcement=1.0)
    assert not np.array_equal(before, policy.action_weights)


def test_zero_reinforcement_is_a_noop(rng):
    policy = CentralizedPredatorPolicy(obs_dim=4, n_actions=9, rng=rng)
    before_weights = policy.action_weights.copy()
    before_bias = policy.action_bias.copy()
    policy.update(features=np.array([1.0, 0.0, 0.0, 0.0]), action=3, reinforcement=0.0)
    np.testing.assert_array_equal(before_weights, policy.action_weights)
    np.testing.assert_array_equal(before_bias, policy.action_bias)


def test_every_agents_experience_updates_the_same_shared_weights(rng):
    """The whole point of "centralized": two different agents' transitions both
    move the SAME weight matrix, not two separate ones."""
    policy = CentralizedPredatorPolicy(obs_dim=4, n_actions=9, rng=rng)
    policy.update(features=np.array([1.0, 0.0, 0.0, 0.0]), action=0, reinforcement=1.0)
    after_agent_a = policy.action_weights.copy()
    policy.update(features=np.array([0.0, 1.0, 0.0, 0.0]), action=1, reinforcement=-1.0)
    after_agent_b = policy.action_weights.copy()
    # Both updates landed on the same object -- agent B's update further changed
    # what agent A's update had already produced, not a fresh/separate matrix.
    assert not np.array_equal(after_agent_a, after_agent_b)


def test_action_weight_absmean_matches_manual_computation(rng):
    policy = CentralizedPredatorPolicy(obs_dim=4, n_actions=9, rng=rng)
    expected = float(np.abs(policy.action_weights).mean())
    assert policy.action_weight_absmean() == pytest.approx(expected)


def test_reproducible_with_same_rng_seed():
    rng_a = np.random.default_rng(5)
    rng_b = np.random.default_rng(5)
    policy_a = CentralizedPredatorPolicy(obs_dim=4, n_actions=9, rng=rng_a)
    policy_b = CentralizedPredatorPolicy(obs_dim=4, n_actions=9, rng=rng_b)
    np.testing.assert_array_equal(policy_a.action_weights, policy_b.action_weights)
    np.testing.assert_array_equal(policy_a.action_bias, policy_b.action_bias)
