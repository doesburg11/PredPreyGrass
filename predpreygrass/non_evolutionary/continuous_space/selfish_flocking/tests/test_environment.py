import numpy as np
import pytest
import torch

from predpreygrass.non_evolutionary.continuous_space.selfish_flocking.environment import (
    DISCRETE_TURNS,
    SelfishFlockingEnv,
    Settings,
    wrap_delta,
)
from predpreygrass.non_evolutionary.continuous_space.selfish_flocking.rl import (
    QNetwork,
    ReplayBuffer,
    dqn_update,
)
from predpreygrass.non_evolutionary.continuous_space.selfish_flocking.evaluate import (
    cluster_labels,
    component_sizes,
)
from predpreygrass.non_evolutionary.continuous_space.selfish_flocking.train import (
    _peer_action,
)
from predpreygrass.non_evolutionary.continuous_space.selfish_flocking.rl import Actor


def test_seeded_reset_and_observations_are_reproducible():
    settings = Settings(prey_count=10, neighbors=5)
    first, second = SelfishFlockingEnv(settings), SelfishFlockingEnv(settings)
    obs_a = first.reset(seed=17)
    obs_b = second.reset(seed=17)

    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(first.positions, second.positions)
    assert obs_a.shape == (21,)
    assert np.all((0 <= obs_a) & (obs_a <= 1))


def test_wrap_delta_uses_shortest_path_around_edges():
    delta = wrap_delta(np.array([[39.0, -39.0]]), 40.0, 40.0)
    np.testing.assert_allclose(delta, [[-1.0, 1.0]])


def test_cluster_metrics_use_transitive_and_toroidal_connectivity():
    positions = np.array([[39.5, 0.0], [0.5, 0.0], [1.5, 0.0], [20.0, 20.0]])
    labels = cluster_labels(positions, 40.0, 40.0, 1.1)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]
    sizes, noise_agents = component_sizes(positions, 40.0, 40.0, 1.1)
    np.testing.assert_array_equal(sizes, [3])
    assert noise_agents == 1


def test_step_moves_prey_and_validates_discrete_actions():
    env = SelfishFlockingEnv(Settings(prey_count=4, neighbors=2, max_steps=5))
    observation = env.reset(seed=2)
    before = env.positions.copy()
    next_observation, reward, done, info = env.step(
        2,
        lambda observations: np.full(len(observations), 2),
    )
    assert not np.array_equal(env.positions, before)
    assert next_observation.shape == observation.shape
    assert reward == 1.0
    assert not done
    assert info["steps"] == 1
    with pytest.raises(ValueError):
        env.step(5)


def test_continuous_actions_accept_batched_peer_actions():
    env = SelfishFlockingEnv(Settings(prey_count=4, neighbors=1))
    env.reset(seed=4)
    env.step(
        0.5,
        lambda observations: np.zeros(len(observations), dtype=float),
        discrete=False,
    )
    assert env.steps == 1
    assert np.all((env.positions >= 0) & (env.positions < 40))


def test_dqn_update_runs_on_a_sampled_batch():
    device = torch.device("cpu")
    online, target = QNetwork(9), QNetwork(9)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    replay = ReplayBuffer(10)
    for index in range(4):
        replay.add(np.full(9, index / 4), index % len(DISCRETE_TURNS), 1, np.zeros(9), False)

    loss = dqn_update(online, target, optimizer, replay, 2, 0.999999, device)
    assert np.isfinite(loss)


def test_policy_action_handles_single_and_batched_observations():
    device = torch.device("cpu")
    observations = np.zeros((3, 9), dtype=np.float32)
    for algo, model in (("dqn", QNetwork(9)), ("ddpg", Actor(9))):
        single = _peer_action(model, algo, observations[0], device)
        batch = _peer_action(model, algo, observations, device)
        assert np.isscalar(single)
        assert np.asarray(batch).shape == (3,)
