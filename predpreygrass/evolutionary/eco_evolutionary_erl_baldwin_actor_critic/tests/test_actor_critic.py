import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_actor_critic.config import config_actor_critic
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_actor_critic.networks import actor_critic_update
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_actor_critic.world import ActorCriticWorld


def small_config(**overrides):
    cfg = dict(config_actor_critic)
    cfg.update(grid_size=12, n_initial_agents=1, n_initial_carnivores=0, min_plants=0,
               min_trees=0, wall_interior_density=0.0, plant_growth_prob=0.0,
               tree_birth_prob=0.0, tree_death_prob=0.0,
               carnivore_spawn_interval=10**9, mutation_rate=0.0)
    cfg.update(overrides)
    return cfg


def test_actor_critic_update_matches_hand_calculation():
    actor_w = np.zeros((2, 2)); actor_b = np.zeros(2)
    critic_w = np.array([0.5, -0.25]); critic_b = np.array([0.1])
    obs = np.array([2.0, 1.0]); next_obs = np.array([1.0, 2.0])
    probs = np.array([0.4, 0.6])
    td, actor_norm, critic_norm = actor_critic_update(
        actor_w, actor_b, critic_w, critic_b, obs, probs, 1, 0.5, next_obs,
        actor_alpha=0.1, critic_beta=0.2, gamma=0.9,
        actor_max_update_norm=10.0, critic_max_update_norm=10.0)
    # V(s)=0.85, V(s')=0.10 before update; delta=0.5+0.09-0.85=-0.26.
    assert td == pytest.approx(-0.26)
    assert np.allclose(critic_w, [0.396, -0.302])
    assert critic_b[0] == pytest.approx(0.048)
    assert actor_norm > 0 and critic_norm > 0
    assert actor_b[1] < 0


def test_update_norm_caps_are_enforced():
    aw = np.zeros((2, 2)); ab = np.zeros(2); cw = np.zeros(2); cb = np.zeros(1)
    td, an, cn = actor_critic_update(
        aw, ab, cw, cb, np.ones(2), np.array([0.5, 0.5]), 0, 1000.0, np.ones(2),
        actor_alpha=1.0, critic_beta=1.0, gamma=1.0,
        actor_max_update_norm=0.1, critic_max_update_norm=0.2)
    assert td == 1000.0
    assert an == pytest.approx(0.1)
    assert cn == pytest.approx(0.2)


def test_terminal_update_has_zero_bootstrap():
    aw = np.zeros((1, 2)); ab = np.zeros(2); cw = np.array([2.0]); cb = np.array([1.0])
    td, _, _ = actor_critic_update(
        aw, ab, cw, cb, np.array([1.0]), np.array([0.5, 0.5]), 0, -1.0, None,
        actor_alpha=0.01, critic_beta=0.01, gamma=0.9,
        actor_max_update_norm=1.0, critic_max_update_norm=1.0, terminal=True)
    assert td == pytest.approx(-4.0)


def test_newborn_critic_is_zero_and_not_inherited():
    world = ActorCriticWorld(small_config(), np.random.default_rng(4))
    parent = world.agents[0]
    parent.critic_weights.fill(9.0); parent.critic_bias.fill(8.0)
    parent.energy = world.cfg["reproduction_energy_threshold_agent"] + 1
    world._handle_agent_reproduction()
    child = max(world.agents, key=lambda agent: agent.agent_id)
    assert np.all(child.critic_weights == 0.0)
    assert np.all(child.critic_bias == 0.0)


def test_learning_changes_live_actor_not_genome():
    world = ActorCriticWorld(small_config(), np.random.default_rng(5))
    agent = world.agents[0]
    genomic = agent.genome.action_weights.copy()
    for _ in range(30):
        world.step()
    assert np.array_equal(agent.genome.action_weights, genomic)


@pytest.mark.parametrize("key,value", [("actor_alpha", 0.0), ("critic_beta", -1.0),
                                         ("actor_critic_gamma", 1.1)])
def test_invalid_config_rejected(key, value):
    with pytest.raises(ValueError):
        ActorCriticWorld(small_config(**{key: value}), np.random.default_rng(0))
