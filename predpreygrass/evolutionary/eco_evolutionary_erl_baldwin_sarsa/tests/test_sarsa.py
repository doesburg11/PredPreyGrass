import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin import run_erl_simulation

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_sarsa.config import config_sarsa
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_sarsa.networks import (
    q_values,
    sarsa_lambda_update,
    softmax_probs,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_sarsa.world import (
    N_ACTIONS,
    SarsaAgent,
    SarsaWorld,
)


def small_config(**overrides):
    cfg = dict(config_sarsa)
    cfg.update(
        grid_size=12,
        n_initial_agents=1,
        n_initial_carnivores=0,
        min_plants=0,
        min_trees=0,
        wall_interior_density=0.0,
        plant_growth_prob=0.0,
        tree_birth_prob=0.0,
        tree_death_prob=0.0,
        carnivore_spawn_interval=10**9,
        mutation_rate=0.0,
    )
    cfg.update(overrides)
    return cfg


def test_q_values_and_temperature_softmax():
    obs = np.array([2.0, -1.0])
    weights = np.array([[1.0, 0.0], [0.0, 1.0]])
    bias = np.array([0.5, -0.5])
    values = q_values(obs, weights, bias)
    assert np.allclose(values, [2.5, -1.5])
    assert np.isclose(softmax_probs(values, 0.7).sum(), 1.0)
    assert softmax_probs(values, 0.2)[0] > softmax_probs(values, 2.0)[0]


def test_softmax_rejects_nonfinite_q_values():
    with pytest.raises(FloatingPointError, match="non-finite"):
        softmax_probs(np.array([0.0, np.inf]), 1.0)


def test_one_step_sarsa_update_matches_hand_calculation():
    weights = np.zeros((2, 2))
    bias = np.zeros(2)
    traces_w = np.zeros_like(weights)
    traces_b = np.zeros_like(bias)
    obs = np.array([2.0, 3.0])
    next_obs = np.array([1.0, 1.0])
    weights[:, 1] = [0.5, 0.25]
    bias[1] = 0.5  # next Q = 1.25

    delta = sarsa_lambda_update(
        weights,
        bias,
        traces_w,
        traces_b,
        obs,
        0,
        reward=2.0,
        next_obs=next_obs,
        next_action=1,
        alpha=0.1,
        gamma=0.8,
        trace_lambda=0.5,
    )

    assert delta == pytest.approx(3.0)
    assert np.allclose(weights[:, 0], [0.6, 0.9])
    assert bias[0] == pytest.approx(0.3)
    assert np.allclose(traces_w[:, 0], obs)
    assert np.allclose(traces_w[:, 1], 0.0)


def test_accumulating_trace_carries_credit_backward():
    weights = np.zeros((1, 2))
    bias = np.zeros(2)
    traces_w = np.zeros_like(weights)
    traces_b = np.zeros_like(bias)
    sarsa_lambda_update(
        weights, bias, traces_w, traces_b, np.array([1.0]), 0, 0.0,
        np.array([1.0]), 1, alpha=0.1, gamma=1.0, trace_lambda=0.5,
    )
    sarsa_lambda_update(
        weights, bias, traces_w, traces_b, np.array([1.0]), 1, 1.0,
        np.array([1.0]), 1, alpha=0.1, gamma=1.0, trace_lambda=0.5,
    )
    assert weights[0, 0] > 0.0
    assert weights[0, 1] > 0.0


def test_terminal_update_has_zero_bootstrap_and_clears_traces():
    weights = np.zeros((1, 2))
    bias = np.zeros(2)
    traces_w = np.full_like(weights, 2.0)
    traces_b = np.full_like(bias, 2.0)
    delta = sarsa_lambda_update(
        weights,
        bias,
        traces_w,
        traces_b,
        np.array([1.0]),
        0,
        reward=-1.0,
        next_obs=None,
        next_action=None,
        alpha=0.1,
        gamma=1.0,
        trace_lambda=0.5,
        terminal=True,
    )
    assert delta == pytest.approx(-1.0)
    assert np.allclose(traces_w, 0.0)
    assert np.allclose(traces_b, 0.0)


def test_nonterminal_update_requires_cached_next_action():
    with pytest.raises(ValueError, match="next_obs and next_action"):
        sarsa_lambda_update(
            np.zeros((1, 2)), np.zeros(2), np.zeros((1, 2)), np.zeros(2),
            np.ones(1), 0, 0.0, np.ones(1), None,
            alpha=0.1, gamma=1.0, trace_lambda=0.5,
        )


def test_founder_has_private_q_copy_and_zero_traces():
    world = SarsaWorld(small_config(), np.random.default_rng(0))
    agent = world.agents[0]
    assert isinstance(agent, SarsaAgent)
    assert np.array_equal(agent.action_weights, agent.genome.action_weights)
    assert not np.shares_memory(agent.action_weights, agent.genome.action_weights)
    assert not np.shares_memory(agent.action_bias, agent.genome.action_bias)
    assert np.allclose(agent.eligibility_weights, 0.0)
    assert np.allclose(agent.eligibility_bias, 0.0)


def test_child_does_not_inherit_parent_learning_or_trace():
    cfg = small_config(strategy="L")
    world = SarsaWorld(cfg, np.random.default_rng(1))
    parent = world.agents[0]
    parent.action_weights += 100.0
    parent.eligibility_weights += 5.0
    parent.energy = cfg["reproduction_energy_threshold_agent"] + 1.0
    world._handle_agent_reproduction()
    child = next(agent for agent in world.agents if agent.generation == 1)
    assert np.array_equal(child.action_weights, child.genome.action_weights)
    assert np.array_equal(child.genome.action_weights, parent.genome.action_weights)
    assert not np.shares_memory(child.action_weights, parent.action_weights)
    assert not np.shares_memory(child.action_bias, parent.action_bias)
    assert np.allclose(child.eligibility_weights, 0.0)


def test_learning_disabled_strategy_does_not_change_live_q():
    world = SarsaWorld(small_config(strategy="E"), np.random.default_rng(2))
    agent = world.agents[0]
    before_weights = agent.action_weights.copy()
    before_bias = agent.action_bias.copy()
    for _ in range(10):
        world.step()
    assert np.array_equal(agent.action_weights, before_weights)
    assert np.array_equal(agent.action_bias, before_bias)


def test_learning_strategy_changes_live_q_but_not_genome():
    world = SarsaWorld(small_config(strategy="ERL", sarsa_alpha=0.1), np.random.default_rng(3))
    agent = world.agents[0]
    genome_weights = agent.genome.action_weights.copy()
    for _ in range(10):
        world.step()
    assert not np.array_equal(agent.action_weights, genome_weights)
    assert np.array_equal(agent.genome.action_weights, genome_weights)


def test_death_uses_final_evaluation_delta_once_and_is_idempotent():
    cfg = small_config(strategy="ERL", sarsa_alpha=0.1, sarsa_terminal_bonus=-1.0)
    world = SarsaWorld(cfg, np.random.default_rng(4))
    agent = world.agents[0]
    agent.prev_obs = np.ones(world.obs_dim)
    agent.prev_action = 0
    agent.prev_eval = 0.0
    before = agent.action_weights.copy()
    world._kill_agent(agent)
    after_first = agent.action_weights.copy()
    world._kill_agent(agent)
    assert not np.array_equal(after_first, before)
    assert np.array_equal(agent.action_weights, after_first)
    assert np.allclose(agent.eligibility_weights, 0.0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("sarsa_alpha", 0.0),
        ("sarsa_gamma", 1.1),
        ("sarsa_lambda", -0.1),
        ("sarsa_temperature", 0.0),
        ("sarsa_terminal_bonus", np.inf),
    ],
)
def test_invalid_sarsa_config_rejected(key, value):
    with pytest.raises(ValueError):
        SarsaWorld(small_config(**{key: value}), np.random.default_rng(0))


def test_action_space_still_has_four_actions():
    assert N_ACTIONS == 4


def test_checkpoint_preserves_sarsa_type_rng_transition_and_traces(tmp_path):
    world = SarsaWorld(small_config(), np.random.default_rng(5))
    world.step()
    agent = world.agents[0]
    agent.eligibility_weights += 2.0
    path = tmp_path / "checkpoint_step_1.pkl"
    save_checkpoint(world, path)
    restored = load_checkpoint(path)
    restored_agent = restored.agents[0]
    assert type(restored) is SarsaWorld
    assert np.array_equal(restored_agent.prev_obs, agent.prev_obs)
    assert restored_agent.prev_action == agent.prev_action
    assert np.array_equal(restored_agent.eligibility_weights, agent.eligibility_weights)
    assert restored.rng.bit_generator.state == world.rng.bit_generator.state


def test_baseline_runner_rejects_sarsa_checkpoint_type():
    assert type(SarsaWorld(small_config(), np.random.default_rng(6))) \
        is not run_erl_simulation.EXPECTED_WORLD_CLASS


def test_step_samples_each_action_only_once(monkeypatch):
    world = SarsaWorld(small_config(strategy="ERL"), np.random.default_rng(7))
    agent = world.agents[0]
    agent.prev_obs = np.zeros(world.obs_dim)
    agent.prev_action = 0
    agent.prev_eval = 0.0
    calls = []

    def choose_once(selected_agent, obs):
        calls.append((selected_agent.agent_id, obs.copy()))
        return 2

    monkeypatch.setattr(world, "_choose_action", choose_once)
    monkeypatch.setattr(world, "_resolve_agent_action", lambda *args, **kwargs: None)
    world._step_agents()
    assert len(calls) == 1
    assert agent.prev_action == 2
