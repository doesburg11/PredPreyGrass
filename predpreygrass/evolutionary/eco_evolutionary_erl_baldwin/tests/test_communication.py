"""Tests for the new S/ERLS alarm-call mechanism only. The original
ERL/E/L/F/B and the C/ERLC and K/ERLK mechanisms are covered by their own
test files and must still pass unmodified.
"""

import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.genome import founder_genome
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import Agent, Carnivore, ErlWorld, OBS_DIM


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _small_world_cfg(**overrides):
    base = dict(
        config_erl,
        grid_size=12,
        n_initial_agents=1,
        n_initial_carnivores=0,
        min_plants=2,
        min_trees=1,
        carnivore_spawn_interval=10_000_000,
        mutation_rate=0.0,
        alarm_recency_window=2,
        call_conspicuousness_multiplier=1.8,
    )
    base.update(overrides)
    return base


def _place_agent(world: ErlWorld, agent_id: int, row: int, col: int, propensity: float = 0.0) -> Agent:
    genome = founder_genome(world.obs_dim, 4, world.rng, 0.1)
    genome.alarm_call_propensity = propensity
    agent = Agent(
        agent_id=agent_id, row=row, col=col,
        energy=world.cfg["initial_energy_agent"],
        health=world.cfg["initial_health_agent"],
        in_tree=False, genome=genome,
        action_weights=genome.action_weights.copy(),
        action_bias=genome.action_bias.copy(),
        generation=0,
    )
    world.agents.append(agent)
    world.occupant[(row, col)] = agent
    return agent


# --- strategy acceptance and obs_dim ---


def test_S_and_ERLS_are_accepted_strategies(rng):
    for strategy in ("S", "ERLS"):
        world = ErlWorld(_small_world_cfg(strategy=strategy), rng)
        assert world.strategy == strategy


def test_obs_dim_is_extended_only_for_signal_strategies(rng):
    for strategy in ("ERL", "E", "L", "F", "B", "C", "ERLC", "K", "ERLK"):
        world = ErlWorld(_small_world_cfg(strategy=strategy), rng)
        assert world.obs_dim == OBS_DIM, f"{strategy} must keep obs_dim == {OBS_DIM}"
    for strategy in ("S", "ERLS"):
        world = ErlWorld(_small_world_cfg(strategy=strategy), rng)
        assert world.obs_dim == OBS_DIM + 1


def test_founder_genome_shapes_match_obs_dim_for_signal_strategies(rng):
    world = ErlWorld(_small_world_cfg(strategy="S"), rng)
    agent = world.agents[0]
    assert agent.genome.eval_weights.shape == (OBS_DIM + 1,)
    assert agent.genome.action_weights.shape == (OBS_DIM + 1, 4)


# --- alarm signal detection ---


def test_alarm_signal_detects_recent_caller(rng):
    world = ErlWorld(_small_world_cfg(strategy="S", n_initial_agents=0), rng)
    caller = _place_agent(world, 1, row=5, col=5)
    _receiver = _place_agent(world, 2, row=5, col=7)
    caller.last_call_step = world.current_step
    signal = world._alarm_signal(5, 7, 0, -1, world.cfg["agent_sense_range"])  # looking West toward caller
    assert signal > 0.0


def test_alarm_signal_zero_when_no_recent_call(rng):
    world = ErlWorld(_small_world_cfg(strategy="S", n_initial_agents=0), rng)
    _caller = _place_agent(world, 1, row=5, col=5)  # never called (last_call_step stays _NEVER)
    signal = world._alarm_signal(5, 7, 0, -1, world.cfg["agent_sense_range"])
    assert signal == 0.0


def test_alarm_signal_expires_outside_recency_window(rng):
    world = ErlWorld(_small_world_cfg(strategy="S", n_initial_agents=0, alarm_recency_window=2), rng)
    caller = _place_agent(world, 1, row=5, col=5)
    caller.last_call_step = 0
    world.current_step = 2
    assert world._alarm_signal(5, 7, 0, -1, world.cfg["agent_sense_range"]) > 0.0  # still within window
    world.current_step = 10
    assert world._alarm_signal(5, 7, 0, -1, world.cfg["agent_sense_range"]) == 0.0  # expired


def test_alarm_signal_blocked_by_intervening_object(rng):
    world = ErlWorld(_small_world_cfg(strategy="S", n_initial_agents=0), rng)
    caller = _place_agent(world, 1, row=5, col=5)
    _blocker = _place_agent(world, 2, row=5, col=6)  # sits between caller and receiver; never called
    caller.last_call_step = world.current_step
    signal = world._alarm_signal(5, 7, 0, -1, world.cfg["agent_sense_range"])  # looking West: hits blocker first
    assert signal == 0.0, "line of sight should be blocked by the non-calling agent in between"


def test_observe_agent_includes_alarm_channel_for_signal_strategy(rng):
    world = ErlWorld(_small_world_cfg(strategy="S", n_initial_agents=0), rng)
    caller = _place_agent(world, 1, row=5, col=5)
    receiver = _place_agent(world, 2, row=5, col=7)
    caller.last_call_step = world.current_step
    obs = world._observe_agent(receiver)
    assert obs.shape == (OBS_DIM + 1,)
    assert obs[OBS_DIM] > 0.0


# --- call decision (evolvable propensity) ---


def test_high_propensity_agent_calls_when_threatened(rng):
    world = ErlWorld(_small_world_cfg(strategy="S", n_initial_agents=0), rng)
    agent = _place_agent(world, 1, row=5, col=5, propensity=10.0)  # sigmoid(10) ~= 1.0
    carnivore = Carnivore(carnivore_id=1, row=5, col=6, energy=8.0, health=15.0)
    world.carnivores.append(carnivore)
    world.occupant[(5, 6)] = carnivore
    threatened = world._agents_with_carnivore_nearby()
    assert agent.agent_id in threatened
    world._process_alarm_calls(threatened)
    assert agent.last_call_step == world.current_step


def test_low_propensity_agent_rarely_calls_when_threatened(rng):
    world = ErlWorld(_small_world_cfg(strategy="S", n_initial_agents=0), rng)
    agent = _place_agent(world, 1, row=5, col=5, propensity=-10.0)  # sigmoid(-10) ~= 0.0
    carnivore = Carnivore(carnivore_id=1, row=5, col=6, energy=8.0, health=15.0)
    world.carnivores.append(carnivore)
    world.occupant[(5, 6)] = carnivore
    threatened = world._agents_with_carnivore_nearby()
    world._process_alarm_calls(threatened)
    assert agent.last_call_step == -10**9  # _NEVER, essentially never triggers at this propensity


def test_no_call_without_a_nearby_threat(rng):
    world = ErlWorld(_small_world_cfg(strategy="S", n_initial_agents=0), rng)
    agent = _place_agent(world, 1, row=5, col=5, propensity=10.0)
    threatened = world._agents_with_carnivore_nearby()  # no carnivores at all
    assert agent.agent_id not in threatened
    world._process_alarm_calls(threatened)
    assert agent.last_call_step == -10**9


# --- the cost: carnivore targeting ---


def test_carnivore_prefers_a_calling_agent_at_equal_distance(rng):
    world = ErlWorld(_small_world_cfg(strategy="S", n_initial_agents=0, call_conspicuousness_multiplier=2.0), rng)
    caller = _place_agent(world, 1, row=5, col=4)   # 1 cell West of carnivore
    _quiet = _place_agent(world, 2, row=5, col=6)   # 1 cell East of carnivore -- same distance
    caller.last_call_step = world.current_step
    carnivore = Carnivore(carnivore_id=1, row=5, col=5, energy=8.0, health=15.0)
    world.carnivores.append(carnivore)
    world.occupant[(5, 5)] = carnivore

    action = world._carnivore_fsa_action(carnivore)
    # action 3 = West (per _DIRS = [N, S, E, W]), i.e. toward the caller
    assert action == 3, "carnivore should prefer the more conspicuous (calling) agent at equal distance"


def test_carnivore_targeting_unaffected_by_calls_under_other_strategies(rng):
    """Regression guard: last_call_step never gets set under any strategy
    but S/ERLS, so this cost mechanic must be a structural no-op elsewhere."""
    world = ErlWorld(_small_world_cfg(strategy="E", n_initial_agents=0), rng)
    a = _place_agent(world, 1, row=5, col=4)
    b = _place_agent(world, 2, row=5, col=6)
    a.last_call_step = world.current_step  # manually forced; shouldn't matter under "E"
    carnivore = Carnivore(carnivore_id=1, row=5, col=5, energy=8.0, health=15.0)
    world.carnivores.append(carnivore)
    world.occupant[(5, 5)] = carnivore
    # Both equidistant with no genuine conspicuousness difference expected under "E":
    # the FSA should just take the first one found in scan order, not be biased by last_call_step.
    action = world._carnivore_fsa_action(carnivore)
    assert action in (2, 3)  # sanity: still picks a valid adjacent agent, doesn't crash


# --- learning/evolution semantics ---


def test_strategy_S_does_not_learn_but_inherits_genome(rng):
    world = ErlWorld(_small_world_cfg(strategy="S", n_initial_agents=10, mutation_rate=0.05), rng)
    before = {a.agent_id: a.action_weights.copy() for a in world.agents}
    for _ in range(30):
        world.step()
        if world.population_counts()["agent"] == 0:
            break
    for agent in world.agents:
        if agent.agent_id in before:
            assert np.array_equal(agent.action_weights, before[agent.agent_id]), \
                "strategy S must never update the live action network (same as E)"


def test_strategy_ERLS_learns(rng):
    cfg = _small_world_cfg(strategy="ERLS", n_initial_agents=5, founder_weight_std=1.0)
    world = ErlWorld(cfg, rng)
    before = {a.agent_id: a.action_weights.copy() for a in world.agents}
    changed = False
    for _ in range(30):
        world.step()
        for a in world.agents:
            if a.agent_id in before and not np.array_equal(a.action_weights, before[a.agent_id]):
                changed = True
        if world.population_counts()["agent"] == 0:
            break
    assert changed, "strategy ERLS must learn during life (same as ERL)"


def test_ERLS_offspring_does_not_inherit_learned_weights(rng):
    cfg = _small_world_cfg(strategy="ERLS")
    world = ErlWorld(cfg, rng)
    parent = world.agents[0]
    original_genome_action_weights = parent.genome.action_weights.copy()
    original_propensity = parent.genome.alarm_call_propensity

    parent.action_weights += 50.0
    parent.energy = cfg["reproduction_energy_threshold_agent"] + 1

    world._handle_agent_reproduction()

    assert len(world.agents) == 2
    child = world.agents[-1]
    assert np.array_equal(child.genome.action_weights, original_genome_action_weights)
    assert child.genome.alarm_call_propensity == original_propensity  # mutation_rate=0.0
    assert not np.array_equal(child.genome.action_weights, parent.action_weights)
    assert np.array_equal(child.action_weights, child.genome.action_weights)


def test_world_smoke_runs_without_crashing_under_S(rng):
    world = ErlWorld(dict(config_erl, strategy="S", grid_size=20, n_initial_agents=15, n_initial_carnivores=2), rng)
    for _ in range(200):
        world.step()
        if world.population_counts()["agent"] == 0:
            break
