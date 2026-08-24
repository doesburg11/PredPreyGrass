"""Tests for the new C/ERLC cooperation mechanism only. The original
ERL/E/L/F/B behavior is covered by test_erl_baldwin.py (copied unchanged
from the validated module) and must still pass unmodified -- that's the
regression check that this addition didn't disturb the validated conditions.
"""

import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import Agent, Carnivore, ErlWorld


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
        cooperation_radius=3,
        competency_window=50,
        coop_threshold_discount_frac=0.5,
    )
    base.update(overrides)
    return base


def _make_bare_agent(world: ErlWorld, agent_id: int, row: int, col: int) -> Agent:
    """A minimal extra agent placed directly (bypassing energy/founder setup)
    for group-composition tests."""
    from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.genome import founder_genome

    genome = founder_genome(7, 4, world.rng, 0.1)
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


# --- strategy acceptance ---


def test_C_and_ERLC_are_accepted_strategies(rng):
    for strategy in ("C", "ERLC"):
        world = ErlWorld(_small_world_cfg(strategy=strategy), rng)
        assert world.strategy == strategy


# --- group-fitness detection (the core new mechanism) ---


def test_group_not_fit_when_no_competency_demonstrated(rng):
    world = ErlWorld(_small_world_cfg(strategy="C"), rng)
    agent = world.agents[0]
    assert not world._agent_group_is_cooperative_fit(agent)


def test_group_not_fit_with_only_two_of_three_competencies(rng):
    world = ErlWorld(_small_world_cfg(strategy="C"), rng)
    agent = world.agents[0]
    agent.last_forage_step = world.current_step
    agent.last_evade_step = world.current_step
    # last_reproduce_step still _NEVER
    assert not world._agent_group_is_cooperative_fit(agent)


def test_group_fit_when_agent_alone_demonstrates_all_three(rng):
    world = ErlWorld(_small_world_cfg(strategy="C"), rng)
    agent = world.agents[0]
    agent.last_forage_step = world.current_step
    agent.last_evade_step = world.current_step
    agent.last_reproduce_step = world.current_step
    assert world._agent_group_is_cooperative_fit(agent)


def test_group_fit_via_complementary_neighbors_not_individual(rng):
    """The Houghton-style credit assignment: no single member has all three
    competencies, but the GROUP does -- and that's what should count."""
    world = ErlWorld(_small_world_cfg(strategy="C", n_initial_agents=0), rng)
    a = _make_bare_agent(world, 1, row=5, col=5)
    b = _make_bare_agent(world, 2, row=5, col=6)  # within cooperation_radius=3
    c = _make_bare_agent(world, 3, row=6, col=5)  # within cooperation_radius=3

    a.last_forage_step = world.current_step
    b.last_evade_step = world.current_step
    c.last_reproduce_step = world.current_step

    # None of a, b, c individually has all three -- but the group does.
    assert world._agent_group_is_cooperative_fit(a)
    assert world._agent_group_is_cooperative_fit(b)
    assert world._agent_group_is_cooperative_fit(c)


def test_group_not_fit_when_neighbor_outside_cooperation_radius(rng):
    world = ErlWorld(_small_world_cfg(strategy="C", n_initial_agents=0, cooperation_radius=1), rng)
    a = _make_bare_agent(world, 1, row=1, col=1)
    b = _make_bare_agent(world, 2, row=5, col=5)  # far outside radius=1
    a.last_forage_step = world.current_step
    a.last_evade_step = world.current_step
    b.last_reproduce_step = world.current_step  # only b has this one, and b is out of range
    assert not world._agent_group_is_cooperative_fit(a)


def test_competency_expires_outside_window(rng):
    world = ErlWorld(_small_world_cfg(strategy="C", competency_window=10), rng)
    agent = world.agents[0]
    agent.last_forage_step = 0
    agent.last_evade_step = 0
    agent.last_reproduce_step = 0
    world.current_step = 5
    assert world._agent_group_is_cooperative_fit(agent)  # still within window
    world.current_step = 50
    assert not world._agent_group_is_cooperative_fit(agent)  # expired


# --- breeding bonus actually applies, and only under C/ERLC ---


def test_cooperative_fit_group_gets_lower_reproduction_threshold(rng):
    cfg = _small_world_cfg(strategy="C", coop_threshold_discount_frac=0.5)
    world = ErlWorld(cfg, rng)
    agent = world.agents[0]
    agent.last_forage_step = world.current_step
    agent.last_evade_step = world.current_step
    agent.last_reproduce_step = world.current_step

    discounted_threshold = cfg["reproduction_energy_threshold_agent"] * 0.5
    # Energy above the discounted threshold but below the full threshold:
    # should reproduce ONLY because of the cooperation bonus.
    agent.energy = discounted_threshold + 0.5
    assert agent.energy < cfg["reproduction_energy_threshold_agent"]

    world._handle_agent_reproduction()
    assert len(world.agents) == 2, "cooperative bonus should have allowed reproduction below the base threshold"


def test_non_cooperative_agent_does_not_get_bonus_even_under_C(rng):
    cfg = _small_world_cfg(strategy="C", coop_threshold_discount_frac=0.5)
    world = ErlWorld(cfg, rng)
    agent = world.agents[0]
    # No competencies demonstrated -> group not fit -> full threshold applies.
    discounted_threshold = cfg["reproduction_energy_threshold_agent"] * 0.5
    agent.energy = discounted_threshold + 0.5
    world._handle_agent_reproduction()
    assert len(world.agents) == 1, "without a fit group, no bonus should apply"


def test_bonus_never_applies_under_original_five_strategies(rng):
    """Regression guard: E/L/F/B/ERL must be byte-identical to the validated
    module -- the coop bonus code path must never be entered for them."""
    for strategy in ("ERL", "E", "L", "F", "B"):
        cfg = _small_world_cfg(strategy=strategy, coop_threshold_discount_frac=0.99)
        world = ErlWorld(cfg, rng)
        agent = world.agents[0]
        agent.last_forage_step = world.current_step
        agent.last_evade_step = world.current_step
        agent.last_reproduce_step = world.current_step  # would trigger bonus under C/ERLC

        discounted_threshold = cfg["reproduction_energy_threshold_agent"] * 0.01
        agent.energy = discounted_threshold + 0.5  # far below full threshold
        world._handle_agent_reproduction()
        assert len(world.agents) == 1, f"strategy {strategy} must ignore the coop bonus entirely"


# --- learning/evolution semantics for the new strategies ---


def test_strategy_C_does_not_learn_but_inherits_genome(rng):
    world = ErlWorld(_small_world_cfg(strategy="C", n_initial_agents=10, mutation_rate=0.05), rng)
    before = {a.agent_id: a.action_weights.copy() for a in world.agents}
    for _ in range(30):
        world.step()
        if world.population_counts()["agent"] == 0:
            break
    for agent in world.agents:
        if agent.agent_id in before:
            assert np.array_equal(agent.action_weights, before[agent.agent_id]), \
                "strategy C must never update the live action network (same as E)"


def test_strategy_ERLC_learns(rng):
    cfg = _small_world_cfg(strategy="ERLC", n_initial_agents=5, founder_weight_std=1.0)
    world = ErlWorld(cfg, rng)
    before = {a.agent_id: a.action_weights.copy() for a in world.agents}
    changed = False
    for _ in range(30):
        world.step()
        # Check while agents are still alive -- this small test world can go
        # fully extinct within ~20 steps, at which point there's nothing left
        # to check (that would be a false negative, not evidence against
        # learning; see the eval-value trace used to diagnose this).
        for a in world.agents:
            if a.agent_id in before and not np.array_equal(a.action_weights, before[a.agent_id]):
                changed = True
        if world.population_counts()["agent"] == 0:
            break
    assert changed, "strategy ERLC must learn during life (same as ERL)"


def test_ERLC_offspring_does_not_inherit_learned_weights(rng):
    """The same Darwinian-not-Lamarckian invariant tested for ERL in
    test_erl_baldwin.py, re-checked under ERLC since it adds a new
    reproduction-eligibility path (the coop threshold discount) that
    could plausibly have bypassed the genome/live-weights separation."""
    cfg = _small_world_cfg(strategy="ERLC")
    world = ErlWorld(cfg, rng)
    parent = world.agents[0]
    original_genome_action_weights = parent.genome.action_weights.copy()

    parent.action_weights += 50.0
    parent.last_forage_step = world.current_step
    parent.last_evade_step = world.current_step
    parent.last_reproduce_step = world.current_step
    parent.energy = cfg["reproduction_energy_threshold_agent"] + 1

    world._handle_agent_reproduction()

    assert len(world.agents) == 2
    child = world.agents[-1]
    assert np.array_equal(child.genome.action_weights, original_genome_action_weights)
    assert not np.array_equal(child.genome.action_weights, parent.action_weights)
    assert np.array_equal(child.action_weights, child.genome.action_weights)


def test_evasion_recorded_only_when_threatened_and_unattacked(rng):
    cfg = _small_world_cfg(strategy="C", n_initial_carnivores=0, carnivore_sense_range=6)
    world = ErlWorld(cfg, rng)
    agent = world.agents[0]
    carnivore = Carnivore(
        carnivore_id=999, row=agent.row, col=agent.col + 1,
        energy=cfg["initial_energy_carnivore"], health=cfg["initial_health_carnivore"],
    )
    world.carnivores.append(carnivore)
    world.occupant[(carnivore.row, carnivore.col)] = carnivore

    threatened = world._agents_with_carnivore_nearby()
    assert agent.agent_id in threatened

    world._attacked_this_step = set()  # carnivore didn't get to act yet in this synthetic check
    world._record_evasions(threatened)
    assert agent.last_evade_step == world.current_step


def test_no_evasion_recorded_when_agent_was_attacked(rng):
    cfg = _small_world_cfg(strategy="C")
    world = ErlWorld(cfg, rng)
    agent = world.agents[0]
    world._attacked_this_step = {agent.agent_id}
    world._record_evasions(frozenset({agent.agent_id}))
    assert agent.last_evade_step != world.current_step
