"""Tests for the new K/ERLK kin-selection mechanism only. The original
ERL/E/L/F/B and the C/ERLC cooperation mechanism are covered by their own
test files and must still pass unmodified -- that's the regression check
that this addition didn't disturb either.
"""

import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.genome import founder_genome, genome_similarity
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import Agent, ErlWorld


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
        kinship_similarity_scale=2.0,
        kinship_discount_cap=0.9,
    )
    base.update(overrides)
    return base


def _place_agent(world: ErlWorld, agent_id: int, row: int, col: int, genome) -> Agent:
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


def test_K_and_ERLK_are_accepted_strategies(rng):
    for strategy in ("K", "ERLK"):
        world = ErlWorld(_small_world_cfg(strategy=strategy), rng)
        assert world.strategy == strategy


# --- genome_similarity itself ---


def test_genome_similarity_identical_genomes_is_one(rng):
    g = founder_genome(7, 4, rng, 0.5)
    assert genome_similarity(g, g.copy(), scale=2.0) == pytest.approx(1.0)


def test_genome_similarity_decreases_with_distance(rng):
    a = founder_genome(7, 4, rng, 0.1)
    b = a.copy()
    b.eval_weights += 0.5  # small perturbation: "close kin"
    c = a.copy()
    c.eval_weights += 5.0  # large perturbation: "unrelated"
    sim_close = genome_similarity(a, b, scale=2.0)
    sim_far = genome_similarity(a, c, scale=2.0)
    assert 0.0 < sim_far < sim_close < 1.0


def test_genome_similarity_ignores_kinship_sensitivity_itself(rng):
    a = founder_genome(7, 4, rng, 0.5)
    b = a.copy()
    b.kinship_sensitivity += 100.0  # wildly different nepotism trait, identical behavior genes
    assert genome_similarity(a, b, scale=2.0) == pytest.approx(1.0)


# --- genome heritability of the new trait ---


def test_kinship_sensitivity_is_heritable_and_mutatable(rng):
    from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.genome import crossover, mutate

    g = founder_genome(7, 4, rng, init_std=1.0)
    assert isinstance(g.kinship_sensitivity, float)

    c = g.copy()
    assert c.kinship_sensitivity == g.kinship_sensitivity
    c.kinship_sensitivity += 1.0
    assert c.kinship_sensitivity != g.kinship_sensitivity  # copy is independent

    m = mutate(g, rng, rate=1.0, std=0.5)
    assert m.kinship_sensitivity != g.kinship_sensitivity  # rate=1.0 -> always mutates

    a = founder_genome(7, 4, rng, 1.0)
    b = founder_genome(7, 4, rng, 1.0)
    child = crossover(a, b, rng)
    assert child.kinship_sensitivity in (a.kinship_sensitivity, b.kinship_sensitivity)


# --- the core mechanism: damage discount between kin under K/ERLK ---


def test_kinship_discount_reduces_damage_between_similar_genomes_under_K(rng):
    cfg = _small_world_cfg(strategy="K", n_initial_agents=0)
    world = ErlWorld(cfg, rng)

    base_genome = founder_genome(7, 4, rng, 0.1)
    attacker_genome = base_genome.copy()
    attacker_genome.kinship_sensitivity = 10.0  # sigmoid(10) ~= 1.0: "maximally nepotistic"
    victim_genome = base_genome.copy()  # near-identical behavioral genes -> high relatedness

    attacker = _place_agent(world, 1, row=5, col=5, genome=attacker_genome)
    victim = _place_agent(world, 2, row=5, col=6, genome=victim_genome)

    starting_health = victim.health
    world._resolve_agent_action(attacker, action=2)  # action 2 = East, per _DIRS
    damage_taken = starting_health - victim.health

    full_damage = cfg["agent_attack_damage"]
    assert damage_taken < full_damage, "kin-similar victim should take discounted damage"
    assert damage_taken == pytest.approx(full_damage * (1 - cfg["kinship_discount_cap"]), rel=0.05)


def test_no_kinship_discount_for_unrelated_agents_even_under_K(rng):
    cfg = _small_world_cfg(strategy="K", n_initial_agents=0)
    world = ErlWorld(cfg, rng)

    attacker_genome = founder_genome(7, 4, rng, 0.1)
    attacker_genome.kinship_sensitivity = 10.0
    victim_genome = founder_genome(7, 4, rng, 0.1)
    victim_genome.eval_weights += 20.0  # force a huge genome distance -> ~0 similarity
    victim_genome.action_weights += 20.0

    attacker = _place_agent(world, 1, row=5, col=5, genome=attacker_genome)
    victim = _place_agent(world, 2, row=5, col=6, genome=victim_genome)

    starting_health = victim.health
    world._resolve_agent_action(attacker, action=2)
    damage_taken = starting_health - victim.health

    assert damage_taken == pytest.approx(cfg["agent_attack_damage"], rel=0.01)


def test_kinship_discount_never_applies_under_strategies_without_it(rng):
    """Regression guard: ERL/E/L/F/B/C/ERLC must be byte-identical to their
    validated behavior -- the kinship discount must never be entered for
    them, even with a maximally nepotistic genome and identical victims."""
    for strategy in ("ERL", "E", "L", "F", "B", "C", "ERLC"):
        cfg = _small_world_cfg(strategy=strategy, n_initial_agents=0)
        world = ErlWorld(cfg, rng)

        base_genome = founder_genome(7, 4, rng, 0.1)
        attacker_genome = base_genome.copy()
        attacker_genome.kinship_sensitivity = 10.0
        victim_genome = base_genome.copy()

        attacker = _place_agent(world, 1, row=5, col=5, genome=attacker_genome)
        victim = _place_agent(world, 2, row=5, col=6, genome=victim_genome)

        starting_health = victim.health
        world._resolve_agent_action(attacker, action=2)
        damage_taken = starting_health - victim.health

        assert damage_taken == pytest.approx(cfg["agent_attack_damage"], rel=0.01), (
            f"strategy {strategy} must ignore kin selection entirely"
        )


# --- learning/evolution semantics for the new strategies ---


def test_strategy_K_does_not_learn_but_inherits_genome(rng):
    world = ErlWorld(_small_world_cfg(strategy="K", n_initial_agents=10, mutation_rate=0.05), rng)
    before = {a.agent_id: a.action_weights.copy() for a in world.agents}
    for _ in range(30):
        world.step()
        if world.population_counts()["agent"] == 0:
            break
    for agent in world.agents:
        if agent.agent_id in before:
            assert np.array_equal(agent.action_weights, before[agent.agent_id]), \
                "strategy K must never update the live action network (same as E)"


def test_strategy_ERLK_learns(rng):
    cfg = _small_world_cfg(strategy="ERLK", n_initial_agents=5, founder_weight_std=1.0)
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
    assert changed, "strategy ERLK must learn during life (same as ERL)"


def test_ERLK_offspring_does_not_inherit_learned_weights(rng):
    """The Darwinian-not-Lamarckian invariant, re-checked under ERLK since
    kinship_sensitivity is a new heritable field that could plausibly have
    been wired up incorrectly (e.g. copied from live state instead of the
    genome record)."""
    cfg = _small_world_cfg(strategy="ERLK")
    world = ErlWorld(cfg, rng)
    parent = world.agents[0]
    original_genome_action_weights = parent.genome.action_weights.copy()
    original_kinship_sensitivity = parent.genome.kinship_sensitivity

    parent.action_weights += 50.0
    parent.energy = cfg["reproduction_energy_threshold_agent"] + 1

    world._handle_agent_reproduction()

    assert len(world.agents) == 2
    child = world.agents[-1]
    assert np.array_equal(child.genome.action_weights, original_genome_action_weights)
    assert child.genome.kinship_sensitivity == original_kinship_sensitivity  # mutation_rate=0.0
    assert not np.array_equal(child.genome.action_weights, parent.action_weights)
    assert np.array_equal(child.action_weights, child.genome.action_weights)
