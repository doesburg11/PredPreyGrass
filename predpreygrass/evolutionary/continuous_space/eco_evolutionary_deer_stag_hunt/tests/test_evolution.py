"""Genomes in the environment: inheritance, founders, gene pools and
selection differentials."""

import json

import numpy as np
import pytest

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.environment import MOVES, EvolvingDeerStagHunt, pooled_gene_pool


def run_episode(env, seed, rng, decisions=10000):
    env.reset(seed=seed)
    for _ in range(decisions):
        _, _, terms, truncs, _ = env.step({a: int(rng.integers(MOVES)) for a in env.active})
        if terms["__all__"] or truncs["__all__"]:
            return True
    return False


def fed_world(**extra):
    """Well-fed, frugal hunters that breed under random actions and live to
    the episode's end."""
    base = dict(energy_predator_initial=11.5, energy_predator_resting_metabolic_cost=0.0005, energy_predator_speed_cost=0.0, max_time_steps=3000)
    return EvolvingDeerStagHunt(dict(base, **extra))


def test_first_founders_and_bounds():
    env = EvolvingDeerStagHunt(dict(gene_founder_std=5.0))
    env.reset(seed=0)
    genomes = np.array([p.genome for p in env.predators])
    assert genomes.shape == (6, 4) and np.abs(genomes).max() <= 1.0 and (np.abs(genomes) == 1.0).any()
    assert all(p.generation == 0 and p.genome is not None for p in env.predators)
    assert all(d.genome is None for d in env.deer)


def test_newborns_inherit_with_mutation():
    env = fed_world(gene_mutation_std=0.0)
    rng = np.random.default_rng(0)
    run_episode(env, 1, rng)
    family = env.family
    assert env.births["predator"] > 0
    parents = {pid: rec for pid, rec in family.items() if rec[2] > 0}
    assert parents and sum(rec[2] for rec in family.values()) == env.births["predator"]
    founders = {tuple(rec[0]) for pid, rec in family.items() if rec[1] == 0}
    assert all(tuple(rec[0]) in founders for rec in family.values())  # no mutation: every genome a founder's

    env = fed_world(gene_mutation_std=0.05)
    run_episode(env, 1, np.random.default_rng(0))
    children = [rec for rec in env.family.values() if rec[1] > 0]
    founders = np.array([rec[0] for rec in env.family.values() if rec[1] == 0])
    assert children
    for genome, generation, _ in children:
        nearest = np.abs(founders - genome).sum(axis=1).min()
        assert nearest > 0  # mutated
    assert env.lifetime["max_generation"] == max(rec[1] for rec in env.family.values())
    assert len(env.archive) == env.births["predator"]


def test_next_founders_are_the_survivors():
    env = fed_world()
    run_episode(env, 2, np.random.default_rng(2))
    alive = sorted(tuple(p.genome) for p in env.predators)
    assert alive
    genomes, generations = env.gene_pool
    assert sorted(map(tuple, genomes)) == alive
    env.reset(seed=None)
    assert all(tuple(p.genome) in set(alive) for p in env.predators)


def test_after_extinction_the_founders_are_recent_births():
    env = EvolvingDeerStagHunt(dict(energy_predator_initial=11.5, energy_predator_resting_metabolic_cost=0.01, max_time_steps=8000))
    rng = np.random.default_rng(4)
    for seed in range(20):
        run_episode(env, seed, rng)
        if not env.predators and env.archive:
            break
    else:
        pytest.skip("no extinction after births in 20 episodes")
    genomes, _ = env.gene_pool
    assert sorted(map(tuple, genomes)) == sorted(tuple(g) for g, _ in env.archive)


def test_seeded_founders_are_reproducible_and_pools_round_trip(tmp_path):
    env = fed_world()
    run_episode(env, 5, np.random.default_rng(5))
    state = env.gene_pool_state()
    path = tmp_path / "gene_pools.json"
    path.write_text(json.dumps([state, state]))
    genomes, generations = pooled_gene_pool(path)
    assert len(genomes) == 2 * len(state["alive"])
    other = EvolvingDeerStagHunt(dict(gene_initial_pool=str(path)))
    other.reset(seed=9)
    first = [tuple(p.genome) for p in other.predators]
    assert set(first) <= set(map(tuple, genomes))
    other.set_founder_pool(genomes, generations)
    other.reset(seed=9)
    assert [tuple(p.genome) for p in other.predators] == first


def test_selection_differential():
    env = EvolvingDeerStagHunt()
    env.reset(seed=0)
    env.family = {"a": [np.array([1.0, 0, 0, 0]), 0, 3], "b": [np.array([-1.0, 0, 0, 0]), 0, 1]}
    stats = env.gene_stats()
    # relative offspring 1.5, 0.5; cov with gene (+1, -1) = 0.5.
    assert stats["gene_a_hunter_selection"] == pytest.approx(0.5)
    assert stats["gene_a_young_selection"] == 0.0
    assert stats["gene_a_hunter_mean"] == 0.0


def test_lifetime_counts_physics_steps_and_episodes():
    env = EvolvingDeerStagHunt(dict(max_time_steps=400))
    rng = np.random.default_rng(0)
    total = 0
    for seed in range(3):
        run_episode(env, seed, rng)
        total += env.time_step
    assert env.lifetime["physics_steps"] == total and env.lifetime["episodes"] == 3


def test_episode_stats_carry_gene_statistics():
    env = fed_world()
    run_episode(env, 1, np.random.default_rng(0))
    assert "gene_a_adult_mean" in env.episode_stats and "physics_steps" in env.episode_stats


def test_lifetime_selection_sums_add_up_the_episodes():
    env = fed_world()
    rng = np.random.default_rng(3)
    sums, count = np.zeros(4), 0
    for seed in range(3):
        run_episode(env, seed, rng)
        stats = env.episode_stats
        if "gene_a_hunter_selection" in stats:
            count += 1
            sums += [stats[f"gene_{g}_selection"] for g in ("a_hunter", "a_young", "a_adult", "neutral")]
    assert count > 0 and env.lifetime["selection_episodes"] == count
    np.testing.assert_allclose([env.lifetime["selection_sum"][g] for g in ("a_hunter", "a_young", "a_adult", "neutral")], sums)
