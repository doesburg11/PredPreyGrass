"""Heritable traits: phenotypes, costs, inheritance, the gene pool across
episodes, and the neutral control."""

import math

import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.environment import (
    FIXES,
    TRAITS,
    PredPreyGrassContinuous,
)
from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.geometry import Vector
from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.tests.compare import random_actions, snapshot
from predpreygrass.non_evolutionary.base_environment_continuous.environment import (
    PredPreyGrassContinuous as BaseEnvironment,
)

STOP = 32


@pytest.mark.parametrize("fixes", [{}, FIXES])
def test_neutral_traits_leave_the_dynamics_of_the_base_environment(fixes):
    """Without expression, genes are carried but change nothing: the full
    state matches base_environment_continuous bit for bit."""
    for seed in (0, 1, 2):
        ours = PredPreyGrassContinuous({**fixes, "trait_expression": False})
        base = BaseEnvironment(dict(fixes))
        obs, _ = ours.reset(seed=seed)
        base.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(300):
            actions = random_actions(rng, obs)
            obs, _, terms, truncs, _ = ours.step(actions)
            base.step(actions)
            assert snapshot(ours) == snapshot(base)
            if terms["__all__"] or truncs["__all__"]:
                break
            obs = {a: o for a, o in obs.items() if not (terms.get(a) or truncs.get(a))}


def test_zero_genome_is_the_base_animal():
    env = PredPreyGrassContinuous()
    for species in ("predator", "prey"):
        settings, multiplier = env._phenotype(species, np.zeros(len(TRAITS)))
        assert settings == env.settings[species] and multiplier == 1.0


def test_phenotype_and_cost():
    env = PredPreyGrassContinuous({"trait_cost_speed": 1.0, "trait_cost_vision": 0.5, "trait_cost_fov": 0.25})
    genome = np.array([math.log(1.5), math.log(0.8), math.log(1.2)])
    (_, _, speed, _, _, view, fov), multiplier = env._phenotype("predator", genome)
    assert speed == pytest.approx(5 * 1.5) and view == pytest.approx(200 * 0.8) and fov == pytest.approx(150 * 1.2)
    assert multiplier == pytest.approx(1 + 1.0 * 0.5 + 0.5 * -0.2 + 0.25 * 0.2)


def test_vision_and_fov_are_capped():
    env = PredPreyGrassContinuous()
    (*_, view, fov), _ = env._phenotype("prey", np.array([0.0, 0.7, 0.7]))
    assert view == pytest.approx(400.0) and fov == pytest.approx(360.0)


def test_cost_multiplier_has_a_floor():
    env = PredPreyGrassContinuous({"trait_min_cost_multiplier": 0.3})
    _, multiplier = env._phenotype("prey", np.full(len(TRAITS), -0.7))
    assert multiplier == 0.3


def test_resting_cost_scales_with_the_multiplier():
    env = PredPreyGrassContinuous({"prey_count": 1, "predator_count": 1})
    env.reset(seed=0)
    prey, predator = env.prey[0], env.predators[0]
    prey.position, predator.position = Vector(100.0, 100.0), Vector(600.0, 600.0)
    for animal in (prey, predator):
        animal.velocity = Vector(0.0, 0.0)
    prey.cost_multiplier = 2.0
    before = prey.energy
    env._physics_step({"prey_0": STOP, "predator_0": STOP})
    assert before - prey.energy == pytest.approx(2.0 * env.resting_cost["prey"])


def test_observation_shows_own_traits_only_when_expressed():
    for express in (True, False):
        env = PredPreyGrassContinuous({"trait_expression": express})
        env.reset(seed=0)
        animal = env.prey[0]
        animal.genome = np.array([0.35, -0.7, 0.0])
        obs = env._observe(animal)
        assert obs.shape == (36,)
        expected = animal.genome / env.trait_bound if express else np.zeros(len(TRAITS))
        assert np.allclose(obs[4 : 4 + len(TRAITS)], expected)


def test_offspring_inherit_the_genome_with_mutation():
    env = PredPreyGrassContinuous({"prey_count": 1, "predator_count": 1, "trait_mutation_std": 0.0})
    env.reset(seed=0)
    parent = env.prey[0]
    parent.genome = np.array([0.2, -0.1, 0.3])
    parent.energy = 9.0
    env.predators[0].position = Vector(600.0, 600.0)
    env._physics_step({"prey_0": STOP, "predator_0": STOP})
    child = env.prey[-1]
    assert child is not parent and np.array_equal(child.genome, parent.genome)
    assert np.array_equal(env.archive["prey"][-1], parent.genome)


def test_mutation_stays_within_bounds():
    env = PredPreyGrassContinuous({"trait_mutation_std": 5.0})
    env.reset(seed=0)
    for _ in range(100):
        genome = env._mutated(np.zeros(len(TRAITS)))
        assert np.all(np.abs(genome) <= env.trait_bound)


def test_founders_come_from_the_survivors_of_the_last_episode():
    env = PredPreyGrassContinuous({"trait_mutation_std": 0.0})
    env.reset(seed=0)
    marks = {"predator": np.array([0.5, 0.0, 0.0]), "prey": np.array([0.0, -0.5, 0.0])}
    for species, animals in (("predator", env.predators), ("prey", env.prey)):
        for animal in animals:
            animal.genome = marks[species]
    env.reset(seed=1)
    for species, animals in (("predator", env.predators), ("prey", env.prey)):
        assert all(np.array_equal(a.genome, marks[species]) for a in animals)


def test_an_extinct_species_restarts_from_its_recent_births():
    env = PredPreyGrassContinuous({"trait_mutation_std": 0.0})
    env.reset(seed=0)
    env.archive["predator"].append(np.array([0.0, 0.0, 0.4]))
    env.predators = []
    env.reset(seed=1)
    assert all(np.array_equal(a.genome, [0.0, 0.0, 0.4]) for a in env.predators)


def test_a_frozen_gene_pool_does_not_change():
    env = PredPreyGrassContinuous({"trait_mutation_std": 0.0})
    pool = np.array([[0.1, 0.2, 0.3]])
    env.gene_pool = {"predator": pool, "prey": pool}
    env.freeze_gene_pool = True
    for seed in (0, 1):
        env.reset(seed=seed)
        assert all(np.array_equal(a.genome, pool[0]) for a in env.predators + env.prey)
        for animal in env.predators + env.prey:
            animal.genome = np.zeros(len(TRAITS))


def test_seeded_episodes_with_traits_are_reproducible():
    traces = []
    for _ in range(2):
        env = PredPreyGrassContinuous(FIXES)
        obs, _ = env.reset(seed=4)
        rng = np.random.default_rng(4)
        trace = []
        for _ in range(200):
            obs, _, terms, truncs, _ = env.step(random_actions(rng, obs))
            trace.append((snapshot(env)["animals"], [a.genome.tolist() for a in env.predators + env.prey]))
            if terms["__all__"] or truncs["__all__"]:
                break
            obs = {a: o for a, o in obs.items() if not (terms.get(a) or truncs.get(a))}
        traces.append(trace)
    assert traces[0] == traces[1]


def test_invalid_trait_settings_are_rejected():
    for settings in ({"trait_bound": 0}, {"trait_cost_speed": -1}, {"trait_expression": 1}, {"trait_archive_size": 0}):
        with pytest.raises(ValueError):
            PredPreyGrassContinuous(settings)


def test_founders_are_drawn_without_replacement_and_not_mutated():
    env = PredPreyGrassContinuous({"trait_mutation_std": 1.0})
    pool = np.array([[0.01 * k, 0.0, 0.0] for k in range(8)])
    env.gene_pool = {"predator": pool, "prey": pool}
    env.freeze_gene_pool = True
    env.reset(seed=0)
    drawn = sorted(a.genome[0] for a in env.prey)
    assert drawn == pytest.approx(sorted(pool[:, 0]))  # all eight, each once
    assert all(any(np.array_equal(a.genome, g) for g in pool) for a in env.predators)


def test_a_small_pool_is_drawn_with_replacement():
    env = PredPreyGrassContinuous()
    env.gene_pool = {"predator": np.array([[0.2, 0.0, 0.0]]), "prey": np.array([[0.0, 0.3, 0.0]])}
    env.freeze_gene_pool = True
    env.reset(seed=0)
    assert len(env.prey) == 8 and all(np.array_equal(a.genome, [0.0, 0.3, 0.0]) for a in env.prey)


def test_births_count_towards_the_parents_offspring():
    env = PredPreyGrassContinuous({"prey_count": 1, "predator_count": 1})
    env.reset(seed=0)
    parent = env.prey[0]
    parent.energy = 9.0
    env.predators[0].position = Vector(600.0, 600.0)
    env._physics_step({"prey_0": STOP, "predator_0": STOP})
    assert env.family["prey_0"][2] == 1 and env.family["prey_1"][2] == 0


def test_selection_differential():
    env = PredPreyGrassContinuous()
    env.reset(seed=0)
    env.family = {
        "prey_0": ["prey", np.array([0.0, 0.0, 0.0]), 0],
        "prey_1": ["prey", np.array([0.2, 0.0, 0.0]), 2],
        "predator_0": ["predator", np.array([0.1, 0.0, 0.0]), 0],
    }
    # Relative fitness 0 and 2, speed genes 0 and 0.2: cov = mean((-1)(-0.1), (1)(0.1)) = 0.1.
    assert env.selection_differentials("prey") == pytest.approx([0.1, 0.0, 0.0])
    assert env.selection_differentials("predator") is None  # one animal


def test_no_offspring_means_no_selection_differential():
    env = PredPreyGrassContinuous()
    env.reset(seed=0)
    assert env.selection_differentials("prey") is None


def test_trait_stats_describe_living_animals_only():
    env = PredPreyGrassContinuous()
    env.reset(seed=0)
    env.archive["predator"].append(np.array([0.5, 0.5, 0.5]))
    env.predators = []
    stats = env.trait_stats()
    assert stats["predator"]["alive"] == 0 and "speed_mean" not in stats["predator"]
    assert stats["prey"]["speed_mean"] == pytest.approx(np.mean([a.genome[0] for a in env.prey]))


def test_without_survivors_or_births_the_previous_pool_is_kept():
    env = PredPreyGrassContinuous()
    env.reset(seed=0)
    old = np.array([[0.3, 0.0, 0.0]])
    env.gene_pool["predator"] = old
    env.predators = []
    env.reset(seed=1)
    assert env.gene_pool["predator"] is old


def test_gene_pools_from_a_checkpoint(tmp_path):
    from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.checkpoints import (
        GENE_POOLS,
        load_gene_pool,
        use_gene_pool,
    )

    workers = [
        {"predator": {"alive": [[0.1, 0, 0]], "pool": None, "archive": []}, "prey": {"alive": [], "pool": [[0, 0.2, 0]], "archive": []}},
        {"predator": {"alive": [], "pool": None, "archive": [[0.3, 0, 0]]}, "prey": {"alive": [[0, 0.4, 0]], "pool": None, "archive": []}},
    ]
    (tmp_path / GENE_POOLS).write_text(__import__("json").dumps(workers))
    pool = load_gene_pool(tmp_path)
    assert pool["predator"].tolist() == [[0.1, 0, 0], [0.3, 0, 0]]
    assert pool["prey"].tolist() == [[0, 0.2, 0], [0, 0.4, 0]]
    assert load_gene_pool(tmp_path / "missing") is None
    env = PredPreyGrassContinuous({"trait_mutation_std": 0.5})
    use_gene_pool(env, pool)
    for seed in range(3):
        env.reset(seed=seed)
        for _ in range(20):
            env.step({agent: STOP for agent in env.active})
    assert env.gene_pool["predator"].tolist() == [[0.1, 0, 0], [0.3, 0, 0]]
