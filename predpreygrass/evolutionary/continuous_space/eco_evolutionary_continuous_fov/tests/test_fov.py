"""The field-of-view trade-off (binocular overlap) and per-species gene-pool
freezing (DESIGN_fov_tradeoff.md in eco_evolutionary_continuous_traits)."""

import json
import math

import numpy as np
import pytest

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.config.config_env import config_env
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.environment import (
    FIXES,
    TRAITS,
    PredPreyGrassContinuous,
    pooled_gene_pool,
)
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.geometry import Vector
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.tests.compare import random_actions, snapshot
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.environment import (
    PredPreyGrassContinuous as TraitsEnvironment,
)

TRADE_OFF_KEYS = ("fov_monocular_field", "fov_monocular_distance_noise", "trait_frozen_species")
PREDATOR_SLOT = 4 + len(TRAITS)  # prey observer: the first slot is the nearest predator


def traits_config(**overrides):
    """config_env without this module's own keys, for both environments."""
    return {key: value for key, value in {**config_env, **overrides}.items() if key not in TRADE_OFF_KEYS}


def run(env, seed, steps=300):
    """Observations and full state per step under seeded random actions."""
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    trace = []
    for _ in range(steps):
        actions = random_actions(rng, obs)
        obs, rewards, terms, truncs, _ = env.step(actions)
        trace.append((snapshot(env), {a: o.tobytes() for a, o in obs.items()}, rewards))
        if terms["__all__"] or truncs["__all__"]:
            break
        obs = {a: o for a, o in obs.items() if not (terms.get(a) or truncs.get(a))}
    return trace


@pytest.mark.parametrize("seed", [0, 1])
def test_without_noise_it_is_the_continuous_traits_environment(seed):
    """Noise 0: observations, rewards and state match
    eco_evolutionary_continuous_traits bit for bit."""
    ours = PredPreyGrassContinuous({**traits_config(), "fov_monocular_distance_noise": 0.0})
    theirs = TraitsEnvironment(traits_config())
    assert run(ours, seed) == run(theirs, seed)


def test_noise_changes_observations_only():
    """The noise draws from its own generator: under the same actions the
    world evolves exactly as without it."""
    noisy = PredPreyGrassContinuous({**config_env, "fov_monocular_distance_noise": 0.3})
    clean = PredPreyGrassContinuous({**config_env, "fov_monocular_distance_noise": 0.0})
    for env in (noisy, clean):
        env.reset(seed=3)
    rng = np.random.default_rng(3)
    obs = noisy._observations()
    differ = False
    for _ in range(200):
        actions = random_actions(rng, obs)
        obs, _, terms, truncs, _ = noisy.step(actions)
        obs_clean, *_ = clean.step(actions)
        assert snapshot(noisy) == snapshot(clean)
        differ |= any(not np.array_equal(obs[a], obs_clean[a]) for a in obs)
        if terms["__all__"] or truncs["__all__"]:
            break
        obs = {a: o for a, o in obs.items() if not (terms.get(a) or truncs.get(a))}
    assert differ


def scene(bearing, prey_fov_gene=0.0, noise=0.3, orientation=0.0):
    """A prey at the centre facing the given orientation and one predator 100
    units away at the given bearing from it; returns (env, prey, predator)."""
    env = PredPreyGrassContinuous({**FIXES, "prey_count": 1, "predator_count": 1, "fov_monocular_distance_noise": noise})
    env.reset(seed=0)
    prey, predator = env.prey[0], env.predators[0]
    genome = np.array([0.0, 0.0, prey_fov_gene])
    prey.genome = genome
    settings, prey.cost_multiplier = env._phenotype("prey", genome)
    _, _, prey.max_speed, _, _, prey.view_distance, prey.fov = settings
    prey.position, prey.orientation = Vector(400.0, 400.0), orientation
    # Orientation and bearing as in geometry.in_view: angle = -atan2(dy, dx).
    radians = math.radians(orientation + bearing)
    predator.position = Vector(400.0 + 100.0 * math.cos(radians), 400.0 - 100.0 * math.sin(radians))
    return env, prey, predator


def predator_slot(env, prey, fields=4):
    obs = env._observe(prey)
    return obs[PREDATOR_SLOT : PREDATOR_SLOT + fields]


def test_overlap_formula():
    env = PredPreyGrassContinuous({"fov_monocular_field": 150.0})

    class Eye:
        orientation = 0.0

    for fov, overlap in ((75.0, 75.0), (150.0, 150.0), (240.0, 60.0), (300.0, 0.0), (360.0, 0.0)):
        Eye.fov = fov
        assert min(max(2 * env.monocular_field - fov, 0.0), fov) == overlap
        # just inside and just outside the overlap
        if 0 < overlap < fov:
            inside = math.radians(overlap / 2 - 1)
            outside = math.radians(overlap / 2 + 1)
            assert env._monocular_factor(Eye, math.cos(inside), -math.sin(inside), 0.5) == 1.0
            env.monocular_noise = 0.3
            assert env._monocular_factor(Eye, math.cos(outside), -math.sin(outside), 0.5) != 1.0
            env.monocular_noise = 0.0


def test_binocular_distances_are_exact():
    """Prey base field 240 degrees: overlap 60, so a predator at 20 degrees
    is seen exactly, every time."""
    env, prey, _ = scene(bearing=20.0)
    clean, clean_prey, _ = scene(bearing=20.0, noise=0.0)
    for _ in range(20):
        assert np.array_equal(predator_slot(env, prey), predator_slot(clean, clean_prey))


def test_monocular_distances_are_noisy_but_point_the_right_way():
    """A predator at 70 degrees (outside the 60-degree overlap): the seen
    flag and direction are kept, the distance varies around the true one."""
    env, prey, _ = scene(bearing=70.0)
    clean, clean_prey, _ = scene(bearing=70.0, noise=0.0)
    truth = predator_slot(clean, clean_prey)
    ratios = []
    for _ in range(400):
        slot = predator_slot(env, prey)
        assert slot[0] == 1.0
        assert math.atan2(slot[2], slot[1]) == pytest.approx(math.atan2(truth[2], truth[1]), abs=1e-5)
        assert slot[3] <= 1.0
        ratios.append(slot[3] / truth[3])
    log_ratios = np.log(ratios)
    assert abs(log_ratios.mean()) < 0.06 and log_ratios.std() == pytest.approx(0.3, abs=0.05)


@pytest.mark.parametrize("orientation", [0.0, 170.0, -100.0])
@pytest.mark.parametrize("bearing, binocular", [(20.0, True), (-20.0, True), (70.0, False), (-70.0, False)])
def test_both_sides_and_any_orientation(orientation, bearing, binocular):
    """The overlap is symmetric about the heading, also where the bearing
    wraps around 180 degrees; velocities are never noisy."""
    env, prey, predator = scene(bearing=bearing, orientation=orientation)
    clean, clean_prey, clean_predator = scene(bearing=bearing, orientation=orientation, noise=0.0)
    for animal in (predator, clean_predator):
        animal.velocity = Vector(1.5, -0.5)
    truth = predator_slot(clean, clean_prey, fields=6)
    assert truth[0] == 1.0
    slots = [predator_slot(env, prey, fields=6) for _ in range(20)]
    assert all(np.array_equal(slot[4:], truth[4:]) for slot in slots)
    exact = [np.array_equal(slot, truth) for slot in slots]
    assert all(exact) if binocular else not any(exact)


def test_noise_is_capped_at_the_observation_scale():
    """A predator near the edge of view: draws that would push its distance
    beyond the view distance are capped at 1."""
    env, prey, predator = scene(bearing=70.0, noise=2.0)
    radians = math.radians(70.0)
    reach = 0.95 * prey.view_distance
    predator.position = Vector(400.0 + reach * math.cos(radians), 400.0 - reach * math.sin(radians))
    distances = [predator_slot(env, prey)[3] for _ in range(200)]
    assert max(distances) == pytest.approx(1.0) and all(d <= 1.0 for d in distances)


def test_noise_is_reproducible_under_a_seed():
    a, b = (PredPreyGrassContinuous({**config_env, "fov_monocular_distance_noise": 0.3}) for _ in range(2))
    obs_a, _ = a.reset(seed=5)
    obs_b, _ = b.reset(seed=5)
    assert all(np.array_equal(obs_a[agent], obs_b[agent]) for agent in obs_a)
    rng = np.random.default_rng(5)
    for _ in range(50):
        actions = random_actions(rng, obs_a)
        obs_a, *_ = a.step(actions)
        obs_b, *_ = b.step(actions)
        assert obs_a.keys() == obs_b.keys()
        assert all(np.array_equal(obs_a[agent], obs_b[agent]) for agent in obs_a)


def test_a_narrow_field_is_all_binocular():
    """Prey with the field of view gene at -0.7 (240 * 0.5 = 120 degrees,
    below one eye's 150): fully binocular, no noise anywhere in view."""
    env, prey, _ = scene(bearing=55.0, prey_fov_gene=-0.7)
    clean, clean_prey, _ = scene(bearing=55.0, prey_fov_gene=-0.7, noise=0.0)
    assert predator_slot(clean, clean_prey)[0] == 1.0
    for _ in range(20):
        assert np.array_equal(predator_slot(env, prey), predator_slot(clean, clean_prey))


def test_grass_outside_the_overlap_is_noisy():
    env, prey, _ = scene(bearing=0.0)
    clean, clean_prey, _ = scene(bearing=0.0, noise=0.0)
    for e in (env, clean):
        e.grass.patches = e.grass.patches[:1]
    # Same grass patch for both, at 80 degrees and 60 units.
    radians = math.radians(80.0)
    for e in (env, clean):
        e.grass.patches[0].position = Vector(400.0 + 60.0 * math.cos(radians), 400.0 - 60.0 * math.sin(radians))
    grass_index = slice(-5, -1)
    noisy_values = [env._observe(prey)[grass_index] for _ in range(50)]
    truth = clean._observe(clean_prey)[grass_index]
    assert truth[0] == 1.0
    assert all(v[0] == 1.0 and v[3] <= 1.0 for v in noisy_values)
    assert len({round(float(v[3]), 6) for v in noisy_values}) > 10


def test_frozen_species_keeps_its_gene_pool(tmp_path):
    pool = {
        "predator": {"alive": np.full((10, len(TRAITS)), 0.3).tolist(), "pool": None, "archive": []},
        "prey": {"alive": np.full((12, len(TRAITS)), -0.2).tolist(), "pool": None, "archive": []},
    }
    path = tmp_path / "gene_pools.json"
    path.write_text(json.dumps([pool]))
    env = PredPreyGrassContinuous(
        {**FIXES, "trait_initial_gene_pool": str(path), "trait_frozen_species": ["prey"], "max_time_steps": 400, "trait_mutation_std": 0.2}
    )
    start = {species: env.gene_pool[species].copy() for species in ("predator", "prey")}
    rng = np.random.default_rng(0)
    for episode in range(4):
        obs, _ = env.reset(seed=episode)
        while True:
            obs, _, terms, truncs, _ = env.step(random_actions(rng, obs))
            if terms["__all__"] or truncs["__all__"]:
                break
            obs = {a: o for a, o in obs.items() if not (terms.get(a) or truncs.get(a))}
    env.reset(seed=99)
    assert np.array_equal(env.gene_pool["prey"], start["prey"])
    assert not np.array_equal(env.gene_pool["predator"], start["predator"])


def play(env, episodes, rng):
    founders = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        founders.append(env.trait_values("prey").copy())
        while True:
            obs, _, terms, truncs, _ = env.step(random_actions(rng, obs))
            if terms["__all__"] or truncs["__all__"]:
                break
            obs = {a: o for a, o in obs.items() if not (terms.get(a) or truncs.get(a))}
    return founders


def rows(genomes):
    return sorted(map(tuple, np.asarray(genomes).tolist()))


def test_frozen_species_without_an_initial_pool_keeps_its_first_founders():
    """No trait_initial_gene_pool: the first episode's founders become the
    frozen pool, not a fresh N(0, founder_std) draw each episode."""
    env = PredPreyGrassContinuous({**FIXES, "trait_frozen_species": ["prey"], "max_time_steps": 300, "trait_founder_std": 0.3})
    founders = play(env, 3, np.random.default_rng(1))
    assert all(rows(f) == rows(founders[0]) for f in founders)


def test_a_frozen_pool_survives_a_checkpoint(tmp_path):
    """gene_pool_state and pooled_gene_pool: a frozen species restarts from
    its frozen pool, not from its (mutated) living animals or recent births;
    an evolving species from its living animals."""
    start = np.random.default_rng(0).normal(0.0, 0.3, size=(12, len(TRAITS)))
    path = tmp_path / "gene_pools.json"
    path.write_text(json.dumps([{"predator": {"alive": [], "pool": None, "archive": []}, "prey": {"alive": start.tolist(), "pool": None, "archive": []}}]))
    config = {**FIXES, "trait_initial_gene_pool": str(path), "trait_frozen_species": ["prey"], "max_time_steps": 300, "trait_mutation_std": 0.2}
    env = PredPreyGrassContinuous(config)
    play(env, 2, np.random.default_rng(2))
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps([env.gene_pool_state(), env.gene_pool_state()]))
    pool = pooled_gene_pool(saved)
    assert rows(pool["prey"]) == rows(np.concatenate([start, start]))
    assert rows(pool["predator"]) == rows(np.concatenate([env.trait_values("predator")] * 2))
    # After the species died out: still the frozen pool, not the archive.
    env.prey, env.archive["prey"] = [], [np.full(len(TRAITS), 0.5)]
    saved.write_text(json.dumps([env.gene_pool_state()]))
    assert rows(pooled_gene_pool(saved)["prey"]) == rows(start)
    # And the branch continues from it.
    resumed = PredPreyGrassContinuous({**config, "trait_initial_gene_pool": str(saved)})
    assert rows(resumed.gene_pool["prey"]) == rows(start)


@pytest.mark.parametrize(
    "bad",
    [
        {"fov_monocular_field": 0},
        {"fov_monocular_field": 400},
        {"fov_monocular_distance_noise": -0.1},
        {"fov_monocular_distance_noise": float("nan")},
        {"trait_frozen_species": "prey"},
        {"trait_frozen_species": ["wolf"]},
        {"trait_frozen_species": None},
        {"trait_frozen_species": ("prey",)},
        {"trait_frozen_species": ["prey", "prey"]},
        {"trait_frozen_species": [1]},
    ],
)
def test_invalid_settings(bad):
    with pytest.raises(ValueError):
        PredPreyGrassContinuous(bad)


def test_module_config_is_the_planned_experiment():
    assert config_env["fov_monocular_field"] == 150.0
    assert config_env["fov_monocular_distance_noise"] == 0.3
    assert config_env["trait_cost_fov"] == 0.0
    assert (config_env["trait_cost_speed"], config_env["trait_cost_vision"]) == (0.25, 0.125)
    PredPreyGrassContinuous(config_env)


def test_workers_keep_their_own_frozen_founders(tmp_path):
    """Two simulation workers without an initial pool: each keeps its own
    first founders, and a checkpoint pools both."""
    config = {**FIXES, "trait_frozen_species": ["prey"], "max_time_steps": 200, "trait_founder_std": 0.3}
    workers = [PredPreyGrassContinuous(config) for _ in range(2)]
    founders = []
    for k, env in enumerate(workers):
        rng = np.random.default_rng(k)
        seen = []
        for episode in range(2):
            obs, _ = env.reset(seed=100 * k + episode)
            seen.append(rows(env.trait_values("prey")))
            for _ in range(20):
                obs, _, terms, truncs, _ = env.step(random_actions(rng, obs))
                obs = {a: o for a, o in obs.items() if not (terms.get(a) or truncs.get(a))}
        assert seen[0] == seen[1]
        founders.append(seen[0])
    assert founders[0] != founders[1]
    saved = tmp_path / "gene_pools.json"
    saved.write_text(json.dumps([env.gene_pool_state() for env in workers]))
    assert rows(pooled_gene_pool(saved)["prey"]) == sorted(founders[0] + founders[1])
