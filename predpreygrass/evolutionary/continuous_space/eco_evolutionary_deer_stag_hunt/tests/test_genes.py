"""The innate bias (genes.py) and its torch twin (rl_module.py)."""

import math

import numpy as np
import pytest
import torch

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import genes
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.environment import MOVES, EvolvingDeerStagHunt
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.geometry import Vector
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.rl_module import torch_bias


def world(**config):
    env = EvolvingDeerStagHunt(dict(config, predator_count=2, deer_count=2))
    env.reset(seed=0)
    return env


def place(env, hunter_xy=(400, 400), other_xy=(400, 300), deer=()):
    """Hunter 0 at hunter_xy facing -y (up), hunter 1 at other_xy, deer at
    the given (x, y, energy)."""
    a, b = env.predators
    a.position, b.position = Vector(*hunter_xy), Vector(*other_xy)
    for p in (a, b):
        p.velocity, p.orientation = Vector(0, -1), 90.0
    for d, (x, y, energy) in zip(env.deer, deer):
        d.position, d.energy = Vector(x, y), energy
    for d in env.deer[len(deer):]:
        d.position = Vector(10, 790)  # out of view
    return a


def obs_of(env, animal, genome):
    animal.genome = np.array(genome, dtype=float)
    return env._observe(animal)[None, :]


def best_move(env, obs):
    b = genes.bias(obs, genes.layout(env), genes.BETA0)[0]
    return int(np.argmax(b)), b


def test_beta0_meets_its_rule():
    assert genes.beta_rule_probability(0.0) == pytest.approx(3 / 48)
    assert genes.beta_rule_probability(genes.BETA0) == pytest.approx(0.5, abs=1e-9)
    assert genes.BETA0 == pytest.approx(3.4417, abs=1e-3)


def test_move_table_matches_the_environment():
    env = world()
    animal = env.predators[0]
    for move in range(MOVES):
        v = env._desired_velocity(move, animal)
        expected = genes.DIRECTION_TABLE[move] * genes.SPEED_TABLE[move] * animal.max_speed
        assert (v.x, v.y) == pytest.approx(tuple(expected), abs=1e-9)


def test_layout_points_at_the_inputs():
    env = world()
    lay = genes.layout(env)
    assert lay["adult_threshold"] == 5 / 16
    assert lay["gene_start"] + len(genes.GENES) == lay["obs_size"] == env.obs_size["predator"]
    a = env.predators[0]
    a.genome = np.array([0.1, -0.2, 0.3, -0.4])
    np.testing.assert_allclose(env._observe(a)[lay["gene_start"] :], a.genome, atol=1e-7)


@pytest.mark.parametrize("sign", [1, -1])
def test_hunter_gene_steers_toward_or_away(sign):
    env = world()
    a = place(env, other_xy=(400, 300))  # the other hunter straight ahead (-y)
    move, b = best_move(env, obs_of(env, a, [sign, 0, 0, 0]))
    assert genes.SPEED_TABLE[move] == 1.0
    assert tuple(genes.DIRECTION_TABLE[move]) == pytest.approx((0.0, -float(sign)), abs=1e-9)
    assert np.all(b[~genes.MOVING] == 0)


def test_deer_genes_tell_young_from_adult():
    env = world()
    # Young deer ahead (-y), adult deer to the right (+x); other hunter out of view.
    a = place(env, other_xy=(10, 10), deer=[(400, 300, 3.0), (480, 400, 6.0)])
    move, _ = best_move(env, obs_of(env, a, [0, 1, 0, 0]))
    assert tuple(genes.DIRECTION_TABLE[move]) == pytest.approx((0.0, -1.0), abs=1e-9)
    # The adult is at 90 degrees, inside the 150 degree cone only if the
    # hunter faces it: turn the hunter right (+x, heading 0).
    a.velocity, a.orientation = Vector(1, 0), 0.0
    env.deer[0].position = Vector(10, 790)
    move, _ = best_move(env, obs_of(env, a, [0, 0, 1, 0]))
    assert tuple(genes.DIRECTION_TABLE[move]) == pytest.approx((1.0, 0.0), abs=1e-9)
    # The young gene ignores an adult, and the neutral gene does nothing.
    for genome in ([0, 1, 0, 0], [0, 0, 0, 1]):
        _, b = best_move(env, obs_of(env, a, genome))
        assert np.all(b == 0)


def test_nothing_in_view_means_no_bias_and_cruise():
    env = world()
    a = place(env, other_xy=(10, 10))
    a.velocity, a.orientation = Vector(-1, 0), 180.0  # heading -x
    obs = obs_of(env, a, [1, 1, 1, 1])
    lay = genes.layout(env)
    assert not genes.anything_in_view(obs, lay)[0]
    assert np.all(genes.bias(obs, lay, genes.BETA0) == 0)
    logits = genes.e_arm_logits(obs, lay, genes.BETA0)[0]
    cruise = int(np.argmax(logits))
    assert np.isfinite(logits).sum() == 1
    assert tuple(genes.DIRECTION_TABLE[cruise]) == pytest.approx((-1.0, 0.0), abs=1e-9)
    assert genes.SPEED_TABLE[cruise] == 1.0


def test_e_arm_masks_stops_when_something_is_in_view():
    env = world()
    a = place(env, other_xy=(400, 300))
    logits = genes.e_arm_logits(obs_of(env, a, [0, 0, 0, 0]), genes.layout(env), genes.BETA0)[0]
    assert np.all(np.isneginf(logits[~genes.MOVING])) and np.all(logits[genes.MOVING] == 0)
    moves = genes.sample(np.repeat(logits[None], 20000, axis=0), np.random.default_rng(0))
    assert genes.MOVING[moves].all()
    assert np.bincount(moves, minlength=MOVES)[genes.MOVING].min() > 400  # about uniform over 32


def test_numpy_and_torch_bias_agree_on_real_observations():
    env = EvolvingDeerStagHunt(dict(width=300, height=300, predator_view_distance=150, deer_view_distance=150, predator_count=10, deer_count=14))
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    lay = genes.layout(env)
    rows = []
    for p in env.predators:
        p.genome = rng.uniform(-1, 1, 4)
    for _ in range(40):
        obs, *_ = env.step({a: int(rng.integers(MOVES)) for a in env.active})
        rows += [obs[p.id] for p in env.predators if p.id in obs]
    batch = np.stack(rows).astype(np.float32)
    assert genes.anything_in_view(batch, lay).mean() > 0.5
    expected = genes.bias(batch, lay, genes.BETA0)
    got = torch_bias(torch.from_numpy(batch), lay, genes.BETA0, torch.tensor(genes.DIRECTION_TABLE, dtype=torch.float32), torch.tensor(genes.SPEED_TABLE, dtype=torch.float32))
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-5)
    assert np.abs(expected).max() > 1


def test_sample_follows_the_probabilities():
    logits = np.log(np.array([[0.2, 0.5, 0.3] + [1e-300] * (MOVES - 3)]))
    moves = genes.sample(np.repeat(logits, 50000, axis=0), np.random.default_rng(1))
    np.testing.assert_allclose(np.bincount(moves, minlength=3)[:3] / 50000, [0.2, 0.5, 0.3], atol=0.01)
