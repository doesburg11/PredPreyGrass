"""The fix_* settings: each fix changes what it should and, with all of them
off, nothing changes (test_regression.py checks that bit for bit)."""

import random

import numpy as np
import pytest

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.environment import (
    DEFAULT_CONFIG,
    FIXES,
    PredPreyGrassContinuous,
)
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.geometry import Vector
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.tests.compare import random_actions, snapshot

STOP = 32  # any action from 32 to 47: speed 0
EAST = 4  # full speed towards +x


def world(fixes=(), prey=(), predators=(), **settings):
    """An environment after reset whose animals are replaced by the given
    ones, each (x, y, vx, vy), with plenty of energy and the movement
    order prey first, then predators."""
    # Base animals: founders without trait variation.
    config = {"prey_count": len(prey), "predator_count": len(predators), "trait_founder_std": 0.0, **settings}
    config.update({f"fix_{name}": True for name in fixes})
    env = PredPreyGrassContinuous(config)
    env.reset(seed=0)
    for animal, (x, y, vx, vy) in zip(env.prey + env.predators, list(prey) + list(predators)):
        animal.position = Vector(float(x), float(y))
        animal.velocity = Vector(float(vx), float(vy))
        animal.energy = 6.0 if animal.species == "prey" else 8.0  # no births
    return env


def physics_step(env, actions=None):
    actions = actions or {agent: STOP for agent in env.active}
    return env._physics_step(actions)


def test_fixes_are_off_by_default_and_listed():
    assert len(FIXES) == 8
    assert all(DEFAULT_CONFIG[key] is False for key in FIXES)
    with pytest.raises(ValueError):
        PredPreyGrassContinuous({"fix_wrap_borders": 1})


@pytest.mark.parametrize("fixed, expected_x", [(False, 0.0), (True, 3.0)])
def test_wrap_borders(fixed, expected_x):
    env = world(["wrap_borders"] if fixed else [], prey=[(100, 100, 0, 0)], predators=[(798, 400, 5, 0)])
    predator = env.predators[0]
    physics_step(env, {"prey_0": STOP, "predator_0": EAST})
    assert predator.position.x == pytest.approx(expected_x)
    assert predator.position.y == 400.0


@pytest.mark.parametrize("velocity", [(0.0, 0.0), (1e-9, -1e-9)])
@pytest.mark.parametrize("fixed", [False, True])
def test_keep_heading_when_stopped(fixed, velocity):
    env = world(["keep_heading_when_stopped"] if fixed else [], prey=[(100, 100, *velocity)], predators=[(600, 600, 0, 0)])
    prey = env.prey[0]
    prey.orientation = 45.0
    physics_step(env)
    if fixed:
        assert prey.orientation == 45.0
    else:
        assert prey.orientation != 45.0


@pytest.mark.parametrize("fixed, expected", [(False, 52), (True, 50)])
def test_episode_length(fixed, expected):
    env = world(
        ["episode_length"] if fixed else [],
        prey=[(100, 100, 0, 0)],
        predators=[(600, 600, 0, 0)],
        max_time_steps=50,
        action_repeat=1,
    )
    steps = 0
    while True:
        _, _, terms, truncs, _ = env.step({agent: STOP for agent in env.active})
        steps += 1
        if terms["__all__"] or truncs["__all__"]:
            break
    assert truncs["__all__"] and steps == expected


@pytest.mark.parametrize("fixed", [False, True])
def test_no_spare_spawn_draw(fixed):
    env = PredPreyGrassContinuous({"fix_no_spare_spawn_draw": fixed})
    env.reset(seed=3)
    spawn = random.Random(3)
    prey = []
    for _ in range(8):
        prey.append((spawn.randint(20, 780), spawn.randint(20, 780)))
        if not fixed:
            spawn.randint(150, 250)
    predators = [(spawn.randint(0, 800), spawn.randint(0, 800)) for _ in range(6)]
    assert [(a.position.x, a.position.y) for a in env.prey] == prey
    assert [(a.position.x, a.position.y) for a in env.predators] == predators


@pytest.mark.parametrize("fixed", [False, True])
def test_caught_prey_does_not_move(fixed):
    env = world(
        ["caught_prey_does_not_move"] if fixed else [],
        prey=[(420, 400, 0, -6), (100, 100, 0, 0)],
        predators=[(400, 400, 0, 0)],
    )
    caught, predator = env.prey[0], env.predators[0]
    energy = predator.energy
    physics_step(env)
    assert caught not in env.prey and caught not in env.entities and not caught.alive
    assert predator.energy > energy - 0.01  # fed by the catch
    assert (caught.position.y == 400.0) == fixed


@pytest.mark.parametrize("fixed, speed", [(False, 1.0), (True, 6.0 / 8)])
def test_same_collisions_for_both(fixed, speed):
    """A stopped prey bumped by another: the push replaces its acceleration
    (unit length) or adds to it (top speed / 8, as for predators)."""
    env = world(
        ["same_collisions_for_both"] if fixed else [],
        prey=[(400, 400, 0, 0), (410, 400, 0, 0)],
        predators=[(100, 100, 0, 0)],
    )
    bumped = env.prey[0]
    physics_step(env)
    assert bumped.velocity.x == pytest.approx(-speed)
    assert bumped.velocity.y == pytest.approx(0.0)


@pytest.mark.parametrize("fixed, speed", [(False, 1.0), (True, 0.0)])
def test_all_contacts(fixed, speed):
    """A prey between two others: pushed away from the first only, or from
    both (cancelling out)."""
    env = world(
        ["all_contacts"] if fixed else [],
        prey=[(400, 400, 0, 0), (410, 400, 0, 0), (390, 400, 0, 0)],
        predators=[(100, 100, 0, 0)],
    )
    middle = env.prey[0]
    physics_step(env)
    assert middle.velocity.mag() == pytest.approx(speed)


def crowd(fixes):
    """Prey bumping into prey and predators into predators, all moving."""
    return world(
        fixes,
        prey=[(400, 400, 1, 0), (415, 405, -1, 0), (430, 398, 0, 1), (200, 200, 2, 2)],
        predators=[(600, 600, 0, 3), (612, 610, 1, -1), (625, 600, -2, 0)],
    )


@pytest.mark.parametrize("fixed", [False, True])
def test_simultaneous_movement_ignores_the_order(fixed):
    """With simultaneous movement the order of the entity list does not
    matter; with sequential movement it does."""
    fixes = ["simultaneous_movement"] if fixed else []
    forward, backward = crowd(fixes), crowd(fixes)
    backward.entities.reverse()
    actions = {"prey_0": 4, "prey_1": 12, "prey_2": 0, "prey_3": STOP, "predator_0": 8, "predator_1": 4, "predator_2": 12}
    physics_step(forward, actions)
    physics_step(backward, actions)
    same = snapshot(forward)["animals"] == snapshot(backward)["animals"]
    assert same == fixed


def test_all_fixes_play_reproducible_episodes_in_bounds():
    results = []
    for _ in range(2):
        env = PredPreyGrassContinuous(FIXES)
        rng = np.random.default_rng(5)
        obs, _ = env.reset(seed=5)
        trace = []
        for _ in range(400):
            obs, rewards, terms, truncs, infos = env.step(random_actions(rng, obs))
            for animal in env.predators + env.prey:
                assert 0 <= animal.position.x < env.width and 0 <= animal.position.y < env.height
            assert all(np.all(np.abs(o) <= 1) for o in obs.values())
            trace.append(snapshot(env)["animals"])
            if terms["__all__"] or truncs["__all__"]:
                break
            obs = {agent: o for agent, o in obs.items() if not (terms.get(agent) or truncs.get(agent))}
        results.append(trace)
    assert results[0] == results[1]


@pytest.mark.parametrize("fixed", [False, True])
def test_catches_use_the_positions_at_the_start_of_the_step(fixed):
    """Predators moved first (reversed order): sequentially, a predator that
    moves into reach catches in the same step and one that moves out of
    reach does not; simultaneously, only the start of the step counts."""
    fixes = ["simultaneous_movement"] if fixed else []
    # predator_0 starts 35 from prey_0 and runs at it; predator_1 starts 30
    # from prey_1 and runs away.
    env = world(fixes, prey=[(400, 400, 0, 0), (100, 400, 0, 0)], predators=[(435, 400, -5, 0), (70, 400, -5, 0)])
    env.entities.reverse()
    approaching, fleeing = env.prey
    physics_step(env, {"prey_0": STOP, "prey_1": STOP, "predator_0": 12, "predator_1": 12})  # 12: towards -x
    assert approaching.alive == fixed
    assert fleeing.alive == (not fixed)


@pytest.mark.parametrize("start, velocity, expected", [((2, 3), (-3, -4), (799, 799)), ((799, 1), (4, -3), (3, 798))])
def test_wrap_borders_both_ways(start, velocity, expected):
    """Overshoot below 0 and past the width, in x and y (speed 5, the top)."""
    env = world(["wrap_borders"], prey=[(100, 100, 0, 0)], predators=[(*start, *velocity)])
    predator = env.predators[0]
    env._finish_move(predator)
    assert (predator.position.x, predator.position.y) == pytest.approx(expected)


def test_shortest_episode_with_fixed_length():
    env = world(["episode_length"], prey=[(100, 100, 0, 0)], predators=[(600, 600, 0, 0)], max_time_steps=1, action_repeat=8)
    _, _, terms, truncs, _ = env.step({agent: STOP for agent in env.active})
    assert truncs["__all__"] and env.time_step == 1


def test_a_caught_prey_that_does_not_move_feeds_its_catcher_its_energy():
    env = world(["caught_prey_does_not_move"], prey=[(420, 400, 0, -6), (100, 100, 0, 0)], predators=[(400, 400, 0, 0)])
    predator = env.predators[0]
    physics_step(env)
    cost = env.resting_cost["predator"] + env.speed_cost["predator"] * predator.velocity.mag()
    assert predator.energy == pytest.approx(8.0 + 0.5 * 6.0 - cost)
    assert env.births == {"predator": 0, "prey": 0}


@pytest.mark.parametrize("all_contacts", [False, True])
def test_vectorized_contacts_match_the_per_animal_search(all_contacts):
    """_all_contacts (simultaneous movement) gives exactly what _contacts
    gives animal by animal, in dense crowds that wrap around the edges."""
    rng = np.random.default_rng(1)
    fixes = ["simultaneous_movement"] + (["all_contacts"] if all_contacts else [])
    for trial in range(30):
        prey = [(float(x), float(y), 0, 0) for x, y in rng.uniform(-60, 60, size=(40, 2)) % 800]
        predators = [(float(x), float(y), 0, 0) for x, y in rng.uniform(-60, 60, size=(15, 2)) % 800]
        env = world(fixes, prey=prey, predators=predators)
        expected = [(animal, *env._contacts(animal)) for animal in env.entities]
        got = env._all_contacts()
        assert [(a, c) for a, c, _ in got] == [(a, c) for a, c, _ in expected]
        assert [[(v.x, v.y) for v in aways] for _, _, aways in got] == [
            [(v.x, v.y) for v in aways] for _, _, aways in expected
        ]


@pytest.mark.parametrize("all_contacts", [False, True])
@pytest.mark.parametrize(
    "prey, predators",
    [
        # Exactly tangent (distance 32: no contact) and just inside.
        ([(400, 400, 0, 0), (432, 400, 0, 0), (400, 431.9, 0, 0)], [(468, 400, 0, 0)]),
        # Two prey at the same spot, touched by two predators (first in list catches).
        ([(400, 400, 0, 0), (400, 400, 0, 0)], [(410, 400, 0, 0), (390, 400, 0, 0)]),
        # Across the corner of the torus.
        ([(2, 2, 0, 0), (798, 798, 0, 0), (2, 798, 0, 0)], [(798, 2, 0, 0)]),
        # No predators; one prey.
        ([(400, 400, 0, 0)], []),
    ],
)
def test_vectorized_contacts_edge_cases(all_contacts, prey, predators):
    fixes = ["simultaneous_movement"] + (["all_contacts"] if all_contacts else [])
    for reverse in (False, True):
        env = world(fixes, prey=prey, predators=predators)
        if reverse:
            env.entities.reverse()
        expected = [(animal, *env._contacts(animal)) for animal in env.entities]
        got = env._all_contacts()
        assert [(a, c, [(v.x, v.y) for v in w]) for a, c, w in got] == [
            (a, c, [(v.x, v.y) for v in w]) for a, c, w in expected
        ]
