"""Tests of the prey's rules."""

import random

import numpy as np

import pytest

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import StagHuntRulePrey
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.geometry import Vector, direction_vector
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.rules import DIRECTIONS, FULL, HALF, STOPPED, direction_towards, move_of, prey_move


class NoDraws:
    def randrange(self, *_):
        raise AssertionError("no draw expected")

    choice = randrange


def world(**settings):
    env = StagHuntRulePrey(dict(dict(predator_count=1, rabbit_count=1, mammoth_count=1, grass_count=0, steppe_count=0), **settings))
    env.reset(seed=0)
    env.predators[0].position = Vector(400, 700)
    env.rabbits[0].position = Vector(400, 400)
    env.mammoths[0].position = Vector(100, 100)
    return env


def face(animal, degrees):
    animal.orientation = degrees


def speed_of(move):
    return move // DIRECTIONS


@pytest.mark.parametrize("direction", range(DIRECTIONS))
def test_direction_towards_inverts_the_movement_directions(direction):
    v = direction_vector(direction, DIRECTIONS)
    assert direction_towards(v.x, v.y) == direction


def test_a_prey_flees_directly_away_at_full_speed():
    env = world()
    rabbit = env.rabbits[0]
    env.predators[0].position = Vector(400, 350)  # above, -y
    face(rabbit, 90)  # orientation convention: 90 is -y
    assert prey_move(env, rabbit, NoDraws()) == move_of(direction_towards(0, 1), FULL)


def test_flight_uses_the_wrapped_distance():
    env = world()
    rabbit = env.rabbits[0]
    rabbit.position = Vector(790, 400)
    env.predators[0].position = Vector(30, 400)  # 40 to the right, across the edge
    face(rabbit, 0)
    assert prey_move(env, rabbit, random.Random(0)) == move_of(direction_towards(-1, 0), FULL)


def test_a_predator_beyond_the_flee_distance_is_ignored():
    env = world(rabbit_flee_distance=50.0)
    rabbit = env.rabbits[0]
    env.predators[0].position = Vector(400, 340)
    face(rabbit, 90)
    assert speed_of(prey_move(env, rabbit, random.Random(0))) == HALF


def test_a_predator_outside_the_view_cone_is_ignored():
    env = world()
    rabbit = env.rabbits[0]
    env.predators[0].position = Vector(400, 450)  # behind it
    face(rabbit, 90)
    assert speed_of(prey_move(env, rabbit, random.Random(0))) == HALF


def test_the_nearest_of_two_predators_is_fled():
    env = world(predator_count=2)
    rabbit = env.rabbits[0]
    env.predators[0].position = Vector(400, 300)
    env.predators[1].position = Vector(450, 400)
    face(rabbit, 45)
    assert prey_move(env, rabbit, NoDraws()) == move_of(direction_towards(-1, 0), FULL)


def test_a_prey_goes_to_food_in_view_and_stops_when_it_can_eat():
    env = world(grass_count=1, grass_clustered=False, grass_dispersal=False, grass_overgrazing=False)
    rabbit = env.rabbits[0]
    patch = env.food["grass"].patches[0]
    env.predators[0].position = Vector(400, 700)
    patch.position = Vector(400, 300)
    face(rabbit, 90)
    assert prey_move(env, rabbit, NoDraws()) == move_of(direction_towards(0, -1), FULL)
    patch.position = Vector(400, 395)
    assert speed_of(prey_move(env, rabbit, NoDraws())) == STOPPED


def test_a_threat_comes_before_food():
    env = world(grass_count=1, grass_clustered=False, grass_dispersal=False, grass_overgrazing=False)
    rabbit = env.rabbits[0]
    env.food["grass"].patches[0].position = Vector(400, 395)
    env.predators[0].position = Vector(400, 350)
    face(rabbit, 90)
    assert speed_of(prey_move(env, rabbit, NoDraws())) == FULL


def test_mammoths_use_the_steppe_and_their_own_flee_distance():
    env = world(steppe_count=1, steppe_clustered=False, steppe_dispersal=False, steppe_overgrazing=False, mammoth_flee_distance=30.0)
    mammoth = env.mammoths[0]
    mammoth.position = Vector(400, 400)
    env.rabbits[0].position = Vector(100, 100)
    env.food["steppe"].patches[0].position = Vector(400, 300)
    env.predators[0].position = Vector(400, 350)  # within rabbits' flee distance, not this one
    face(mammoth, 90)
    assert prey_move(env, mammoth, NoDraws()) == move_of(direction_towards(0, -1), FULL)


def test_wandering_keeps_close_to_the_current_heading():
    env = world()
    rabbit = env.rabbits[0]
    rabbit.velocity = direction_vector(4, DIRECTIONS)
    rng = random.Random(1)
    directions = {prey_move(env, rabbit, rng) % DIRECTIONS for _ in range(50)}
    assert directions == {3, 4, 5}


def test_episodes_repeat_with_a_seed():
    def play(seed):
        env = StagHuntRulePrey()
        env.reset(seed=seed)
        rng = random.Random(seed)
        for _ in range(40):
            env.step({a: np.array([rng.randrange(48), rng.randrange(2)]) for a in env.active})
        return [(a.id, round(a.position.x, 6), round(a.position.y, 6), round(a.energy, 6)) for a in env.living()]

    assert play(3) == play(3)
    assert play(3) != play(4)
