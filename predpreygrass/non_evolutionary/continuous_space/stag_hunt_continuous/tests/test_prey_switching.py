"""Tests of the prey-switching measures."""

import math

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import MOVES, StagHuntContinuous, species_of
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.geometry import Vector, heading_degrees
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.prey_switching import (
    bin_rates,
    diet_slope,
    new_bins,
    play_episode,
    pooled,
    pursuit_along,
    rabbit_bin,
    rabbits_visible_to_hunters,
    summarise,
)


def random_policy(seed):
    rng = np.random.default_rng(seed)

    def choose(obs):
        return {a: (np.array([rng.integers(MOVES), rng.integers(2)]) if species_of(a) == "predator" else int(rng.integers(MOVES))) for a in obs}

    return choose


def short_world():
    return StagHuntContinuous({"predator_count": 12, "mammoth_count": 6, "max_time_steps": 600})


@pytest.mark.parametrize("rabbits, index", [(0, None), (1, 0), (5, 0), (6, 1), (10, 1), (11, 2), (20, 2), (21, 3), (40, 3), (41, 4), (70, 4), (71, 5), (500, 5)])
def test_rabbit_bin_edges(rabbits, index):
    assert rabbit_bin(rabbits) == index


def test_pursuit_is_a_share_of_top_speed_across_the_wrap():
    env = short_world()
    before, target = Vector(795, 400), Vector(30, 400)  # 35 east, across the edge
    assert pursuit_along(env, before, Vector(5, 400), target, 5, 2) == pytest.approx(1.0)
    assert pursuit_along(env, before, Vector(785, 400), target, 5, 2) == pytest.approx(-1.0)
    assert pursuit_along(env, before, Vector(795, 410), target, 5, 2) == pytest.approx(0.0)
    assert pursuit_along(env, before, Vector(5, 400), before, 5, 2) == 0.0
    assert pursuit_along(env, before, before, target, 5, 0) == 0.0


def test_diet_slope_recovers_known_values():
    rng = np.random.default_rng(0)
    for slope in (0.5, 1.0, 2.0):
        captures = []
        for _ in range(4000):
            rabbits, mammoths = int(rng.integers(1, 100)), int(rng.integers(1, 20))
            p = 1 / (1 + math.exp(-(0.3 + slope * math.log(rabbits / mammoths))))
            caught = int(rng.random() < p)
            captures.append((rabbits, mammoths, caught, 1 - caught))
        assert diet_slope(captures)[1] == pytest.approx(slope, abs=0.15)


def test_diet_slope_needs_both_prey_and_variation():
    assert math.isnan(diet_slope([])[1])
    assert math.isnan(diet_slope([(5, 5, 3, 0), (9, 2, 1, 0)])[1])
    assert math.isnan(diet_slope([(5, 5, 3, 1), (5, 5, 2, 2)])[1])


def test_diet_slope_is_unidentified_under_separation():
    captures = [(5, 50, 0, 3), (10, 50, 0, 2), (50, 5, 4, 0), (50, 2, 3, 0)]
    assert math.isnan(diet_slope(captures)[1])


def test_diet_slope_weights_aggregated_captures():
    single = [(r, m, 1, 0) for r, m, c, k in [(10, 5, 3, 1), (40, 5, 5, 1), (5, 10, 1, 4)] for _ in range(c)]
    single += [(r, m, 0, 1) for r, m, c, k in [(10, 5, 3, 1), (40, 5, 5, 1), (5, 10, 1, 4)] for _ in range(k)]
    assert diet_slope([(10, 5, 3, 1), (40, 5, 5, 1), (5, 10, 1, 4)])[1] == pytest.approx(diet_slope(single)[1])


def placed_world(place, **settings):
    """A world whose reset ends with place(env), so one decision runs from a known state."""

    class Placed(StagHuntContinuous):
        def reset(self, **kwargs):
            result = super().reset(**kwargs)
            place(self)
            return result

    return Placed({"predator_count": 1, "rabbit_count": 2, "mammoth_count": 1, "action_repeat": 3, "max_time_steps": 3, **settings})


def test_exposure_stops_when_a_rabbit_is_caught():
    def place(env):
        env.predators[0].position = Vector(400, 400)
        env.rabbits[0].position = Vector(400, 400)
        env.rabbits[1].position = Vector(100, 100)
        env.mammoths[0].position = Vector(700, 100)

    env = placed_world(place)
    row = play_episode(env, random_policy(0), 0)
    assert env.hunt_stats["rabbits_caught"] == 1
    assert env.time_step == 3
    # 2 rabbits x 1 hunter at the step of the catch, then 1 x 1 twice; 6 if
    # the counts at the decision's start were charged throughout.
    assert sum(b["rabbit_hunter_steps"] for b in row["bins"]) == 4
    assert "_physics_step" not in vars(env)


def test_hunter_dead_within_a_decision_leaves_pursuit_not_the_sightings():
    def place(env):
        env.predators[0].position = Vector(400, 400)
        env.predators[0].energy = 0.001
        env.predators[0].orientation = heading_degrees(Vector(1, 0))
        env.rabbits[0].position = Vector(550, 400)
        env.rabbits[1].position = Vector(100, 100)
        env.mammoths[0].position = Vector(700, 100)

    env = placed_world(place, max_time_steps=30)
    row = play_episode(env, random_policy(1), 1)
    assert not env.predators
    assert sum(b["sightings_at_start"] for b in row["bins"]) >= 1
    assert sum(b["sightings"] for b in row["bins"]) == 0


def test_hidden_rabbits_are_not_sighted():
    settings = {"burrow_count": 1, "burrow_radius": 24.0, "burrow_hides_rabbits": True}
    env = StagHuntContinuous({"predator_count": 1, "rabbit_count": 2, "mammoth_count": 1, **settings})
    env.reset(seed=0)
    env.burrows[0] = (400, 400)
    env.rabbits[0].position = Vector(400, 400)
    env.rabbits[1].position = Vector(100, 100)
    visible = rabbits_visible_to_hunters(env)
    assert env.rabbits[0] not in visible and env.rabbits[1] in visible
    env = short_world()
    env.reset(seed=0)
    assert rabbits_visible_to_hunters(env) == env.rabbits


def test_bin_rates_and_empty_bins():
    row = dict(new_bins()[0], rabbit_hunter_steps=1e5, catches=3, sightings=4, pursuit=2, sightings_at_start=6)
    rates = bin_rates(row)
    assert rates["catch_rate_x1e5"] == 3 and rates["pursuit"] == 0.5 and rates["catches_per_sighting"] == 0.5
    assert set(bin_rates(new_bins()[0]).values()) == {None}


def test_episode_counters_match_the_environment():
    env = short_world()
    row = play_episode(env, random_policy(3), 3)
    stats = env.hunt_stats
    # Rabbits exist at the start of every decision with a catch.
    assert sum(b["catches"] for b in row["bins"]) == stats["rabbits_caught"]
    assert sum(b["kills"] for b in row["bins"]) <= sum(stats[f"kills_party_{n}"] for n in range(1, 5))
    assert sum(b["physics_steps"] for b in row["bins"]) <= env.time_step
    assert sum(c[2] for c in row["captures"]) <= stats["rabbits_caught"]
    assert all(-1.5 <= b["pursuit"] / b["sightings"] <= 1.5 for b in row["bins"] if b["sightings"])
    assert all(b["sightings"] <= b["sightings_at_start"] for b in row["bins"])


def test_measuring_does_not_change_the_episode():
    env = short_world()
    row = play_episode(env, random_policy(5), 5)
    measured = row["hunt_stats"]
    other = short_world()
    obs, _ = other.reset(seed=5)
    choose = random_policy(5)
    while True:
        obs, _, terms, truncs, _ = other.step(choose({a: obs[a] for a in sorted(obs) if a in other.active}))
        if terms["__all__"] or truncs["__all__"]:
            break
    assert other.episode_hunt_stats == measured


def test_summary_pools_and_gives_intervals():
    env = short_world()
    rows = [play_episode(env, random_policy(seed), seed) for seed in range(3)]
    summary = summarise(rows)
    assert summary["episodes"] == 3
    assert sum(b["catches"] for b in summary["bins"]) == sum(b["catches"] for b in pooled(rows))
    for entry in summary["bins"]:
        if entry["catch_rate_x1e5"] is not None:
            low, high = entry["catch_rate_x1e5_ci"]
            assert low <= high
