"""Sharing of adult catches (environment.py docstring, DESIGN.md): who the
bystanders are, how the energy is split, what is never shared, the
counters, the settings and this module's world."""

import json
import math

import pytest

from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.environment import DEFAULT_CONFIG, DIRECTIONS
from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.tests.test_environment import deer_at, place, stop, world

SHARE = {"deer_catch_rule": "touch", "adult_share_fraction": 0.5, "adult_share_radius": 128.0}


def adult_caught_by_two(env, energy=6.0):
    deer = deer_at(env, energy)
    a, b = env.predators[:2]
    place(a, 420, 300)
    place(b, 380, 300)
    return deer, a, b


def test_without_sharing_a_hunter_near_an_adult_catch_gets_nothing():
    env = world(deer_catch_rule="touch")
    deer, a, b = adult_caught_by_two(env)
    near = env.predators[2]
    place(near, 400, 400)  # 100 from the deer
    before = near.energy
    env.step(stop(env))
    assert not deer.alive and env.stats["adults_caught"] == 1
    assert near.energy == pytest.approx(before - env.resting_cost["predator"])
    assert env.stats["adults_shared"] == 0 and env.stats["energy_adult_to_bystanders"] == 0


def test_a_bystander_gets_the_share_and_the_catchers_split_the_rest():
    env = world(**SHARE)
    deer, a, b = adult_caught_by_two(env)
    near = env.predators[2]
    place(near, 400, 400)
    before = [p.energy for p in (a, b, near)]
    env.step(stop(env))
    cost = env.resting_cost["predator"]
    food = 0.5 * 6.0
    assert a.energy == pytest.approx(before[0] + food * 0.5 / 2 - cost)
    assert b.energy == pytest.approx(before[1] + food * 0.5 / 2 - cost)
    assert near.energy == pytest.approx(before[2] + food * 0.5 - cost)
    assert env.stats["adults_shared"] == 1 and env.stats["share_bystanders"] == 1
    assert env.stats["energy_adult_to_bystanders"] == pytest.approx(food * 0.5)
    assert env.stats["energy_from_adult"] == pytest.approx(food)


def test_bystanders_split_their_part_evenly_and_hunters_beyond_the_radius_get_nothing():
    env = world(**SHARE)
    deer, a, b = adult_caught_by_two(env)
    near1, near2, far = env.predators[2:5]
    place(near1, 400, 172)  # 128 from the deer: inside (<=)
    place(near2, 300, 300)  # 100
    place(far, 529, 300)  # 129: outside
    before = {p.id: p.energy for p in (a, near1, near2, far)}
    env.step(stop(env))
    cost = env.resting_cost["predator"]
    food = 0.5 * 6.0
    assert near1.energy == pytest.approx(before[near1.id] + food * 0.5 / 2 - cost)
    assert near2.energy == pytest.approx(before[near2.id] + food * 0.5 / 2 - cost)
    assert far.energy == pytest.approx(before[far.id] - cost)
    assert a.energy == pytest.approx(before[a.id] + food * 0.5 / 2 - cost)
    assert env.stats["share_bystanders"] == 2


def test_without_a_bystander_the_catchers_keep_it_all():
    env = world(**SHARE)
    deer, a, b = adult_caught_by_two(env)
    before = a.energy
    env.step(stop(env))
    assert a.energy == pytest.approx(before + 0.5 * 6.0 / 2 - env.resting_cost["predator"])
    assert env.stats["adults_shared"] == 0


@pytest.mark.parametrize("energy", [2.0, 3.5, 4.9])
def test_young_deer_are_never_shared_beyond_their_catchers(energy):
    env = world(**SHARE)
    deer, a, b = adult_caught_by_two(env, energy)  # two touching: caught without a draw
    near = env.predators[2]
    place(near, 400, 400)
    before = near.energy
    env.step(stop(env))
    assert not deer.alive and env.stats["young_caught_party"] == 1
    assert near.energy == pytest.approx(before - env.resting_cost["predator"])
    assert env.stats["adults_shared"] == 0


def test_bystanders_are_those_near_at_the_start_of_the_step_whatever_they_do_in_it():
    env = world(**SHARE)
    deer, a, b = adult_caught_by_two(env)
    leaving = env.predators[2]
    place(leaving, 400, 427)  # just inside, and moving at full speed this step
    actions = stop(env)
    actions[leaving.id] = DIRECTIONS // 4  # one of the 16 directions, full speed
    before = leaving.energy
    env.step(actions)
    assert env.stats["share_bystanders"] == 1
    assert leaving.energy > before


def test_a_bystander_at_the_energy_cap_counts_only_what_it_gained():
    env = world(**SHARE)
    deer, a, b = adult_caught_by_two(env, 10.0)
    near = env.predators[2]
    place(near, 400, 400)
    near.energy = env.max_energy["predator"] - 0.25
    env.step(stop(env))
    assert env.stats["energy_adult_to_bystanders"] == pytest.approx(0.25)


def test_the_party_rule_shares_adults_too():
    env = world(adult_share_fraction=0.5, adult_share_radius=128.0)
    deer, a, b = adult_caught_by_two(env)
    near = env.predators[2]
    place(near, 400, 400)
    env.step(stop(env))
    assert env.stats["adults_caught"] == 1 and env.stats["adults_shared"] == 1


@pytest.mark.parametrize("settings", [{"adult_share_fraction": -0.1}, {"adult_share_fraction": 1.1}, {"adult_share_fraction": math.nan}, {"adult_share_radius": -1.0}, {"adult_share_radius": math.inf}])
def test_sharing_settings_are_validated(settings):
    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.environment import DeerStagHunt

    with pytest.raises(ValueError):
        DeerStagHunt(dict(DEFAULT_CONFIG, **settings))


def test_this_module_s_world_is_the_touch_rule_shared_at_one_half_within_128():
    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.config.config_env import config_env

    assert {k: config_env[k] for k in SHARE} == SHARE
    assert {k: v for k, v in config_env.items() if k not in SHARE} == {k: v for k, v in DEFAULT_CONFIG.items() if k not in SHARE}
    assert DEFAULT_CONFIG["adult_share_fraction"] == 0.0


def test_deer_stag_hunt_run_settings_load_without_sharing(tmp_path):
    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.checkpoints import checkpoint_env_config

    old = {k: v for k, v in DEFAULT_CONFIG.items() if not k.startswith("adult_share")}
    (tmp_path / "source_code").mkdir()
    (tmp_path / "source_code" / "settings.json").write_text(json.dumps({"config_env": old, "config_ppo": {}}))
    (tmp_path / "checkpoint").mkdir()
    assert checkpoint_env_config(tmp_path / "checkpoint")["adult_share_fraction"] == 0.0


def test_a_hunter_arriving_within_the_radius_during_the_step_gets_nothing():
    env = world(**SHARE)
    deer, a, b = adult_caught_by_two(env)
    arriving = env.predators[2]
    place(arriving, 400, 429)  # 129: outside at the start of the step
    arriving.velocity.y = -5.0  # already moving towards the deer at full speed
    actions = stop(env)
    actions[arriving.id] = 3 * DIRECTIONS // 4  # keep going
    before = arriving.energy
    env.step(actions)
    assert env.torus.distance(deer.position, arriving.position) < 128
    assert env.stats["adults_shared"] == 0 and arriving.energy < before


def test_bystanders_at_the_cap_lose_what_they_cannot_take_it_is_not_passed_on():
    env = world(**SHARE)
    deer, a, b = adult_caught_by_two(env, 10.0)
    full, hungry = env.predators[2:4]
    place(full, 400, 400)
    place(hungry, 300, 300)
    full.energy = env.max_energy["predator"] - 0.25
    before = hungry.energy
    env.step(stop(env))
    food = 0.5 * 10.0
    assert hungry.energy == pytest.approx(before + food * 0.5 / 2 - env.resting_cost["predator"])
    assert env.stats["energy_adult_to_bystanders"] == pytest.approx(0.25 + food * 0.5 / 2)


def test_derived_rates_include_the_bystanders_share_of_adult_energy():
    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.callbacks import derived_rates
    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.environment import new_stats

    stats = dict(new_stats(), energy_from_adult=4.0, energy_adult_to_bystanders=1.0, energy_from_fawn=4.0)
    assert derived_rates(stats)["adult_energy_to_bystanders"] == 0.25
    assert "adult_energy_to_bystanders" not in derived_rates(new_stats())
