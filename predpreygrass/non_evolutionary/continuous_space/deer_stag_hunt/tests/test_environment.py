"""Tests of the deer world's rules: catching fawns, middle deer and adults,
sharing, cooldowns, observations, births and settings."""

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.environment import DEFAULT_CONFIG, DIRECTIONS, MOVES, DeerStagHunt
from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.geometry import Vector

STOP = 2 * DIRECTIONS  # any direction, speed 0


def world(**config):
    """A reset world, one physics step per decision, with no grass to eat
    and every animal stopped and parked apart, away from the test spot."""
    env = DeerStagHunt(dict(DEFAULT_CONFIG, **config))
    env.reset(seed=0)
    env.action_repeat = 1
    for patch in env.grass.patches:
        patch.ready_at = float("inf")
    for i, animal in enumerate(env.living()):
        animal.velocity = Vector(0, 0)
        animal.position = Vector(40 + 60 * (i % 5), 560 + 60 * (i // 5))
    return env


def place(animal, x, y):
    animal.position = Vector(x, y)
    animal.velocity = Vector(0, 0)


def stop(env):
    return {agent: STOP for agent in env.active}


def deer_at(env, energy, x=400, y=300):
    deer = env.deer[0]
    place(deer, x, y)
    deer.energy = energy
    return deer


def test_a_fawn_is_caught_by_a_lone_hunter_who_gains_half_its_energy():
    env = world()
    deer = deer_at(env, 2.0)
    hunter = env.predators[0]
    place(hunter, 420, 300)
    before = hunter.energy
    env.step(stop(env))
    assert not deer.alive and env.stats["young_caught_solo"] == 1
    assert hunter.energy == pytest.approx(before + 0.5 * 2.0 - env.resting_cost["predator"])


def test_an_adult_escapes_a_lone_toucher_even_with_a_partner_near_and_the_toucher_waits():
    env = world()
    deer = deer_at(env, 6.0)
    toucher, near = env.predators[:2]
    place(toucher, 420, 300)
    place(near, 400, 350)  # within the party radius, not touching
    env.step(stop(env))
    assert deer.alive and env.stats["adult_failures"] == 1 and env.stats["deer_caught"] == 0
    assert toucher.catch_ready_at == env.time_step + env.catch_cooldown
    assert near.catch_ready_at == 0


def test_an_adult_touched_by_two_at_once_is_caught_and_shared_by_the_touchers_only():
    env = world()
    deer = deer_at(env, 6.0)
    a, b, near = env.predators[:3]
    place(a, 420, 300)
    place(b, 380, 300)
    place(near, 400, 350)
    before = [p.energy for p in (a, b, near)]
    env.step(stop(env))
    assert not deer.alive and env.stats["adults_caught"] == 1
    cost = env.resting_cost["predator"]
    assert a.energy == pytest.approx(before[0] + 0.5 * 6.0 / 2 - cost)
    assert b.energy == pytest.approx(before[1] + 0.5 * 6.0 / 2 - cost)
    assert near.energy == pytest.approx(before[2] - cost)
    assert env.stats["energy_from_adult"] == pytest.approx(0.5 * 6.0)


def test_touches_within_the_window_catch_an_adult_and_further_apart_do_not():
    for gap, caught in ((8, True), (9, False)):
        env = world(deer_catch_cooldown=0)
        deer = deer_at(env, 6.0)
        first, second = env.predators[:2]
        place(first, 420, 300)
        place(second, 380, 380)  # not touching yet
        env.step(stop(env))
        place(first, 420, 420)  # leaves
        for _ in range(gap - 1):
            env.step(stop(env))
        place(second, 380, 300)
        env.step(stop(env))
        assert (not deer.alive) is caught, gap
        assert env.stats["adults_caught"] == int(caught)


def test_touches_are_forgotten_when_the_deer_stops_being_adult():
    env = world(deer_catch_cooldown=0)
    deer = deer_at(env, 5.0)
    first, second = env.predators[:2]
    place(first, 420, 300)
    env.step(stop(env))
    assert deer.id in env.adult_touches
    place(first, 420, 420)
    deer.energy = 4.9  # no longer adult: a middle deer
    env.catch_rng.random = lambda: 0.999  # its lone catch fails
    env.step(stop(env))
    assert deer.id not in env.adult_touches
    deer.energy = 5.5
    place(second, 380, 300)
    env.step(stop(env))
    assert deer.alive and env.stats["adults_caught"] == 0


def test_a_middle_deer_is_caught_alone_with_a_chance_falling_linearly():
    for draw, caught in ((0.33, True), (0.34, False)):
        env = world()
        deer = deer_at(env, 4.0)  # chance (5 - 4) / (5 - 2) = 1/3
        place(env.predators[0], 420, 300)
        env.catch_rng.random = lambda draw=draw: draw
        env.step(stop(env))
        assert (not deer.alive) is caught
        assert env.stats["young_caught_solo" if caught else "young_failures"] == 1


def test_a_young_deer_with_a_partner_near_is_caught_and_shared_by_everyone_near():
    env = world()
    deer = deer_at(env, 4.5)
    toucher, near, far = env.predators[:3]
    place(toucher, 420, 300)
    place(near, 400, 360)  # within 64 of the deer
    place(far, 400, 400)
    env.catch_rng.random = lambda: pytest.fail("a party catch draws no number")
    before = [p.energy for p in (toucher, near, far)]
    env.step(stop(env))
    assert not deer.alive and env.stats["young_caught_party"] == 1
    cost = env.resting_cost["predator"]
    assert toucher.energy == pytest.approx(before[0] + 0.5 * 4.5 / 2 - cost)
    assert near.energy == pytest.approx(before[1] + 0.5 * 4.5 / 2 - cost)
    assert far.energy == pytest.approx(before[2] - cost)
    assert env.stats["energy_from_middle"] == pytest.approx(0.5 * 4.5)


def test_kinds_by_energy():
    env = world()
    deer = env.deer[0]
    for energy, kind in ((1.0, "fawn"), (2.0, "fawn"), (2.01, "middle"), (4.99, "middle"), (5.0, "adult"), (9.0, "adult")):
        deer.energy = energy
        assert env.kind(deer) == kind


def test_predators_see_deer_energy_and_their_own_cooldown():
    env = world()
    deer = deer_at(env, 6.0)
    hunter = env.predators[0]
    place(hunter, 300, 300)
    hunter.velocity = Vector(1, 0)  # facing the deer
    hunter.orientation = 0.0
    hunter.catch_ready_at = 4
    obs = env._observe(hunter)
    assert obs.shape == (env.obs_size["predator"],) == (48,)
    assert obs[-3] == pytest.approx(6.0 / 16.0)  # the nearest deer's energy
    assert obs[-2] == 0.0  # second deer slot empty
    assert obs[-1] == pytest.approx(4 / env.catch_cooldown)
    assert env.obs_size["deer"] == 45


def test_observations_fit_their_spaces_through_an_episode():
    env = DeerStagHunt()
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(200):
        for agent, value in obs.items():
            assert env.observation_spaces[agent].contains(value), agent
        obs, _, terms, truncs, _ = env.step({a: int(rng.integers(MOVES)) for a in env.active})
        if terms["__all__"] or truncs["__all__"]:
            break


def test_a_deer_birth_costs_the_parent_3_and_the_child_starts_at_2():
    env = world()
    parent = env.deer[0]
    parent.energy = 7.5
    _, rewards, *_ = env.step(stop(env))
    assert env.births["deer"] == 1 and rewards[parent.id] == 10.0
    child = env.deer[-1]
    assert child.energy == 2.0  # born after this step's costs
    assert parent.energy == pytest.approx(7.5 - env.resting_cost["deer"] - 3.0)


def test_actions_are_movement_integers():
    env = world()
    agent = env.predators[0].id
    for bad in (np.array([1, 0]), MOVES, -1, True, 1.5):
        with pytest.raises(ValueError):
            env.step(dict(stop(env), **{agent: bad}))
    env.step(dict(stop(env), **{agent: np.array([3])}))


@pytest.mark.parametrize(
    "settings",
    [
        {"deer_adult_energy": 2.0},
        {"deer_fawn_energy": -1.0},
        {"deer_touch_window": 1.5},
        {"deer_catch_cooldown": -1},
        {"reproduction_deer_cost": 7.0},
        {"reproduction_deer_cost": 1.0},
        {"deer_view_distance": 500},
        {"hunter_calls": True},
        {"mammoth_count": 0},
    ],
)
def test_settings_are_validated(settings):
    with pytest.raises(ValueError):
        DeerStagHunt(dict(DEFAULT_CONFIG, **settings))


def test_the_episode_ends_when_the_deer_are_gone():
    env = world()
    for deer in env.deer[1:]:
        deer.alive = False
    env.animals["deer"] = env.deer[:1]
    env.active = [a.id for a in env.living()]
    deer = deer_at(env, 2.0)
    place(env.predators[0], 420, 300)
    _, _, terms, truncs, _ = env.step(stop(env))
    assert not deer.alive and terms["__all__"] and not truncs["__all__"]
    assert env.episode_stats["deer_caught"] == 1 and env.extinct_at == {"deer": 1}


def test_a_lone_fawn_is_caught_whatever_the_draw():
    """A lone catch draws a number even at chance 1, as stag_hunt_continuous
    does, so both worlds' catch streams stay in step."""
    env = world()
    deer = deer_at(env, 1.5)
    place(env.predators[0], 420, 300)
    env.catch_rng.random = lambda: 1.0 - 1e-12
    env.step(stop(env))
    assert not deer.alive and env.stats["young_caught_solo"] == 1


def test_an_adult_caught_across_the_window_is_shared_by_both_touchers_only():
    env = world(deer_catch_cooldown=0)
    deer = deer_at(env, 6.0)
    first, second, third = env.predators[:3]
    place(first, 420, 300)
    place(second, 380, 380)
    place(third, 400, 340)  # near the deer, never touching it
    start = [p.energy for p in (first, second, third)]
    env.step(stop(env))
    place(first, 420, 420)
    for _ in range(3):
        env.step(stop(env))
    place(second, 380, 300)
    env.step(stop(env))
    assert not deer.alive and env.stats["adults_caught"] == 1
    cost = 5 * env.resting_cost["predator"]
    assert first.energy == pytest.approx(start[0] + 0.5 * deer.energy / 2 - cost)
    assert second.energy == pytest.approx(start[1] + 0.5 * deer.energy / 2 - cost)
    assert third.energy == pytest.approx(start[2] - cost)


def test_a_catch_above_the_energy_cap_counts_only_what_the_hunter_gained():
    env = world()
    deer = deer_at(env, 2.0)
    hunter = env.predators[0]
    place(hunter, 420, 300)
    hunter.energy = env.max_energy["predator"] - 0.25
    env.step(stop(env))
    assert not deer.alive
    assert env.stats["energy_from_fawn"] == pytest.approx(0.25)


def test_the_episode_is_truncated_at_the_time_limit():
    env = world(max_time_steps=5)
    for _ in range(4):
        _, _, terms, truncs, _ = env.step(stop(env))
        assert not truncs["__all__"]
    _, _, terms, truncs, _ = env.step(stop(env))
    assert truncs["__all__"] and not terms["__all__"] and env.time_step == 5
    assert env.episode_stats["physics_steps"] == 5 and env.extinct_at == {}


# --- deer_catch_rule "touch" (RESULTS.md section 2) ----------------------


def test_the_default_catch_rule_is_party():
    assert DEFAULT_CONFIG["deer_catch_rule"] == "party"
    with pytest.raises(ValueError):
        DeerStagHunt({"deer_catch_rule": "other"})


def test_touch_rule_a_partner_near_but_not_touching_plays_no_part_in_a_young_catch():
    for draw, caught in ((0.16, True), (0.17, False)):
        env = world(deer_catch_rule="touch")
        deer = deer_at(env, 4.5)  # solo chance (5 - 4.5) / 3 = 1/6
        toucher, near = env.predators[:2]
        place(toucher, 420, 300)
        place(near, 400, 350)  # within 64, not touching
        env.catch_rng.random = lambda draw=draw: draw
        before = near.energy
        env.step(stop(env))
        assert (not deer.alive) is caught
        assert env.stats["young_caught_solo" if caught else "young_failures"] == 1
        assert env.stats["young_caught_party"] == 0
        assert near.energy == pytest.approx(before - env.resting_cost["predator"])
        if not caught:
            assert toucher.catch_ready_at == env.time_step + env.catch_cooldown


def test_touch_rule_a_fawn_alone_is_caught_without_a_draw():
    env = world(deer_catch_rule="touch")
    deer = deer_at(env, 2.0)
    place(env.predators[0], 420, 300)
    env.catch_rng.random = lambda: pytest.fail("a fawn draws no number")
    env.step(stop(env))
    assert not deer.alive and env.stats["young_caught_solo"] == 1


def test_touch_rule_two_touching_at_once_catch_a_young_deer_without_a_draw_and_share_it():
    env = world(deer_catch_rule="touch")
    deer = deer_at(env, 4.0)
    a, b, near = env.predators[:3]
    place(a, 420, 300)
    place(b, 380, 300)
    place(near, 400, 350)
    env.catch_rng.random = lambda: pytest.fail("two touchers draw no number")
    before = [p.energy for p in (a, b, near)]
    env.step(stop(env))
    assert not deer.alive and env.stats["young_caught_party"] == 1
    cost = env.resting_cost["predator"]
    assert a.energy == pytest.approx(before[0] + 0.5 * 4.0 / 2 - cost)
    assert b.energy == pytest.approx(before[1] + 0.5 * 4.0 / 2 - cost)
    assert near.energy == pytest.approx(before[2] - cost)


def test_touch_rule_a_second_touch_within_the_window_after_a_failed_attempt_catches_and_later_does_not():
    for gap, caught in ((8, True), (9, False)):
        env = world(deer_catch_rule="touch")
        deer = deer_at(env, 4.0)
        first, second = env.predators[:2]
        place(first, 420, 300)
        place(second, 380, 380)
        env.catch_rng.random = lambda: 0.999  # every solo attempt fails
        env.step(stop(env))
        assert env.stats["young_failures"] == 1
        place(first, 420, 420)
        for _ in range(gap - 1):
            env.step(stop(env))
        place(second, 380, 300)
        env.step(stop(env))
        assert (not deer.alive) is caught, gap
        assert env.stats["young_caught_party"] == int(caught)


def test_touch_rule_adults_as_in_the_party_rule_and_touches_survive_growing_up():
    env = world(deer_catch_rule="touch")
    deer = deer_at(env, 4.9)
    first, second = env.predators[:2]
    place(first, 420, 300)
    place(second, 400, 350)  # near, not touching
    env.catch_rng.random = lambda: 0.999
    env.step(stop(env))
    assert deer.alive and env.stats["young_failures"] == 1
    place(first, 420, 420)
    deer.energy = 5.5  # now adult; the failed touch still counts
    place(second, 380, 300)
    env.step(stop(env))
    assert not deer.alive and env.stats["adults_caught"] == 1


def test_touch_rule_a_lone_adult_touch_fails():
    env = world(deer_catch_rule="touch")
    deer = deer_at(env, 6.0)
    toucher = env.predators[0]
    place(toucher, 420, 300)
    env.catch_rng.random = lambda: pytest.fail("an adult draws no number")
    env.step(stop(env))
    assert deer.alive and env.stats["adult_failures"] == 1
    assert toucher.catch_ready_at == env.time_step + env.catch_cooldown


def test_old_run_settings_without_the_catch_rule_load_as_party(tmp_path):
    import json

    from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.checkpoints import checkpoint_env_config

    old = {k: v for k, v in DEFAULT_CONFIG.items() if k != "deer_catch_rule"}
    (tmp_path / "source_code").mkdir()
    (tmp_path / "source_code" / "settings.json").write_text(json.dumps({"config_env": old, "config_ppo": {}}))
    (tmp_path / "checkpoint").mkdir()
    assert checkpoint_env_config(tmp_path / "checkpoint")["deer_catch_rule"] == "party"


def test_touch_rule_one_hunter_touching_again_after_its_wait_is_not_two_touchers():
    env = world(deer_catch_rule="touch", deer_catch_cooldown=2)
    deer = deer_at(env, 4.0)
    hunter = env.predators[0]
    place(hunter, 420, 300)
    env.catch_rng.random = lambda: 0.999
    for _ in range(6):
        env.step(stop(env))
    assert deer.alive and env.stats["young_caught_party"] == 0
    assert env.stats["young_failures"] == 2  # touches at steps 1 and 4 only


def test_touch_rule_a_remembered_toucher_that_died_gets_no_share():
    env = world(deer_catch_rule="touch")
    deer = deer_at(env, 4.0)
    first, second, third = env.predators[:3]
    place(first, 420, 300)
    env.catch_rng.random = lambda: 0.999
    env.step(stop(env))
    first.energy = 1e-9  # starves at the next step, touching nothing
    place(first, 420, 420)
    env.step(stop(env))
    assert not first.alive and deer.id in env.adult_touches
    place(second, 380, 300)
    env.step(stop(env))
    assert deer.alive and env.stats["young_failures"] == 2  # the dead toucher does not count
    place(second, 380, 420)
    place(third, 400, 280)
    gains = []
    gain = env._gain
    env._gain = lambda animal, amount: (gains.append((animal.id, amount)), gain(animal, amount))
    env.step(stop(env))
    assert not deer.alive and env.stats["young_caught_party"] == 1
    shares = dict(g for g in gains if g[0].startswith("predator"))
    assert set(shares) == {second.id, third.id}  # the dead first toucher is not there
    assert shares[second.id] == shares[third.id] == pytest.approx(1.0, abs=0.01)


def test_touch_rule_three_touching_at_once_share_evenly_and_touches_survive_shrinking():
    env = world(deer_catch_rule="touch")
    deer = deer_at(env, 6.0)
    a, b, c = env.predators[:3]
    for p, (x, y) in zip((a, b, c), ((425, 300), (387.5, 321.65), (387.5, 278.35))):  # 25 from the deer, 43 apart
        place(p, x, y)
    before = [p.energy for p in (a, b, c)]
    env.step(stop(env))
    assert not deer.alive and env.stats["adults_caught"] == 1
    for p, e in zip((a, b, c), before):
        assert p.energy == pytest.approx(e + 0.5 * 6.0 / 3 - env.resting_cost["predator"])
    env = world(deer_catch_rule="touch")
    deer = deer_at(env, 5.5)
    first, second = env.predators[:2]
    place(first, 420, 300)
    env.step(stop(env))
    assert env.stats["adult_failures"] == 1
    place(first, 420, 420)
    deer.energy = 4.0  # no longer adult; the failed touch still counts
    env.catch_rng.random = lambda: pytest.fail("a second toucher draws no number")
    place(second, 380, 300)
    env.step(stop(env))
    assert not deer.alive and env.stats["young_caught_party"] == 1


def test_party_rule_given_explicitly_is_the_default_world_draw_for_draw():
    a, b = DeerStagHunt({}), DeerStagHunt({"deer_catch_rule": "party"})
    obs_a, _ = a.reset(seed=3)
    obs_b, _ = b.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(300):
        actions = {agent: int(rng.integers(MOVES)) for agent in a.active}
        obs_a, *_ = a.step(actions)
        obs_b, *_ = b.step(actions)
        assert a.stats == b.stats and obs_a.keys() == obs_b.keys()
        assert all(np.array_equal(obs_a[k], obs_b[k]) for k in obs_a)
        if not a.active:
            break
    assert a.catch_rng.random() == b.catch_rng.random()
