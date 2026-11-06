"""Tests of the lean and fat rabbits (rabbit_fat_energy) and of predators
seeing the rabbits' condition (observe_rabbit_condition), ported from
stag_hunt_rule_prey."""

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import StagHuntContinuous
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.geometry import Vector
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.tests.test_environment import random_actions, stopped_actions, world


def rabbit_beside(env, *numbers, energy):
    """Puts the rabbit touching predator numbers[0], with the others near."""
    rabbit = env.rabbits[0]
    rabbit.energy = energy
    rabbit.velocity = Vector(0, 0)
    first = env.predators[numbers[0]]
    rabbit.position = Vector(first.position.x + 20, first.position.y)
    for k, n in enumerate(numbers[1:]):
        env.predators[n].position = Vector(first.position.x + 20, first.position.y + 40 + 5 * k)
    env.action_repeat = 1
    return rabbit


def test_a_fat_rabbit_escapes_a_lone_hunter_who_then_waits():
    env = world(rabbit_fat_energy=6.0)
    rabbit = rabbit_beside(env, 0, energy=7.0)
    env.step(stopped_actions(env))
    assert env.rabbits and env.hunt_stats["rabbit_catch_failures"] == 1
    assert env.predators[0].catch_ready_at == env.time_step + env.config["rabbit_catch_cooldown"]
    assert rabbit.alive
    # In its cooldown the hunter does not try again.
    env.step(stopped_actions(env))
    assert env.hunt_stats["rabbit_catch_failures"] == 1


def test_a_lean_rabbit_is_caught_alone_and_the_catch_goes_to_the_hunter():
    env = world(rabbit_fat_energy=6.0)
    rabbit_beside(env, 0, energy=2.0)
    before = env.predators[0].energy
    env.step(stopped_actions(env))
    assert not env.rabbits and env.hunt_stats["rabbits_caught_solo"] == 1
    assert env.predators[0].energy == pytest.approx(before + 0.5 * 2.0)


def test_a_fat_rabbit_is_caught_by_two_who_share_it():
    env = world(rabbit_fat_energy=6.0)
    rabbit_beside(env, 0, 1, energy=7.0)
    before = [p.energy for p in env.predators[:2]]
    env.step(stopped_actions(env))
    assert not env.rabbits and env.hunt_stats["rabbits_caught_party"] == 1 and env.hunt_stats["rabbits_caught"] == 1
    for p, e in zip(env.predators[:2], before):
        assert p.energy == pytest.approx(e + 0.5 * 7.0 / 2)


def test_without_the_rule_one_hunter_catches_any_rabbit():
    env = world()
    rabbit_beside(env, 0, 1, energy=7.0)
    before = [p.energy for p in env.predators[:2]]
    env.step(stopped_actions(env))
    assert not env.rabbits and env.hunt_stats["rabbits_caught"] == 1
    # The whole catch to one hunter, none to the one beside it.
    assert env.predators[0].energy == pytest.approx(before[0] + 0.5 * 7.0)
    assert env.predators[1].energy == pytest.approx(before[1])


def test_a_toucher_shares_even_with_a_party_radius_below_touching():
    env = world(rabbit_fat_energy=6.0, rabbit_party_radius=0.0)
    rabbit_beside(env, 0, energy=2.0)
    before = env.predators[0].energy
    env.step(stopped_actions(env))
    assert not env.rabbits and env.predators[0].energy == pytest.approx(before + 0.5 * 2.0)


def test_a_rabbit_in_a_burrow_is_not_caught_and_costs_no_cooldown():
    env = world(rabbit_fat_energy=6.0, burrow_count=1, burrow_radius=30.0)
    rabbit = rabbit_beside(env, 0, energy=7.0)
    env.burrows = np.array([[rabbit.position.x, rabbit.position.y]])
    rabbit.in_burrow = True
    env.step(stopped_actions(env))
    assert rabbit.alive and env.hunt_stats["catches_blocked"] == 1
    assert env.hunt_stats["rabbit_catch_failures"] == 0 and env.predators[0].catch_ready_at == 0


def test_predators_see_the_rabbits_energy_and_their_catch_cooldown():
    env = world(rabbit_fat_energy=6.0, observe_rabbit_condition=True, predator_fov=360)
    plain = world(rabbit_fat_energy=6.0, predator_fov=360)
    n = env.observe[("predator", "rabbit")]
    assert env.obs_size["predator"] == plain.obs_size["predator"] + n + 1
    assert env.obs_size["rabbit"] == plain.obs_size["rabbit"]
    rabbit_beside(env, 0, energy=7.0)
    obs, *_ = env.step(stopped_actions(env))
    tail = obs["predator_0"][-(n + 1):]
    assert tail[0] == pytest.approx(env.rabbits[0].energy / env.config["energy_rabbit_max"], abs=1e-6)
    assert list(tail[1:n]) == [0.0] * (n - 1)
    assert tail[n] == pytest.approx(1.0)  # a full cooldown just started
    assert obs["predator_1"][-1] == 0.0
    assert env.observation_spaces["predator_0"].contains(obs["predator_0"])


def test_the_default_world_is_unchanged_by_the_new_settings():
    """Off by default: the same random play as with the settings stated
    explicitly off, and no new counters move."""

    def run(**config):
        env = StagHuntContinuous(dict(mammoth_count=0, max_time_steps=1500, **config))
        env.reset(seed=3)
        rng = np.random.default_rng(1)
        trace = []
        while True:
            _, rewards, terms, truncs, _ = env.step(random_actions(env, rng))
            trace.append((sorted(rewards.items()), [(a.id, round(a.position.x, 9), round(a.energy, 9)) for a in env.living()]))
            if terms["__all__"] or truncs["__all__"]:
                return trace, env

    plain, env = run()
    explicit, _ = run(rabbit_fat_energy=0.0, observe_rabbit_condition=False, rabbit_fat_party_unsplit=False)
    assert plain == explicit
    assert env.hunt_stats["rabbits_caught"] > 0
    assert env.hunt_stats["rabbits_caught_solo"] == env.hunt_stats["rabbits_caught_party"] == env.hunt_stats["rabbit_catch_failures"] == 0


@pytest.mark.parametrize(
    "settings",
    [{"rabbit_fat_energy": 2.0}, {"rabbit_catch_cooldown": -1}, {"rabbit_party_radius": -1.0}, {"observe_rabbit_condition": 1}],
)
def test_lean_and_fat_settings_are_validated(settings):
    with pytest.raises(ValueError):
        StagHuntContinuous(settings)


def test_the_arm_is_round_1a_with_the_rule():
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.launchers.calibration import ARMS, INIT_FROM

    arm = dict(ARMS["lean_fat_learning"])
    assert arm.pop("rabbit_fat_energy") == 6.0 and arm.pop("observe_rabbit_condition") is True
    assert arm == ARMS["rabbits_only"]
    assert "lean_fat_learning" not in INIT_FROM


def test_a_middle_rabbit_is_caught_alone_by_chance_from_its_own_seeded_stream():
    import random

    outcomes = []
    for seed in range(12):
        env = world(rabbit_fat_energy=6.0)
        env.reset(seed=seed)
        rabbit_beside(env, 0, energy=4.5)  # catch chance (6 - 4.5) / (6 - 3) = 0.5
        env.step(stopped_actions(env))
        expected = random.Random(seed + 6).random() < 0.5
        assert (not env.rabbits) == expected
        assert env.hunt_stats["rabbits_caught_solo"] + env.hunt_stats["rabbit_catch_failures"] == 1
        outcomes.append(expected)
    assert any(outcomes) and not all(outcomes)


def test_coop_fair_summary_handles_empty_counts():
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.launchers.coop_fair import KINDS, summarise

    row = {"hunter_steps": 0, "with_partner": 0, **{f"{k}_{x}": 0 for k in KINDS for x in ("approaches", "with_partner")}}
    s = summarise([row])
    assert s["chance"] is None and s["cooperation_shown"] is False
    row.update(hunter_steps=10, with_partner=2, fat_approaches=4, fat_with_partner=2, lean_approaches=5, lean_with_partner=1)
    s = summarise([row])
    assert s["chance"] == 0.2 and s["share"]["fat"] == 0.5 and s["cooperation_shown"] is True


def test_an_unsplit_fat_rabbit_pays_each_member_of_the_party_in_full():
    env = world(rabbit_fat_energy=6.0, rabbit_fat_party_unsplit=True)
    rabbit_beside(env, 0, 1, energy=7.0)
    before = [p.energy for p in env.predators[:2]]
    env.step(stopped_actions(env))
    assert not env.rabbits and env.hunt_stats["rabbits_caught_fat_unsplit"] == 1
    for p, e in zip(env.predators[:2], before):
        assert p.energy == pytest.approx(e + 0.5 * 7.0)


def test_unsplit_leaves_middle_party_catches_and_lone_catches_split_as_before():
    env = world(rabbit_fat_energy=6.0, rabbit_fat_party_unsplit=True)
    rabbit_beside(env, 0, 1, energy=4.0)  # a middle rabbit, caught by the party
    before = [p.energy for p in env.predators[:2]]
    env.step(stopped_actions(env))
    assert not env.rabbits and env.hunt_stats["rabbits_caught_fat_unsplit"] == 0
    for p, e in zip(env.predators[:2], before):
        assert p.energy == pytest.approx(e + 0.5 * 4.0 / 2)
    env = world(rabbit_fat_energy=6.0, rabbit_fat_party_unsplit=True)
    rabbit_beside(env, 0, energy=2.0)
    before = env.predators[0].energy
    env.step(stopped_actions(env))
    assert env.predators[0].energy == pytest.approx(before + 0.5 * 2.0)
    assert env.hunt_stats["rabbits_caught_fat_unsplit"] == 0


@pytest.mark.parametrize("settings", [{"rabbit_fat_party_unsplit": True}, {"rabbit_fat_energy": 6.0, "rabbit_fat_party_unsplit": 1}])
def test_unsplit_is_validated(settings):
    with pytest.raises(ValueError):
        StagHuntContinuous(settings)


def test_the_unsplit_arm_is_the_lean_fat_arm_with_the_flag():
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.launchers.calibration import ARMS, INIT_FROM

    arm = dict(ARMS["lean_fat_unsplit"])
    assert arm.pop("rabbit_fat_party_unsplit") is True
    assert arm == ARMS["lean_fat_learning"] and "lean_fat_unsplit" not in INIT_FROM


@pytest.mark.parametrize("energy, unsplit", [(6.0, True), (2.5, False)])
def test_unsplit_at_the_fat_boundary_and_for_a_lean_party_catch(energy, unsplit):
    env = world(rabbit_fat_energy=6.0, rabbit_fat_party_unsplit=True)
    rabbit_beside(env, 0, 1, energy=energy)
    before = [p.energy for p in env.predators[:2]]
    env.step(stopped_actions(env))
    assert not env.rabbits and env.hunt_stats["rabbits_caught_fat_unsplit"] == unsplit
    for p, e in zip(env.predators[:2], before):
        assert p.energy == pytest.approx(e + 0.5 * energy / (1 if unsplit else 2))


def test_every_party_member_gains_in_full_and_a_hunter_beyond_the_radius_nothing():
    env = world(rabbit_fat_energy=6.0, rabbit_fat_party_unsplit=True)
    rabbit = rabbit_beside(env, 0, 1, 2, energy=7.0)
    far = env.predators[3]
    far.position = Vector(rabbit.position.x + 300, rabbit.position.y)
    before = [p.energy for p in env.predators[:4]]
    env.step(stopped_actions(env))
    for p, e in zip(env.predators[:3], before):
        assert p.energy == pytest.approx(e + 0.5 * 7.0)
    assert far.energy <= before[3]


def test_flag_off_is_unchanged_in_the_lean_and_fat_world():
    def run(**config):
        env = StagHuntContinuous(dict(mammoth_count=0, max_time_steps=1500, rabbit_fat_energy=6.0, **config))
        env.reset(seed=3)
        rng = np.random.default_rng(1)
        trace = []
        while True:
            _, rewards, terms, truncs, _ = env.step(random_actions(env, rng))
            trace.append((sorted(rewards.items()), [(a.id, round(a.position.x, 9), round(a.energy, 9)) for a in env.living()]))
            if terms["__all__"] or truncs["__all__"]:
                return trace, env

    plain, env = run()
    explicit, _ = run(rabbit_fat_party_unsplit=False)
    assert plain == explicit
    assert env.hunt_stats["rabbits_caught_solo"] + env.hunt_stats["rabbits_caught_party"] + env.hunt_stats["rabbit_catch_failures"] > 0
    assert env.hunt_stats["rabbits_caught_fat_unsplit"] == 0
