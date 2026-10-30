"""Tests of the stag-hunt environment: spaces, determinism, contact
catching, food layers, episode ends and the hunt rule (DESIGN.md, 3.2)."""

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import (
    MOVES,
    StagHuntContinuous,
    species_of,
)
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.geometry import Vector, heading_degrees

STOP = MOVES - 1  # speed 0


def stopped_actions(env, join=0):
    return {a: (np.array([STOP, join]) if species_of(a) == "predator" else STOP) for a in env.active}


def random_actions(env, rng):
    return {a: (np.array([rng.integers(MOVES), rng.integers(2)]) if species_of(a) == "predator" else int(rng.integers(MOVES))) for a in env.active}


def place(env, positions):
    """Put animals at given positions, standing still; others far away in a
    corner band so they touch nothing of interest."""
    for animal in env.living():
        animal.velocity = Vector(0, 0)
        if animal.id in positions:
            animal.position = Vector(*positions[animal.id])
    return env


def world(**config):
    """A world with hunters, mammoths and no rabbits or food in the way."""
    settings = dict(
        predator_count=4,
        rabbit_count=1,
        mammoth_count=2,
        grass_count=0,
        steppe_count=0,
        energy_predator_resting_metabolic_cost=0.0,
        energy_predator_speed_cost=0.0,
        energy_mammoth_resting_metabolic_cost=0.0,
        energy_mammoth_speed_cost=0.0,
        energy_rabbit_resting_metabolic_cost=0.0,
        energy_rabbit_speed_cost=0.0,
    )
    settings.update(config)
    env = StagHuntContinuous(settings)
    env.reset(seed=0)
    # Everything far apart: predators along y = 700, mammoths at y = 100,
    # the rabbit at (400, 400).
    positions = {f"predator_{i}": (100 + 150 * i, 700) for i in range(4)}
    positions.update({"mammoth_0": (200, 100), "mammoth_1": (600, 100), "rabbit_0": (400, 400)})
    return place(env, positions)


def hunt(env, joins):
    terminateds, truncateds, infos = {}, {}, {}
    env._hunts(joins, terminateds, truncateds, infos)
    return terminateds, truncateds, infos


def energies(env):
    return {p.id: p.energy for p in env.predators}


def test_spaces_and_observations():
    env = StagHuntContinuous()
    obs, _ = env.reset(seed=3)
    sizes = {species_of(a): o.shape[0] for a, o in obs.items()}
    assert sizes == {"predator": 58, "rabbit": 45, "mammoth": 39}
    rng = np.random.default_rng(3)
    for _ in range(30):
        obs, _, te, tr, _ = env.step(random_actions(env, rng))
        for agent, o in obs.items():
            assert env.observation_spaces[agent].contains(o)
        if te["__all__"] or tr["__all__"]:
            break


def test_reset_is_deterministic():
    a, b = StagHuntContinuous(), StagHuntContinuous()
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    assert oa.keys() == ob.keys()
    for agent in oa:
        np.testing.assert_array_equal(oa[agent], ob[agent])
    ra, rb = np.random.default_rng(1), np.random.default_rng(1)
    for _ in range(20):
        sa = a.step(random_actions(a, ra))
        sb = b.step(random_actions(b, rb))
        for agent in sa[0]:
            np.testing.assert_array_equal(sa[0][agent], sb[0][agent])
        assert sa[1] == sb[1]


def test_invalid_actions_and_settings():
    env = StagHuntContinuous()
    env.reset(seed=0)
    actions = stopped_actions(env)
    actions["predator_0"] = np.array([STOP, 2])
    with pytest.raises(ValueError):
        env.step(actions)
    actions["predator_0"] = STOP  # a predator needs [move, join]
    with pytest.raises(ValueError):
        env.step(actions)
    with pytest.raises(ValueError):
        StagHuntContinuous({"hunt_success_by_party": (0.1, 0.2, 0.3)})
    with pytest.raises(ValueError):
        StagHuntContinuous({"hunt_scavenger_fraction": 1.5})
    with pytest.raises(ValueError):
        StagHuntContinuous({"no_such_setting": 1})


def test_rabbit_caught_on_contact_mammoth_not():
    env = world()
    place(env, {"predator_0": (400, 410), "predator_1": (200, 110)})
    before = env.predators[0].energy
    rabbit_energy = env.rabbits[0].energy
    _, _, te, _, _ = env.step(stopped_actions(env, join=0))
    assert te["rabbit_0"] and not env.rabbits
    assert env.predators[0].energy == pytest.approx(min(24.0, before + 0.5 * rabbit_energy))
    # predator_1 touches mammoth_0 with join off: nothing happens to it.
    assert any(m.id == "mammoth_0" for m in env.mammoths)


def test_food_layers_are_separate():
    env = StagHuntContinuous({"predator_count": 1, "rabbit_count": 1, "mammoth_count": 1})
    env.reset(seed=0)
    grass, steppe = env.food["grass"], env.food["steppe"]
    # The rabbit stands on a steppe patch, the mammoth on a grass patch.
    place(env, {"rabbit_0": (steppe.patches[0].position.x, steppe.patches[0].position.y),
                "mammoth_0": (grass.patches[0].position.x, grass.patches[0].position.y),
                "predator_0": (5, 5)})
    rabbit, mammoth = env.rabbits[0], env.mammoths[0]
    eaten_r = grass.advance([rabbit])
    eaten_m = steppe.advance([mammoth])
    assert eaten_r.get("rabbit_0", 0) == 0
    assert eaten_m.get("mammoth_0", 0) == 0


def test_successful_pair_hunt_shares_carcass():
    env = world(hunt_success_by_party=(1.0, 1.0, 1.0, 1.0))
    # Two joiners touch mammoth_0, one stander-by within 96, one far away.
    place(env, {"predator_0": (200, 130), "predator_1": (230, 100), "predator_2": (200, 180), "predator_3": (700, 700)})
    start = energies(env)
    te, tr, infos = hunt(env, {"predator_0": 1, "predator_1": 1, "predator_2": 0, "predator_3": 1})
    assert te["mammoth_0"] and [m.id for m in env.mammoths] == ["mammoth_1"]
    carcass, s, cost = 32.0, 0.2, 0.05
    after = energies(env)
    for p in ("predator_0", "predator_1"):
        assert after[p] == pytest.approx(start[p] - cost + carcass * (1 - s) / 2)
    assert after["predator_2"] == pytest.approx(start["predator_2"] + carcass * s)
    assert after["predator_3"] == start["predator_3"]
    gained = sum(after[p] - start[p] for p in after) + 2 * cost
    assert gained == pytest.approx(carcass)
    assert env.hunt_stats["hunts_party_2"] == 1 and env.hunt_stats["kills_party_2"] == 1
    assert env.hunt_stats["scavenger_energy"] == pytest.approx(carcass * s)


def test_without_bystanders_contributors_take_all():
    env = world(hunt_success_by_party=(1.0, 1.0, 1.0, 1.0))
    place(env, {"predator_0": (200, 130), "predator_1": (230, 100)})
    start = energies(env)
    hunt(env, {"predator_0": 1, "predator_1": 1, "predator_2": 0, "predator_3": 0})
    for p in ("predator_0", "predator_1"):
        assert env.predators[int(p[-1])].energy == pytest.approx(start[p] - 0.05 + 16.0)


def test_failed_hunt_costs_and_cooldown():
    env = world(hunt_success_by_party=(0.0, 0.0, 0.0, 0.0))
    place(env, {"predator_0": (200, 130), "predator_1": (230, 100)})
    start = energies(env)
    te, _, _ = hunt(env, {"predator_0": 1, "predator_1": 1, "predator_2": 0, "predator_3": 0})
    assert "mammoth_0" not in te and len(env.mammoths) == 2
    for p in ("predator_0", "predator_1"):
        assert energies(env)[p] == pytest.approx(start[p] - 0.05 - 0.05 * 32.0 / 2)
    # Cooldown: the next two decisions they cannot contribute.
    for step in range(3):
        env.decision += 1
        parties = env.hunt_parties({"predator_0": 1, "predator_1": 1, "predator_2": 0, "predator_3": 0})
        assert bool(parties) == (step == 2)
    # Observed remaining cooldown: decisions 1 and 2 blocked.
    for decision, remaining in ((1, 1.0), (2, 0.5), (3, 0.0)):
        env.decision = decision
        assert env._observe(env.predators[0])[-1] == pytest.approx(remaining)


def test_party_rules():
    env = world(hunt_success_by_party=(1.0, 1.0, 1.0, 1.0))
    # predator_0 touches both mammoths (moved close together); it contributes
    # to the nearest only. predator_1 touches mammoth_0 with join off: a
    # bystander, not a contributor.
    place(env, {"mammoth_0": (300, 100), "mammoth_1": (360, 100), "predator_0": (325, 100), "predator_1": (300, 130)})
    parties = env.hunt_parties({"predator_0": 1, "predator_1": 0, "predator_2": 0, "predator_3": 0})
    assert [m.id for m in parties] == ["mammoth_0"]
    contributors, bystanders = parties[env.mammoths[0]]
    assert [p.id for p in contributors] == ["predator_0"]
    assert [p.id for p in bystanders] == ["predator_1"]
    # A predator that did not act this decision (not in joins) cannot contribute.
    assert not env.hunt_parties({"predator_1": 0})


def test_solo_success_table():
    env = world(hunt_success_by_party=(1.0, 0.0, 0.0, 0.0))
    place(env, {"predator_0": (200, 130)})
    te, _, _ = hunt(env, {"predator_0": 1, "predator_1": 0, "predator_2": 0, "predator_3": 0})
    assert te["mammoth_0"]
    assert env.hunt_stats["kills_party_1"] == 1


def test_hunts_happen_in_step_once_per_decision():
    env = world(hunt_success_by_party=(0.0, 0.0, 0.0, 0.0), hunt_cooldown=0)
    place(env, {"predator_0": (200, 130), "predator_1": (230, 100)})
    env.step(stopped_actions(env, join=1))
    assert env.hunt_stats["hunts_party_2"] == 1
    assert env.hunt_stats["contact_decisions"] == 2 and env.hunt_stats["contact_with_partner"] == 2


def test_starvation_by_hunt_costs():
    env = world(hunt_success_by_party=(0.0, 0.0, 0.0, 0.0))
    place(env, {"predator_0": (200, 130)})
    env.predators[0].energy = 0.01
    te, _, infos = hunt(env, {"predator_0": 1, "predator_1": 0, "predator_2": 0, "predator_3": 0})
    assert te["predator_0"] and infos["predator_0"]["starved"]
    assert env.hunt_stats["predators_starved_in_hunts"] == 1


def test_episode_continues_without_rabbits_and_ends_without_prey():
    env = world(hunt_success_by_party=(1.0, 1.0, 1.0, 1.0))
    env.animals["rabbit"].clear()
    _, _, te, tr, _ = env.step(stopped_actions(env))
    assert not te["__all__"] and not tr["__all__"]
    # Kill both mammoths in one decision: nothing left to hunt.
    place(env, {"predator_0": (200, 130), "predator_1": (600, 130)})
    _, _, te, tr, _ = env.step(stopped_actions(env, join=1))
    assert te["__all__"] and not tr["__all__"]
    assert env.episode_hunt_stats["kills_party_1"] == 2


def test_episode_ends_when_predators_die():
    env = world()
    for predator in env.predators:
        predator.energy = 1e-9
    env.config["energy_predator_resting_metabolic_cost"] = 0.0
    env.resting_cost["predator"] = 1.0
    _, _, te, tr, _ = env.step(stopped_actions(env))
    assert tr["__all__"] or te["__all__"]
    assert not env.predators


def test_observation_layout():
    """Self, 4 predator, 2 mammoth, 2 rabbit slots, grass, energy, cooldown."""
    env = world(rabbit_count=1)
    me = env.predators[0]
    place(env, {"predator_0": (400, 400), "mammoth_0": (400, 300), "rabbit_0": (400, 250), "predator_1": (400, 350)})
    me.orientation = heading_degrees(Vector(0, -1))  # facing -y, where the others are
    obs = env._observe(me)
    predator_slots, mammoth_slots, rabbit_slots = obs[4:28], obs[28:40], obs[40:52]
    assert predator_slots[0] == 1 and predator_slots[3] == pytest.approx(50 / 200)
    assert mammoth_slots[0] == 1 and mammoth_slots[3] == pytest.approx(100 / 200)
    assert rabbit_slots[0] == 1 and rabbit_slots[3] == pytest.approx(150 / 200)
    assert obs[56] == pytest.approx(me.energy / 24.0)


def test_agents_are_the_acting_agents():
    env = world(hunt_success_by_party=(1.0, 1.0, 1.0, 1.0))
    place(env, {"predator_0": (200, 130)})
    _, _, te, _, _ = env.step(stopped_actions(env, join=1))
    assert te["mammoth_0"]
    assert "mammoth_0" not in env.agents
    assert sorted(env.agents) == sorted(env.active)


def test_tie_goes_to_lower_agent_number():
    env = world(mammoth_count=2)
    # mammoth_1 is first in the list but has the higher number.
    env.animals["mammoth"].reverse()
    place(env, {"mammoth_0": (300, 100), "mammoth_1": (340, 100), "predator_0": (320, 100)})
    parties = env.hunt_parties({"predator_0": 1})
    assert [m.id for m in parties] == ["mammoth_0"]


def test_contact_counts_include_cooldown():
    env = world(hunt_success_by_party=(0.0, 0.0, 0.0, 0.0))
    place(env, {"predator_0": (200, 130)})
    joins = {"predator_0": 1, "predator_1": 0, "predator_2": 0, "predator_3": 0}
    hunt(env, joins)  # fails: cooldown
    env.decision += 1
    hunt(env, joins)  # in cooldown: a contact, not ready, no hunt
    stats = env.hunt_stats
    assert stats["contact_decisions"] == 2 and stats["contact_ready"] == 1
    assert stats["hunts_party_1"] == 1


def test_capped_carcass_gain():
    env = world(hunt_success_by_party=(1.0, 1.0, 1.0, 1.0))
    place(env, {"predator_0": (200, 130)})
    env.predators[0].energy = 20.0
    hunt(env, {"predator_0": 1, "predator_1": 0, "predator_2": 0, "predator_3": 0})
    assert env.predators[0].energy == 24.0
    assert env.hunt_stats["mammoth_energy"] == pytest.approx(24.0 - (20.0 - 0.05))


def test_joint_arrivals():
    env = world(hunt_success_by_party=(0.0, 0.0, 0.0, 0.0), hunt_cooldown=0)
    joins = {f"predator_{i}": 0 for i in range(4)}
    # predator_0 arrives at decision 0, predator_1 at decision 2: joint.
    place(env, {"predator_0": (200, 130)})
    hunt(env, joins)
    env.decision = 2
    place(env, {"predator_1": (230, 100)})
    hunt(env, joins)
    # Both leave: the encounter episode closes.
    env.decision = 3
    place(env, {"predator_0": (100, 700), "predator_1": (250, 700)})
    hunt(env, joins)
    # A second episode at mammoth_1: arrivals 3 decisions apart, not joint.
    env.decision = 4
    place(env, {"predator_2": (600, 130)})
    hunt(env, joins)
    env.decision = 7
    place(env, {"predator_3": (630, 100)})
    hunt(env, joins)
    env.decision = 8
    place(env, {"predator_2": (400, 700), "predator_3": (550, 700)})
    hunt(env, joins)
    stats = env.hunt_stats
    assert stats["encounter_episodes"] == 2
    assert stats["touched_episodes"] == 2
    assert stats["joint_arrivals"] == 1


def test_spawn_matches_the_base_world():
    from predpreygrass.non_evolutionary.continuous_space.base_environment_continuous.environment import FIXES, PredPreyGrassContinuous

    base = PredPreyGrassContinuous(FIXES)
    base.reset(seed=5)
    env = StagHuntContinuous()
    env.reset(seed=5)
    assert [(a.position.x, a.position.y) for a in base.prey] == [(a.position.x, a.position.y) for a in env.rabbits]
    assert [(a.position.x, a.position.y) for a in base.predators] == [(a.position.x, a.position.y) for a in env.predators]


def test_time_limit():
    env = StagHuntContinuous({"max_time_steps": 16})
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    steps = 0
    while True:
        _, _, te, tr, _ = env.step(random_actions(env, rng))
        steps += 1
        if te["__all__"] or tr["__all__"]:
            break
    assert env.time_step <= 16 and steps <= 2
    assert env.episode_hunt_stats is not None and env.agents == []


def test_expected_values_and_rates():
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.calibrate import expected_values
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.callbacks import derived_rates
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import DEFAULT_CONFIG, new_hunt_stats

    rows = expected_values(DEFAULT_CONFIG)
    # A solo hunt never succeeds: it costs the join cost and the struggle.
    assert rows[0][2] == pytest.approx(-(0.05 * 32.0 + 0.05))
    # A pair: 0.4 x 16 - 0.6 x 0.8 - 0.05.
    assert rows[1][2] == pytest.approx(0.4 * 16 - 0.6 * 0.8 - 0.05)
    stats = new_hunt_stats()
    stats.update(decisions=10, joins=5, contact_decisions=4, contact_joins=3, encounter_episodes=4, joint_arrivals=1, hunts_party_1=3, hunts_party_2=1)
    rates = derived_rates(stats)
    assert rates["join_rate"] == 0.5 and rates["contact_join_rate"] == 0.75
    assert rates["joint_arrival_rate"] == 0.25 and rates["party_2plus_share"] == 0.25
