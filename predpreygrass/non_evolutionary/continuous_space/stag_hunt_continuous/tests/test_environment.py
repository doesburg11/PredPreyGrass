"""Tests of the stag-hunt environment: spaces, determinism, contact
catching, food layers, episode ends and the hunt rule (DESIGN.md, 3.2)."""

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import (
    BURROW_SIZE,
    EXIT_WINDOW,
    MOVES,
    OBSERVED,
    SELF_SIZE,
    SLOT_SIZE,
    StagHuntContinuous,
    species_of,
)
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.callbacks import derived_rates
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


def test_truncate_on_prey_extinction():
    # Rabbits gone, mammoths left: truncated, not terminated.
    env = world(truncate_on_prey_extinction=True)
    env.animals["rabbit"].clear()
    _, _, te, tr, _ = env.step(stopped_actions(env))
    assert tr["__all__"] and not te["__all__"]
    assert all(tr[p.id] and not te[p.id] for p in env.predators)
    # Mammoths hunted out, the rabbit left: truncated too.
    env = world(truncate_on_prey_extinction=True, hunt_success_by_party=(1.0, 1.0, 1.0, 1.0))
    place(env, {"predator_0": (200, 130), "predator_1": (600, 130)})
    _, _, te, tr, _ = env.step(stopped_actions(env, join=1))
    assert not env.mammoths and env.rabbits
    assert tr["__all__"] and not te["__all__"]
    assert env.episode_hunt_stats["kills_party_1"] == 2
    # No prey of either kind: terminated, as without the setting.
    env = world(truncate_on_prey_extinction=True)
    env.animals["rabbit"].clear()
    env.animals["mammoth"].clear()
    _, _, te, tr, _ = env.step(stopped_actions(env))
    assert te["__all__"] and not tr["__all__"]
    # The last rabbit caught, or the mammoths starved: the dead terminated
    # only, the survivors truncated only.
    env = world(truncate_on_prey_extinction=True)
    place(env, {"predator_0": (400, 410)})
    _, _, te, tr, _ = env.step(stopped_actions(env))
    assert te["rabbit_0"] and not tr["rabbit_0"]
    assert tr["__all__"] and not te["__all__"]
    assert all(tr[a.id] and not te[a.id] for a in env.living())
    assert env.episode_hunt_stats["rabbits_caught"] == 1
    env = world(truncate_on_prey_extinction=True)
    for mammoth in env.mammoths:
        mammoth.energy = 1e-9
    env.resting_cost["mammoth"] = 1.0
    _, _, te, tr, _ = env.step(stopped_actions(env))
    assert not env.mammoths and env.rabbits
    assert all(te[f"mammoth_{i}"] and not tr[f"mammoth_{i}"] for i in range(2))
    assert tr["__all__"] and not te["__all__"]
    assert all(tr[a.id] and not te[a.id] for a in env.living())
    # A species absent from the start does not end the episode.
    env = world(truncate_on_prey_extinction=True, mammoth_count=0)
    _, _, te, tr, _ = env.step(stopped_actions(env))
    assert not te["__all__"] and not tr["__all__"]
    with pytest.raises(ValueError):
        StagHuntContinuous({"truncate_on_prey_extinction": 1})


def test_episode_ends_when_predators_die():
    env = world()
    for predator in env.predators:
        predator.energy = 1e-9
    env.config["energy_predator_resting_metabolic_cost"] = 0.0
    env.resting_cost["predator"] = 1.0
    _, _, te, tr, _ = env.step(stopped_actions(env))
    assert tr["__all__"] and not te["__all__"]
    assert not env.predators
    # The dead are terminated only, the surviving prey truncated only.
    assert all(te[f"predator_{i}"] and not tr[f"predator_{i}"] for i in range(4))
    assert all(tr[a.id] and not te[a.id] for a in env.living())


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


def test_birth_interval():
    for interval, births in ((0, [1, 1, 1, 1]), (3, [1, 0, 0, 1])):
        env = world(reproduction_predator_interval=interval)
        counts = []
        for step in range(4):
            env.predators[0].energy = 24.0
            env.time_step = 10 + step
            counts.append(sum(parent == "predator_0" for _, parent in env._reproduce({}, {})))
        assert counts == births
        # Newborns start below the threshold, so only predator_0 gave birth.
        assert env.births["predator"] == sum(births)
    with pytest.raises(ValueError):
        StagHuntContinuous({"reproduction_predator_interval": -1})
    with pytest.raises(ValueError):
        StagHuntContinuous({"reproduction_rabbit_interval": 2.5})


def test_birth_interval_in_steps():
    # One physics step per decision, so every step is visible.
    env = world(action_repeat=1, reproduction_predator_interval=3)
    for _ in range(2):
        born_at = []
        while env.time_step < 8:
            env.predators[0].energy = 24.0
            _, _, _, _, infos = env.step(stopped_actions(env))
            if infos.get("predator_0", {}).get("births"):
                born_at.append(env.time_step)
        assert born_at == [1, 4, 7]
        # A reset clears the wait.
        env.reset(seed=0)
        place(env, {})


def test_old_checkpoint_settings_get_added_settings(tmp_path):
    import json
    import pickle

    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.checkpoints import ADDED_SETTINGS, checkpoint_env_config
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import DEFAULT_CONFIG

    old = {k: v for k, v in DEFAULT_CONFIG.items() if k not in ADDED_SETTINGS}
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (tmp_path / "source_code").mkdir()
    (tmp_path / "source_code" / "settings.json").write_text(json.dumps({"config_env": old}))
    with open(checkpoint / "class_and_ctor_args.pkl", "wb") as f:
        pickle.dump({"ctor_args_and_kwargs": (({"env_config": old},), {})}, f)
    config = checkpoint_env_config(checkpoint)
    assert all(config[k] == v for k, v in ADDED_SETTINGS.items()) and config["burrow_count"] == 0
    # Other missing settings are still refused.
    del old["hunt_cooldown"]
    (tmp_path / "source_code" / "settings.json").write_text(json.dumps({"config_env": old}))
    with pytest.raises(ValueError):
        checkpoint_env_config(checkpoint)


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


def burrow_world(**config):
    """world() with one burrow of radius 24 at (400, 400), on the rabbit."""
    env = world(burrow_count=1, **config)
    env.burrows = np.array([[400.0, 400.0]])
    for rabbit, inside in zip(env.rabbits, env._inside(env.rabbits)):
        rabbit.in_burrow = inside
    return env


def test_burrows_leave_the_rest_of_the_world_unchanged():
    plain, burrows = StagHuntContinuous(), StagHuntContinuous({"burrow_count": 10})
    plain.reset(seed=7)
    burrows.reset(seed=7)
    assert burrows.burrows.shape == (10, 2)
    assert [(a.id, a.position.x, a.position.y) for a in plain.living()] == [(a.id, a.position.x, a.position.y) for a in burrows.living()]
    for layer in ("grass", "steppe"):
        assert [(p.position.x, p.position.y) for p in plain.food[layer].patches] == [(p.position.x, p.position.y) for p in burrows.food[layer].patches]
        assert plain.food[layer].rng.getstate() == burrows.food[layer].rng.getstate()
    # Grass inside a burrow is blocked; the steppe never is.
    grass = burrows.food["grass"]
    expected = [any(grass.torus.distance(p.position, Vector(x, y)) < 24.0 for x, y in burrows.burrows) for p in grass.patches]
    assert [p.blocked for p in grass.patches] == expected
    assert not any(p.blocked for p in burrows.food["steppe"].patches)
    again = StagHuntContinuous({"burrow_count": 10})
    again.reset(seed=7)
    np.testing.assert_array_equal(again.burrows, burrows.burrows)
    obs, _ = burrows.reset(seed=8)
    sizes = {species_of(a): o.shape[0] for a, o in obs.items()}
    assert sizes == {"predator": 64, "rabbit": 49, "mammoth": 39}
    rng = np.random.default_rng(8)
    for _ in range(30):
        obs, _, te, tr, _ = burrows.step(random_actions(burrows, rng))
        for agent, o in obs.items():
            assert burrows.observation_spaces[agent].contains(o)
        if te["__all__"] or tr["__all__"]:
            break


def test_burrow_settings_are_checked():
    with pytest.raises(ValueError, match="burrow_count"):
        StagHuntContinuous({"burrow_count": -1})
    with pytest.raises(ValueError, match="burrow_radius"):
        StagHuntContinuous({"burrow_radius": 0})
    with pytest.raises(ValueError, match="burrow_radius"):
        StagHuntContinuous({"burrow_radius": 400})


def test_rabbit_in_burrow_cannot_be_caught():
    env = burrow_world()
    place(env, {"predator_0": (400, 410)})
    _, _, te, _, _ = env.step(stopped_actions(env))
    assert not te.get("rabbit_0") and env.rabbits
    stats = env.hunt_stats
    assert stats["catches_blocked"] == stats["rabbit_steps_inside"] == stats["rabbit_steps"] == env.action_repeat
    assert stats["predator_steps_at_burrow_edge"] == env.action_repeat  # 10 from the centre, within 16 of the edge
    # Out of the burrow (centre 30 from the burrow's), still touching: caught
    # in the step it left, an exit catch.
    place(env, {"rabbit_0": (400, 430), "predator_0": (400, 440)})
    _, _, te, _, _ = env.step(stopped_actions(env))
    assert te["rabbit_0"] and not env.rabbits
    assert stats["burrow_exits"] == 1 and stats["catches_after_exit"] == 1 and stats["rabbits_caught"] == 1


def test_catch_long_after_exit_is_not_an_exit_catch():
    env = burrow_world(action_repeat=1)
    place(env, {"rabbit_0": (400, 500)})
    for _ in range(EXIT_WINDOW + 1):
        env.step(stopped_actions(env))
    assert env.hunt_stats["burrow_exits"] == 1
    place(env, {"predator_0": (400, 510)})
    env.step(stopped_actions(env))
    assert not env.rabbits
    assert env.hunt_stats["catches_after_exit"] == 0


def test_grass_in_burrow_cannot_be_eaten_or_seen():
    env = StagHuntContinuous({"predator_count": 1, "rabbit_count": 1, "mammoth_count": 1, "burrow_count": 1})
    env.reset(seed=0)
    grass = env.food["grass"]
    patch = grass.patches[0]
    grass.set_burrows([patch.position.copy()], 24.0)
    assert patch.blocked and sum(p.blocked for p in grass.patches) >= 1
    rabbit = env.rabbits[0]
    place(env, {"rabbit_0": (patch.position.x, patch.position.y)})
    others = [p for p in grass.patches if not p.blocked and grass.torus.distance(p.position, patch.position) <= grass.consume_radius]
    eaten = grass.advance([rabbit])
    assert eaten.get("rabbit_0", 0) == len(others)
    assert grass.available(patch) and grass.total_consumed == len(others)
    rabbit.orientation = 0.0
    food = grass.observation(rabbit, 267, 360)
    assert food[0] == 0 or food[3] > 0  # the blocked patch underfoot is not the nearest food


def test_burrow_observations():
    env = burrow_world(rabbit_count=2)
    me = env.predators[0]
    place(env, {"predator_0": (400, 500), "rabbit_0": (400, 400), "rabbit_1": (450, 450)})
    me.orientation = heading_degrees(Vector(0, -1))  # facing the burrow
    obs = env._observe(me)
    assert obs.shape == (64,)
    np.testing.assert_allclose(obs[58:62], [1.0, 0.0, -100 / 200, 100 / 200])
    # Rabbit slots nearest first: rabbit_1 (outside), then rabbit_0 (inside).
    assert obs[40] == 1 and obs[46] == 1
    np.testing.assert_array_equal(obs[62:64], [0.0, 1.0])
    # Facing away, the burrow is not seen.
    me.orientation = heading_degrees(Vector(0, 1))
    np.testing.assert_array_equal(env._observe(me)[58:62], 0.0)
    # Rabbits: the nearest burrow anywhere, [dx, dy, distance, inside].
    inside, outside = env.rabbits
    np.testing.assert_allclose(env._observe(inside)[45:49], [0.0, 0.0, 0.0, 1.0])
    place(env, {"rabbit_1": (400, 400 + 300)})  # 300 away, beyond the 267 view
    np.testing.assert_allclose(env._observe(outside)[45:49], [0.0, -1.0, 1.0, 0.0], atol=1e-6)


def test_newborn_rabbit_in_burrow_is_counted():
    env = burrow_world(rabbit_count=1, energy_rabbit_max=16.0)
    env.rabbits[0].energy = 15.0
    env.time_step = 1
    env.birth_rng.seed(0)
    rewards, infos = {}, {}
    born = env._reproduce(rewards, infos)
    assert len(born) == 1
    child = env.rabbits[-1]
    # Born 32 from its parent at the burrow's centre: outside a radius-24 burrow.
    assert not child.in_burrow and env.hunt_stats["births_rabbit_in_burrow"] == 0
    env.burrow_radius = 40.0
    env.rabbits[0].energy = 15.0
    env._reproduce(rewards, infos)
    assert env.rabbits[-1].in_burrow and env.hunt_stats["births_rabbit_in_burrow"] == 1


def test_steps_at_predator_cap():
    env = world(reproduction_max_predator=4)
    env.step(stopped_actions(env))
    assert env.hunt_stats["steps_at_predator_cap"] == env.action_repeat


@pytest.mark.parametrize("wait, counted", [(EXIT_WINDOW - 1, True), (EXIT_WINDOW, False)])
def test_exit_window_boundary(wait, counted):
    """A catch counts as at the exit when fewer than EXIT_WINDOW physics
    steps have passed since the step the rabbit left."""
    env = burrow_world(action_repeat=1)
    place(env, {"rabbit_0": (400, 500)})
    env.step(stopped_actions(env))  # leaves at this step
    left = env.rabbits[0].left_burrow_at
    for _ in range(wait - 1):
        env.step(stopped_actions(env))
    place(env, {"predator_0": (400, 510)})
    env.step(stopped_actions(env))  # caught at step left + wait
    assert not env.rabbits and env.time_step - 1 - left == wait
    assert env.hunt_stats["catches_after_exit"] == int(counted)


def test_latest_exit_counts_and_rabbit_walks_out():
    env = burrow_world(action_repeat=1)
    rabbit = env.rabbits[0]
    place(env, {"rabbit_0": (400, 500)})
    env.step(stopped_actions(env))
    place(env, {"rabbit_0": (400, 400)})
    env.step(stopped_actions(env))
    assert rabbit.in_burrow and env.hunt_stats["burrow_exits"] == 1
    # Walking out at full speed along +x (move 0) until it is out.
    rabbit.velocity = Vector(0, 0)
    moves = {a: (np.array([STOP, 0]) if species_of(a) == "predator" else (0 if a == "rabbit_0" else STOP)) for a in env.active}
    while env.hunt_stats["burrow_exits"] < 2:
        env.step({a: moves[a] for a in env.active})
    assert not rabbit.in_burrow and rabbit.left_burrow_at == env.time_step - 1
    assert env.hunt_stats["rabbit_steps"] == env.time_step


def test_burrows_wrap_around_the_torus():
    env = burrow_world()
    env.burrows = np.array([[5.0, 400.0]])
    place(env, {"rabbit_0": (795, 400), "predator_0": (795, 415)})  # 10 from the burrow across the edge
    assert env._inside(env.rabbits) == [True]
    env.rabbits[0].in_burrow = True
    env.step(stopped_actions(env))
    assert env.rabbits and env.hunt_stats["catches_blocked"] > 0
    np.testing.assert_allclose(env._observe(env.rabbits[0])[45:49], [10 / 267, 0.0, 10 / 267, 1.0], atol=1e-6)
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.grass import GrassPatch

    grass = env.food["grass"]
    grass.set_burrows([Vector(5.0, 400.0)], 24.0)
    near, far = GrassPatch(Vector(790, 400)), GrassPatch(Vector(770, 400))
    grass.mark(near)
    grass.mark(far)
    assert near.blocked and not far.blocked


def test_counters_without_burrows_and_extinction_time():
    env = world()
    place(env, {"predator_0": (400, 410)})
    env.step(stopped_actions(env))
    stats = env.hunt_stats
    assert stats["predator_steps"] == 4 * env.action_repeat and stats["rabbit_steps"] == 1
    assert stats["rabbit_steps_inside"] == 0 and env.rabbit_burrow_steps == {}
    assert env.extinct_at == {"rabbit": 1}


def cluster_world(seed=7, **config):
    env = StagHuntContinuous({"burrow_count": 20, "burrow_placement": "cluster", **config})
    env.reset(seed=seed)
    return env


def warren_distances(env):
    """Torus distance from each grass cluster's centre to its warren."""
    grass = env.food["grass"]
    return [grass.torus.distance(centre, Vector(*xy)) for centre, xy in zip(grass.centres, env.burrows)]


def test_cluster_warrens_beside_their_cluster_and_world_unchanged():
    plain, env = StagHuntContinuous(), cluster_world()
    plain.reset(seed=7)
    assert env.burrows.shape == (20, 2)
    assert warren_distances(env) == pytest.approx([24.0 + 24.0] * 20)
    assert [(a.id, a.position.x, a.position.y) for a in plain.living()] == [(a.id, a.position.x, a.position.y) for a in env.living()]
    grass = env.food["grass"]
    assert [(p.position.x, p.position.y) for p in plain.food["grass"].patches] == [(p.position.x, p.position.y) for p in grass.patches]
    assert plain.food["grass"].rng.getstate() == grass.rng.getstate()
    expected = [any(grass.torus.distance(p.position, Vector(x, y)) < 24.0 for x, y in env.burrows) for p in grass.patches]
    assert [p.blocked for p in grass.patches] == expected
    np.testing.assert_array_equal(cluster_world().burrows, env.burrows)
    assert not np.array_equal(cluster_world(seed=8).burrows, env.burrows)


def test_cluster_warren_settings_are_checked():
    with pytest.raises(ValueError, match="burrow_placement"):
        StagHuntContinuous({"burrow_count": 10, "burrow_placement": "grid"})
    with pytest.raises(ValueError, match="grass_cluster_count"):
        StagHuntContinuous({"burrow_count": 10, "burrow_placement": "cluster"})
    with pytest.raises(ValueError, match="clustered"):
        StagHuntContinuous({"burrow_count": 20, "burrow_placement": "cluster", "grass_clustered": False, "grass_dispersal": False, "grass_overgrazing": False})
    StagHuntContinuous({"burrow_placement": "cluster"})  # no burrows: nothing to place


def test_warren_moves_with_its_collapsed_cluster():
    env = cluster_world(rabbit_count=1)
    grass = env.food["grass"]
    rabbit = env.rabbits[0]
    rabbit.velocity = Vector(0, 0)
    rabbit.position = Vector(*env.burrows[3])
    env._burrow_step()
    assert rabbit.in_burrow
    before = env.burrows.copy()
    grass.collapse(3, [p for p in grass.patches if p.cluster == 3])
    exits = env.hunt_stats["burrow_exits"]
    env._burrow_step()
    assert env.hunt_stats["burrow_moves"] == 1
    assert warren_distances(env) == pytest.approx([48.0] * 20)
    np.testing.assert_array_equal(np.delete(env.burrows, 3, axis=0), np.delete(before, 3, axis=0))
    assert not np.array_equal(env.burrows[3], before[3])
    expected = [any(grass.torus.distance(p.position, Vector(x, y)) < 24.0 for x, y in env.burrows) for p in grass.patches]
    assert [p.blocked for p in grass.patches] == expected
    # A rabbit in the abandoned warren is in the open (unless the new one covers it).
    if not env._inside([rabbit])[0]:
        assert not rabbit.in_burrow and env.hunt_stats["burrow_exits"] == exits + 1
    env._burrow_step()
    assert env.hunt_stats["burrow_moves"] == 1


def test_rabbit_in_abandoned_warren_is_caught_in_that_step():
    env = world(burrow_count=20, burrow_placement="cluster", grass_overgrazing=False)  # empty clusters would collapse every step
    env.burrows = np.array([(50.0, 50.0 + 30 * i) for i in range(20)])
    env.burrows[3] = (400.0, 400.0)
    place(env, {"predator_0": (400, 410)})
    _, _, te, _, _ = env.step(stopped_actions(env))
    assert env.rabbits and env.hunt_stats["catches_blocked"] == env.action_repeat
    # The cluster collapses and moves far away: at the start of the next
    # physics step the warren follows it, the rabbit is in the open, an exit,
    # and the touching predator catches it in that same step.
    env.food["grass"].centres[3] = Vector(700.0, 300.0)
    _, _, te, _, _ = env.step(stopped_actions(env))
    stats = env.hunt_stats
    assert te["rabbit_0"] and not env.rabbits
    assert stats["burrow_moves"] == 1 and stats["burrow_exits"] == 1
    assert stats["catches_after_exit"] == stats["rabbits_caught"] == 1


def test_several_collapses_move_their_warrens_in_cluster_order():
    env = cluster_world()
    grass = env.food["grass"]
    for cluster in (5, 2):
        grass.collapse(cluster, [p for p in grass.patches if p.cluster == cluster])
    expected = env.burrows.copy()
    state = env.burrow_rng.getstate()
    for cluster in (2, 5):
        expected[cluster] = env._warren(grass.centres[cluster])
    env.burrow_rng.setstate(state)
    env._burrow_step()
    np.testing.assert_array_equal(env.burrows, expected)
    assert env.hunt_stats["burrow_moves"] == 2
    marked = [any(grass.torus.distance(p.position, Vector(x, y)) < 24.0 for x, y in env.burrows) for p in grass.patches]
    assert [p.blocked for p in grass.patches] == marked


def test_cluster_warrens_follow_the_cluster_count():
    env = cluster_world(grass_cluster_count=10, burrow_count=10)
    assert env.burrows.shape == (10, 2) and warren_distances(env) == pytest.approx([48.0] * 10)
    with pytest.raises(ValueError, match="grass_cluster_count"):
        StagHuntContinuous({"grass_cluster_count": 10, "burrow_count": 9, "burrow_placement": "cluster"})


def test_calibration_arms_build():
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.config.config_env import config_env
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.launchers.calibration import ARMS

    for settings in ARMS.values():
        StagHuntContinuous({**config_env, **settings})
    r10b, r12 = ARMS["with_mammoths_r10b"], ARMS["with_mammoths_r12"]
    assert {k: v for k, v in r12.items() if r10b.get(k) != v} == {"burrow_count": 20, "burrow_placement": "cluster"}


def rabbit_slots(env, predator):
    """The values of a predator's rabbit slots."""
    start = SELF_SIZE + SLOT_SIZE * sum(env.observe[("predator", other)] for other in OBSERVED[: OBSERVED.index("rabbit")])
    return env._observe(predator)[start : start + SLOT_SIZE * env.observe[("predator", "rabbit")]]


def test_hidden_rabbits_unseen_by_predators_and_still_safe():
    shown, hidden = burrow_world(), burrow_world(burrow_hides_rabbits=True)
    assert shown.obs_size["predator"] == 64 and hidden.obs_size["predator"] == 62
    assert hidden.obs_size["rabbit"] == shown.obs_size["rabbit"] == 49
    for env in (shown, hidden):
        place(env, {"predator_0": (400, 430)})
        env.predators[0].orientation = heading_degrees(Vector(0, -1))  # towards the rabbit
    assert rabbit_slots(shown, shown.predators[0])[0] == 1.0
    assert not rabbit_slots(hidden, hidden.predators[0]).any()
    obs = hidden._observations()["predator_0"]
    assert obs.shape == (62,) and hidden.observation_spaces["predator_0"].contains(obs)
    # Outside the burrow the rabbit is seen again.
    place(hidden, {"rabbit_0": (400, 370)})
    assert rabbit_slots(hidden, hidden.predators[0])[0] == 1.0
    # Hidden or not, a rabbit inside cannot be caught.
    place(hidden, {"rabbit_0": (400, 400), "predator_0": (400, 410)})
    hidden.step(stopped_actions(hidden))
    assert hidden.rabbits and hidden.hunt_stats["catches_blocked"] == hidden.action_repeat
    with pytest.raises(ValueError, match="burrow_hides_rabbits"):
        StagHuntContinuous({"burrow_hides_rabbits": 1})


def test_visible_rabbit_takes_the_slot_of_a_hidden_one():
    env = burrow_world(burrow_hides_rabbits=True, rabbit_count=2)
    place(env, {"predator_0": (400, 430), "rabbit_0": (400, 400), "rabbit_1": (400, 360)})
    predator = env.predators[0]
    predator.orientation = heading_degrees(Vector(0, -1))
    slots = rabbit_slots(env, predator)
    assert slots[0] == 1.0 and slots[3] == pytest.approx(70 / predator.view_distance)
    assert not slots[SLOT_SIZE:].any()


@pytest.mark.parametrize(
    "config, size",
    [
        ({"burrow_count": 10, "burrow_hides_rabbits": True, "hunt_condition_factor": 0.5}, 64),
        ({"burrow_count": 0, "burrow_hides_rabbits": True}, 58),
        ({"burrow_count": 10, "burrow_hides_rabbits": True, "observe_predator_rabbit": 0}, 50),
        ({"burrow_count": 10, "burrow_hides_rabbits": False, "observe_predator_rabbit": 3}, 71),
    ],
)
def test_hidden_rabbit_observation_sizes(config, size):
    env = StagHuntContinuous(config)
    obs, _ = env.reset(seed=3)
    assert env.obs_size["predator"] == size
    rng = np.random.default_rng(3)
    for _ in range(5):
        for agent, o in obs.items():
            assert env.observation_spaces[agent].contains(o)
        obs, _, te, tr, _ = env.step(random_actions(env, rng))


def test_hidden_check_arm_differs_only_in_hiding():
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.launchers.calibration import ARMS

    shown, hidden = ARMS["rabbits_only_warrens"], ARMS["rabbits_only_warrens_hidden"]
    assert {k: v for k, v in hidden.items() if shown.get(k) != v} == {"burrow_hides_rabbits": True}


def test_grass_near_burrow_counters():
    env = cluster_world()
    env._burrow_step()
    stats, grass = env.hunt_stats, env.food["grass"]
    edible = [p for p in grass.patches if grass.edible(p)]
    near = sum(min(grass.torus.distance(p.position, Vector(x, y)) for x, y in env.burrows) <= 24.0 + 12.0 for p in edible)
    assert stats["grass_edible_patch_steps"] == len(edible)
    assert stats["grass_near_burrow_patch_steps"] == near > 0
    assert derived_rates(stats)["grass_near_burrow_share"] == near / len(edible)


class FixedDraw:
    """A hunt_rng whose random() returns a given value."""

    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


@pytest.mark.parametrize("energy, draw, killed", [(24.0, 0.15, True), (24.0, 0.25, False), (12.0, 0.55, True), (12.0, 0.65, False), (48.0, 0.25, False), (0.0, 0.95, True)])
def test_well_fed_mammoths_are_harder_to_kill(energy, draw, killed):
    """Success probability 1 x (1 - 0.8 x min(1, energy / 24))."""
    env = world(hunt_success_by_party=(1.0, 1.0, 1.0, 1.0), hunt_condition_factor=0.8, energy_mammoth_max=48.0)
    env.mammoths[0].energy = energy
    env.hunt_rng = FixedDraw(draw)
    place(env, {"predator_0": (200, 130), "predator_1": (230, 100)})
    te, _, _ = hunt(env, {"predator_0": 1, "predator_1": 1, "predator_2": 0, "predator_3": 0})
    assert bool(te.get("mammoth_0")) == killed
    condition = min(1.0, energy / 24.0)
    assert env.hunt_stats["hunted_condition_total"] == pytest.approx(condition)
    assert env.hunt_stats["killed_condition_total"] == pytest.approx(condition if killed else 0.0)


def test_condition_observed_and_off_by_default():
    assert StagHuntContinuous().obs_size["predator"] == 58
    with pytest.raises(ValueError, match="hunt_condition_factor"):
        StagHuntContinuous({"hunt_condition_factor": 1.5})
    env = world(hunt_condition_factor=0.5)
    assert env.obs_size == {"predator": 60, "rabbit": 45, "mammoth": 39}
    me = env.predators[0]
    place(env, {"predator_0": (400, 400), "mammoth_0": (400, 300), "mammoth_1": (400, 600)})
    env.mammoths[0].energy = 6.0
    me.orientation = heading_degrees(Vector(0, -1))  # sees mammoth_0 only
    obs = env._observe(me)
    np.testing.assert_allclose(obs[58:60], [6.0 / 24.0, 0.0])
    # With burrows too: condition first, then the burrow features.
    both = world(hunt_condition_factor=0.5, burrow_count=1)
    assert both.obs_size["predator"] == 58 + 2 + 4 + 2


def test_condition_factor_zero_keeps_the_hunts():
    a, b = StagHuntContinuous(), StagHuntContinuous({"hunt_condition_factor": 0.0})
    a.reset(seed=4)
    b.reset(seed=4)
    ra, rb = np.random.default_rng(4), np.random.default_rng(4)
    for _ in range(40):
        sa, sb = a.step(random_actions(a, ra)), b.step(random_actions(b, rb))
        assert sa[1] == sb[1]
        if sa[2]["__all__"] or sa[3]["__all__"]:
            break
    assert a.hunt_stats == b.hunt_stats


class CountingDraw:
    def __init__(self, value):
        self.value, self.calls = value, 0

    def random(self):
        self.calls += 1
        return self.value


@pytest.mark.parametrize("draw, killed", [(0.29, True), (0.31, False)])
def test_condition_scales_the_party_size_chance(draw, killed):
    """Pair chance 0.4, k = 0.5, condition 0.5: p = 0.3, one draw."""
    env = world(hunt_success_by_party=(0.0, 0.4, 1.0, 1.0), hunt_condition_factor=0.5)
    env.mammoths[0].energy = 12.0
    env.hunt_rng = CountingDraw(draw)
    place(env, {"predator_0": (200, 130), "predator_1": (230, 100)})
    te, _, _ = hunt(env, {"predator_0": 1, "predator_1": 1, "predator_2": 0, "predator_3": 0})
    assert bool(te.get("mammoth_0")) == killed and env.hunt_rng.calls == 1


@pytest.mark.parametrize("draw, killed", [(0.39, True), (0.41, False)])
def test_without_condition_the_old_chance_and_one_draw(draw, killed):
    env = world(hunt_success_by_party=(0.0, 0.4, 1.0, 1.0))
    env.mammoths[0].energy = 24.0
    env.hunt_rng = CountingDraw(draw)
    place(env, {"predator_0": (200, 130), "predator_1": (230, 100)})
    te, _, _ = hunt(env, {"predator_0": 1, "predator_1": 1, "predator_2": 0, "predator_3": 0})
    assert bool(te.get("mammoth_0")) == killed and env.hunt_rng.calls == 1


def test_condition_means():
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.callbacks import derived_rates

    env = world(hunt_success_by_party=(1.0, 1.0, 1.0, 1.0), hunt_condition_factor=1.0)
    env.mammoths[0].energy, env.mammoths[1].energy = 24.0, 6.0  # condition 1 (never killed), 0.25
    place(env, {"predator_0": (200, 130), "predator_1": (600, 130)})
    env.hunt_rng = FixedDraw(0.5)
    hunt(env, {"predator_0": 1, "predator_1": 1, "predator_2": 0, "predator_3": 0})
    rates = derived_rates(dict(env.hunt_stats))
    assert rates["hunted_condition_mean"] == pytest.approx((1.0 + 0.25) / 2)
    assert rates["killed_condition_mean"] == pytest.approx(0.25)
    assert "killed_condition_mean" not in derived_rates(dict(env.hunt_stats, kills_party_1=0))


@pytest.mark.parametrize("energy, ratio, killed", [(9.9, 1.0, True), (10.0, 1.0, False), (6.6, 1.5, True), (6.7, 1.5, False)])
def test_energy_rule_summed_energy_against_the_mammoths(energy, ratio, killed):
    """Pair with 4 + 6 = 10 energy (before the join cost), no random draw."""
    env = world(hunt_rule="energy", hunt_energy_ratio=ratio)
    env.mammoths[0].energy = energy
    env.predators[0].energy, env.predators[1].energy = 4.0, 6.0
    env.hunt_rng = CountingDraw(0.0)
    place(env, {"predator_0": (200, 130), "predator_1": (230, 100)})
    te, _, _ = hunt(env, {"predator_0": 1, "predator_1": 1, "predator_2": 0, "predator_3": 0})
    assert bool(te.get("mammoth_0")) == killed and env.hunt_rng.calls == 0
    stats = env.hunt_stats
    assert stats["hunted_party_energy_total"] == pytest.approx(10.0) and stats["hunted_mammoth_energy_total"] == pytest.approx(energy)
    assert stats["kills_party_2"] == killed
    rates = derived_rates(stats)
    assert rates["hunted_party_energy_mean"] == pytest.approx(10.0) and rates["hunted_mammoth_energy_mean"] == pytest.approx(energy)


def test_energy_rule_lets_a_well_fed_hunter_kill_alone():
    env = world(hunt_rule="energy")
    env.mammoths[0].energy, env.predators[0].energy = 8.0, 12.0
    place(env, {"predator_0": (200, 130)})
    te, _, _ = hunt(env, {"predator_0": 1, "predator_1": 0, "predator_2": 0, "predator_3": 0})
    assert te["mammoth_0"] and env.hunt_stats["kills_party_1"] == 1


def test_energy_rule_observations():
    env = world(hunt_rule="energy")
    assert env.obs_size["predator"] == 58 + 2 + 4 and env.obs_size["rabbit"] == 45 and env.obs_size["mammoth"] == 39
    predator = env.predators[0]
    place(env, {"predator_0": (200, 130), "predator_1": (200, 160), "mammoth_1": (600, 600)})
    predator.orientation = heading_degrees(Vector(0, -1))  # towards mammoth_0, away from predator_1
    env.mammoths[0].energy = 20.0
    obs = env._observe(predator)
    assert obs.shape == (64,) and env.observation_spaces[predator.id].contains(obs)
    # Tail: energy, cooldown, mammoth energies (2), predator energies (4).
    mammoths, predators = obs[-6:-4], obs[-4:]
    assert mammoths[0] == pytest.approx(20.0 / 40.0) and mammoths[1] == 0.0
    seen = []
    env._slots(predator, env.predators, 4, seen)
    assert list(predators[: len(seen)]) == pytest.approx([p.energy / 24.0 for p in seen])
    assert not predators[len(seen) :].any()
    # The chance rule keeps its observation.
    assert world().obs_size["predator"] == 58


def test_energy_rule_settings_are_checked():
    with pytest.raises(ValueError, match="hunt_rule"):
        StagHuntContinuous({"hunt_rule": "summed"})
    with pytest.raises(ValueError, match="hunt_energy_ratio"):
        StagHuntContinuous({"hunt_rule": "energy", "hunt_energy_ratio": 0})
    with pytest.raises(ValueError, match="hunt_condition_factor"):
        StagHuntContinuous({"hunt_rule": "energy", "hunt_condition_factor": 0.5})


def test_energy_rule_episode_runs():
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.config.config_env import config_env
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.launchers.calibration import ARMS

    env = StagHuntContinuous({**config_env, **ARMS["with_mammoths_r13a"]})
    obs, _ = env.reset(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(40):
        for agent, o in obs.items():
            assert env.observation_spaces[agent].contains(o)
        obs, _, te, tr, _ = env.step(random_actions(env, rng))
        if te["__all__"] or tr["__all__"]:
            break


@pytest.mark.parametrize(
    "config, size",
    [
        ({"burrow_count": 10}, 64 + 6),
        ({"burrow_count": 10, "burrow_hides_rabbits": True}, 62 + 6),
        ({"burrow_count": 10, "burrow_hides_rabbits": True, "observe_predator_rabbit": 0}, 50 + 6),
        ({"observe_predator_predator": 2, "observe_predator_mammoth": 3}, 58 - 12 + 6 + 3 + 2),
    ],
)
def test_energy_rule_observation_layout_with_other_settings(config, size):
    env = world(hunt_rule="energy", **config)
    assert env.obs_size["predator"] == size
    predator = env.predators[0]
    obs = env._observe(predator)
    assert obs.shape == (size,)
    # The energies follow own energy and cooldown in the tail.
    mammoth_slots, predator_slots = env.observe[("predator", "mammoth")], env.observe[("predator", "predator")]
    start = size - (BURROW_SIZE + (0 if env.burrow_hides else env.observe[("predator", "rabbit")]) if env.burrow_count else 0) - mammoth_slots - predator_slots
    assert obs[start - 2] == pytest.approx(predator.energy / 24.0)
    seen = {"mammoth": [], "predator": []}
    for other in seen:
        env._slots(predator, env.animals[other], env.observe[("predator", other)], seen[other])
    expected = [m.energy / 40.0 for m in seen["mammoth"]] + [0.0] * (mammoth_slots - len(seen["mammoth"]))
    expected += [p.energy / 24.0 for p in seen["predator"]] + [0.0] * (predator_slots - len(seen["predator"]))
    assert list(obs[start : start + mammoth_slots + predator_slots]) == pytest.approx(expected)


def test_energy_rule_compares_before_the_join_cost():
    """4 + 6 = 10 > 9.5 before a join cost of 1 each (8 after): a kill."""
    env = world(hunt_rule="energy", hunt_join_cost=1.0)
    env.mammoths[0].energy = 9.5
    env.predators[0].energy, env.predators[1].energy = 4.0, 6.0
    place(env, {"predator_0": (200, 130), "predator_1": (230, 100)})
    te, _, _ = hunt(env, {"predator_0": 1, "predator_1": 1, "predator_2": 0, "predator_3": 0})
    assert te["mammoth_0"]


def test_energy_observations_at_the_maxima_stay_in_range():
    env = world(hunt_rule="energy")
    for p in env.predators:
        p.energy = 24.0
    for m in env.mammoths:
        m.energy = 40.0
    place(env, {"predator_0": (200, 130), "predator_1": (200, 110), "mammoth_1": (600, 600)})
    env.predators[0].orientation = heading_degrees(Vector(0, -1))
    obs = env._observe(env.predators[0])
    assert obs[-6] == 1.0 and obs[-4] == 1.0 and env.observation_spaces["predator_0"].contains(obs)
