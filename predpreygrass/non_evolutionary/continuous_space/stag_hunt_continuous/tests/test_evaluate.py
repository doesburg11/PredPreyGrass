"""Tests of the evaluation's joint-arrival count and chance level."""

import random

import numpy as np

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import MOVES, StagHuntContinuous, species_of
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.evaluate import joint_arrivals, outcome, shifted


def test_joint_arrivals_on_tracks():
    # One mammoth standing at (200, 100); two predators arrive 2 decisions
    # apart (joint), leave, then one arrives alone (touched, not joint).
    mammoth = {"mammoth_0": {d: (200, 100) for d in range(10)}}
    far = (200, 600)
    a = {0: (200, 130), 1: (200, 130), 2: (200, 130), 3: far, 4: far, 5: (200, 130), 6: far}
    b = {0: far, 1: far, 2: (230, 100), 3: far, 4: far, 5: far, 6: far}
    result = joint_arrivals({"predator_0": a, "predator_1": b}, mammoth, list(range(7)), 800, 800, 48, 96)
    assert result == (2, 2, 1)


def test_shift_keeps_tracks_rigid():
    tracks = {"predator_0": {0: (10.0, 20.0), 1: (790.0, 30.0)}}
    moved = shifted(tracks, random.Random(1), 800, 800)["predator_0"]
    step = [(moved[1][i] - moved[0][i]) % 800 for i in range(2)]
    assert np.allclose(step, [780.0, 10.0])


def test_tracked_count_matches_the_environment():
    env = StagHuntContinuous({"predator_count": 12, "mammoth_count": 6})
    predators, mammoths, decisions = {}, {}, []
    record = env._record

    def recording(joins):
        decisions.append(env.decision)
        for a in env.predators:
            predators.setdefault(a.id, {})[env.decision] = (a.position.x, a.position.y)
        for a in env.mammoths:
            mammoths.setdefault(a.id, {})[env.decision] = (a.position.x, a.position.y)
        record(joins)

    env._record = recording
    env.reset(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(150):
        actions = {a: (np.array([rng.integers(MOVES), rng.integers(2)]) if species_of(a) == "predator" else int(rng.integers(MOVES))) for a in env.active}
        _, _, te, tr, _ = env.step(actions)
        if te["__all__"] or tr["__all__"]:
            break
    else:
        for mammoth_id in list(env.encounters):
            env._close_encounter(mammoth_id)
    stats = env.hunt_stats
    assert stats["encounter_episodes"] > 0
    assert joint_arrivals(predators, mammoths, decisions, 800, 800, 48, 96) == (
        stats["encounter_episodes"],
        stats["touched_episodes"],
        stats["joint_arrivals"],
    )


def test_outcome():
    env = StagHuntContinuous()
    env.reset(seed=0)
    assert outcome(env) == "coexisting"
    env.animals["mammoth"].clear()
    assert outcome(env) == "only_predator_rabbit"


def test_tracked_count_with_births_kills_and_an_ending():
    """Births of both species, successful hunts and an episode that ends in
    a physics step: the tracks still give the environment's counts."""
    env = StagHuntContinuous(
        {
            "predator_count": 12,
            "mammoth_count": 8,
            "hunt_success_by_party": (1.0, 1.0, 1.0, 1.0),
            "energy_predator_initial": 11.5,
            "energy_mammoth_initial": 23.5,
            "max_time_steps": 1600,
        }
    )
    predators, mammoths, decisions = {}, {}, []
    record = env._record

    def recording(joins):
        decisions.append(env.decision)
        for a in env.predators:
            predators.setdefault(a.id, {})[env.decision] = (a.position.x, a.position.y)
        for a in env.mammoths:
            mammoths.setdefault(a.id, {})[env.decision] = (a.position.x, a.position.y)
        record(joins)

    env._record = recording
    env.reset(seed=7)
    rng = np.random.default_rng(7)
    while True:
        actions = {a: (np.array([rng.integers(MOVES), 1]) if species_of(a) == "predator" else int(rng.integers(MOVES))) for a in env.active}
        _, _, te, tr, _ = env.step(actions)
        if te["__all__"] or tr["__all__"]:
            break
    stats = env.episode_hunt_stats
    assert stats["births_predator"] > 0 and stats["births_mammoth"] > 0
    assert sum(stats[f"kills_party_{n}"] for n in range(1, 5)) > 0
    assert joint_arrivals(predators, mammoths, decisions, 800, 800, 48, 96) == (
        stats["encounter_episodes"],
        stats["touched_episodes"],
        stats["joint_arrivals"],
    )
