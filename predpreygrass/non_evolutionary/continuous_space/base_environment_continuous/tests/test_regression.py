"""Exact equivalence with add_walls without add_walls installed.

Replays the reference trajectories' random actions and compares a digest
(SHA-256, first 64 bits) of every reset and decision (actions,
observations, rewards, flags, infos and the full state, bit for bit) with
the digests of add_walls's results, stored in data/reference_digests.json
by make_fixture.py. test_equivalence.py does the same against add_walls
itself and shows what differs; this test only tells at which decision the
first difference is.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent

from predpreygrass.non_evolutionary.continuous_space.base_environment_continuous.tests.compare import (
    CASES,
    count_events,
    digest,
    random_actions,
    snapshot,
)
from predpreygrass.non_evolutionary.continuous_space.base_environment_continuous.environment import PredPreyGrassContinuous

FIXTURE = json.loads((HERE / "data" / "reference_digests.json").read_text())


def test_fixture_covers_every_case():
    """The stored digests are those of compare.CASES, in full: a shortened
    or missing case would silently test less."""
    assert sorted(FIXTURE) == sorted(CASES)
    for name, (episodes, decisions, first_seed, overrides, _) in CASES.items():
        case = FIXTURE[name]
        assert case["settings"] == overrides
        assert [e["seed"] for e in case["episodes"]] == list(range(first_seed, first_seed + episodes))
        assert all(1 <= len(e["steps"]) <= decisions for e in case["episodes"])


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_add_walls_digests(name):
    case = FIXTURE[name]
    env = PredPreyGrassContinuous(case["settings"])
    events = Counter()
    for episode in case["episodes"]:
        seed = episode["seed"]
        rng = np.random.default_rng(seed)
        obs, infos = env.reset(seed=seed)
        assert digest((obs, infos, snapshot(env))) == episode["reset"], (name, seed)
        actions = random_actions(rng, obs)
        for i, expected in enumerate(episode["steps"]):
            sent = actions
            population = {"predator": len(env.predators), "prey": len(env.prey)}
            obs, rewards, terms, truncs, infos = env.step(actions)
            result = (sent, obs, rewards, terms, truncs, infos, snapshot(env))
            assert digest(result) == expected, (name, seed, i)
            count_events(env, population, terms, truncs, infos, events)
            actions = random_actions(rng, obs, terms, truncs)
        assert terms["__all__"] or truncs["__all__"] or i + 1 == CASES[name][1]
        events["collapses"] += env.grass.total_collapses
    # Identical digests mean add_walls had the same events.
    for event in CASES[name][4]:
        assert events[event], (event, events)
