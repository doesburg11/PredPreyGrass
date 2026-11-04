"""Tests of the hunters' energy accounts."""

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import MOVES, PARTY_SIZES, StagHuntRulePrey
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.payoffs import (
    GROUPS,
    add_life,
    approach_start,
    new_row,
    play_episode,
    rate,
    split_bouts,
    summarise,
)


def random_policy(seed):
    rng = np.random.default_rng(seed)

    def choose(obs):
        return {a: np.array([rng.integers(MOVES), rng.integers(2)]) for a in obs}

    return choose


def small_env(**settings):
    return StagHuntRulePrey(dict(dict(max_time_steps=800, hunt_min_party=1), **settings))


def record(near=False, touching=False, party=0, distance=500.0, **values):
    base = {"physics_steps": 8, "rabbit": 0.0, "metabolic": 0.1, "hunt_net": 0.0, "scavenged": 0.0, "kill": 0, "group_kill": 0}
    return dict(base, near=near, touching=touching, party=party, distance=distance, **values)


def test_split_bouts_merges_short_gaps():
    near = [False, True, True, False, False, True, False, False, False, True]
    assert split_bouts(near, gap=2) == [(1, 5), (9, 9)]
    assert split_bouts(near, gap=1) == [(1, 2), (5, 5), (9, 9)]
    assert split_bouts([False, False]) == []


def test_approach_stops_when_the_distance_stops_falling():
    assert approach_start([5.0, 9.0, 7.0, 4.0, 2.0], 4) == 1
    assert approach_start([1.0, 2.0], 1) == 1
    assert approach_start([9.0, 7.0, 4.0], 2, earliest=1) == 1


def test_groups_follow_what_the_hunter_did():
    far = [record(), record(), record()]
    row = new_row()
    add_life(
        row,
        [
            record(distance=300.0, rabbit=2.0),
            record(distance=200.0),
            record(distance=90.0, near=True),
            record(distance=10.0, near=True, touching=True),
            *far,
            record(distance=10.0, near=True, touching=True, party=1, hunt_net=-0.66),
            *far,
            record(distance=10.0, near=True, touching=True, party=2, hunt_net=19.0, kill=1, group_kill=1),
            *far,
            record(distance=90.0, near=True, scavenged=3.0),
        ],
    )
    groups = row["groups"]
    assert groups["touched_no_attempt"]["bouts"] == 1
    assert groups["touched_no_attempt"]["approach_steps"] == 16
    assert groups["touched_no_attempt"]["approach_rabbit"] == 2.0
    assert groups["attempted_alone"]["hunt_net"] == pytest.approx(-0.66)
    assert groups["attempted_alone"]["approach_steps"] == 8
    assert groups["attempted_in_party"]["group_kill_credits"] == 1
    assert groups["attempted_in_party"]["approach_steps"] == 8
    assert groups["near"]["physics_steps"] == 8
    assert groups["near"]["bouts"] == 0
    assert groups["near"]["scavenged"] == 3.0
    assert groups["elsewhere"]["physics_steps"] == 8 * 7
    assert sum(groups[g]["physics_steps"] + groups[g]["approach_steps"] for g in GROUPS) == 8 * 16


def test_a_bout_takes_its_group_from_its_largest_party():
    row = new_row()
    add_life(row, [record(near=True, touching=True, party=1), record(near=True, touching=True, party=2)])
    assert row["groups"]["attempted_in_party"]["bouts"] == 1
    assert row["groups"]["attempted_alone"]["bouts"] == 0


def test_rate_counts_the_approach_only_when_asked():
    totals = dict.fromkeys(("physics_steps", "approach_steps", "rabbit", "hunt_net", "metabolic", "approach_rabbit", "approach_metabolic"), 0.0)
    totals.update(physics_steps=10, approach_steps=10, hunt_net=20.0, metabolic=2.0, approach_metabolic=2.0)
    assert rate(totals) == pytest.approx(800.0)
    assert rate(totals, approach=False) == pytest.approx(1800.0)
    assert rate(dict.fromkeys(totals, 0.0)) is None


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_accounts_explain_the_hunters_energy(seed):
    assert play_episode(small_env(), random_policy(seed), seed)["unexplained_energy"] == pytest.approx(0.0, abs=1e-6)


def test_accounts_match_the_environments_counters():
    env = small_env(max_time_steps=1600)
    row = play_episode(env, random_policy(3), 3)
    stats = env.hunt_stats
    for size in range(1, PARTY_SIZES + 1):
        assert row["attempts"][str(size)]["hunts"] == stats[f"hunts_party_{size}"]
        assert row["attempts"][str(size)]["kills"] == stats[f"kills_party_{size}"]
    kills = sum(row["attempts"][str(size)]["kills"] for size in range(1, PARTY_SIZES + 1))
    assert sum(row["attempts"][str(size)]["hunts"] for size in range(1, PARTY_SIZES + 1)) > 0
    assert sum(row["groups"][g]["kill_credits"] for g in GROUPS) >= kills or kills == 0
    rabbit = sum(row["groups"][g]["rabbit"] + row["groups"][g]["approach_rabbit"] for g in GROUPS)
    assert rabbit <= stats["rabbit_energy"] + 1e-6
    assert sum(row["groups"][g]["scavenged"] for g in GROUPS) == pytest.approx(stats["scavenger_energy"], abs=1e-6)
    net = sum(row["attempts"][str(size)]["contributor_net"] for size in range(1, PARTY_SIZES + 1))
    assert sum(row["groups"][g]["hunt_net"] for g in GROUPS) == pytest.approx(net, abs=1e-6)


def test_contacts_count_only_hunters_ready_to_hunt():
    env = small_env(hunt_energy_ratio=1.0, max_time_steps=1600)
    row = play_episode(env, random_policy(5), 5)
    stats = env.hunt_stats
    assert sum(row["contacts"][key]["decisions"] for key in row["contacts"]) == stats["contact_ready"]
    assert sum(row["contacts"][key]["joins"] for key in row["contacts"]) == stats["contact_ready_joins"]


def test_the_environment_is_left_as_it_was():
    env = small_env()
    play_episode(env, random_policy(0), 0)
    for name in ("_gain", "_physics_step", "_hunts"):
        assert name not in vars(env)


def test_energy_rule_attempts_with_enough_energy_all_kill():
    env = small_env(hunt_energy_ratio=1.0, max_time_steps=1600)
    row = play_episode(env, random_policy(4), 4)
    for size in range(1, PARTY_SIZES + 1):
        counters = row["attempts"][str(size)]
        assert counters["kills"] == counters["enough_energy"]


def test_minimum_party_stops_solo_kills_and_counts_as_not_enough():
    env = small_env(hunt_energy_ratio=0.1, hunt_min_party=2, max_time_steps=1600)
    row = play_episode(env, random_policy(4), 4)
    assert row["attempts"]["1"]["hunts"] > 0
    assert row["attempts"]["1"]["kills"] == row["attempts"]["1"]["enough_energy"] == 0
    assert row["contacts"]["alone"]["enough"] == 0
    assert row["unexplained_energy"] == pytest.approx(0.0, abs=1e-6)


def test_summary_reports_each_group():
    rows = [play_episode(small_env(), random_policy(seed), seed) for seed in range(2)]
    summary = summarise(rows)
    assert summary["episodes"] == 2
    assert set(summary["groups"]) == set(GROUPS)
    low, high = summary["hunt_net_interval"]
    assert low <= high
    assert sum(g["time_share"] for g in summary["groups"].values()) == pytest.approx(1.0)
