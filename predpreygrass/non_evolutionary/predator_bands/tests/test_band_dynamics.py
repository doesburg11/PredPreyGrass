"""Tests for band fission, fusion and drift-out (all default off): the split geometry, id handling, children, fusion hysteresis, drift-out
timing, integration with step(), snapshots and validation."""
import copy

import numpy as np
import pytest

from predpreygrass.non_evolutionary.predator_bands.config_env import config_env as _base
from predpreygrass.non_evolutionary.predator_bands.predpreygrass_rllib_env import PredPreyGrass


def _env(**kw):
    c = copy.deepcopy(_base)
    c.update({"max_steps": 200, "diet_required": False})
    c.update(kw)
    env = PredPreyGrass(c)
    env.reset(seed=1)
    return env


def _members(env, band):
    return sorted(a for a, b in env.agent_band.items() if b == band and a in env.predator_positions)


def _place(env, agent, pos):
    env.grid_world_state[1, *env.agent_positions[agent]] = 0
    env.agent_positions[agent] = pos
    env.predator_positions[agent] = pos
    env.grid_world_state[1, *pos] = env.agent_energies.get(agent, 0.0)


def _park(env, keep, start_row=24):
    i = 0
    for a in list(env.predator_positions):
        if a not in keep:
            _place(env, a, (start_row - i // 8, 16 + i % 8))
            i += 1


def _roles(env, band=0):
    band0 = _members(env, band)
    mother = next(a for a in band0 if "female" in a and a in env.agent_mate)
    father = env.agent_mate[mother]
    kids = [a for a in band0 if env.agent_parents.get(a) == (father, mother)]
    singles = [a for a in band0 if a not in kids and a not in (mother, father)]
    return mother, father, kids, singles


def _cluster(env, members, cells):
    for a, c in zip(members, cells):
        _place(env, a, c)


LEFT = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)]
RIGHT = [(20, 20), (20, 21), (21, 20), (21, 21), (20, 22), (22, 20)]


# ---- defaults, validation -------------------------------------------------------------------------------------------------
def test_everything_is_off_by_default_and_bands_never_change_on_their_own():
    env = _env()
    assert not (env.band_fission or env.band_fusion) and env.band_drift_steps == 0
    before = dict(env.agent_band)
    for _ in range(25):
        env.step({a: env.noop_action_id for a in env.agents})
        if len(env.predator_positions) < len(before):
            break
    assert all(env.agent_band.get(a) == b for a, b in before.items() if a in env.agent_band)
    assert env.band_fissions == env.band_fusions == env.band_drift_outs == 0


@pytest.mark.parametrize("kw", [
    {"band_fission": True, "num_bands": 0, "n_initial_active_predator_male": 2, "n_initial_active_predator_female": 2, "scripted_prey": False},
    {"band_fission": True, "band_max_size": 1},
    {"band_fission": True, "band_check_interval": 0},
    {"band_fission": True, "band_max_size": 4, "band_min_split_size": 4},
    {"band_fusion": True, "band_fuse_steps": -1},
    {"band_drift_steps": -5, "band_fusion": True},
])
def test_invalid_dynamics_config_is_rejected(kw):
    with pytest.raises(ValueError):
        _env(**kw)


# ---- fission ------------------------------------------------------------------------------------------------------------------
def test_a_band_above_the_size_cap_splits_into_its_two_clusters_and_the_stronger_keeps_the_id():
    env = _env(num_bands=1, band_fission=True, band_max_size=5, band_min_split_size=2)  # one band: only it can split
    mother, father, kids, singles = _roles(env)
    members = _members(env, 0)
    assert len(members) == 6
    _park(env, set(members))
    left, right = [mother, father, singles[0]], [kids[0], kids[1], singles[1]]
    _cluster(env, left, LEFT)
    _cluster(env, right, RIGHT)
    for a in members:
        env.agent_energies[a] = 5.0
    for a in right:
        env.agent_energies[a] = 9.0  # the right cluster is stronger: it keeps id 0
    env._update_bands()
    assert env.band_fissions == 1 and env._next_band_id == env.num_bands + 1
    right_ids = {env.agent_band[a] for a in right if a not in kids}
    assert right_ids == {0}
    # the children follow their (left) mother, not their position
    assert env.agent_band[kids[0]] == env.agent_band[mother] == env.num_bands
    assert env.agent_band[father] == env.num_bands and env.agent_band[singles[0]] == env.num_bands


def test_no_split_at_the_cap_or_when_a_part_would_be_too_small_and_ids_are_never_reused():
    env = _env(num_bands=1, band_fission=True, band_max_size=6, band_min_split_size=2)
    env._update_bands()
    assert env.band_fissions == 0  # size 6 is not above the cap of 6
    env = _env(num_bands=1, band_fission=True, band_max_size=5, band_min_split_size=2)
    mother, father, kids, singles = _roles(env)
    members = _members(env, 0)
    _park(env, set(members))
    _cluster(env, members[:5], LEFT[:5])
    _place(env, members[5], (22, 22))  # one outlier: a part of 1 < min split size
    env._update_bands()
    assert env.band_fissions == 0
    env = _env(num_bands=2, band_fission=True, band_max_size=5, band_min_split_size=2)
    for band in (0, 1):
        m = _members(env, band)
        _park(env, set(m), start_row=24 - 3 * band)
        _cluster(env, m[:3], LEFT[:3] if band == 0 else [(10, 2), (10, 3), (11, 2)])
        _cluster(env, m[3:], RIGHT[:3] if band == 0 else [(10, 20), (10, 21), (11, 20)])
    env._update_bands()
    assert env.band_fissions == 2 and env._next_band_id == env.num_bands + 2
    assert sorted({v for v in env.agent_band.values()}) == list(range(env.num_bands + 2))  # ids 0..3, none reused


# ---- fusion -------------------------------------------------------------------------------------------------------------------
def _small_bands_env(**kw):
    cfg = dict(num_bands=2, band_couples=1, band_children_per_couple=0, band_singles_male=1, band_singles_female=1, band_fusion=True,
               band_max_size=12, band_fuse_distance=3, band_fuse_steps=30, band_check_interval=10)
    cfg.update(kw)
    return _env(**cfg)


def test_two_close_small_bands_fuse_after_enough_consecutive_checks_and_the_lower_id_survives():
    env = _small_bands_env()
    a, b = _members(env, 0), _members(env, 1)
    _park(env, set(a + b))
    _cluster(env, a, [(5, 5), (5, 6), (6, 5), (6, 6)])
    _cluster(env, b, [(7, 5), (7, 6), (8, 5), (8, 6)])
    env._update_bands()
    env._update_bands()
    assert env.band_fusions == 0  # 20 of 30 steps in contact
    env._update_bands()
    assert env.band_fusions == 1 and {env.agent_band[x] for x in a + b} == {0}


def test_fusion_resets_when_contact_breaks_and_a_union_above_the_cap_does_not_fuse():
    env = _small_bands_env()
    a, b = _members(env, 0), _members(env, 1)
    _park(env, set(a + b))
    _cluster(env, a, [(5, 5), (5, 6), (6, 5), (6, 6)])
    _cluster(env, b, [(7, 5), (7, 6), (8, 5), (8, 6)])
    env._update_bands()
    env._update_bands()
    _cluster(env, b, [(20, 5), (20, 6), (21, 5), (21, 6)])  # contact breaks
    env._update_bands()
    _cluster(env, b, [(7, 5), (7, 6), (8, 5), (8, 6)])
    env._update_bands()
    env._update_bands()
    assert env.band_fusions == 0  # the counter restarted
    env2 = _small_bands_env(band_max_size=6)  # union of 8 > 0.75 * 6
    a, b = _members(env2, 0), _members(env2, 1)
    _park(env2, set(a + b))
    _cluster(env2, a, [(5, 5), (5, 6), (6, 5), (6, 6)])
    _cluster(env2, b, [(7, 5), (7, 6), (8, 5), (8, 6)])
    for _ in range(5):
        env2._update_bands()
    assert env2.band_fusions == 0


# ---- drift-out ----------------------------------------------------------------------------------------------------------------
def test_a_member_out_of_range_for_long_enough_leaves_after_the_configured_steps():
    env = _env(band_drift_steps=50, band_check_interval=10)
    mother, father, kids, singles = _roles(env)
    members = _members(env, 0)
    _park(env, set(members))
    _cluster(env, [father, singles[0], singles[1], kids[0], kids[1]], [(5, 5), (5, 6), (6, 5), (6, 6), (5, 7)])
    _place(env, mother, (20, 20))  # far from everyone in her band
    for i in range(4):
        env._update_bands()
        assert env.agent_band[mother] == 0  # 10, 20, 30, 40 steps out
    env._update_bands()
    new_id = env.agent_band[mother]
    assert new_id == env.num_bands and env.band_drift_outs == 1
    assert env.agent_band[kids[0]] == 0 and env.agent_band[kids[1]] == 0  # her children, in range of the band, stay
    assert env.agent_band[father] == 0


def test_coming_back_into_range_resets_the_drift_counter():
    env = _env(band_drift_steps=50, band_check_interval=10)
    mother, father, kids, singles = _roles(env)
    members = _members(env, 0)
    _park(env, set(members))
    _cluster(env, [father, singles[0], singles[1], kids[0], kids[1]], [(5, 5), (5, 6), (6, 5), (6, 6), (5, 7)])
    _place(env, mother, (20, 20))
    for _ in range(4):
        env._update_bands()
    _place(env, mother, (7, 7))  # back within 5 cells of a band-mate
    env._update_bands()
    _place(env, mother, (20, 20))
    for _ in range(4):
        env._update_bands()
    assert env.agent_band[mother] == 0 and env.band_drift_outs == 0


# ---- integration, snapshot, metrics --------------------------------------------------------------------------------------------
def test_step_runs_the_update_every_check_interval():
    env = _env(num_bands=1, band_fission=True, band_max_size=5, band_min_split_size=2, band_check_interval=5, diet_required=False)
    mother, father, kids, singles = _roles(env)
    members = _members(env, 0)
    _park(env, set(members))
    _cluster(env, members[:3], LEFT[:3])
    _cluster(env, members[3:], RIGHT[:3])
    for a in env.predator_positions:
        env.agent_energies[a] = 12.0
    for _ in range(4):
        env.step({a: env.noop_action_id for a in env.agents})
        assert env.band_fissions == 0
    env.step({a: env.noop_action_id for a in env.agents})  # the 5th step triggers the check
    assert env.band_fissions == 1


def test_snapshot_round_trip_and_metrics():
    env = _env(band_drift_steps=50, band_check_interval=10, band_fusion=True)
    snap = env.get_state_snapshot()
    env._next_band_id = 9
    env.band_fissions = 3
    env._band_out_steps = {"predator_male_0": 20}
    env.restore_state_snapshot(snap)
    assert env._next_band_id == env.num_bands and env.band_fissions == 0 and env._band_out_steps == {}
    m = env._build_episode_training_metrics()
    for k in ("band_fissions", "band_fusions", "band_drift_outs", "mean_band_size", "bands_alive"):
        assert k in m
    assert m["mean_band_size"] == pytest.approx(6.0)


# ---- fixes from the Codex review of the band dynamics ---------------------------------------------------------------------------
def test_the_drift_counter_restarts_when_the_band_changes_and_a_lone_member_does_not_accumulate():
    env = _env(band_drift_steps=50, band_check_interval=10)
    mother, father, kids, singles = _roles(env)
    members = _members(env, 0)
    _park(env, set(members))
    _cluster(env, [father, singles[0], singles[1], kids[0], kids[1]], [(5, 5), (5, 6), (6, 5), (6, 6), (5, 7)])
    _place(env, mother, (2, 20))  # far from everyone, including the parked members of the other bands
    env._band_out_steps[mother] = (0, 40)
    env.agent_band[mother] = 1  # e.g. a marriage moved her: the old counter must not count in the new band
    env._update_bands()
    assert env._band_out_steps[mother] == (1, 10)  # restarted, not 50
    # a lone member (no band-mate to be near) never accumulates and never drifts when join is off
    lone_env = _env(band_drift_steps=20, band_check_interval=10)
    lone = _members(lone_env, 0)[0]
    for x in _members(lone_env, 0)[1:]:
        lone_env.agent_band[x] = 1
    for _ in range(5):
        lone_env._update_bands()
    assert lone not in lone_env._band_out_steps and lone_env.agent_band[lone] == 0


def test_a_drifting_member_leaves_alone_its_children_in_range_of_the_band_stay():
    env = _env(band_drift_steps=10, band_check_interval=10)
    mother, father, kids, singles = _roles(env)
    members = _members(env, 0)
    _park(env, set(members))
    _cluster(env, [father, singles[0], singles[1], kids[0], kids[1]], [(5, 5), (5, 6), (6, 5), (6, 6), (5, 7)])
    _place(env, mother, (2, 20))
    env._update_bands()
    new_id = env.agent_band[mother]
    assert new_id == env.num_bands and env.band_drift_outs == 1
    assert env.agent_band[kids[0]] == 0 and env.agent_band[kids[1]] == 0  # the children keep their band


def test_doomed_predators_do_not_count_as_band_members():
    env = _env()
    band0 = _members(env, 0)
    env.agent_energies[band0[0]] = -1.0
    assert band0[0] not in env._bands_now()[0] and len(env._bands_now()[0]) == 5
    env2 = _env(diet_required=True)
    env2.agent_fruit_store[band0[1]] = 0.0
    assert band0[1] not in env2._bands_now()[0]


def test_a_very_large_band_keeps_splitting_until_every_part_is_within_the_cap_and_fresh_bands_do_not_fuse_at_once():
    env = _env(num_bands=1, band_couples=2, band_children_per_couple=0, band_singles_male=3, band_singles_female=3,  # no children: exact clusters
              
               band_fission=True, band_max_size=4, band_min_split_size=2, band_fusion=True, band_fuse_steps=0, band_fuse_distance=100)
    members = _members(env, 0)
    assert len(members) == 10
    _park(env, set(members))
    cells = [(2, 2), (2, 3), (3, 2), (2, 20), (2, 21), (3, 20), (20, 2), (20, 3), (20, 20), (20, 21)]  # clusters of 3, 3, 2, 2
    _cluster(env, members, cells)
    env._update_bands()
    sizes = [len(m) for m in env._bands_now().values()]
    assert env.band_fissions >= 2 and max(sizes) <= 4
    assert env.band_fusions == 0  # bands created in this call are not fused in it


def test_drift_join_moves_a_member_into_the_nearest_other_band_in_range_else_it_leaves_alone():
    env = _env(band_drift_steps=10, band_drift_join=True, band_check_interval=10)
    b0, b1 = _members(env, 0), _members(env, 1)
    _park(env, set(b0 + b1))
    _cluster(env, b0[1:], [(5, 5), (5, 6), (6, 5), (6, 6), (5, 7)])
    _place(env, b0[0], (20, 20))                      # far from its own band ...
    _cluster(env, b1[:2], [(20, 23), (21, 23)])        # ... but within 5 cells of two band-1 members
    _cluster(env, b1[2:], [(12, 2), (12, 3), (13, 2), (13, 3)])
    env._update_bands()
    assert env.agent_band[b0[0]] == 1 and env.band_drift_outs >= 1
    env2 = _env(band_drift_steps=10, band_drift_join=True, band_check_interval=10)
    b0 = _members(env2, 0)
    _park(env2, set(b0))
    _cluster(env2, b0[1:], [(5, 5), (5, 6), (6, 5), (6, 6), (5, 7)])
    _place(env2, b0[0], (2, 20))                      # nobody else near: a one-member band
    env2._update_bands()
    assert env2.agent_band[b0[0]] == env2.num_bands


def test_a_singleton_joins_a_nearby_band_when_drift_join_is_on():
    env = _env(band_drift_steps=10, band_drift_join=True, band_check_interval=10)
    b0, b1 = _members(env, 0), _members(env, 1)
    _park(env, set(b0 + b1))
    env.agent_band[b0[0]] = 7  # alone in band 7
    _place(env, b0[0], (10, 10))
    _cluster(env, b0[1:], [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4)])
    _cluster(env, b1, [(10, 12), (10, 13), (11, 12), (11, 13), (12, 12), (12, 13)])  # a band right next to it
    env._update_bands()
    assert env.agent_band[b0[0]] == 1


def test_snapshot_keeps_nonempty_dynamics_state():
    env = _env(band_drift_steps=50, band_fusion=True)
    env._band_out_steps = {"predator_male_0": (0, 20)}
    env._band_contact = {(0, 1): 20}
    snap = env.get_state_snapshot()
    env._band_out_steps, env._band_contact = {}, {}
    env.restore_state_snapshot(snap)
    assert env._band_out_steps == {"predator_male_0": (0, 20)} and env._band_contact == {(0, 1): 20}
