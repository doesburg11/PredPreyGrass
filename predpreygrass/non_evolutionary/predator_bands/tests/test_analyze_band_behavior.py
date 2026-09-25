"""Tests for analyze_band_behavior.py: instrumentation bookkeeping, founder roles, cohesion features, approach geometry,
the bootstrap helper and a short end-to-end rollout."""
import copy

import numpy as np
import pytest

from predpreygrass.non_evolutionary.predator_bands import analyze_band_behavior as ab
from predpreygrass.non_evolutionary.predator_bands.config_env import config_env as _base


def _cfg(**kw):
    c = copy.deepcopy(_base)
    c.update({"max_steps": 60, "diet_required": False})
    c.update(kw)
    return c


def _env(**kw):
    env = ab.InstrumentedBandsEnv(_cfg(**kw))
    env.reset(seed=1)
    return env


def _members(env, band):
    return [a for a, b in env.agent_band.items() if b == band]


def _place(env, agent, pos):
    env.grid_world_state[1, *env.agent_positions[agent]] = 0
    env.agent_positions[agent] = pos
    env.predator_positions[agent] = pos
    env.grid_world_state[1, *pos] = env.agent_energies[agent]


def test_founder_roles_counts():
    env = _env()
    roles = ab.founder_roles(env)
    counts = {r: sum(1 for v in roles.values() if v == r) for r in ab.ROLES}
    assert counts == {"couple_male": 5, "couple_female": 5, "child": 10, "single_male": 5, "single_female": 5, "born": 0}


def test_instrumented_env_records_own_forage_and_conserves_shared_energy():
    env = _env(band_meat_share_rate=0.6, band_fruit_share_rate=0.3, band_share_range=30)
    band = _members(env, 0)
    forager = band[0]
    total_before = sum(env.agent_energies[a] for a in env.predator_positions)
    env._apply_band_share(forager, 2.0, is_fruit=False)
    env._apply_band_share(forager, 1.0, is_fruit=True)
    assert env.own["meat"][forager] == 2.0 and env.own["fruit"][forager] == 1.0
    assert env.n_items["meat"][forager] == 1 and env.n_items["fruit"][forager] == 1
    assert env.given["band"]["meat"][forager] == pytest.approx(1.2) and env.given["band"]["fruit"][forager] == pytest.approx(0.3)
    for kind in ("meat", "fruit"):
        assert sum(env.recv["band"][kind].values()) == pytest.approx(sum(env.given["band"][kind].values()))  # only moves energy
    assert sum(env.agent_energies[a] for a in env.predator_positions) == pytest.approx(total_before)
    assert forager not in env.recv["band"]["meat"]
    assert sum(env.recv["care"]["meat"].values()) == 0 and sum(env.recv["gift"]["meat"].values()) == 0


def test_own_forage_is_recorded_even_when_sharing_is_off():
    env = _env(band_share_rate=0.0, band_meat_share_rate=0.0, band_fruit_share_rate=0.0)
    f = _members(env, 0)[0]
    env._apply_band_share(f, 2.0, is_fruit=True)
    assert env.own["fruit"][f] == 2.0 and sum(env.given["band"]["fruit"].values()) == 0.0


def test_cohesion_features_distances_and_range():
    env = _env(band_share_range=5)
    band = _members(env, 0)
    other = _members(env, 1)[0]
    a = band[0]
    for x in env.predator_positions:
        if x not in (a, band[1], band[2], other):
            _place(env, x, (24, 24))
    _place(env, a, (10, 10))
    _place(env, band[1], (10, 13))
    _place(env, band[2], (10, 17))
    _place(env, other, (12, 10))
    d_same, d_other, n_in_range, same, other_pos = ab.cohesion_features(env, a)
    assert d_same == 3 and d_other == 2 and n_in_range == 1
    lone = _members(env, 2)[0]
    _place(env, lone, (0, 0))
    for x in _members(env, 2):
        if x != lone:
            _place(env, x, (24, 23))
    _, _, n, _, _ = ab.cohesion_features(env, lone)
    assert n == 0


def test_action_geometry_sign_and_visibility():
    moves = np.array([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)])
    pos = np.array([5, 5])
    east = np.array([[8, 5]])
    d = ab.action_geometry(pos, east, moves, 20, 3)
    assert d[1] < d[0] < d[2]  # stepping east is closest, west farthest
    assert ab.action_geometry(pos, np.array([[12, 5]]), moves, 20, 3) is None  # not visible
    assert ab.action_geometry(pos, np.array([[5, 5]]), moves, 20, 3) is None  # co-located
    assert ab.action_geometry(pos, np.empty((0, 2), dtype=int), moves, 20, 3) is None


def test_ratio_ci_needs_episodes_and_recovers_constants():
    rng = np.random.default_rng(0)
    assert ab.ratio_ci([0, 0], [0, 0], rng) is None
    p, lo, hi = ab.ratio_ci([1.0, 2.0], [1, 1], rng)
    assert p == 1.5 and np.isnan(lo)
    p, lo, hi = ab.ratio_ci([2.0] * 6, [1] * 6, rng)
    assert (p, lo, hi) == (2.0, 2.0, 2.0)


def test_short_random_rollout_and_summary_run_end_to_end():
    cfg = _cfg(max_steps=40)
    episodes, lives = ab.rollout(cfg, None, 3, 10, random_policy=True)
    assert len(episodes) == 3 and lives
    assert {"episode", "sex", "role", "life", "own_meat", "recv_meat", "given_fruit", "censored"} <= set(lives[0])
    assert {l["role"] for l in lives} <= set(ab.ROLES)
    # every founder appears once per episode with a role other than "born"
    for ep in range(3):
        founders = [l for l in lives if l["episode"] == ep and l["role"] != "born"]
        assert len(founders) == 30
    lines = ab.summarize("test", episodes, lives, np.random.default_rng(0), episodes)
    text = "\n".join(lines)
    for token in ("A. energy sources", "B. founders by role", "C. approach bias", "D. cohesion", "single_female"):
        assert token in text


def test_care_and_gifts_are_tracked_separately_from_band_sharing():
    env = _env(parent_offspring_share_rate=0.2, band_share_rate=0.0, band_meat_share_rate=0.0, band_fruit_share_rate=0.0,
               male_gift_donation_rate=0.3, female_gift_donation_rate=0.3)
    mother = next(a for a in env.agent_mate if "female" in a)
    father = env.agent_mate[mother]
    kids = [a for a, p in env.agent_parents.items() if p == (father, mother)]
    for a in env.predator_positions:
        if a not in [mother, father] + kids:
            _place(env, a, (24, 24))
    _place(env, mother, (10, 10))
    _place(env, father, (10, 11))
    for k, pos in zip(kids, [(11, 10), (11, 11)]):
        _place(env, k, pos)
    env._share_energy_with_offspring(mother, 2.0, is_fruit=True)
    assert env.given["care"]["fruit"][mother] == pytest.approx(0.4)
    assert sum(env.recv["care"]["fruit"].values()) == pytest.approx(0.4)
    env._apply_male_gift(father, 2.0)
    assert env.given["gift"]["meat"][father] == pytest.approx(0.6) and env.recv["gift"]["meat"][mother] == pytest.approx(0.6)
    assert sum(env.recv["band"]["meat"].values()) == 0


def test_action_geometry_treats_moves_onto_occupied_cells_as_staying():
    moves = np.array([(0, 0), (1, 0), (-1, 0)])
    pos = np.array([5, 5])
    target = np.array([[8, 5]])
    free = ab.action_geometry(pos, target, moves, 20, 3)
    blocked = ab.action_geometry(pos, target, moves, 20, 3, occupied={(6, 5)})
    assert free[1] < free[0] and blocked[1] == blocked[0]  # the step east is blocked, so it ends where staying does
    assert blocked[2] == free[2]


def test_random_baseline_is_only_printed_for_the_first_bucket():
    cfg = _cfg(max_steps=150)
    eps, lives = ab.rollout(cfg, None, 6, 3, random_policy=True)
    lines = ab.summarize("t", eps, lives, np.random.default_rng(0), eps)
    d = [l for l in lines if "steps" in l and "same-band dist" in l]
    assert all(("random same-band" in l) == ("steps 0-100" in l) for l in d)
