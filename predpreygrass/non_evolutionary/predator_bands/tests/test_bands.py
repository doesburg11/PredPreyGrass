"""Tests for the band mechanics of predator_bands: the initial layout, within-band sharing, kin exclusion, marriage,
the band observation channels, snapshots and config validation. The inherited diet / scripted-prey / mechanics tests
run with bands off in the other two test files."""
import copy

import numpy as np
import pytest

from predpreygrass.non_evolutionary.predator_bands.config_env import config_env as _base
from predpreygrass.non_evolutionary.predator_bands.predpreygrass_rllib_env import PredPreyGrass


N_BANDS = _base["num_bands"]  # the default layout is N_BANDS bands of 6 (couple + 2 children + 2 singles)


def _env(**overrides):
    config = copy.deepcopy(_base)
    config.update({"max_steps": 100})
    config.update(overrides)
    env = PredPreyGrass(config)
    env.reset(seed=1)
    return env


def _members(env, band):
    return [a for a, b in env.agent_band.items() if b == band and a in env.predator_positions]


def _place(env, agent, pos):
    old = env.agent_positions.get(agent)
    if old is not None:
        env.grid_world_state[1, *old] = 0
    env.agent_positions[agent] = pos
    env.predator_positions[agent] = pos
    env.grid_world_state[1, *pos] = env.agent_energies.get(agent, 0.0)


def _noop(env):
    return {a: env.noop_action_id for a in env.agents}


def _quiet(**kw):
    """Band env with no other predators moving/eating surprises: only what a test sets up."""
    return _env(diet_required=False, **kw)


# ---- initial layout -------------------------------------------------------------------------------------------
def test_default_layout_five_bands_of_six_balanced_and_unique_cells():
    env = _env()
    assert len(env.predator_positions) == 6 * N_BANDS
    assert env.n_initial_active_predator_male == 3 * N_BANDS and env.n_initial_active_predator_female == 3 * N_BANDS
    assert len(set(env.predator_positions.values())) == 6 * N_BANDS
    assert sorted(len(_members(env, b)) for b in range(N_BANDS)) == [6] * N_BANDS
    assert len(env._scripted_prey_ids) == 40


def test_band_composition_couples_children_singles():
    env = _env()
    for b in range(N_BANDS):
        members = _members(env, b)
        couples = [(m, env.agent_mate[m]) for m in members if "male" in m and "female" not in m and m in env.agent_mate]
        assert len(couples) == 1
        m, f = couples[0]
        assert env.agent_mate[f] == m and env.agent_band[f] == b
        kids = [a for a in members if env.agent_parents.get(a) == (m, f)]
        assert len(kids) == 2 and {("male" in k and "female" not in k) for k in kids} == {True, False}
        singles = [a for a in members if a not in kids and a not in (m, f)]
        assert len(singles) == 2 and all(a not in env.agent_mate for a in singles)
        assert m in env.has_reproduced and f in env.has_reproduced and not any(k in env.has_reproduced for k in kids)


def test_bands_are_spatially_clustered_and_separated():
    env = _env()
    spread, centroids = [], []
    for b in range(N_BANDS):
        pts = np.array([env.predator_positions[a] for a in _members(env, b)])
        c = pts.mean(0)
        centroids.append(c)
        spread.append(np.abs(pts - c).max())
    assert max(spread) <= 3.0
    d = [np.abs(centroids[i] - centroids[j]).max() for i in range(N_BANDS) for j in range(i + 1, N_BANDS)]
    assert min(d) >= 4.0


@pytest.mark.parametrize("couples,kids,sm,sf", [(2, 3, 0, 2), (1, 0, 2, 1), (3, 1, 1, 0)])
def test_initial_counts_follow_the_composition(couples, kids, sm, sf):
    env = _env(num_bands=3, band_couples=couples, band_children_per_couple=kids, band_singles_male=sm,
               band_singles_female=sf, n_initial_active_prey=5)
    males = 3 * (couples * (1 + (kids + 1) // 2) + sm)
    females = 3 * (couples * (1 + kids // 2) + sf)
    assert env.n_initial_active_predator_male == males and env.n_initial_active_predator_female == females
    assert len(env.predator_positions) == males + females
    assert len(env.agent_band) == males + females


def test_num_bands_zero_turns_bands_off():
    env = _env(num_bands=0, n_initial_active_predator_male=2, n_initial_active_predator_female=2, scripted_prey=False)
    assert env.agent_band == {}
    obs = env._get_observation(next(a for a in env.agents if "predator" in a))
    assert obs[6:].sum() == 0


# ---- sharing --------------------------------------------------------------------------------------------------
def _sharing_setup(**kw):
    env = _quiet(band_share_rate=0.3, band_share_range=5, parent_offspring_share_rate=0.0, **kw)
    band0 = _members(env, 0)
    forager, near, far = band0[0], band0[1], band0[2]
    stranger = _members(env, 1)[0]
    _place(env, forager, (10, 10))
    _place(env, near, (10, 12))
    _place(env, far, (10, 20))  # same band, out of range
    _place(env, stranger, (10, 11))  # in range, other band
    for a in _members(env, 0) + [stranger]:
        env.agent_energies[a] = 5.0
        env.agent_fruit_store[a] = 2.5
    # park everyone else out of the way
    for a in env.predator_positions:
        if a not in (forager, near, far, stranger) and env.agent_band[a] == 0:
            _place(env, a, (24, 24 - list(env.predator_positions).index(a) % 20))
    return env, forager, near, far, stranger


def test_share_splits_fruit_equally_within_range_and_band_only():
    env, forager, near, far, stranger = _sharing_setup()
    others_in_range = [a for a in _members(env, 0) if a != forager and
                       max(abs(env.predator_positions[a][0] - 10), abs(env.predator_positions[a][1] - 10)) <= 5]
    assert near in others_in_range and far not in others_in_range and stranger not in others_in_range
    before = {a: (env.agent_energies[a], env.agent_fruit_store[a]) for a in env.predator_positions}
    env._apply_band_share(forager, 2.0, is_fruit=True)
    total = 0.3 * 2.0
    share = total / len(others_in_range)
    assert env.agent_energies[forager] == pytest.approx(before[forager][0] - total)
    assert env.agent_fruit_store[forager] == pytest.approx(before[forager][1] - total)
    for a in others_in_range:
        assert env.agent_energies[a] == pytest.approx(before[a][0] + share)
        assert env.agent_fruit_store[a] == pytest.approx(before[a][1] + share)
    for a in (far, stranger):
        assert env.agent_energies[a] == before[a][0] and env.agent_fruit_store[a] == before[a][1]
    assert env.band_share_events == 1 and env.band_share_fruit_total == pytest.approx(total)


def test_meat_share_leaves_fruit_stores_untouched():
    env, forager, near, *_ = _sharing_setup()
    f_forager, f_near = env.agent_fruit_store[forager], env.agent_fruit_store[near]
    e_near = env.agent_energies[near]
    env._apply_band_share(forager, 3.0, is_fruit=False)
    assert env.agent_fruit_store[forager] == f_forager and env.agent_fruit_store[near] == f_near
    assert env.agent_energies[near] > e_near
    assert env.band_share_meat_total == pytest.approx(0.9)


def test_share_does_not_rescue_a_doomed_member_and_can_be_switched_off():
    env, forager, near, *_ = _sharing_setup()
    env.diet_required = True
    env.agent_energies[near] = 8.0
    env.agent_fruit_store[near] = 0.0  # fruit store empty: already doomed
    others = [a for a in env.predator_positions if a not in (forager, near) and env.agent_band[a] == 0
              and max(abs(env.predator_positions[a][0] - 10), abs(env.predator_positions[a][1] - 10)) <= 5]
    e_near = env.agent_energies[near]
    env._apply_band_share(forager, 2.0, is_fruit=True)
    assert env.agent_energies[near] == e_near
    env2 = _quiet(band_share_rate=0.0)
    f = _members(env2, 0)[0]
    env2._apply_band_share(f, 2.0, is_fruit=True)
    assert env2.band_share_events == 0


def test_a_step_with_fruit_shares_it_with_the_band():
    env, forager, near, far, stranger = _sharing_setup()
    fruit = next(iter(env.fruit_positions))
    env.fruit_positions[fruit] = (10, 10)
    env.fruit_energies[fruit] = 2.0
    env.grid_world_state[4, 10, 10] = 2.0
    # keep every other fruit away from every predator so that only the forager eats this step
    taken = set(env.predator_positions.values())
    free = next((x, y) for x in range(env.grid_size) for y in range(env.grid_size) if (x, y) not in taken)
    for other_fruit in env.fruit_positions:
        if other_fruit != fruit:
            env.grid_world_state[4, *env.fruit_positions[other_fruit]] = 0
            env.fruit_positions[other_fruit] = free
            env.fruit_energies[other_fruit] = 0.0
    env.rng = np.random.default_rng(0)
    env.step(_noop(env))
    assert env.band_share_events == 1 and env.band_share_fruit_total == pytest.approx(0.6)


# ---- kin exclusion and marriage -------------------------------------------------------------------------------
def _births(env):
    return env.episode_births["predator_male"] + env.episode_births["predator_female"]


def test_kin_exclusion_blocks_parent_child_and_siblings_but_not_strangers():
    env = _quiet()
    m, f = next((a, b) for a, b in env.agent_mate.items() if "female" not in a)
    kids = [a for a in env.agent_parents if env.agent_parents[a] == (m, f)]
    kid_m = next(k for k in kids if "female" not in k)
    kid_f = next(k for k in kids if "female" in k)
    other_male = next(a for a in env.predator_positions if "female" not in a and a not in (m, kid_m)
                      and env.agent_band[a] != env.agent_band[m])
    assert env._kin_blocked(m, kid_f)        # father-daughter
    assert env._kin_blocked(kid_m, f)        # son-mother
    assert env._kin_blocked(kid_m, kid_f)    # siblings
    assert not env._kin_blocked(other_male, kid_f)
    assert env.kin_blocked_checks == 3
    env2 = _quiet(kin_exclusion=False)
    m2, f2 = next((a, b) for a, b in env2.agent_mate.items() if "female" not in a)
    kid_f2 = next(a for a in env2.agent_parents if env2.agent_parents[a] == (m2, f2) and "female" in a)
    assert not env2._kin_blocked(m2, kid_f2)


def test_a_blocked_kin_pair_does_not_breed_but_a_stranger_does():
    env = _quiet(mate_search_radius=3)
    m, f = next((a, b) for a, b in env.agent_mate.items() if "female" not in a)
    daughter = next(a for a in env.agent_parents if env.agent_parents[a] == (m, f) and "female" in a)
    for a in env.predator_positions:
        env.agent_energies[a] = 1.0
    _place(env, m, (10, 10))
    _place(env, daughter, (10, 11))
    for a in (m, daughter):
        env.agent_energies[a] = 13.0
        env.agent_fruit_store[a] = 6.5
    env.step(_noop(env))
    assert _births(env) == 0


def _marriage_setup(rule="female_joins_male"):
    env = _quiet(marriage_rule=rule, mate_search_radius=3)
    bands = {b: _members(env, b) for b in range(N_BANDS)}
    male = next(a for a in bands[0] if "female" not in a and a not in env.agent_mate)         # single male, band 0
    female = next(a for a in bands[1] if "female" in a and a not in env.agent_mate)           # single female, band 1
    for a in env.predator_positions:
        env.agent_energies[a] = 1.0
    _place(env, male, (10, 10))
    _place(env, female, (10, 11))
    for a in (male, female):
        env.agent_energies[a] = 13.0
        env.agent_fruit_store[a] = 6.5
    return env, male, female


def test_cross_band_pairing_is_a_marriage_the_female_joins_and_the_child_follows_the_father():
    env, male, female = _marriage_setup()
    env.step(_noop(env))
    assert _births(env) == 1 and env.marriages == 1 and env.within_band_pairings == 0
    assert env.agent_band[female] == env.agent_band[male] == 0
    child = next(a for a in env.agent_parents if env.agent_parents[a] == (male, female))
    assert env.agent_band[child] == 0


def test_male_joins_female_rule():
    env, male, female = _marriage_setup("male_joins_female")
    env.step(_noop(env))
    assert env.agent_band[male] == env.agent_band[female] == 1
    assert env.marriages == 1


def test_marriage_takes_dependent_children_along_but_not_grown_ones():
    env, male, female = _marriage_setup()
    kid = "predator_female_test_kid"
    grown = "predator_male_test_grown"
    for a, pos in ((kid, (20, 20)), (grown, (20, 21))):
        env.agent_positions[a] = pos
        env.predator_positions[a] = pos
        env.agent_energies[a] = 1.0
        env.agent_fruit_store[a] = 0.5
        env.agent_parents[a] = (None, female)
        env.agent_band[a] = 1
        env.agents.append(a)
        env.cumulative_rewards[a] = 0
    env.has_reproduced.add(grown)
    env._record_pairing(male, female)
    assert env.agent_band[kid] == 0 and env.agent_band[grown] == 1


def test_within_band_pairing_counts_no_marriage():
    env = _quiet(mate_search_radius=3)
    band = _members(env, 0)
    male = next(a for a in band if "female" not in a and a not in env.agent_mate)
    female = next(a for a in band if "female" in a and a not in env.agent_mate)
    env._record_pairing(male, female)
    assert env.within_band_pairings == 1 and env.marriages == 0


# ---- observation, snapshots, validation -------------------------------------------------------------------------
def test_band_observation_channels_mark_same_and_other_band_predators():
    env = _quiet()
    a = _members(env, 0)[0]
    mate_mates = _members(env, 0)[1]
    other = _members(env, 1)[0]
    for x in env.predator_positions:
        if x not in (a, mate_mates, other):
            _place(env, x, (24, 24))  # out of view (cells may collide only among unrelated agents; harmless here)
    _place(env, a, (10, 10))
    _place(env, mate_mates, (10, 12))
    _place(env, other, (12, 10))
    obs = env._get_observation(a)
    off = (env.predator_obs_range - 1) // 2
    assert obs[6, off, off + 2] == 1.0 and obs[7, off + 2, off] == 1.0
    assert obs[6].sum() == 1.0 and obs[7].sum() == 1.0  # the observer itself is not marked
    assert obs[6, off, off] == 0 and obs[7, off, off] == 0


def test_snapshot_round_trip_of_band_state():
    env = _quiet()
    snap = env.get_state_snapshot()
    a = _members(env, 0)[0]
    env.agent_band[a] = 4
    env.marriages = 7
    env.restore_state_snapshot(snap)
    assert env.agent_band[a] == 0 and env.marriages == 0


def test_removed_predators_drop_their_band_entry():
    env = _quiet()
    a = _members(env, 0)[0]
    env.agent_energies[a] = -1.0
    env.step(_noop(env))
    assert a not in env.agent_positions and a not in env.agent_band


@pytest.mark.parametrize("kw", [
    {"band_share_rate": 0.9, "parent_offspring_share_rate": 0.2},
    {"band_share_rate": 0.5, "male_gift_donation_rate": 0.4, "parent_offspring_share_rate": 0.2},
    {"band_share_rate": 1.5},
    {"marriage_rule": "nope"},
    {"num_bands": -1},
    {"band_children_per_couple": -1},
])
def test_invalid_band_config_is_rejected(kw):
    with pytest.raises(ValueError):
        _env(**kw)


def test_full_random_episode_with_default_bands_runs_and_reports_band_metrics():
    env = _env(max_steps=150)
    obs, _ = env.reset(seed=4)
    rng = np.random.default_rng(0)
    for _ in range(150):
        obs, r, t, tr, _ = env.step({a: int(rng.integers(env.num_actions)) for a in obs})
        assert all("prey" not in a and o.shape[0] == 8 and np.isfinite(o).all() for a, o in obs.items())
        assert set(env.agent_band) >= set(env.predator_positions)
        if t.get("__all__") or tr.get("__all__"):
            break
    m = env._build_episode_training_metrics()
    for k in ("band_share_events", "band_share_meat_total", "band_share_fruit_total", "marriages",
              "within_band_pairings", "kin_blocked_checks", "bands_alive"):
        assert k in m
    assert m["band_share_events"] > 0


# ---- fixes from the Codex review ---------------------------------------------------------------------------------
def test_band_members_start_within_the_spawn_radius_of_a_common_centre():
    env = _env(band_spawn_radius=2)
    for b in range(N_BANDS):
        pts = np.array([env.predator_positions[a] for a in _members(env, b)])
        assert np.abs(pts.max(0) - pts.min(0)).max() <= 4  # all within radius 2 of one centre (diameter 4)


def test_layout_that_cannot_fit_the_spawn_radius_is_rejected():
    with pytest.raises(ValueError):
        _env(band_spawn_radius=0)  # 6 members cannot fit on the single centre cell


def test_oversized_layout_and_bad_channel_count_are_rejected():
    with pytest.raises(ValueError):
        _env(grid_size=10)  # the predators + 40 prey + 200 food items do not fit
    with pytest.raises(ValueError):
        _env(num_obs_channels=6)
    with pytest.raises(ValueError):
        _env(n_possible_predator_male=3)


def test_donation_sum_check_ignores_band_share_when_bands_are_off():
    env = _env(num_bands=0, n_initial_active_predator_male=2, n_initial_active_predator_female=2, scripted_prey=False,
               n_initial_active_prey=3, band_share_rate=0.9, parent_offspring_share_rate=0.2)
    assert env.num_bands == 0


def test_unrelated_cross_band_pair_reproduces_when_kin_pairs_do_not():
    env, male, female = _marriage_setup()
    assert not env._kin_blocked(male, female)
    env.step(_noop(env))
    assert _births(env) == 1


# ---- separate meat / fruit sharing rates ---------------------------------------------------------------------------
def test_meat_and_fruit_use_their_own_sharing_rates():
    env, forager, near, *_ = _sharing_setup()
    env.band_meat_share_rate, env.band_fruit_share_rate = 0.6, 0.2
    env._apply_band_share(forager, 2.0, is_fruit=False)
    assert env.band_share_meat_total == pytest.approx(1.2) and env.band_share_fruit_total == 0.0
    env._apply_band_share(forager, 2.0, is_fruit=True)
    assert env.band_share_fruit_total == pytest.approx(0.4)


def test_rates_default_to_band_share_rate_and_validate():
    env = _quiet(band_share_rate=0.35)
    assert env.band_meat_share_rate == 0.35 and env.band_fruit_share_rate == 0.35
    env = _quiet(band_meat_share_rate=0.6, band_fruit_share_rate=0.3)
    assert env.band_meat_share_rate == 0.6 and env.band_fruit_share_rate == 0.3
    with pytest.raises(ValueError):
        _env(band_meat_share_rate=0.7, parent_offspring_share_rate=0.4)  # 0.7 + 0.4 > 1 for a meat gain
    with pytest.raises(ValueError):
        _env(band_fruit_share_rate=1.2)
    _env(band_meat_share_rate=0.6, band_fruit_share_rate=0.3)  # 0.6 + 0.2 care is fine


# ---- band compass ----------------------------------------------------------------------------------------------------
def _compass_env(**kw):
    return _quiet(band_compass=True, **kw)


def test_compass_is_off_by_default_and_adds_four_channels_when_on():
    assert _quiet().num_obs_channels == 8
    env = _compass_env()
    assert env.num_obs_channels == 12
    a = _members(env, 0)[0]
    assert env._get_observation(a).shape == (12, env.predator_obs_range, env.predator_obs_range)


def test_compass_points_to_the_nearest_same_band_member_outside_the_window():
    env = _compass_env()
    band = _members(env, 0)
    a, far_mate, farther_mate = band[0], band[1], band[2]
    other = _members(env, 1)[0]
    for x in env.predator_positions:
        if x not in (a, far_mate, farther_mate, other):
            _place(env, x, (24, 24))
    _place(env, a, (10, 5))
    _place(env, far_mate, (10, 15))     # dx = 0, dy = +10: outside the 7x7 window
    _place(env, farther_mate, (23, 5))  # farther
    _place(env, other, (10, 6))         # other band, adjacent: must not count
    obs = env._get_observation(a)
    assert np.all(obs[8] == 1.0)
    assert np.allclose(obs[9], 0.5) and np.allclose(obs[10], 1.0)  # unit vector (0, +1) -> encoded (0.5, 1.0)
    assert np.allclose(obs[11], 10 / env.grid_size)
    assert obs[8:].min() >= 0.0 and obs.max() <= 100.0


def test_compass_is_zero_without_a_band_mate_and_for_other_observers():
    env = _compass_env()
    band = _members(env, 0)
    lone = band[0]
    for x in list(env.predator_positions):
        if x != lone:
            if env.agent_band[x] == 0:
                env.agent_energies[x] = -1.0
    for x in band[1:]:
        del env.predator_positions[x]
    obs = env._get_observation(lone)
    assert obs[8:].sum() == 0.0
    prey_obs = None
    if env._scripted_prey_ids:
        prey_obs = env._get_observation(env._scripted_prey_ids[0])
        assert prey_obs[8:].sum() == 0.0


def test_compass_points_diagonally_and_needs_enough_channels():
    env = _compass_env()
    band = _members(env, 0)
    a, mate = band[0], band[1]
    for x in env.predator_positions:
        if x not in (a, mate) and env.agent_band[x] == 0:
            _place(env, x, (24, 24))
    _place(env, a, (5, 5))
    _place(env, mate, (12, 12))
    obs = env._get_observation(a)
    r = 1 / np.sqrt(2)
    assert np.allclose(obs[9], (r + 1) / 2) and np.allclose(obs[10], (r + 1) / 2)
    with pytest.raises(ValueError):
        _env(band_compass=True, num_obs_channels=10)


# ---- distance decay of band sharing ----------------------------------------------------------------------------------------
def test_distance_decay_scales_each_share_and_the_donor_pays_only_what_is_delivered():
    env, forager, near, far, stranger = _sharing_setup(band_share_distance_decay=1.0)
    _place(env, far, (10, 15))  # 5 cells away, inside the range of 5
    _place(env, near, (10, 11))  # 1 cell away
    others = [a for a in _members(env, 0) if a != forager]
    inside = [a for a in others if max(abs(env.predator_positions[a][0] - 10), abs(env.predator_positions[a][1] - 10)) <= 5]
    before = {a: env.agent_energies[a] for a in env.predator_positions}
    env._apply_band_share(forager, 2.0, is_fruit=False)
    gains = {a: env.agent_energies[a] - before[a] for a in inside}
    share = 0.3 * 2.0 / len(inside)
    assert gains[near] == pytest.approx(share * (1 - 1 / 6))
    assert gains[far] == pytest.approx(share * (1 - 5 / 6))
    assert gains[near] > gains[far] > 0
    paid = before[forager] - env.agent_energies[forager]
    assert paid == pytest.approx(sum(gains.values())) and paid < 0.6
    assert env.band_share_meat_total == pytest.approx(paid)


def test_distance_decay_zero_is_the_flat_split_and_bad_values_are_rejected():
    env, forager, near, far, stranger = _sharing_setup()
    before = env.agent_energies[forager]
    env._apply_band_share(forager, 2.0, is_fruit=False)
    assert before - env.agent_energies[forager] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        _env(band_share_distance_decay=1.5)
    with pytest.raises(ValueError):
        _env(band_share_distance_decay=-0.1)
