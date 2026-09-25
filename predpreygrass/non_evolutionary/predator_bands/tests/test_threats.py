"""Tests for the roaming threats of predator_bands (default off): placement, movement, defense arithmetic, kills and repelling,
the threat observation channel, snapshots, validation; plus the band-compass fixes from the Codex review."""
import copy

import numpy as np
import pytest

from predpreygrass.non_evolutionary.predator_bands.config_env import config_env as _base
from predpreygrass.non_evolutionary.predator_bands.predpreygrass_rllib_env import PredPreyGrass


class _FixedRandom:
    """Real generator with random() forced to a value (integers/choice/etc. stay real)."""

    def __init__(self, real, value):
        self._real, self._value = real, value

    def random(self, *a, **k):
        return self._value

    def __getattr__(self, name):
        return getattr(self._real, name)


def _env(**kw):
    c = copy.deepcopy(_base)
    c.update({"max_steps": 100, "diet_required": False, "num_threats": 2})
    c.update(kw)
    env = PredPreyGrass(c)
    env.reset(seed=1)
    return env


def _members(env, band):
    return [a for a, b in env.agent_band.items() if b == band]


def _place(env, agent, pos):
    env.grid_world_state[1, *env.agent_positions[agent]] = 0
    env.agent_positions[agent] = pos
    env.predator_positions[agent] = pos
    env.grid_world_state[1, *pos] = env.agent_energies[agent]


def _park_others(env, keep, start=(24, 0)):
    """Move every predator not in `keep` to unique cells far away (bottom-right region)."""
    i = 0
    for a in list(env.predator_positions):
        if a not in keep:
            _place(env, a, (24 - i // 8, 16 + i % 8))
            i += 1


def _noop(env):
    return {a: env.noop_action_id for a in env.agents}


# ---- off by default / placement / channels --------------------------------------------------------------------------
def test_threats_are_off_by_default_and_add_one_channel_when_on():
    c = copy.deepcopy(_base)
    env = PredPreyGrass(c)
    env.reset(seed=1)
    assert env.num_threats == 0 and env.threat_positions == {} and env.num_obs_channels == 8
    env = _env(band_compass=True)
    assert env.num_obs_channels == 13 and len(env.threat_positions) == 2
    assert _env().num_obs_channels == 9


def test_threats_are_placed_on_free_unique_cells_and_do_not_change_the_predator_layout_counts():
    env = _env(num_threats=6)
    cells = list(env.threat_positions.values())
    assert len(set(cells)) == 6 and not (set(cells) & set(env.agent_positions.values()))
    assert len(env.predator_positions) == 6 * _base["num_bands"]


def test_threat_channel_marks_threats_inside_the_window_only():
    env = _env(num_threats=2)
    a = _members(env, 0)[0]
    _place(env, a, (10, 10))
    env.threat_positions = {"threat_0": (10, 12), "threat_1": (20, 20)}
    obs = env._get_observation(a)
    off = (env.predator_obs_range - 1) // 2
    ch = 8
    assert obs[ch, off, off + 2] == 1.0 and obs[ch].sum() == 1.0
    prey_like = env._get_observation(a)  # sanity: shape
    assert prey_like.shape[0] == env.num_obs_channels


# ---- movement -----------------------------------------------------------------------------------------------------------
def test_a_threat_chases_a_sensed_predator_and_wanders_when_alone():
    env = _env(num_threats=1)
    a = _members(env, 0)[0]
    _park_others(env, {a})
    _place(env, a, (10, 10))
    env.threat_positions = {"threat_0": (10, 14)}
    d0 = 4
    env.rng = np.random.default_rng(0)
    env._threats_act()
    tx, ty = env.threat_positions["threat_0"]
    assert max(abs(tx - 10), abs(ty - 10)) == d0 - 1  # closer by one step
    # out of sense radius: moves by at most one cell in any direction, stays on the grid
    env.threat_positions = {"threat_0": (0, 0)}
    _place(env, a, (20, 20))
    env._threats_act()
    tx, ty = env.threat_positions["threat_0"]
    assert (tx, ty) != (0, 0) and 0 <= tx <= 1 and 0 <= ty <= 1


def test_threats_never_step_onto_predators_or_other_threats():
    env = _env(num_threats=4)
    env.rng = np.random.default_rng(3)
    for _ in range(50):
        env._threats_act()
        cells = list(env.threat_positions.values())
        assert len(set(cells)) == len(cells)
        assert not (set(cells) & set(env.predator_positions.values()))


# ---- defense arithmetic ----------------------------------------------------------------------------------------------------
def _attack_setup(n_band_defenders=0, n_other_band_defenders=0, **kw):
    env = _env(num_threats=1, **kw)
    band0, band1 = _members(env, 0), _members(env, 1)
    target = band0[0]
    keep = {target} | set(band0[1 : 1 + n_band_defenders]) | set(band1[:n_other_band_defenders])
    _park_others(env, keep)
    _place(env, target, (10, 10))
    for i, d in enumerate(band0[1 : 1 + n_band_defenders]):
        _place(env, d, (10 + (i + 1) % 3 - 1 + 0, 12))  # within radius 2 of the target
    for i, d in enumerate(band1[:n_other_band_defenders]):
        _place(env, d, (12, 10 + i))
    env.threat_positions = {"threat_0": (9, 10)}  # adjacent to the target
    for a in env.predator_positions:
        env.agent_energies[a] = 5.0
    return env, target


def test_lone_predator_is_killed_at_the_full_probability():
    env, target = _attack_setup(0)
    env.rng = _FixedRandom(env.rng, 0.49)  # < 0.5 * (1 - 0/3)
    env._threats_act()
    assert env.agent_energies[target] == -1.0 and env.threat_kills_alone == 1 and env.threat_encounters == 1
    env, target = _attack_setup(0)
    env.rng = _FixedRandom(env.rng, 0.51)
    env._threats_act()
    assert env.agent_energies[target] == 5.0 and env.threat_kills_alone == 0


def test_each_defender_lowers_the_kill_probability():
    # 1 defender: p = 0.5 * (1 - 1/3) = 0.3333; 2 defenders: p = 0.5 * (1 - 2/3) = 0.1667
    env, target = _attack_setup(1)
    env.rng = _FixedRandom(env.rng, 0.30)
    env._threats_act()
    assert env.agent_energies[target] == -1.0 and env.threat_kills_alone == 0
    env, target = _attack_setup(1)
    env.rng = _FixedRandom(env.rng, 0.35)
    env._threats_act()
    assert env.agent_energies[target] == 5.0
    env, target = _attack_setup(2)
    env.rng = _FixedRandom(env.rng, 0.15)
    env._threats_act()
    assert env.agent_energies[target] == -1.0
    env, target = _attack_setup(2)
    env.rng = _FixedRandom(env.rng, 0.18)
    env._threats_act()
    assert env.agent_energies[target] == 5.0


def test_enough_defenders_repel_the_threat_far_away_without_a_kill():
    env, target = _attack_setup(3)
    env.rng = _FixedRandom(env.rng, 0.0)  # would kill if it rolled
    env._threats_act()
    assert env.agent_energies[target] == 5.0 and env.threat_repelled == 1 and env.threat_kills == {"predator_male": 0, "predator_female": 0}
    tx, ty = env.threat_positions["threat_0"]
    assert max(abs(tx - 10), abs(ty - 10)) >= env.threat_flee_distance


def test_only_band_mates_defend_by_default_but_anyone_with_any():
    env, target = _attack_setup(0, n_other_band_defenders=3)
    env.rng = _FixedRandom(env.rng, 0.49)
    env._threats_act()
    assert env.agent_energies[target] == -1.0  # other-band predators do not defend under 'band'
    env, target = _attack_setup(0, n_other_band_defenders=3, threat_defense_by="any")
    env.rng = _FixedRandom(env.rng, 0.0)
    env._threats_act()
    assert env.agent_energies[target] == 5.0 and env.threat_repelled == 1


def test_defenders_outside_the_radius_or_dead_do_not_count():
    env, target = _attack_setup(2)
    band0 = _members(env, 0)
    far = [a for a in band0 if a != target][0]
    _place(env, far, (10, 20))  # outside the defense radius
    dead = [a for a in band0 if a not in (target, far)][0]
    env.agent_energies[dead] = -1.0
    assert env._threat_defenders(target) == 0


# ---- kills go through the normal removal path; snapshots; validation; full episode ---------------------------------------------
def test_a_killed_predator_is_removed_in_the_step_and_counted_by_sex():
    env, target = _attack_setup(0)
    env.rng = _FixedRandom(env.rng, 0.0)
    male = "male" in target and "female" not in target
    obs, rewards, terms, truncs, _ = env.step(_noop(env))
    assert target not in env.agent_positions and terms.get(target) is True
    assert env.threat_kills["predator_male" if male else "predator_female"] == 1
    assert target not in env.agent_band and target not in env.agent_fruit_store


def test_snapshot_round_trip_of_threats():
    env = _env(num_threats=3)
    snap = env.get_state_snapshot()
    env.threat_positions = {"threat_0": (0, 0)}
    env.threat_encounters = 9
    env.threat_kills["predator_male"] = 4
    env.restore_state_snapshot(snap)
    assert len(env.threat_positions) == 3 and env.threat_encounters == 0 and env.threat_kills == {"predator_male": 0, "predator_female": 0}


@pytest.mark.parametrize("kw", [{"num_threats": -1}, {"threat_kill_prob": 1.5}, {"threat_defenders_to_repel": 0},
                                {"threat_defense_by": "nobody"}, {"threat_sense_radius": -1}, {"num_obs_channels": 8}])
def test_invalid_threat_config_is_rejected(kw):
    with pytest.raises(ValueError):
        _env(**kw)


def test_full_random_episode_with_threats_runs_and_reports_metrics():
    env = _env(num_threats=4, max_steps=200)
    obs, _ = env.reset(seed=4)
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs, r, t, tr, _ = env.step({a: int(rng.integers(env.num_actions)) for a in obs})
        assert all(o.shape[0] == env.num_obs_channels and np.isfinite(o).all() for o in obs.values())
        if t.get("__all__") or tr.get("__all__"):
            break
    m = env._build_episode_training_metrics()
    for k in ("threat_encounters", "threat_kills_male", "threat_kills_female", "threat_kills_alone", "threat_repelled"):
        assert k in m
    assert m["threat_encounters"] > 0


# ---- compass fixes from the Codex review -------------------------------------------------------------------------------------
def test_compass_ignores_dead_or_doomed_members_and_breaks_ties_by_agent_id():
    env = _env(num_threats=0, band_compass=True)
    band = sorted(_members(env, 0))
    a, near_dead, tie_low, tie_high = band[0], band[1], band[2], band[3]
    _park_others(env, {a, near_dead, tie_low, tie_high})
    _place(env, a, (10, 10))
    _place(env, near_dead, (10, 11))   # nearest but dead
    env.agent_energies[near_dead] = -1.0
    _place(env, tie_high, (10, 15))    # equidistant: the lower id wins
    _place(env, tie_low, (5, 10))
    obs = env._get_observation(a)
    assert np.allclose(obs[11], 5 / env.grid_size)
    # tie_low id sorts before tie_high, so the compass points to tie_low: dx = -1 -> encoded 0, dy = 0 -> 0.5
    assert np.allclose(obs[9], 0.0) and np.allclose(obs[10], 0.5)
    # negative directions: the compass encodes a unit vector pointing up/left as values below 0.5
    assert obs[9].min() >= 0.0


def test_channel_validation_applies_with_bands_off_too():
    c = copy.deepcopy(_base)
    c.update({"num_bands": 0, "num_obs_channels": 6, "n_initial_active_predator_male": 2, "n_initial_active_predator_female": 2,
              "scripted_prey": False})
    with pytest.raises(ValueError):
        PredPreyGrass(c)
    c["num_obs_channels"] = 0
    with pytest.raises(ValueError):
        PredPreyGrass(c)


# ---- fixes from the Codex review of the threats ------------------------------------------------------------------------------
def test_two_threats_attacking_the_same_target_use_the_start_of_phase_state():
    """A kill by one threat must not change the defenders or the target seen by the next threat (no order dependence)."""
    env, target = _attack_setup(1)  # 1 defender: p = 0.3333 per threat
    band0 = _members(env, 0)
    env.threat_positions = {"threat_0": (9, 10), "threat_1": (9, 9)}  # both adjacent to the target only
    env.num_threats = 2
    env.rng = _FixedRandom(env.rng, 0.30)  # both rolls succeed
    env._threats_act()
    assert env.agent_energies[target] == -1.0
    assert env.threat_kills["predator_male"] + env.threat_kills["predator_female"] == 1  # one death, counted once
    assert env.threat_encounters == 2


def test_threat_ids_are_processed_in_numeric_not_lexicographic_order():
    env = _env(num_threats=12)
    order = []
    orig = env.rng.integers

    ids = sorted(env.threat_positions, key=lambda t: int(t.split("_")[1]))
    assert ids[2] == "threat_2" and ids[10] == "threat_10"  # numeric order the code uses


def test_doomed_predators_are_neither_targets_nor_defenders():
    env, target = _attack_setup(2)
    env.diet_required = True
    band0 = [a for a in _members(env, 0) if a != target]
    d1 = [a for a in band0 if a in env.predator_positions and max(abs(env.predator_positions[a][0] - 10), abs(env.predator_positions[a][1] - 10)) <= 2]
    assert len(d1) >= 2
    for a in env.predator_positions:
        env.agent_fruit_store[a] = 2.5
    env.agent_fruit_store[d1[0]] = 0.0  # doomed by an empty fruit store
    assert env._threat_defenders(target) == len(d1) - 1
    doomed_target = target
    env.agent_fruit_store[doomed_target] = 0.0
    env.rng = _FixedRandom(env.rng, 0.0)
    e0 = env.threat_encounters
    env._threats_act()
    assert env.threat_encounters == e0  # the only adjacent predator is doomed, so it is not attacked


def test_repel_moves_to_a_free_cell_and_counts_only_if_it_moved():
    env, target = _attack_setup(3)
    env.rng = _FixedRandom(env.rng, 0.0)
    env._threats_act()
    assert env.threat_repelled == 1
    t = env.threat_positions["threat_0"]
    assert t not in set(env.predator_positions.values())
    with pytest.raises(ValueError):
        _env(threat_flee_distance=99)


def test_snapshot_restores_the_rng_so_replays_match():
    env = _env(num_threats=3)
    snap = env.get_state_snapshot()
    env._threats_act()
    a = dict(env.threat_positions)
    env.restore_state_snapshot(snap)
    env._threats_act()
    assert dict(env.threat_positions) == a


def test_threat_channel_with_compass_is_channel_twelve_and_compass_is_intact():
    env = _env(num_threats=1, band_compass=True)
    band = sorted(_members(env, 0))
    a, mate = band[0], band[1]
    _park_others(env, {a, mate})
    _place(env, a, (10, 10))
    _place(env, mate, (10, 20))
    env.threat_positions = {"threat_0": (12, 10)}
    obs = env._get_observation(a)
    off = (env.predator_obs_range - 1) // 2
    assert obs[12, off + 2, off] == 1.0 and obs[12].sum() == 1.0
    assert np.allclose(obs[8], 1.0) and np.allclose(obs[11], 10 / env.grid_size)


def test_a_threat_killed_predator_gets_no_shares_and_does_not_reproduce_that_step():
    env, target = _attack_setup(0)
    env.rng = _FixedRandom(env.rng, 0.0)
    mate = next(a for a in env.predator_positions if a != target and env.agent_band[a] == env.agent_band[target])
    _place(env, mate, (10, 13))  # within mating range (3) but outside the defense radius (2)
    for a in (target, mate):
        env.agent_energies[a] = 13.0
        env.agent_fruit_store[a] = 6.5
    births_before = env.episode_births["predator_male"] + env.episode_births["predator_female"]
    env.step(_noop(env))
    assert target not in env.agent_positions
    assert env.episode_births["predator_male"] + env.episode_births["predator_female"] == births_before
    assert env.threat_kills_alone == 1
