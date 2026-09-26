"""Tests for the mammoth (group big game) option of predator_bands, default off: placement, single-band parties, the party-size
arithmetic, equal split as meat with energy-proportional rewards, deaths on failure, blocking by a larger party of another band,
wandering, respawn, the observation channel, snapshots and validation."""
import copy

import numpy as np
import pytest

from predpreygrass.non_evolutionary.predator_bands.config_env import config_env as _base
from predpreygrass.non_evolutionary.predator_bands.predpreygrass_rllib_env import PredPreyGrass


class _Seq:
    """rng whose random() returns the given values in order (then 0.99); everything else is the real generator."""

    def __init__(self, real, values):
        self._real, self._values = real, list(values)

    def random(self, *a, **k):
        return self._values.pop(0) if self._values else 0.99

    def __getattr__(self, name):
        return getattr(self._real, name)


def _env(**kw):
    c = copy.deepcopy(_base)
    c.update({"max_steps": 100, "diet_required": False, "num_mammoths": 1, "mammoth_move_prob": 0.0, "reward_predator_per_energy": 0.5})
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


def _park(env, keep):
    i = 0
    for a in list(env.predator_positions):
        if a not in keep:
            _place(env, a, (24 - i // 8, 16 + i % 8))
            i += 1


def _noop(env):
    return {a: env.noop_action_id for a in env.agents}


def _setup(band_members=1, other_members=0, other_band=1, mammoth_pos=(10, 10), **kw):
    """The attacker (band 0) stands on the mammoth; `band_members` more band-mates and `other_members` band-`other_band` predators stand
    next to it. Everything else is parked far away."""
    env = _env(**kw)
    band0, other = _members(env, 0), _members(env, other_band)
    attacker = band0[0]
    mates = band0[1 : 1 + band_members]
    strangers = other[:other_members]
    _park(env, {attacker, *mates, *strangers})
    env.mammoth_positions = {"mammoth_0": mammoth_pos}
    _place(env, attacker, mammoth_pos)
    ring = [(mammoth_pos[0] + dx, mammoth_pos[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    for a, cell in zip(mates + strangers, ring):
        _place(env, a, cell)
    for a in env.predator_positions:
        env.agent_energies[a] = 5.0
        env.agent_fruit_store[a] = 2.5
    return env, attacker, mates, strangers


# ---- off / placement / channel ----------------------------------------------------------------------------------------------
def test_mammoths_are_off_by_default_and_add_a_channel_when_on():
    c = copy.deepcopy(_base)
    env = PredPreyGrass(c)
    env.reset(seed=1)
    assert env.num_mammoths == 0 and env.mammoth_positions == {} and env.num_obs_channels == 8
    assert _env().num_obs_channels == 9
    assert _env(num_threats=1, band_compass=True).num_obs_channels == 14


def test_mammoths_are_placed_on_free_cells():
    env = _env(num_mammoths=5)
    cells = list(env.mammoth_positions.values())
    assert len(set(cells)) == 5 and not (set(cells) & set(env.predator_positions.values()))


def test_mammoth_channel_value_is_the_energy_at_the_relative_cell():
    env = _env(num_threats=1, band_compass=True)
    a = _members(env, 0)[0]
    _place(env, a, (10, 10))
    env.mammoth_positions = {"mammoth_0": (10, 12)}
    obs = env._get_observation(a)
    off = (env.predator_obs_range - 1) // 2
    ch = 8 + 4 + 1  # after the compass and threat channels
    assert obs[ch, off, off + 2] == env.mammoth_energy and obs[ch].sum() == env.mammoth_energy


# ---- party and outcome arithmetic ---------------------------------------------------------------------------------------------
def test_party_is_the_attackers_band_mates_within_the_radius_only():
    env, attacker, mates, strangers = _setup(band_members=2, other_members=2)
    viable = env._viable_predators()
    party = env._mammoth_party(attacker, viable)
    assert sorted(party) == sorted([attacker, *mates])
    far_mate = [a for a in _members(env, 0) if a not in party][0]
    _place(env, far_mate, (10, 14))
    assert far_mate not in env._mammoth_party(attacker, viable)
    env.agent_energies[mates[0]] = -1.0  # a dead member is not in the party
    assert mates[0] not in env._mammoth_party(attacker, env._viable_predators())


@pytest.mark.parametrize("n_mates,p", [(0, 0.02), (1, 0.25), (2, 0.6), (3, 0.9), (4, 0.9)])
def test_success_probability_by_party_size(n_mates, p):
    env, attacker, mates, _ = _setup(band_members=n_mates)
    env.rng = _Seq(env.rng, [p - 1e-6])  # success roll just below p
    env._mammoths_act()
    assert sum(env.mammoth_kills) == 1 and env.mammoth_kills[min(n_mates + 1, 4) - 1] == 1
    env, attacker, mates, _ = _setup(band_members=n_mates)
    env.rng = _Seq(env.rng, [p + 1e-6, *([0.99] * 10)])  # just above p: failure, nobody dies
    env._mammoths_act()
    assert sum(env.mammoth_kills) == 0 and env.mammoth_attempts[min(n_mates + 1, 4) - 1] == 1


def test_a_kill_splits_the_energy_equally_as_meat_and_credits_rewards_and_respawns():
    env, attacker, mates, strangers = _setup(band_members=2, other_members=1)
    party = [attacker, *mates]
    e0 = {a: env.agent_energies[a] for a in env.predator_positions}
    f0 = {a: env.agent_fruit_store[a] for a in env.predator_positions}
    env.rng = _Seq(env.rng, [0.0])
    env._mammoths_act()
    share = env.mammoth_energy / 3
    for a in party:
        assert env.agent_energies[a] == pytest.approx(e0[a] + share)
        assert env.agent_fruit_store[a] == f0[a]  # meat: the fruit store does not change
        assert env._pending_rewards[a] == pytest.approx(0.5 * share)
    for a in strangers:
        assert env.agent_energies[a] == e0[a] and a not in env._pending_rewards
    assert env.mammoth_energy_distributed == pytest.approx(20.0)
    assert "mammoth_0" not in env.mammoth_positions and env.mammoth_respawn_at["mammoth_0"] == env.current_step + env.mammoth_respawn_steps


def test_the_mammoth_respawns_at_a_free_cell_after_the_delay():
    env, attacker, mates, _ = _setup(band_members=2, mammoth_respawn_steps=5)
    env.rng = _Seq(env.rng, [0.0])
    env._mammoths_act()
    env.current_step += 4
    env._mammoths_act()
    assert env.mammoth_positions == {}
    env.current_step += 1
    env._mammoths_act()
    pos = env.mammoth_positions["mammoth_0"]
    assert pos not in set(env.predator_positions.values()) and env.mammoth_respawn_at == {}


def test_failure_kills_party_members_with_the_size_specific_probability_and_spares_other_bands():
    env, attacker, mates, strangers = _setup(band_members=1, other_members=1)  # party of 2: death chance 0.15 each
    party = [attacker, *mates]
    env.rng = _Seq(env.rng, [0.99, 0.10, 0.20])  # failed roll, member 1 dies (< 0.15), member 2 survives (>= 0.15)
    env._mammoths_act()
    dead = [a for a in party if env.agent_energies[a] == -1.0]
    assert len(dead) == 1 and env.mammoth_party_deaths[1] == 1
    assert all(env.agent_energies[a] == 5.0 for a in strangers)
    assert "mammoth_0" in env.mammoth_positions  # a failed hunt does not kill the mammoth


def test_a_larger_party_of_another_band_blocks_the_attempt_and_a_tie_does_not():
    env, attacker, mates, strangers = _setup(band_members=1, other_members=3)  # 2 vs 3
    env.rng = _Seq(env.rng, [0.0, 0.0])
    env._mammoths_act()
    assert env.mammoth_blocked == 1 and sum(env.mammoth_attempts) == 0
    assert all(env.agent_energies[a] == 5.0 for a in env.predator_positions)
    env, attacker, mates, strangers = _setup(band_members=1, other_members=2)  # 2 vs 2: ties favour the attacker
    env.rng = _Seq(env.rng, [0.0, 0.0])
    env._mammoths_act()
    assert env.mammoth_blocked == 0 and sum(env.mammoth_kills) == 1


# ---- movement, rewards through step(), snapshot, validation -----------------------------------------------------------------
def test_mammoths_wander_but_never_onto_predators_or_each_other():
    env = _env(num_mammoths=4, mammoth_move_prob=1.0)
    for _ in range(60):
        env._mammoths_act()
        cells = list(env.mammoth_positions.values())
        assert len(set(cells)) == len(cells) and not (set(cells) & set(env.predator_positions.values()))
        assert all(0 <= x < env.grid_size and 0 <= y < env.grid_size for x, y in cells)
    still = _env(num_mammoths=2, mammoth_move_prob=0.0)
    before = dict(still.mammoth_positions)
    still._mammoths_act()
    assert still.mammoth_positions == before


def test_step_returns_the_share_as_reward_and_removes_killed_party_members():
    env, attacker, mates, _ = _setup(band_members=2)
    env.rng = _Seq(env.rng, [0.0])
    obs, rewards, terms, truncs, _ = env.step(_noop(env))
    share = env.mammoth_energy / 3
    for a in [attacker, *mates]:
        assert rewards[a] >= 0.5 * share - 1e-9
    env, attacker, mates, _ = _setup(band_members=0)
    env.rng = _Seq(env.rng, [0.99, 0.0])  # lone hunter fails and dies (death chance 0.30)
    obs, rewards, terms, truncs, _ = env.step(_noop(env))
    assert attacker not in env.agent_positions and terms.get(attacker) is True
    assert env.mammoth_party_deaths[0] == 1


def test_snapshot_round_trip_of_mammoths():
    env, attacker, mates, _ = _setup(band_members=2)
    snap = env.get_state_snapshot()
    env.rng = _Seq(env.rng, [0.0])
    env._mammoths_act()
    assert sum(env.mammoth_kills) == 1
    env.restore_state_snapshot(snap)
    assert "mammoth_0" in env.mammoth_positions and sum(env.mammoth_kills) == 0 and env.mammoth_respawn_at == {}


@pytest.mark.parametrize("kw", [{"num_mammoths": -1}, {"mammoth_energy": 0}, {"mammoth_move_prob": 2.0},
                                {"mammoth_success_by_party": [0.1, 0.2, 0.3]}, {"mammoth_death_by_party": [0.1, 0.2, 0.3, 1.5]},
                                {"mammoth_respawn_steps": -1}, {"num_obs_channels": 8}])
def test_invalid_mammoth_config_is_rejected(kw):
    with pytest.raises(ValueError):
        _env(**kw)


def test_full_random_episode_with_mammoths_runs_and_reports_metrics():
    env = _env(num_mammoths=3, mammoth_move_prob=0.3, max_steps=200)
    obs, _ = env.reset(seed=4)
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs, r, t, tr, _ = env.step({a: int(rng.integers(env.num_actions)) for a in obs})
        assert all(o.shape[0] == env.num_obs_channels and np.isfinite(o).all() for o in obs.values())
        if t.get("__all__") or tr.get("__all__"):
            break
    m = env._build_episode_training_metrics()
    for k in ("mammoth_attempts_n1", "mammoth_kills_n4", "mammoth_party_deaths_n2", "mammoth_blocked", "mammoth_energy_distributed"):
        assert k in m


# ---- fixes from the Codex review of the mammoths ---------------------------------------------------------------------------------
def test_a_member_killed_at_one_mammoth_is_not_revived_or_recounted_at_another():
    env = _env(num_mammoths=2, mammoth_death_by_party=[0.30, 1.0, 0.05, 0.02])
    band0 = _members(env, 0)
    a1, a2, shared = band0[0], band0[1], band0[2]
    _park(env, {a1, a2, shared})
    env.mammoth_positions = {"mammoth_0": (10, 10), "mammoth_1": (10, 12)}
    _place(env, a1, (10, 10))
    _place(env, a2, (10, 12))
    _place(env, shared, (11, 11))  # adjacent to both mammoths
    for a in env.predator_positions:
        env.agent_energies[a] = 5.0
    # mammoth_0: party {a1, shared} (a2 is 2 cells away), fails, both die (death chance 1.0). mammoth_1: a2 hunts; `shared` is dead.
    env.rng = _Seq(env.rng, [0.99, 0.5, 0.5, 0.99, 0.0])  # fail, two death rolls, m0 wander check, m1 success
    env._mammoths_act()
    assert env.agent_energies[a1] == -1.0 and env.agent_energies[shared] == -1.0  # `shared` was not revived by the second success
    assert env.mammoth_party_deaths[1] == 2 and env.mammoth_kills[0] == 1
    assert env.agent_energies[a2] == pytest.approx(5.0 + env.mammoth_energy)  # a party of one at mammoth_1 (dead members excluded)


def test_terminal_agents_keep_the_share_they_earned_and_pending_rewards_do_not_leak():
    env, attacker, mates, _ = _setup(band_members=2)
    env.rng = _Seq(env.rng, [0.0])
    env._mammoths_act()  # success: a share is pending for each member
    env.agent_energies[mates[0]] = -1.0  # ... and one member dies in the same step (e.g. at another mammoth)
    obs, rewards, terms, truncs, _ = env.step(_noop(env))
    assert terms.get(mates[0]) is True and rewards[mates[0]] == pytest.approx(0.5 * env.mammoth_energy / 3)
    assert env._pending_rewards == {}


def test_a_hunt_is_resolved_before_the_mammoth_wanders_and_a_share_counts_as_eating():
    env, attacker, mates, _ = _setup(band_members=2, mammoth_move_prob=1.0, reward_predator_step=-0.1)
    env.rng = _Seq(env.rng, [0.0])  # success first: the hunt comes before any wandering
    env._mammoths_act()
    assert sum(env.mammoth_kills) == 1  # the mammoth had not moved away
    env, attacker, mates, _ = _setup(band_members=2, reward_predator_step=-0.1)
    env.rng = _Seq(env.rng, [0.0])
    obs, rewards, terms, truncs, _ = env.step(_noop(env))
    share = env.mammoth_energy / 3
    assert rewards[attacker] == pytest.approx(0.5 * share)  # no -0.1 no-forage step reward on top of the share


def test_mammoths_need_bands_and_the_grid_must_have_room():
    with pytest.raises(ValueError):
        _env(num_bands=0, n_initial_active_predator_male=2, n_initial_active_predator_female=2, scripted_prey=False)
    with pytest.raises(ValueError):
        _env(grid_size=16, num_mammoths=20)  # the layout (with the mammoths) does not fit
