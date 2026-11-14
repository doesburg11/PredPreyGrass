"""
Tests for the optional mate observation channel (config "mate_obs_channel", RESULTS.md Iteration 18).

Run explicitly:
    pytest predpreygrass/non_evolutionary/discrete_space/predator_sexual_reproduction/tests/test_mate_obs_channel.py -v
"""
import copy

import numpy as np
import pytest

from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.config_env import config_env as _base_config_env
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.fixed_predator_density_env import (
    FixedPredatorDensityEnv,
)
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.fixed_prey_density_env import FixedPreyDensityEnv
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.predpreygrass_rllib_env import PredPreyGrass

MATE_CH = 5


def _make_env(mate_channel=True, overrides=None, cls=PredPreyGrass):
    config = copy.deepcopy(_base_config_env)
    config.update(
        {
            "grid_size": 12,
            "n_initial_active_predator_male": 2,
            "n_initial_active_predator_female": 2,
            "n_initial_active_prey": 2,
            "initial_num_grass": 5,
            "initial_num_fruit": 5,
            "max_steps": 50,
            "mate_obs_channel": mate_channel,
        }
    )
    if overrides:
        config.update(overrides)
    env = cls(config)
    env.reset(seed=42)
    return env


def _move(env, agent, pos):
    """Move an agent consistently in the position dicts AND the predator/prey grid layer."""
    layer = 1 if "predator" in agent else 2
    old = env.agent_positions[agent]
    env.grid_world_state[layer, *old] = 0
    env.agent_positions[agent] = pos
    if "predator" in agent:
        env.predator_positions[agent] = pos
    else:
        env.prey_positions[agent] = pos
    env.grid_world_state[layer, *pos] = env.agent_energies[agent]


def _clear_predators(env):
    """Park every predator in the far corner region so it never lands inside a test window by accident."""
    corners = [(11, 11), (11, 10), (10, 11), (10, 10)]
    preds = [a for a in env.agents if "predator" in a]
    for a, p in zip(preds, corners):
        _move(env, a, p)
    return preds


def _pair(env):
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    env.agent_mate[male] = female
    env.agent_mate[female] = male
    return male, female


def test_default_is_off_and_shapes_unchanged():
    env = _make_env(mate_channel=False)
    assert env.observation_spaces["predator_male_0"].shape == (5, 7, 7)
    assert env.observation_spaces["predator_female_0"].shape == (5, 7, 7)
    assert env.observation_spaces["prey_0"].shape == (5, 9, 9)
    obs = env._get_observation(next(a for a in env.agents if "predator" in a))
    assert obs.shape == (5, 7, 7)
    assert _base_config_env["mate_obs_channel"] is False


def test_off_observation_identical_to_on_observation_first_five_channels():
    """The channel is purely additive: channels 0-4 are the same either way."""
    off, on = _make_env(mate_channel=False), _make_env(mate_channel=True)
    for agent in off.agents:
        a, b = off._get_observation(agent), on._get_observation(agent)
        np.testing.assert_array_equal(a, b[: a.shape[0]])


def test_on_shapes_and_spaces():
    env = _make_env()
    assert env.observation_spaces["predator_male_0"].shape == (6, 7, 7)
    assert env.observation_spaces["predator_female_0"].shape == (6, 7, 7)
    assert env.observation_spaces["prey_0"].shape == (5, 9, 9)
    for agent in env.agents:
        obs = env._get_observation(agent)
        assert env.observation_spaces[agent].contains(obs)


def test_no_mate_means_empty_channel():
    env = _make_env()
    for agent in env.agents:
        if "predator" in agent:
            assert env._get_observation(agent)[MATE_CH].sum() == 0.0


def test_visible_mate_marked_at_its_cell_for_both_partners():
    env = _make_env()
    _clear_predators(env)
    male, female = _pair(env)
    _move(env, male, (5, 5))
    _move(env, female, (6, 3))  # dx=+1, dy=-2
    obs_m = env._get_observation(male)
    obs_f = env._get_observation(female)
    assert obs_m[MATE_CH].sum() == 1.0 and obs_m[MATE_CH, 3 + 1, 3 - 2] == 1.0
    assert obs_f[MATE_CH].sum() == 1.0 and obs_f[MATE_CH, 3 - 1, 3 + 2] == 1.0
    # The mark sits on a cell the predator layer also shows as occupied.
    assert obs_m[1, 4, 1] > 0 and obs_f[1, 2, 5] > 0


def test_non_mate_predators_are_not_marked():
    env = _make_env()
    _clear_predators(env)
    male, female = _pair(env)
    other_female = [a for a in env.agents if "predator_female" in a and a != female][0]
    _move(env, male, (5, 5))
    _move(env, other_female, (5, 6))
    _move(env, female, (11, 0))  # far away
    obs = env._get_observation(male)
    assert obs[1, 3, 4] > 0  # the other female is visible in the predator layer
    assert obs[MATE_CH].sum() == 0.0


def test_mate_outside_window_not_marked():
    env = _make_env()
    _clear_predators(env)
    male, female = _pair(env)
    _move(env, male, (5, 5))
    _move(env, female, (5, 9))  # Chebyshev 4 > offset 3
    assert env._get_observation(male)[MATE_CH].sum() == 0.0
    _move(env, female, (8, 8))  # Chebyshev exactly 3: on the window edge
    obs = env._get_observation(male)
    assert obs[MATE_CH, 6, 6] == 1.0


def test_mark_correct_at_grid_border():
    env = _make_env()
    _clear_predators(env)
    male, female = _pair(env)
    _move(env, male, (0, 0))
    _move(env, female, (1, 2))
    obs = env._get_observation(male)
    assert obs[MATE_CH, 4, 5] == 1.0
    assert obs[MATE_CH].sum() == 1.0
    assert obs[0, 4, 5] == 0.0  # inside the grid, not border


def test_dead_mate_not_marked():
    env = _make_env()
    _clear_predators(env)
    male, female = _pair(env)
    _move(env, male, (5, 5))
    _move(env, female, (5, 6))
    env._remove_agent(female)
    env.grid_world_state[1, 5, 6] = 0
    assert env.agent_mate[male] == female  # stale bond stays; the channel must still be empty
    assert env._get_observation(male)[MATE_CH].sum() == 0.0


def test_real_reproduction_marks_partners_in_returned_observations():
    env = _make_env(overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1})
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    env.agent_energies[male] = env.predator_creation_energy_threshold + 1.0
    env.agent_energies[female] = env.predator_creation_energy_threshold + 1.0
    _move(env, male, (5, 5))
    _move(env, female, (6, 6))
    obs, _, _, _, _ = env.step({a: env.noop_action_id for a in env.agents})
    assert env.agent_mate[male] == female
    assert obs[male][MATE_CH, 4, 4] == 1.0 and obs[male][MATE_CH].sum() == 1.0
    assert obs[female][MATE_CH, 2, 2] == 1.0 and obs[female][MATE_CH].sum() == 1.0
    child = [a for a in obs if "predator" in a and a not in (male, female)][0]
    assert obs[child][MATE_CH].sum() == 0.0  # a newborn has no mate


def test_reset_clears_marks():
    env = _make_env()
    _clear_predators(env)
    male, female = _pair(env)
    _move(env, male, (5, 5))
    _move(env, female, (5, 6))
    assert env._get_observation(male)[MATE_CH].sum() == 1.0
    obs, _ = env.reset(seed=1)
    assert all(o[MATE_CH].sum() == 0.0 for a, o in obs.items() if "predator" in a)


def test_blank_mode_has_six_channels_and_never_marks():
    env = _make_env(mate_channel="blank")
    assert env.observation_spaces["predator_male_0"].shape == (6, 7, 7)
    assert env.observation_spaces["prey_0"].shape == (5, 9, 9)
    assert env.mate_obs_channel is False and env.mate_obs_channel_mode == "blank"
    _clear_predators(env)
    male, female = _pair(env)
    _move(env, male, (5, 5))
    _move(env, female, (5, 6))
    obs = env._get_observation(male)
    assert obs.shape == (6, 7, 7) and obs[MATE_CH].sum() == 0.0
    ref = _make_env(mate_channel=False)
    _clear_predators(ref)
    m2, f2 = _pair(ref)
    _move(ref, m2, (5, 5))
    _move(ref, f2, (5, 6))
    np.testing.assert_array_equal(obs[:5], ref._get_observation(m2))


@pytest.mark.parametrize("value,mode", [(False, "off"), ("off", "off"), (True, "mate"), ("mate", "mate"), ("blank", "blank")])
def test_accepted_modes(value, mode):
    assert _make_env(mate_channel=value).mate_obs_channel_mode == mode


def test_remating_moves_the_mark_to_the_new_partner():
    env = _make_env(overrides={"n_initial_active_predator_male": 2, "n_initial_active_predator_female": 1})
    m1, m2 = [a for a in env.agents if "predator_male" in a]
    female = next(a for a in env.agents if "predator_female" in a)
    env.agent_mate[m1] = female
    env.agent_mate[female] = m1
    env.has_reproduced.update({m1, female})
    _move(env, m1, (3, 3))
    env.agent_energies[m1] = 1.0  # ineligible
    _move(env, female, (5, 5))
    _move(env, m2, (5, 6))
    env.agent_energies[m2] = env.predator_creation_energy_threshold + 1.0
    env.agent_energies[female] = env.predator_creation_energy_threshold + 1.0
    assert env._get_observation(m1)[MATE_CH].sum() == 1.0  # old bond visible before
    obs, _, _, _, _ = env.step({a: env.noop_action_id for a in env.agents})
    assert env.agent_mate[female] == m2
    assert obs[m1][MATE_CH].sum() == 0.0  # abandoned male: no mark although the female is in view
    assert obs[m1][1].sum() > 0
    assert obs[m2][MATE_CH, 3 + 5 - 5, 3 + 5 - 6] == 1.0 and obs[m2][MATE_CH].sum() == 1.0
    fx, fy = 3 + 5 - 5, 3 + 6 - 5
    assert obs[female][MATE_CH, fx, fy] == 1.0 and obs[female][MATE_CH].sum() == 1.0


def test_culled_mate_unmarked_in_survivors_same_step_observation():
    env = _make_env(
        overrides={
            "prey_density_floor": 2,
            "n_possible_prey": 5000,
            "predator_density_target": 3,
            "n_initial_active_predator_male": 2,
            "n_initial_active_predator_female": 2,
        },
        cls=FixedPredatorDensityEnv,
    )
    _clear_predators(env)
    male, female = _pair(env)
    _move(env, male, (5, 5))
    _move(env, female, (5, 6))
    for a in env.agents:
        if "predator" in a:
            env.agent_energies[a] = 5.0
    # Force the cull to pick the female: rng.choice is the only draw in the cull.
    alive = [a for a in env.agent_positions if "predator_male" in a or "predator_female" in a]
    idx = alive.index(female)

    class _Pick:
        def __init__(self, real):
            self.real = real

        def choice(self, n, size=None, replace=True):
            return np.array([idx])

        def __getattr__(self, name):
            return getattr(self.real, name)

    real_rng = env.rng
    env.rng = _Pick(real_rng)
    try:
        obs, _, term, _, _ = env.step({a: env.noop_action_id for a in env.agents})
    finally:
        env.rng = real_rng
    assert term.get(female) is True and female not in env.predator_positions
    assert obs[male][MATE_CH].sum() == 0.0


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_rejects_non_bool(bad):
    with pytest.raises(ValueError):
        _make_env(mate_channel=bad)


@pytest.mark.parametrize(
    "cls,overrides",
    [
        (FixedPreyDensityEnv, {"prey_density_floor": 2, "n_possible_prey": 5000}),
        (FixedPredatorDensityEnv, {"prey_density_floor": 2, "n_possible_prey": 5000, "predator_density_target": 4}),
    ],
)
def test_subclasses_produce_valid_observations_over_an_episode(cls, overrides):
    env = _make_env(overrides=overrides, cls=cls)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=3)
    for _ in range(40):
        for agent, o in obs.items():
            # Shape, not Box.contains: a predator that starves this step gets a terminal observation with its
            # own (slightly negative) energy in the predator layer -- pre-existing, independent of this channel.
            assert o.shape == env.observation_spaces[agent].shape, agent
            if "predator" in agent:
                assert set(np.unique(o[MATE_CH])) <= {0.0, 1.0}
                assert o[MATE_CH].sum() <= 1.0
                cell = env._visible_mate_cell(agent) if agent in env.agent_positions else None
                if o[MATE_CH].sum() == 1.0 and cell is not None:
                    assert o[1, cell[0], cell[1]] > 0
        actions = {a: int(rng.integers(env.num_actions)) for a in obs}
        obs, _, term, trunc, _ = env.step(actions)
        if term.get("__all__") or trunc.get("__all__"):
            break


def test_random_rollout_marks_only_live_mates_inside_window():
    """Property check over a random-policy episode with frequent mating."""
    env = _make_env(
        overrides={
            "grid_size": 10,
            "n_initial_active_predator_male": 4,
            "n_initial_active_predator_female": 4,
            "predator_creation_energy_threshold": 5.5,
            "initial_energy_predator_male": 6.0,
            "initial_energy_predator_female": 6.0,
            "max_steps": 200,
        }
    )
    rng = np.random.default_rng(1)
    obs, _ = env.reset(seed=7)
    marked = 0
    for _ in range(200):
        for agent, o in obs.items():
            if "predator" not in agent or agent not in env.agent_positions:
                continue
            mate = env.agent_mate.get(agent)
            expect = env._visible_mate_cell(agent)
            if expect is None:
                assert o[MATE_CH].sum() == 0.0
            else:
                assert mate in env.predator_positions
                assert o[MATE_CH].sum() == 1.0 and o[MATE_CH, expect[0], expect[1]] == 1.0
                marked += 1
        actions = {a: int(rng.integers(env.num_actions)) for a in obs}
        obs, _, term, trunc, _ = env.step(actions)
        if term.get("__all__") or trunc.get("__all__"):
            break
    assert marked > 0, "the rollout never produced a visible mate; test setup too weak"
