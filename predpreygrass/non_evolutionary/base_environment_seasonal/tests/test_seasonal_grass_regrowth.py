import copy

import pytest

from predpreygrass.non_evolutionary.base_environment_seasonal.config_env import config_env
from predpreygrass.non_evolutionary.base_environment_seasonal.predpreygrass_rllib_env import PredPreyGrass


def _make_env(**overrides):
    config = copy.deepcopy(config_env)
    config.update(overrides)
    return PredPreyGrass(config)


def test_season_multiplier_phase_boundaries():
    env = _make_env(season_length_steps=5, season_high_multiplier=1.5, season_low_multiplier=0.5)

    # First phase (steps 0-4): high multiplier.
    for step in (0, 1, 4):
        env.current_step = step
        assert env._current_season_multiplier() == 1.5

    # Second phase (steps 5-9): low multiplier.
    for step in (5, 6, 9):
        env.current_step = step
        assert env._current_season_multiplier() == 0.5

    # Third phase (steps 10-14): back to high -- confirms the cycle repeats.
    env.current_step = 10
    assert env._current_season_multiplier() == 1.5
    env.current_step = 14
    assert env._current_season_multiplier() == 1.5


def test_season_disabled_reproduces_flat_baseline():
    # With both multipliers pinned to 1.0, the seasonal mechanism must be a
    # no-op regardless of season_length_steps -- a regression guard so this
    # feature can never silently change behavior when "disabled".
    env = _make_env(season_length_steps=3, season_high_multiplier=1.0, season_low_multiplier=1.0)
    for step in range(0, 20):
        env.current_step = step
        assert env._current_season_multiplier() == 1.0


def _stay_actions(env):
    return {agent: 4 for agent in env.agents}  # action 4 == (0, 0), i.e. stay in place


def test_grass_regrows_faster_in_abundant_phase_than_scarce_phase():
    season_length_steps = 3
    high_multiplier = 1.5
    low_multiplier = 0.5
    base_gain = config_env["energy_gain_per_step_grass"]

    env = _make_env(
        season_length_steps=season_length_steps,
        season_high_multiplier=high_multiplier,
        season_low_multiplier=low_multiplier,
    )
    env.reset(seed=0)

    tracked_grass = next(iter(env.grass_positions))
    env.grass_energies[tracked_grass] = 0.0

    # Steps 0, 1, 2 fall in the abundant phase (season_length_steps=3).
    for _ in range(season_length_steps):
        env.step(_stay_actions(env))
    energy_after_abundant_phase = env.grass_energies[tracked_grass]

    # Steps 3, 4, 5 fall in the scarce phase.
    for _ in range(season_length_steps):
        env.step(_stay_actions(env))
    growth_in_scarce_phase = env.grass_energies[tracked_grass] - energy_after_abundant_phase

    assert energy_after_abundant_phase == pytest.approx(season_length_steps * base_gain * high_multiplier)
    assert growth_in_scarce_phase == pytest.approx(season_length_steps * base_gain * low_multiplier)
    assert energy_after_abundant_phase > growth_in_scarce_phase
