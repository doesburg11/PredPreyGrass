"""Tests for prey_energy_yield (richer prey)."""
import copy

import pytest

from predpreygrass.non_evolutionary.predator_bands.config_env import config_env as _base
from predpreygrass.non_evolutionary.predator_bands.predpreygrass_rllib_env import PredPreyGrass


class _Fixed:
    def __init__(self, real, value):
        self._real, self._value = real, value

    def random(self, *a, **k):
        return self._value

    def __getattr__(self, name):
        return getattr(self._real, name)


def _catch(**kw):
    c = copy.deepcopy(_base)
    c.update(kw)
    env = PredPreyGrass(c)
    env.reset(seed=3)
    pid, prey, pos = "predator_male_test", "prey_test", (0, 0)
    env.agent_positions[pid] = pos
    env.predator_positions[pid] = pos
    env.agent_energies[pid] = 5.0
    env.cumulative_rewards[pid] = 0
    env.agent_positions[prey] = pos
    env.prey_positions[prey] = pos
    env.agent_energies[prey] = 3.0
    env.cumulative_rewards[prey] = 0
    env.rng = _Fixed(env.rng, 0.0)  # the hunt succeeds
    outcome, _, _ = env._resolve_hunting_attempt(pid, pos, prey, 0.5, 0.3, {}, {}, {}, {})
    assert outcome == "success"
    return env.agent_energies[pid] - 5.0


def test_default_yield_is_unchanged_and_yield_scales_the_gain():
    assert _catch() == pytest.approx(3.0)
    assert _catch(prey_energy_yield=2.5) == pytest.approx(7.5)


def test_negative_yield_is_rejected():
    c = copy.deepcopy(_base)
    c["prey_energy_yield"] = -1.0
    with pytest.raises(ValueError):
        PredPreyGrass(c)
