"""
Tests for analysis_env.py: rollout-based analysis must evaluate a run in the environment it was trained in
(prey floor / density target), not the plain base env. See analysis_env.py's docstring for the measured
mismatch that motivated it.

Run explicitly:
    pytest predpreygrass/non_evolutionary/predator_sexual_reproduction/tests/ -v
"""
import copy

from predpreygrass.non_evolutionary.predator_sexual_reproduction.analysis_env import env_class_for, make_env
from predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_energy_sources import (
    InstrumentedMixin,
    make_instrumented_env,
)
from predpreygrass.non_evolutionary.predator_sexual_reproduction.config_env import config_env as _base
from predpreygrass.non_evolutionary.predator_sexual_reproduction.fixed_predator_density_env import (
    FixedPredatorDensityEnv,
)
from predpreygrass.non_evolutionary.predator_sexual_reproduction.fixed_prey_density_env import FixedPreyDensityEnv
from predpreygrass.non_evolutionary.predator_sexual_reproduction.predpreygrass_rllib_env import PredPreyGrass


def _cfg(**over):
    c = copy.deepcopy(_base)
    c.update({"grid_size": 10, "n_initial_active_predator_male": 2, "n_initial_active_predator_female": 2,
              "n_initial_active_prey": 6, "initial_num_grass": 5, "initial_num_fruit": 5, "max_steps": 30})
    c.update(over)
    return c


def test_class_selection_from_saved_config_keys():
    plain = _cfg()
    assert env_class_for(plain) is PredPreyGrass  # no prey_density_floor key: a pre-Iteration-11 run
    assert env_class_for(_cfg(prey_density_floor=20)) is FixedPreyDensityEnv
    assert env_class_for(_cfg(prey_density_floor=20, predator_population_cap=26)) is FixedPreyDensityEnv
    assert env_class_for(_cfg(prey_density_floor=20, predator_density_target=None)) is FixedPreyDensityEnv
    assert env_class_for(_cfg(prey_density_floor=20, predator_density_target=26)) is FixedPredatorDensityEnv


def test_make_env_builds_the_trained_env_and_its_mechanism_runs():
    env = make_env(_cfg(prey_density_floor=8, predator_density_target=3))
    assert isinstance(env, FixedPredatorDensityEnv)
    env.reset(seed=1)
    env.step({a: env.noop_action_id for a in env.agents})
    assert env.current_num_predator_male + env.current_num_predator_female == 3  # culled 4 -> 3
    assert env.current_num_prey >= 8  # prey floor still active underneath


def test_instrumented_env_sits_on_the_trained_env_class_and_tracks():
    env = make_instrumented_env(_cfg(prey_density_floor=8, predator_density_target=3))
    assert isinstance(env, FixedPredatorDensityEnv) and isinstance(env, InstrumentedMixin)
    env.reset(seed=1)
    assert env.energy_from_prey == {} and env.energy_from_fruit == {}  # fresh per reset
    for _ in range(3):
        env.step({a: env.noop_action_id for a in env.agents})
    assert env.current_num_predator_male + env.current_num_predator_female == 3

    plain = make_instrumented_env(_cfg())
    assert isinstance(plain, PredPreyGrass) and not isinstance(plain, FixedPreyDensityEnv)
    assert type(env) is not type(plain)
    # class cache: same trained class -> same instrumented class object
    assert type(make_instrumented_env(_cfg(prey_density_floor=8, predator_density_target=3))) is type(env)


# ---- instrumentation actually fires on each generated class (Codex review: the tests above only checked
# inheritance/reset/population, so a skipped or double-counted hook would have passed) ----
import numpy as np
import pytest


class _FixedRandom:
    def __init__(self, real, value):
        self._real, self._value = real, value

    def random(self, *a, **k):
        return self._value

    def __getattr__(self, name):
        return getattr(self._real, name)


_CLASS_CONFIGS = {
    "base": {},
    "prey_floor": {"prey_density_floor": 8},
    "density_target": {"prey_density_floor": 8, "predator_density_target": 4},  # 4 initial predators: no cull
}


@pytest.mark.parametrize("kind", list(_CLASS_CONFIGS))
def test_instrumented_hunt_and_fruit_are_tracked_once_on_every_class(kind):
    env = make_instrumented_env(_cfg(male_gift_donation_rate=0.0, parent_offspring_share_rate=0.0, **_CLASS_CONFIGS[kind]))
    env.reset(seed=42)
    male = next(a for a in env.agents if "predator_male" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))
    fruit = next(iter(env.fruit_positions))
    pos = (5, 5)
    for d1, d2, key in ((env.agent_positions, env.predator_positions, male), (env.agent_positions, env.prey_positions, prey)):
        d1[key] = pos
        d2[key] = pos
    env.fruit_positions[fruit] = pos
    env.fruit_energies[fruit] = 2.0
    prey_energy = env.agent_energies[prey] - env.homeostatic_energy_cost_per_step_prey  # after Step 1
    env.rng = _FixedRandom(env.rng, 0.0)  # force the hunting-success band
    env.step({a: env.noop_action_id for a in env.agents})
    assert env.n_prey_caught[male] == 1 and env.energy_from_prey[male] == pytest.approx(prey_energy)
    assert env.n_fruit_eaten[male] == 1 and env.energy_from_fruit[male] == pytest.approx(2.0)


def test_instrumented_gift_and_care_transfers_are_exact():
    env = make_instrumented_env(
        _cfg(prey_density_floor=8, predator_density_target=4, male_gift_donation_rate=0.3, parent_offspring_share_rate=0.2)
    )
    env.reset(seed=42)
    male, mate = "predator_male_0", "predator_female_0"
    child = "predator_female_1"
    prey = next(a for a in env.agents if a.startswith("prey"))
    env.agent_mate[male], env.agent_mate[mate] = mate, male
    env.agent_parents[child] = (male, mate)
    layout = {male: (5, 5), mate: (5, 6), child: (6, 5)}
    for a, p in layout.items():
        env.agent_positions[a] = p
        env.predator_positions[a] = p
    env.agent_positions[prey] = (5, 5)
    env.prey_positions[prey] = (5, 5)
    env.rng = _FixedRandom(env.rng, 0.0)
    env.step({a: env.noop_action_id for a in env.agents})
    gained = env.energy_from_prey[male]
    assert gained > 0
    assert env.gift_received[mate] == pytest.approx(0.3 * gained)
    assert env.gift_given[male] == pytest.approx(0.3 * gained)
    assert env.care_received[child] > 0
    assert env.care_received[child] == pytest.approx(env.care_given[male])


def test_replacement_predators_start_with_zero_log():
    env = make_instrumented_env(_cfg(prey_density_floor=8, predator_density_target=6))
    env.reset(seed=1)
    before = set(env.agents)
    env.step({a: env.noop_action_id for a in env.agents})
    replacements = [a for a in env.agents if "predator" in a and a not in before]
    assert len(replacements) == 2  # 4 -> 6
    for a in replacements:
        assert env.energy_from_prey.get(a, 0.0) == 0.0 and env.n_prey_caught.get(a, 0) == 0
        assert env.gift_received.get(a, 0.0) == 0.0 and env.care_received.get(a, 0.0) == 0.0


def test_legacy_config_without_either_key_selects_the_base_env():
    c = _cfg()
    c.pop("prey_density_floor", None)
    c.pop("predator_density_target", None)
    assert env_class_for(c) is PredPreyGrass
