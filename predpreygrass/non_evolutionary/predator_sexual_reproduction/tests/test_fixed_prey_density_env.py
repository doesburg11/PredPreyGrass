"""
Validation tests for FixedPreyDensityEnv (../fixed_prey_density_env.py): the prey-replacement
matched-ecology intervention. Modeled on test_predator_sexual_reproduction_validation.py's
_make_test_env + narrowly-scoped test_* pattern.

Run explicitly (not auto-discovered by the repo's pytest testpaths):
    pytest predpreygrass/non_evolutionary/predator_sexual_reproduction/tests/ -v
"""
import copy

import pytest

from predpreygrass.non_evolutionary.predator_sexual_reproduction.config_env import config_env as _base_config_env
from predpreygrass.non_evolutionary.predator_sexual_reproduction.fixed_prey_density_env import FixedPreyDensityEnv


def _make_env(overrides=None):
    config = copy.deepcopy(_base_config_env)
    config.update(
        {
            "grid_size": 10,
            "n_initial_active_predator_male": 2,
            "n_initial_active_predator_female": 2,
            "n_initial_active_prey": 2,
            "initial_num_grass": 5,
            "initial_num_fruit": 5,
            "max_steps": 50,
            "prey_density_floor": 5,
        }
    )
    if overrides:
        config.update(overrides)
    env = FixedPreyDensityEnv(config)
    env.reset(seed=42)
    return env


def _noop_actions(env):
    return {agent: env.noop_action_id for agent in env.agents}


def test_replenishes_up_to_the_floor_when_below_it():
    env = _make_env(overrides={"n_initial_active_prey": 2, "prey_density_floor": 5})
    assert env.current_num_prey == 2

    env.step(_noop_actions(env))

    assert env.current_num_prey == 5
    prey_agents = [a for a in env.agents if a.startswith("prey")]
    assert len(prey_agents) == 5


def test_no_replenishment_when_already_at_or_above_the_floor():
    env = _make_env(overrides={"n_initial_active_prey": 8, "prey_density_floor": 5})
    assert env.current_num_prey == 8

    env.step(_noop_actions(env))

    assert env.current_num_prey == 8  # untouched -- the floor is a minimum, not a cap


def test_replacement_prey_pays_no_reward_and_is_not_counted_as_a_birth():
    env = _make_env(overrides={"n_initial_active_prey": 2, "prey_density_floor": 5})
    original_ids = set(a for a in env.agents if a.startswith("prey"))

    _, rewards, _, _, _ = env.step(_noop_actions(env))

    new_ids = [a for a in env.agents if a.startswith("prey") and a not in original_ids]
    assert len(new_ids) == 3
    for agent in new_ids:
        assert rewards[agent] == 0.0
        assert env.cumulative_rewards[agent] == 0.0
    assert env.episode_births["prey"] == 0  # not a real (energy-threshold-triggered) birth


def test_max_steps_truncation_is_untouched_no_replenishment_attempted():
    env = _make_env(overrides={"n_initial_active_prey": 2, "prey_density_floor": 5, "max_steps": 1})
    env.step(_noop_actions(env))  # current_step 0 -> 1; not yet truncated, replenishes normally
    prey_before_truncation = env.current_num_prey

    _, _, terminations, truncations, _ = env.step(_noop_actions(env))  # current_step 1 >= max_steps 1

    assert truncations.get("__all__") is True
    assert env.current_num_prey == prey_before_truncation  # override returned early; no replenishment


def test_predator_extinction_still_ends_the_episode_prey_not_replenished():
    env = _make_env(
        overrides={
            "n_initial_active_predator_male": 1,
            "n_initial_active_predator_female": 2,
            "n_initial_active_prey": 2,
            "prey_density_floor": 5,
        }
    )
    male = next(a for a in env.agents if "predator_male" in a)
    env.agent_energies[male] = 0.0  # starves this step (Step 1's homeostatic cost makes it <= 0)
    prey_before = env.current_num_prey

    _, _, terminations, _, _ = env.step(_noop_actions(env))

    assert env.current_num_predator_male == 0
    assert terminations.get("__all__") is True  # predator extinction ends the episode, not masked
    assert env.current_num_prey == prey_before  # guarded: no replenishment once predators are extinct


def test_pool_exhaustion_does_not_crash_and_leaves_prey_below_floor():
    env = _make_env(
        overrides={"n_initial_active_prey": 2, "n_possible_prey": 2, "prey_density_floor": 5}
    )
    assert env._next_prey_idx >= env.n_possible_prey  # no room to grow at all

    env.step(_noop_actions(env))  # must not raise

    assert env.current_num_prey == 2  # unchanged: pool was already exhausted


@pytest.mark.parametrize("bad", [-1, -5])
def test_rejects_a_negative_prey_density_floor(bad):
    with pytest.raises(ValueError, match="prey_density_floor"):
        _make_env(overrides={"prey_density_floor": bad})


def test_total_prey_extinction_is_reversed_but_the_dead_prey_stays_terminated():
    """Codex-review regression: the base class already sets terminations['__all__'] = True when
    current_num_prey hits 0 (predpreygrass_rllib_env.py's Step 6); this class must flip it back to
    False once replenishment restores the population, without un-terminating the prey that actually
    died (a live agent can't have been terminated last step and still be in this step's dict)."""
    env = _make_env(overrides={"n_initial_active_prey": 1, "prey_density_floor": 5})
    prey = next(a for a in env.agents if a.startswith("prey"))
    env.agent_energies[prey] = 0.0  # starves this step (Step 1's homeostatic cost makes it <= 0)

    _, _, terminations, _, _ = env.step(_noop_actions(env))

    assert env.current_num_prey == 5  # 0 (extinct) -> replenished to the floor
    assert terminations[prey] is True  # the prey that actually died stays terminated
    assert terminations.get("__all__") is False  # but the episode itself does not end


def test_replenishment_refreshes_every_live_agents_returned_observation():
    """Codex-review regression: the base class generates final observations (Step 6) BEFORE this
    override's replenishment runs, so without a refresh, every OTHER already-observed agent's
    returned observation would be stale -- it would not show the newly spawned prey at all.

    grid_size=4 with predator_obs_range=7 (offset (7-1)//2=3) guarantees every predator's window
    covers the WHOLE grid regardless of its own position (grid_size - 1 <= offset), so the new prey
    is deterministically within the male's window wherever it happens to spawn -- no reliance on
    where `_find_available_spawn_position`'s random fallback happens to land it."""
    env = _make_env(
        overrides={
            "grid_size": 4,
            "n_initial_active_predator_male": 1,
            "n_initial_active_predator_female": 1,
            "n_initial_active_prey": 1,
            "initial_num_grass": 2,
            "initial_num_fruit": 2,
            "prey_density_floor": 3,
        }
    )
    male = next(a for a in env.agents if "predator_male" in a)

    observations, _, _, _, _ = env.step(_noop_actions(env))

    # A fresh observation, computed straight after the step, must match what was returned -- if the
    # returned one were stale (pre-replenishment), these would differ, since the new prey are
    # guaranteed to fall within the male's (whole-grid) window.
    fresh = env._get_observation(male)
    import numpy as np

    assert np.array_equal(observations[male], fresh)
