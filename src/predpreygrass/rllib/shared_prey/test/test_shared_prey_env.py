import copy

import pytest

from predpreygrass.rllib.shared_prey.config.config_env_shared_prey import config_env
from predpreygrass.rllib.shared_prey.predpreygrass_rllib_env import PredPreyGrass


def _make_env(overrides=None):
    cfg = copy.deepcopy(config_env)
    cfg.update(
        {
            "grid_size": 5,
            "n_initial_active_type_1_predator": 1,
            "n_initial_active_type_2_predator": 0,
            "n_initial_active_type_1_prey": 1,
            "n_initial_active_type_2_prey": 0,
            "n_possible_type_2_predators": 0,
            "n_possible_type_2_prey": 0,
            "initial_num_grass": 0,
            "strict_rllib_output": False,
        }
    )
    if overrides:
        cfg.update(overrides)
    return PredPreyGrass(cfg)


def test_reset_returns_observations_for_all_agents():
    env = _make_env()
    obs, info = env.reset(seed=42)

    assert set(obs.keys()) == set(env.agents)
    assert all(isinstance(v, (list, tuple, dict)) or hasattr(v, "shape") for v in obs.values())
    assert info == {}


def test_team_capture_terminates_prey():
    env = _make_env()
    env.reset(seed=123)
    env.rewards = {}

    predator = next(a for a in env.agents if "predator" in a)
    prey = next(a for a in env.agents if "prey" in a)

    # Place predator adjacent to prey (within Moore neighborhood)
    env.agent_positions[predator] = tuple(env.agent_positions[prey])
    env.predator_positions[predator] = env.agent_positions[predator]

    env._handle_team_capture(prey)

    assert env.terminations[prey] is True
    assert env.active_num_prey == 0
    assert predator in env.agents_just_ate
    catch_reward = env._get_type_specific("reward_predator_catch_prey", predator)
    assert env.rewards[predator] == pytest.approx(catch_reward)
