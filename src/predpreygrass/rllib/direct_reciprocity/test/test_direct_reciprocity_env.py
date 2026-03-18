import copy

import pytest

from predpreygrass.rllib.direct_reciprocity.config.config_env_direct_reciprocity import config_env
from predpreygrass.rllib.direct_reciprocity.predpreygrass_rllib_env import PredPreyGrass


def _make_env(overrides=None):
    cfg = copy.deepcopy(config_env)
    cfg.update(
        {
            "grid_size": 5,
            "n_initial_active_type_1_predator": 2,
            "n_initial_active_type_2_predator": 0,
            "n_initial_active_type_1_prey": 1,
            "n_initial_active_type_2_prey": 0,
            "n_possible_type_1_predators": 10,
            "n_possible_type_2_predators": 0,
            "n_possible_type_1_prey": 10,
            "n_possible_type_2_prey": 0,
            "initial_num_grass": 0,
            "strict_rllib_output": False,
            "max_steps": 50,
        }
    )
    if overrides:
        cfg.update(overrides)
    return PredPreyGrass(cfg)


def test_predator_action_space_is_multidiscrete():
    env = _make_env()
    env.reset(seed=42)
    predator = next(a for a in env.agents if "predator" in a)

    assert env.action_spaces[predator].nvec.tolist()[-1] == 2


def test_share_action_transfers_energy_and_updates_trust():
    env = _make_env({"share_fraction": 0.25, "trust_positive_delta": 0.2})
    env.reset(seed=123)

    predators = sorted(a for a in env.agents if "predator" in a)
    prey = next(a for a in env.agents if "prey" in a)
    captor, recipient = predators

    env.agent_positions[captor] = (2, 2)
    env.predator_positions[captor] = (2, 2)
    env.agent_positions[recipient] = (2, 3)
    env.predator_positions[recipient] = (2, 3)
    env.agent_positions[prey] = (3, 2)
    env.prey_positions[prey] = (3, 2)

    env.agent_energies[captor] = 6.0
    env.agent_energies[recipient] = 4.0
    env.agent_energies[prey] = 4.0
    env.predator_share_intent[captor] = True
    env.grid_world_state[1, 2, 2] = 6.0
    env.grid_world_state[1, 2, 3] = 4.0
    env.grid_world_state[2, 3, 2] = 4.0

    trust_before = env._get_trust_value(recipient, captor)

    env._handle_team_capture(prey)

    assert env.agent_energies[captor] == pytest.approx(6.0 + 3.0)
    assert env.agent_energies[recipient] == pytest.approx(4.0 + 1.0)
    assert env._get_trust_value(recipient, captor) > trust_before
    assert env.terminations[prey] is True


def test_refusing_to_share_reduces_neighbor_trust():
    env = _make_env({"share_fraction": 0.25, "trust_negative_delta": 0.2})
    env.reset(seed=123)

    predators = sorted(a for a in env.agents if "predator" in a)
    prey = next(a for a in env.agents if "prey" in a)
    captor, recipient = predators

    env.predator_share_intent[captor] = False
    env.agent_positions[captor] = (2, 2)
    env.predator_positions[captor] = (2, 2)
    env.agent_positions[recipient] = (2, 3)
    env.predator_positions[recipient] = (2, 3)
    env.agent_positions[prey] = (3, 2)
    env.prey_positions[prey] = (3, 2)

    env.agent_energies[captor] = 6.0
    env.agent_energies[recipient] = 4.0
    env.agent_energies[prey] = 4.0
    env.grid_world_state[1, 2, 2] = 6.0
    env.grid_world_state[1, 2, 3] = 4.0
    env.grid_world_state[2, 3, 2] = 4.0

    trust_before = env._get_trust_value(recipient, captor)

    env._handle_team_capture(prey)

    assert env.agent_energies[recipient] == pytest.approx(4.0)
    assert env._get_trust_value(recipient, captor) < trust_before
