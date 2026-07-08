import copy

import pytest

from predpreygrass.non_evolutionary.direct_reciprocity.config.config_env_direct_reciprocity import config_env
from predpreygrass.non_evolutionary.direct_reciprocity.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.direct_reciprocity.utils.reciprocity_metrics import (
    aggregate_direct_reciprocity_metrics,
    aggregate_share_decisions_from_event_log,
)


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
    captor_events = env.agent_event_log[captor]["eating_events"]
    assert captor_events[-1]["share_opportunity"] is True
    assert captor_events[-1]["share_candidates"] == [recipient]
    assert captor_events[-1]["share_recipient"] == recipient


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


def test_refusing_to_share_only_updates_selected_target():
    env = _make_env(
        {
            "grid_size": 6,
            "n_initial_active_type_1_predator": 3,
            "n_initial_active_type_1_prey": 1,
            "n_possible_type_1_predators": 10,
            "n_possible_type_1_prey": 10,
            "trust_negative_delta": 0.2,
        }
    )
    env.reset(seed=123)

    predators = sorted(a for a in env.agents if "predator" in a)
    prey = next(a for a in env.agents if "prey" in a)
    captor = predators[0]
    other_predators = predators[1:]
    selected_target = max(other_predators)
    bystander = min(other_predators)

    env.predator_share_intent[captor] = False
    env.agent_positions[captor] = (2, 2)
    env.predator_positions[captor] = (2, 2)
    env.agent_positions[selected_target] = (2, 3)
    env.predator_positions[selected_target] = (2, 3)
    env.agent_positions[bystander] = (3, 3)
    env.predator_positions[bystander] = (3, 3)
    env.agent_positions[prey] = (3, 2)
    env.prey_positions[prey] = (3, 2)

    env.agent_energies[captor] = 6.0
    env.agent_energies[selected_target] = 4.0
    env.agent_energies[bystander] = 4.0
    env.agent_energies[prey] = 4.0
    env.grid_world_state[1, 2, 2] = 6.0
    env.grid_world_state[1, 2, 3] = 4.0
    env.grid_world_state[1, 3, 3] = 4.0
    env.grid_world_state[2, 3, 2] = 4.0

    trust_before_target = env._get_trust_value(selected_target, captor)
    trust_before_bystander = env._get_trust_value(bystander, captor)

    env._handle_team_capture(prey)

    assert env._get_trust_value(selected_target, captor) < trust_before_target
    assert env._get_trust_value(bystander, captor) == pytest.approx(trust_before_bystander)
    captor_events = env.agent_event_log[captor]["eating_events"]
    assert captor_events[-1]["share_recipient"] == selected_target


def test_predator_reproduction_respects_cooldown():
    env = _make_env(
        {
            "n_initial_active_type_1_predator": 1,
            "n_initial_active_type_1_prey": 1,
            "predator_reproduction_cooldown_steps": 2,
        }
    )
    env.reset(seed=7)

    predator = next(a for a in env.agents if "predator" in a)
    env.agent_energies[predator] = 20.0

    assert env._handle_predator_reproduction(predator) is True
    assert env.agent_offspring_counts[predator] == 1
    assert env._predator_reproduction_cooldown_remaining(predator) == 2

    env.agent_energies[predator] = 20.0
    env.current_step += 1

    assert env._handle_predator_reproduction(predator) is False
    assert env.agent_offspring_counts[predator] == 1
    assert env._predator_reproduction_cooldown_remaining(predator) == 1

    env.agent_energies[predator] = 20.0
    env.current_step += 1

    assert env._handle_predator_reproduction(predator) is True
    assert env.agent_offspring_counts[predator] == 2


def test_reciprocity_metrics_capture_returned_shares():
    event_log = {
        "type_1_predator_0": {
            "eating_events": [
                {
                    "t": 1,
                    "share_opportunity": True,
                    "share_candidates": ["type_1_predator_1"],
                    "shared_energy": 1.0,
                    "share_recipient": "type_1_predator_1",
                },
                {
                    "t": 3,
                    "share_opportunity": True,
                    "share_candidates": ["type_1_predator_1"],
                    "shared_energy": 1.0,
                    "share_recipient": "type_1_predator_1",
                },
            ]
        },
        "type_1_predator_1": {
            "eating_events": [
                {
                    "t": 2,
                    "share_opportunity": True,
                    "share_candidates": ["type_1_predator_0"],
                    "shared_energy": 1.0,
                    "share_recipient": "type_1_predator_0",
                }
            ]
        },
    }

    share_metrics = aggregate_share_decisions_from_event_log(event_log)
    reciprocity_metrics = aggregate_direct_reciprocity_metrics(event_log)

    assert share_metrics["share_opportunities"] == 3
    assert share_metrics["share_events"] == 3
    assert share_metrics["share_refusals"] == 0
    assert share_metrics["share_decision_rate"] == pytest.approx(1.0)

    assert reciprocity_metrics["opportunities_with_prior_helper_available"] == 2
    assert reciprocity_metrics["opportunities_without_prior_helper_available"] == 1
    assert reciprocity_metrics["shares_when_prior_helper_available"] == 2
    assert reciprocity_metrics["shares_when_no_prior_helper_available"] == 1
    assert reciprocity_metrics["share_rate_when_prior_helper_available"] == pytest.approx(1.0)
    assert reciprocity_metrics["share_rate_when_no_prior_helper_available"] == pytest.approx(1.0)
    assert reciprocity_metrics["share_to_prior_helper_events"] == 2
    assert reciprocity_metrics["share_to_non_helper_events"] == 1
    assert reciprocity_metrics["share_to_prior_helper_rate"] == pytest.approx(2.0 / 3.0)
    assert reciprocity_metrics["reciprocal_share_rate"] == pytest.approx(2.0 / 3.0)
    assert reciprocity_metrics["reciprocal_dyads"] == 1
