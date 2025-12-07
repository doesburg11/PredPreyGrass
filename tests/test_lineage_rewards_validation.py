import copy

import pytest

from predpreygrass.rllib.lineage_rewards.config.config_env_lineage_rewards import config_env
from predpreygrass.rllib.lineage_rewards.predpreygrass_rllib_env import PredPreyGrass


def _make_test_env(overrides=None):
    """Return a tiny lineage_rewards env for deterministic unit tests."""
    cfg = copy.deepcopy(config_env)
    cfg.update(
        {
            "n_initial_active_type_1_predator": 1,
            "n_initial_active_type_2_predator": 0,
            "n_initial_active_type_1_prey": 1,
            "n_initial_active_type_2_prey": 0,
            "n_possible_type_2_predators": 0,
            "n_possible_type_2_prey": 0,
            "initial_num_grass": 4,
            # Allow terminated agents to remain in step outputs for testing expectations
            "strict_rllib_output": False,
        }
    )
    if overrides:
        cfg.update(overrides)
    return PredPreyGrass(cfg)


def test_lineage_reward_triggers_on_descendant_gain():
    env = _make_test_env()
    env.reset(seed=321)
    env.rewards = {}

    parent = next(agent for agent in env.agents if agent.startswith("type_1_predator"))
    child_id = env._alloc_new_id("predator", 1)
    assert child_id is not None

    env.agents.append(child_id)
    env._register_new_agent(child_id, parent_agent_id=parent)
    env.agent_positions[child_id] = env.agent_positions[parent]
    env.predator_positions[child_id] = env.agent_positions[parent]
    env.agent_energies[child_id] = env.initial_energy_predator
    env.agent_live_offspring_ids[parent].append(child_id)
    env.agent_offspring_counts[parent] += 1
    parent_record = env.agent_stats_live[parent]
    parent_record["offspring_count"] += 1

    env._apply_lineage_survival_rewards()

    coeff = env._get_type_specific("lineage_reward_coeff", parent)
    assert env.rewards[parent] == pytest.approx(coeff)
    assert parent_record["lineage_reward_total"] == pytest.approx(coeff)
    reward_events = env.agent_event_log[parent].get("reward_events", [])
    assert reward_events and reward_events[-1]["lineage_reward"] == pytest.approx(coeff)


def test_agent_emits_max_age_termination_and_logs_event():
    env = _make_test_env()
    env.reset(seed=123)

    target = next(agent for agent in env.agents if agent.startswith("type_1_predator"))
    limit = env._get_max_age_limit(target)
    assert limit is not None and limit > 1
    env.agent_ages[target] = limit - 1

    actions = {agent: 0 for agent in env.agents}
    _, _, terminations, _, infos = env.step(actions)

    assert terminations[target] is True
    assert infos[target]["terminated_due_to_age"] is True
    assert target not in env.agent_stats_live
    completed = env.agent_stats_completed[target]
    assert completed["death_cause"] == "max_age"
    assert completed["age_expired_step"] == completed["death_step"]


def test_juvenile_predator_blocked_from_live_prey():
    env = _make_test_env(
        overrides={
            "carcass_only_predator_age": {"type_1_predator": 10, "type_2_predator": None},
        }
    )
    env.reset(seed=999)
    env.rewards = {}

    predator = next(agent for agent in env.agents if agent.startswith("type_1_predator"))
    prey = next(agent for agent in env.agents if agent.startswith("type_1_prey"))

    env.agent_positions[predator] = tuple(env.agent_positions[prey])
    env.predator_positions[predator] = tuple(env.agent_positions[prey])
    env.agent_ages[predator] = 0

    env._handle_predator_engagement(predator)

    step_reward = env._get_type_specific("reward_predator_step", predator)
    assert env.rewards[predator] == pytest.approx(step_reward)
    assert predator not in env.agents_just_ate
    assert prey not in env.dead_prey
    assert env.agent_stats_live[predator]["carcass_only_blocks"] == 1
    assert env._pending_infos[predator]["carcass_only_live_prey_blocked"] is True


def test_predator_can_eat_live_prey_after_window():
    env = _make_test_env(
        overrides={
            "carcass_only_predator_age": {"type_1_predator": 2, "type_2_predator": None},
        }
    )
    env.reset(seed=77)
    env.rewards = {}

    predator = next(agent for agent in env.agents if agent.startswith("type_1_predator"))
    prey = next(agent for agent in env.agents if agent.startswith("type_1_prey"))
    env.agent_positions[predator] = tuple(env.agent_positions[prey])
    env.predator_positions[predator] = tuple(env.agent_positions[prey])
    env.agent_ages[predator] = 5

    env._handle_predator_engagement(predator)

    catch_reward = env._get_type_specific("reward_predator_catch_prey", predator)
    assert env.rewards[predator] == pytest.approx(catch_reward)
    assert predator in env.agents_just_ate
    assert env.agent_stats_live[predator]["times_ate"] == 1
    assert prey in env.dead_prey


def test_founder_predator_starts_at_carcass_threshold():
    window = 12
    env = _make_test_env(
        overrides={
            "carcass_only_predator_age": {"type_1_predator": window, "type_2_predator": None},
        }
    )
    env.reset(seed=222)
    env.rewards = {}

    predator = next(agent for agent in env.agents if agent.startswith("type_1_predator"))
    prey = next(agent for agent in env.agents if agent.startswith("type_1_prey"))

    assert env.agent_ages[predator] == window

    env.agent_positions[predator] = tuple(env.agent_positions[prey])
    env.predator_positions[predator] = tuple(env.agent_positions[prey])

    env._handle_predator_engagement(predator)

    assert predator in env.agents_just_ate
    assert env.rewards[predator] == pytest.approx(env._get_type_specific("reward_predator_catch_prey", predator))
