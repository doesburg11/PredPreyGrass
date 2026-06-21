import copy

import numpy as np
import pytest

from predpreygrass.eco_evolutionary.config.config_env_eco_evolutionary import config_env
from predpreygrass.eco_evolutionary.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.eco_evolutionary.utils.genome import Genome


def _make_test_env(overrides=None):
    """Return a tiny eco_evolutionary env for deterministic unit tests."""
    cfg = copy.deepcopy(config_env)
    cfg.update(
        {
            "n_initial_active_predators": 1,
            "n_initial_active_prey": 1,
            "n_possible_predators": 8,
            "n_possible_prey": 8,
            "initial_num_grass": 4,
        }
    )
    if overrides:
        cfg.update(overrides)
    return PredPreyGrass(cfg)


def _place_agent(env, agent, position):
    """Move one agent in both bookkeeping maps and grid-state channels."""
    old_position = env.agent_positions[agent]
    if agent.startswith("predator"):
        env.grid_world_state[0, *old_position] = 0.0
        env.predator_positions[agent] = position
        env.grid_world_state[0, *position] = env.agent_energies[agent]
    else:
        env.grid_world_state[1, *old_position] = 0.0
        env.prey_positions[agent] = position
        env.grid_world_state[1, *position] = env.agent_energies[agent]
    env.agent_positions[agent] = position


def test_lineage_reward_triggers_on_descendant_gain():
    env = _make_test_env()
    env.reset(seed=321)
    env.rewards = {}

    parent = next(agent for agent in env.agents if agent.startswith("predator"))
    child_id = env._alloc_new_id("predator")
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

    coeff = env._get_role_specific("lineage_reward_coeff", parent)
    assert env.rewards[parent] == pytest.approx(coeff)
    assert parent_record["lineage_reward_total"] == pytest.approx(coeff)
    reward_events = env.agent_event_log[parent].get("reward_events", [])
    assert reward_events and reward_events[-1]["lineage_reward"] == pytest.approx(coeff)


def test_agent_emits_max_age_termination_and_logs_event():
    env = _make_test_env()
    env.reset(seed=123)

    target = next(agent for agent in env.agents if agent.startswith("predator"))
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


def test_rllib_output_preserves_terminal_reward_without_terminal_observation():
    env = _make_test_env(
        overrides={
            "predator_creation_energy_threshold": 999.0,
            "prey_creation_energy_threshold": 999.0,
        }
    )
    env.reset(seed=124)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    prey = next(agent for agent in env.agents if agent.startswith("prey"))
    _place_agent(env, predator, (5, 5))
    _place_agent(env, prey, (5, 5))

    stay_action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (0, 0))
    observations, rewards, terminations, truncations, infos = env.step(
        {agent: stay_action for agent in env.agents}
    )

    assert prey not in observations
    assert predator not in observations
    assert rewards[prey] == pytest.approx(env._get_role_specific("penalty_prey_caught", prey))
    assert terminations[prey] is True
    assert truncations[prey] is False
    assert terminations[predator] is True
    assert truncations[predator] is False
    assert terminations["__all__"] is True
    assert truncations["__all__"] is False
    assert "final_cumulative_reward" in infos[predator]


def test_time_limit_truncates_without_next_observations():
    env = _make_test_env(
        overrides={
            "max_steps": 1,
            "predator_creation_energy_threshold": 999.0,
            "prey_creation_energy_threshold": 999.0,
        }
    )
    env.reset(seed=125)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    prey = next(agent for agent in env.agents if agent.startswith("prey"))
    _place_agent(env, predator, (1, 1))
    _place_agent(env, prey, (env.grid_size - 2, env.grid_size - 2))

    stay_action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (0, 0))
    observations, _, terminations, truncations, _ = env.step({agent: stay_action for agent in env.agents})

    assert observations == {}
    assert terminations[predator] is False
    assert terminations[prey] is False
    assert truncations[predator] is True
    assert truncations[prey] is True
    assert terminations["__all__"] is False
    assert truncations["__all__"] is True


def test_juvenile_predator_blocked_from_live_prey():
    env = _make_test_env(
        overrides={
            "carcass_only_predator_age": {"predator": 10},
        }
    )
    env.reset(seed=999)
    env.rewards = {}

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    prey = next(agent for agent in env.agents if agent.startswith("prey"))

    env.agent_positions[predator] = tuple(env.agent_positions[prey])
    env.predator_positions[predator] = tuple(env.agent_positions[prey])
    env.agent_ages[predator] = 0

    env._handle_predator_engagement(predator)

    step_reward = env._get_role_specific("reward_predator_step", predator)
    assert env.rewards[predator] == pytest.approx(step_reward)
    assert predator not in env.agents_just_ate
    assert prey not in env.dead_prey
    assert env.agent_stats_live[predator]["carcass_only_blocks"] == 1
    assert env._pending_infos[predator]["carcass_only_live_prey_blocked"] is True


def test_predator_can_eat_live_prey_after_window():
    env = _make_test_env(
        overrides={
            "carcass_only_predator_age": {"predator": 2},
        }
    )
    env.reset(seed=77)
    env.rewards = {}

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    prey = next(agent for agent in env.agents if agent.startswith("prey"))
    env.agent_positions[predator] = tuple(env.agent_positions[prey])
    env.predator_positions[predator] = tuple(env.agent_positions[prey])
    env.agent_ages[predator] = 5

    env._handle_predator_engagement(predator)

    catch_reward = env._get_role_specific("reward_predator_catch_prey", predator)
    assert env.rewards[predator] == pytest.approx(catch_reward)
    assert predator in env.agents_just_ate
    assert env.agent_stats_live[predator]["times_ate"] == 1
    assert prey in env.dead_prey


def test_founder_predator_starts_at_carcass_threshold():
    window = 12
    env = _make_test_env(
        overrides={
            "carcass_only_predator_age": {"predator": window},
        }
    )
    env.reset(seed=222)
    env.rewards = {}

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    prey = next(agent for agent in env.agents if agent.startswith("prey"))

    assert env.agent_ages[predator] == window

    env.agent_positions[predator] = tuple(env.agent_positions[prey])
    env.predator_positions[predator] = tuple(env.agent_positions[prey])

    env._handle_predator_engagement(predator)

    assert predator in env.agents_just_ate
    assert env.rewards[predator] == pytest.approx(env._get_role_specific("reward_predator_catch_prey", predator))


def test_founders_receive_genomes_in_event_logs():
    env = _make_test_env()
    env.reset(seed=515)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))

    assert predator in env.agent_genomes
    assert env.agent_stats_live[predator]["genome"] == env.agent_genomes[predator].to_dict()
    assert env.agent_event_log[predator]["genome"] == env.agent_genomes[predator].to_dict()


def test_offspring_inherits_mutated_parent_genome_and_energy_fraction():
    env = _make_test_env(
        overrides={
            "predator_creation_energy_threshold": 10.0,
            "founder_genome": {
                "predator": {
                    "movement_cost_multiplier_mean": 1.0,
                    "movement_cost_multiplier_std": 0.0,
                    "reproduction_threshold_multiplier_mean": 1.0,
                    "reproduction_threshold_multiplier_std": 0.0,
                    "offspring_energy_fraction_mean": 0.5,
                    "offspring_energy_fraction_std": 0.0,
                },
            },
            "genome_mutation": {"rate": 1.0, "std": 0.01},
        }
    )
    env.reset(seed=616)
    env.rewards = {}

    parent = next(agent for agent in env.agents if agent.startswith("predator"))
    parent_energy = 20.0
    env.agent_energies[parent] = parent_energy

    env._handle_predator_reproduction(parent)

    children = env.agent_live_offspring_ids[parent]
    assert len(children) == 1
    child = children[0]
    assert child in env.agent_genomes
    assert env.agent_parents[child] == parent
    assert env.agent_energies[child] == pytest.approx(parent_energy * env.agent_genomes[parent].offspring_energy_fraction)
    assert env.agent_energies[parent] == pytest.approx(parent_energy - env.agent_energies[child])
    assert env.agent_genomes[child].to_dict() != env.agent_genomes[parent].to_dict()


def test_reproduction_threshold_uses_parent_genome_multiplier():
    env = _make_test_env(
        overrides={
            "predator_creation_energy_threshold": 10.0,
            "founder_genome": {
                "predator": {
                    "movement_cost_multiplier_mean": 1.0,
                    "movement_cost_multiplier_std": 0.0,
                    "reproduction_threshold_multiplier_mean": 1.5,
                    "reproduction_threshold_multiplier_std": 0.0,
                    "offspring_energy_fraction_mean": 0.5,
                    "offspring_energy_fraction_std": 0.0,
                },
            },
            "genome_mutation": {"rate": 0.0, "std": 0.0},
        }
    )
    env.reset(seed=717)
    env.rewards = {}

    parent = next(agent for agent in env.agents if agent.startswith("predator"))
    env.agent_energies[parent] = 12.0
    env._handle_predator_reproduction(parent)
    assert env.agent_live_offspring_ids[parent] == []

    env.agent_energies[parent] = 16.0
    env._handle_predator_reproduction(parent)
    assert len(env.agent_live_offspring_ids[parent]) == 1


def test_action_space_uses_extended_moore_neighborhood():
    env = _make_test_env()
    env.reset(seed=727)
    assert env.action_spaces["predator_0"].n == 25
    assert (2, 0) in env.action_to_move_tuple_agents.values()


def test_observation_edges_are_clipped_and_zero_padded():
    env = _make_test_env()
    env.reset(seed=808)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    env.agent_positions[predator] = (0, 0)
    env.predator_positions[predator] = (0, 0)

    obs = env._get_observation(predator)

    assert obs.shape == (3, env.predator_obs_range, env.predator_obs_range)
    offset = (env.predator_obs_range - 1) // 2
    assert np.all(obs[:, :offset, :] == 0.0)
    assert np.all(obs[:, :, :offset] == 0.0)


def test_slow_speed_clips_distance_two_move_to_distance_one():
    env = _make_test_env(
        overrides={
            "speed_distance_threshold": 1.5,
        }
    )
    env.reset(seed=818)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    env.agent_positions[predator] = (10, 10)
    env.predator_positions[predator] = (10, 10)
    env.agent_genomes[predator] = Genome(
        speed=1.0,
        movement_cost_multiplier=1.0,
        reproduction_threshold_multiplier=1.0,
        offspring_energy_fraction=0.5,
    )
    action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (2, 0))

    assert env._get_move(predator, action) == (11, 10)


def test_fast_speed_allows_distance_two_move():
    env = _make_test_env(
        overrides={
            "speed_distance_threshold": 1.5,
        }
    )
    env.reset(seed=919)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    env.agent_positions[predator] = (10, 10)
    env.predator_positions[predator] = (10, 10)
    env.agent_genomes[predator] = Genome(
        speed=1.6,
        movement_cost_multiplier=1.0,
        reproduction_threshold_multiplier=1.0,
        offspring_energy_fraction=0.5,
    )
    action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (2, 0))

    assert env._get_move(predator, action) == (12, 10)


def test_fast_speed_pays_configured_energy_cost_multiplier():
    env = _make_test_env(
        overrides={
            "energy_loss_per_step_predator": 0.2,
            "fast_speed_cost_multiplier": 1.8,
        }
    )
    env.reset(seed=1020)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    env.agent_genomes[predator] = Genome(
        speed=1.6,
        movement_cost_multiplier=1.0,
        reproduction_threshold_multiplier=1.0,
        offspring_energy_fraction=0.5,
    )
    start_energy = 10.0
    env.agent_energies[predator] = start_energy

    env._apply_time_step_update()

    assert env.agent_energies[predator] == pytest.approx(start_energy - 0.2 * 1.8)
