import copy

import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_cadence.config.config_env_eco_evolutionary import config_env
from predpreygrass.evolutionary.eco_evolutionary_cadence.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.evolutionary.eco_evolutionary_cadence.utils.episode_return_callback import EpisodeReturn
from predpreygrass.evolutionary.eco_evolutionary_cadence.utils.genome import Genome


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


class _FakeMetricsLogger:
    def __init__(self):
        self.values = {}

    def log_value(self, name, value):
        self.values[name] = value


class _FakeEpisode:
    length = 3

    def get_return(self):
        return 2.0

    def get_rewards(self):
        return {
            "predator_0": [1.0, 2.0],
            "prey_0": [-1.0],
        }

    def get_last_infos(self):
        return {
            "__all__": {
                "training_metrics": {
                    "predator_speed_mean": 1.6,
                    "predator_fraction_fast": 1.0,
                    "prey_speed_mean": 1.2,
                    "prey_fraction_fast": 0.0,
                    "predator_movement_energy_spent_mean": 0.4,
                    "prey_offspring_count_mean": 2.0,
                }
            },
            "predator_0": {
                "lifetime_steps": 3,
                "final_cumulative_reward": 3.0,
            },
        }


class _FakeEpisodeWithoutInfos(_FakeEpisode):
    def get_last_infos(self):
        return {}


class _FakeMetricsEnv:
    def _build_episode_training_metrics(self):
        return {
            "predator_speed_mean": 1.7,
            "prey_fraction_fast": 0.25,
        }


class _FakeVectorEnv:
    def __init__(self, envs):
        self.envs = envs


def test_episode_return_callback_logs_eco_evolution_metrics():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(episode=_FakeEpisode(), metrics_logger=logger)

    assert logger.values["eco_evolution/predator_speed_mean"] == pytest.approx(1.6)
    assert logger.values["eco_evolution/predator_fraction_fast"] == pytest.approx(1.0)
    assert logger.values["eco_evolution/prey_speed_mean"] == pytest.approx(1.2)
    assert logger.values["eco_evolution/prey_fraction_fast"] == pytest.approx(0.0)
    assert logger.values["eco_evolution/predator_movement_energy_spent_mean"] == pytest.approx(0.4)
    assert logger.values["eco_evolution/prey_offspring_count_mean"] == pytest.approx(2.0)
    assert logger.values["predator_episode_return_p50"] == pytest.approx(3.0)


def test_episode_return_callback_logs_eco_metrics_from_env_fallback():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(
        episode=_FakeEpisodeWithoutInfos(),
        metrics_logger=logger,
        env=_FakeMetricsEnv(),
    )

    assert logger.values["eco_evolution/predator_speed_mean"] == pytest.approx(1.7)
    assert logger.values["eco_evolution/prey_fraction_fast"] == pytest.approx(0.25)


def test_episode_return_callback_logs_eco_metrics_from_vector_env_fallback():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(
        episode=_FakeEpisodeWithoutInfos(),
        metrics_logger=logger,
        env=_FakeVectorEnv([_FakeMetricsEnv()]),
        env_index=0,
    )

    assert logger.values["eco_evolution/predator_speed_mean"] == pytest.approx(1.7)
    assert logger.values["eco_evolution/prey_fraction_fast"] == pytest.approx(0.25)


def test_every_acted_agent_gets_next_or_final_observation():
    env = _make_test_env(
        {
            "seed": 456,
            "max_steps": 120,
            "grid_size": 12,
            "n_initial_active_predators": 4,
            "n_initial_active_prey": 6,
            "n_possible_predators": 80,
            "n_possible_prey": 160,
            "initial_num_grass": 30,
        }
    )
    observations, _ = env.reset(seed=456)

    for _ in range(10):
        actions = {
            agent: int(env.rng.integers(env.action_spaces[agent].n))
            for agent in observations
        }
        acted_agents = set(actions)
        observations, _, terminations, truncations, _ = env.step(actions)

        for agent in acted_agents:
            is_done = terminations.get(agent, False) or truncations.get(agent, False)
            assert agent in observations or is_done
            if is_done:
                assert agent in observations

        if terminations.get("__all__") or truncations.get("__all__"):
            break


def test_agent_emits_max_age_termination_and_logs_event():
    env = _make_test_env(overrides={"max_agent_age": {"predator": 4, "prey": 400}})
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


def test_configured_none_max_agent_age_means_unlimited():
    env = _make_test_env()
    env.reset(seed=122)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    prey = next(agent for agent in env.agents if agent.startswith("prey"))

    assert env._get_max_age_limit(predator) is None
    assert env._get_max_age_limit(prey) == 400


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

    assert prey in observations
    assert predator in observations
    assert rewards[prey] == pytest.approx(env._get_role_specific("penalty_prey_caught", prey))
    assert terminations[prey] is True
    assert truncations[prey] is False
    assert terminations[predator] is True
    assert truncations[predator] is False
    assert terminations["__all__"] is True
    assert truncations["__all__"] is False
    assert "final_cumulative_reward" in infos[predator]
    assert env.agents == []


def test_terminal_reward_agent_is_not_returned_as_next_actor():
    env = _make_test_env(
        overrides={
            "n_initial_active_prey": 2,
            "predator_creation_energy_threshold": 999.0,
            "prey_creation_energy_threshold": 999.0,
        }
    )
    env.reset(seed=126)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    eaten_prey, survivor_prey = [agent for agent in env.agents if agent.startswith("prey")]
    _place_agent(env, predator, (5, 5))
    _place_agent(env, eaten_prey, (5, 5))
    _place_agent(env, survivor_prey, (env.grid_size - 2, env.grid_size - 2))

    stay_action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (0, 0))
    observations, rewards, terminations, truncations, _ = env.step({agent: stay_action for agent in env.agents})

    assert eaten_prey in rewards
    assert terminations[eaten_prey] is True
    assert truncations[eaten_prey] is False
    assert terminations["__all__"] is False
    assert eaten_prey in observations
    assert eaten_prey not in env.agents
    assert survivor_prey in env.agents


def test_time_limit_truncates_with_final_bootstrap_observations():
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
    observations, _, terminations, truncations, infos = env.step({agent: stay_action for agent in env.agents})

    n_ch = env._n_obs_channels()
    n_actions = env.action_range ** 2
    assert set(observations) == {predator, prey}
    assert observations[predator]["observations"].shape == (n_ch, env.predator_obs_range, env.predator_obs_range)
    assert observations[predator]["action_mask"].shape == (n_actions,)
    assert observations[prey]["observations"].shape == (n_ch, env.prey_obs_range, env.prey_obs_range)
    assert observations[prey]["action_mask"].shape == (n_actions,)
    assert terminations[predator] is False
    assert terminations[prey] is False
    assert truncations[predator] is True
    assert truncations[prey] is True
    assert terminations["__all__"] is False
    assert truncations["__all__"] is True
    assert env.agents == []
    assert "training_metrics" in infos["__all__"]
    metrics = infos["__all__"]["training_metrics"]
    assert "predator_speed_mean" in metrics
    assert "prey_speed_mean" in metrics
    assert "predator_fraction_mobile" in metrics
    assert "predator_fraction_fast_cadence" in metrics


def test_founders_receive_genomes_in_event_logs():
    env = _make_test_env()
    env.reset(seed=515)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))

    assert predator in env.agent_genomes
    assert env.agent_stats_live[predator]["genome"] == env.agent_genomes[predator].to_dict()
    assert env.agent_event_log[predator]["genome"] == env.agent_genomes[predator].to_dict()


def test_offspring_inherits_mutated_parent_genome_and_fixed_initial_energy():
    env = _make_test_env(
        overrides={
            "predator_creation_energy_threshold": 10.0,
            "founder_genome": {
                "predator": {
                    "speed_mean": 1.0,
                    "speed_std": 0.0,
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
    assert env.agent_energies[child] == pytest.approx(env.initial_energy_predator)
    assert env.agent_energies[parent] == pytest.approx(parent_energy - env.agent_energies[child])
    assert env.agent_genomes[child].to_dict() != env.agent_genomes[parent].to_dict()


def test_reproduction_threshold_uses_fixed_base_threshold():
    env = _make_test_env(
        overrides={
            "predator_creation_energy_threshold": 10.0,
            "founder_genome": {
                "predator": {
                    "speed_mean": 1.0,
                    "speed_std": 0.0,
                },
            },
            "genome_mutation": {"rate": 0.0, "std": 0.0},
        }
    )
    env.reset(seed=717)
    env.rewards = {}

    parent = next(agent for agent in env.agents if agent.startswith("predator"))
    env.agent_energies[parent] = 9.0
    env._handle_predator_reproduction(parent)
    assert env.agent_live_offspring_ids[parent] == []

    env.agent_energies[parent] = 10.0
    env._handle_predator_reproduction(parent)
    assert len(env.agent_live_offspring_ids[parent]) == 1


def test_action_space_uses_moore_neighborhood():
    env = _make_test_env()
    env.reset(seed=727)
    assert env.action_spaces["predator_0"].n == 9
    assert (1, 0) in env.action_to_move_tuple_agents.values()
    assert (2, 0) not in env.action_to_move_tuple_agents.values()


def test_observation_edges_are_clipped_and_zero_padded():
    env = _make_test_env()
    env.reset(seed=808)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    env.agent_positions[predator] = (0, 0)
    env.predator_positions[predator] = (0, 0)

    obs = env._get_observation(predator)

    n_ch = env._n_obs_channels()
    spatial = obs["observations"]
    assert spatial.shape == (n_ch, env.predator_obs_range, env.predator_obs_range)
    assert obs["action_mask"].shape == (env.action_range ** 2,)
    offset = (env.predator_obs_range - 1) // 2
    assert np.all(spatial[:env.num_obs_channels, :offset, :] == 0.0)
    assert np.all(spatial[:env.num_obs_channels, :, :offset] == 0.0)


def test_slow_speed_is_frozen_when_cooldown_nonzero():
    env = _make_test_env()
    env.reset(seed=818)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    _place_agent(env, predator, (10, 10))
    # speed=0.0 → cooldown=max_cooldown (10); force counter to non-zero
    env.agent_genomes[predator] = Genome(speed=0.0)
    env.agent_move_cooldowns[predator] = 5

    action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (1, 0))
    env._process_agent_movements({predator: action})

    # Agent is frozen — should not have moved
    assert env.agent_positions[predator] == (10, 10)
    assert env.agent_move_cooldowns[predator] == 4  # decremented


def test_fast_speed_moves_every_step():
    env = _make_test_env()
    env.reset(seed=919)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    _place_agent(env, predator, (10, 10))
    # speed=1.0 → cooldown=1; always 0 after phase init
    env.agent_genomes[predator] = Genome(speed=1.0)
    env.agent_move_cooldowns[predator] = 0

    action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (1, 0))
    env._process_agent_movements({predator: action})

    assert env.agent_positions[predator] == (11, 10)
    assert env.agent_move_cooldowns[predator] == 0  # cooldown-1 = 1-1 = 0


def test_fast_speed_pays_only_basal_cost_when_stationary():
    env = _make_test_env(
        overrides={
            "energy_loss_per_step_predator": 0.2,
            "movement_energy_cost_per_cell_predator": 0.05,
            "movement_speed_cost_exponent": 2.0,
            "metabolic_speed_coeff": 1.0,
        }
    )
    env.reset(seed=1020)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    _place_agent(env, predator, (10, 10))
    env.agent_genomes[predator] = Genome(speed=1.0)  # fastest: cooldown=1
    env.agent_move_cooldowns[predator] = 0
    start_energy = 10.0
    env.agent_energies[predator] = start_energy
    env.grid_world_state[0, *env.agent_positions[predator]] = start_energy

    env._apply_time_step_update()
    stay_action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (0, 0))
    env._process_agent_movements({predator: stay_action})

    # speed=1.0, coeff=1.0 → basal = 0.2 × (1 + 1.0 × 1.0) = 0.4; no locomotion cost
    expected_basal = 0.2 * (1.0 + 1.0 * 1.0)
    assert env.agent_energies[predator] == pytest.approx(start_energy - expected_basal)


def test_movement_cost_uses_actual_distance_and_superlinear_speed():
    env = _make_test_env(
        overrides={
            "energy_loss_per_step_predator": 0.2,
            "movement_energy_cost_per_cell_predator": 0.05,
            "movement_speed_cost_exponent": 2.0,
            "metabolic_speed_coeff": 1.0,
        }
    )
    env.reset(seed=1030)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    _place_agent(env, predator, (10, 10))
    env.agent_genomes[predator] = Genome(speed=0.5)  # mid-range genome
    env.agent_move_cooldowns[predator] = 0  # ensure movement executes this step
    start_energy = 10.0
    env.agent_energies[predator] = start_energy
    env.grid_world_state[0, *env.agent_positions[predator]] = start_energy

    env._apply_time_step_update()
    action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (1, 0))
    env._process_agent_movements({predator: action})

    # speed=0.5, coeff=1.0 → basal = 0.2 × (1 + 1.0 × 0.5) = 0.3
    # distance=1, speed=0.5, exponent=2.0 → locomotion = 0.05 × 1.0 × 0.5² = 0.0125
    expected_basal = 0.2 * (1.0 + 1.0 * 0.5)
    expected_movement_cost = 0.05 * 1.0 * (0.5 ** 2)
    assert env.agent_energies[predator] == pytest.approx(start_energy - expected_basal - expected_movement_cost)
    assert env._per_agent_step_deltas[predator]["move"] == pytest.approx(-expected_movement_cost)
    assert env.agent_stats_live[predator]["movement_energy_spent"] == pytest.approx(expected_movement_cost)
