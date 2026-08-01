import copy

import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_cultural_plasticity.config.config_env_eco_evolutionary import config_env
from predpreygrass.evolutionary.eco_evolutionary_cultural_plasticity.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.evolutionary.eco_evolutionary_cultural_plasticity.utils.episode_return_callback import EpisodeReturn
from predpreygrass.evolutionary.eco_evolutionary_cultural_plasticity.utils.genome import (
    Genome,
    founder_genome,
    mutate_genome,
)


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
    # Pin min == max so initial population is deterministic in tests.
    cfg["n_initial_active_predators_min"] = cfg["n_initial_active_predators"]
    cfg["n_initial_active_prey_min"] = cfg["n_initial_active_prey"]
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


class _FakeAgentEpisode:
    def __init__(self, total):
        self._total = total

    def get_return(self):
        return self._total


class _FakeEpisode:
    length = 3
    agent_episodes = {
        "predator_0": _FakeAgentEpisode(3.0),
        "prey_0": _FakeAgentEpisode(-1.0),
    }

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
                    "predator_plasticity_mean": 0.15,
                    "predator_plasticity_repro_spearman": 0.2,
                    "prey_plasticity_mean": 0.08,
                    "prey_dialect_match_rate": 0.4,
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
            "predator_plasticity_mean": 0.12,
            "prey_dialect_match_rate": 0.33,
        }


class _FakeVectorEnv:
    def __init__(self, envs):
        self.envs = envs


# ---- Callback tests ----


def test_episode_return_callback_logs_eco_evolution_metrics():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(episode=_FakeEpisode(), metrics_logger=logger)

    assert logger.values["eco_evolution/predator_plasticity_mean"] == pytest.approx(0.15)
    assert logger.values["eco_evolution/predator_plasticity_repro_spearman"] == pytest.approx(0.2)
    assert logger.values["eco_evolution/prey_plasticity_mean"] == pytest.approx(0.08)
    assert logger.values["eco_evolution/prey_dialect_match_rate"] == pytest.approx(0.4)
    assert logger.values["predator_episode_return_p50"] == pytest.approx(3.0)


def test_episode_return_callback_logs_eco_metrics_from_env_fallback():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(
        episode=_FakeEpisodeWithoutInfos(),
        metrics_logger=logger,
        env=_FakeMetricsEnv(),
    )

    assert logger.values["eco_evolution/predator_plasticity_mean"] == pytest.approx(0.12)
    assert logger.values["eco_evolution/prey_dialect_match_rate"] == pytest.approx(0.33)


def test_episode_return_callback_logs_eco_metrics_from_vector_env_fallback():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(
        episode=_FakeEpisodeWithoutInfos(),
        metrics_logger=logger,
        env=_FakeVectorEnv([_FakeMetricsEnv()]),
        env_index=0,
    )

    assert logger.values["eco_evolution/predator_plasticity_mean"] == pytest.approx(0.12)
    assert logger.values["eco_evolution/prey_dialect_match_rate"] == pytest.approx(0.33)


# ---- RLlib contract tests (generic, not genome-specific) ----


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
    assert set(observations) == {predator, prey}
    assert observations[predator].shape == (n_ch, env.predator_obs_range, env.predator_obs_range)
    assert observations[prey].shape == (n_ch, env.prey_obs_range, env.prey_obs_range)
    assert terminations[predator] is False
    assert terminations[prey] is False
    assert truncations[predator] is True
    assert truncations[prey] is True
    assert terminations["__all__"] is False
    assert truncations["__all__"] is True
    assert env.agents == []
    assert "training_metrics" in infos["__all__"]
    metrics = infos["__all__"]["training_metrics"]
    assert "predator_plasticity_mean" in metrics
    assert "prey_plasticity_mean" in metrics
    assert "predator_dialect_match_rate" in metrics
    assert "prey_dialect_match_rate" in metrics


def test_action_space_uses_extended_moore_neighborhood():
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

    assert obs.shape == (env._n_obs_channels(), env.predator_obs_range, env.predator_obs_range)
    offset = (env.predator_obs_range - 1) // 2
    assert np.all(obs[:env.num_obs_channels, :offset, :] == 0.0)
    assert np.all(obs[:env.num_obs_channels, :, :offset] == 0.0)


def test_insufficient_energy_blocks_reproduction_regardless_of_genome():
    env = _make_test_env(overrides={"predator_creation_energy_threshold": 10.0})
    env.reset(seed=88)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    env.agent_genomes[parent] = Genome(plasticity=0.5, dialect=0)
    env.agent_energies[parent] = 9.99  # just below threshold

    env._handle_predator_reproduction(parent)

    assert env.agent_live_offspring_ids[parent] == []
    assert env.agent_offspring_counts[parent] == 0


def test_energy_exactly_at_threshold_triggers_one_offspring():
    threshold = 10.0
    env = _make_test_env(overrides={"predator_creation_energy_threshold": threshold})
    env.reset(seed=99)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    env.agent_energies[parent] = threshold  # exactly at threshold

    env._handle_predator_reproduction(parent)

    assert len(env.agent_live_offspring_ids[parent]) == 1


def test_multi_generation_ancestry_chain():
    env = _make_test_env(overrides={"predator_creation_energy_threshold": 10.0})
    env.reset(seed=77)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))

    # generation 1: parent -> child
    env.agent_energies[parent] = 20.0
    env._handle_predator_reproduction(parent)
    child = env.agent_live_offspring_ids[parent][0]

    # generation 2: child -> grandchild
    env.agent_energies[child] = 20.0
    env._handle_predator_reproduction(child)
    grandchild = env.agent_live_offspring_ids[child][0]

    assert env.agent_parents[child] == parent
    assert env.agent_parents[grandchild] == child
    assert grandchild in env.agent_genomes
    assert grandchild in env.agent_live_dialect


def test_offspring_investment_fraction_is_fixed_and_genome_independent():
    fraction = 0.4
    env = _make_test_env(
        overrides={"predator_creation_energy_threshold": 10.0, "offspring_investment_fraction": fraction}
    )
    env.reset(seed=646)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    parent_energy = 20.0
    env.agent_energies[parent] = parent_energy
    env.agent_genomes[parent] = Genome(plasticity=0.9, dialect=1)

    env._handle_predator_reproduction(parent)
    child = env.agent_live_offspring_ids[parent][0]

    assert env.agent_energies[child] == pytest.approx(parent_energy * fraction)
    assert env.agent_energies[parent] == pytest.approx(parent_energy * (1 - fraction))


# ---- Genome / dual-inheritance tests ----


def test_founders_receive_genomes_in_event_logs():
    env = _make_test_env()
    env.reset(seed=515)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))

    assert predator in env.agent_genomes
    assert env.agent_stats_live[predator]["genome"] == env.agent_genomes[predator].to_dict()
    assert env.agent_event_log[predator]["genome"] == env.agent_genomes[predator].to_dict()
    assert predator in env.agent_live_dialect
    assert env.agent_live_dialect[predator] == env.agent_genomes[predator].dialect


def test_genome_disabled_produces_no_genomes_and_no_live_dialect():
    env = _make_test_env(overrides={"genome_enabled": False})
    env.reset(seed=11)

    assert env.agent_genomes == {}
    assert env.agent_live_dialect == {}
    for agent in env.agents:
        assert env.agent_event_log[agent]["genome"] is None
        assert env.agent_stats_live[agent]["genome"] is None


def test_founder_genome_plasticity_respects_bounds():
    rng = np.random.default_rng(1)
    config = {
        "trait_bounds": {"plasticity": (0.0, 1.0)},
        "founder_genome": {"predator": {"plasticity_mean": 0.9, "plasticity_std": 0.5}},
        "n_dialects": 4,
    }
    for _ in range(200):
        genome = founder_genome("predator", config, rng)
        assert 0.0 <= genome.plasticity <= 1.0
        assert 0 <= genome.dialect < 4


def test_zero_mutation_rate_produces_exact_genome_copy():
    rng = np.random.default_rng(3)
    parent = Genome(plasticity=0.3, dialect=2)
    config = {"genome_mutation": {"rate": 0.0}, "dialect_mutation": {"rate": 0.0}}
    child = mutate_genome(parent, config, rng)
    assert child.plasticity == pytest.approx(parent.plasticity)
    assert child.dialect == parent.dialect


def test_mutation_always_stays_in_valid_state_space():
    rng = np.random.default_rng(4)
    config = {
        "trait_bounds": {"plasticity": (0.0, 1.0)},
        "genome_mutation": {"rate": 1.0, "std": 0.5},
        "dialect_mutation": {"rate": 1.0},
        "n_dialects": 5,
    }
    genome = Genome(plasticity=0.5, dialect=0)
    for _ in range(200):
        genome = mutate_genome(genome, config, rng)
        assert 0.0 <= genome.plasticity <= 1.0
        assert 0 <= genome.dialect < 5


def test_neutral_drift_control_template_is_a_live_conspecific_not_necessarily_parent():
    env = _make_test_env(
        overrides={
            "predator_creation_energy_threshold": 10.0,
            "n_initial_active_predators": 2,
            "genome_neutral_drift_control": True,
            "genome_mutation": {"rate": 0.0},
            "dialect_mutation": {"rate": 0.0},
        }
    )
    env.reset(seed=656)
    env.rewards = {}

    predators = [a for a in env.agents if a.startswith("predator")]
    assert len(predators) == 2
    env.agent_genomes[predators[0]] = Genome(plasticity=0.9, dialect=0)
    env.agent_genomes[predators[1]] = Genome(plasticity=0.1, dialect=3)

    env.agent_energies[predators[0]] = 20.0
    env._handle_predator_reproduction(predators[0])
    child = env.agent_live_offspring_ids[predators[0]][0]

    # With zero mutation, the child's genome must be an exact copy of ONE of the
    # two live conspecifics' genomes -- not necessarily the reproducing parent's,
    # since the template is a uniformly random live conspecific under the control.
    possible = {
        (env.agent_genomes[predators[0]].plasticity, env.agent_genomes[predators[0]].dialect),
        (env.agent_genomes[predators[1]].plasticity, env.agent_genomes[predators[1]].dialect),
    }
    assert (env.agent_genomes[child].plasticity, env.agent_genomes[child].dialect) in possible


def test_live_culture_metrics_reflect_actual_genomes():
    env = _make_test_env()
    env.reset(seed=111)

    for agent in env.agents:
        env.agent_genomes[agent] = Genome(plasticity=0.25, dialect=1)
        env.agent_live_dialect[agent] = 1

    metrics = env._build_live_culture_metrics()

    assert metrics["predator_plasticity_mean"] == pytest.approx(0.25)
    assert metrics["prey_plasticity_mean"] == pytest.approx(0.25)
    assert metrics["predator_count"] == pytest.approx(1.0)


def test_dialect_entropy_zero_when_fixed_and_positive_when_diverse():
    env = _make_test_env()
    env.reset(seed=222)

    assert env._dialect_entropy([2, 2, 2, 2]) == pytest.approx(0.0)
    assert env._dialect_entropy([]) == pytest.approx(0.0)
    diverse_entropy = env._dialect_entropy([0, 1, 2, 3])
    assert diverse_entropy > 0.0
    # Uniform distribution over n_dialects maximizes entropy at ln(n_dialects).
    assert diverse_entropy == pytest.approx(np.log(4), rel=1e-6)


def test_spearman_corr_perfect_and_no_variance_cases():
    env = _make_test_env()
    env.reset(seed=333)

    assert env._spearman_corr([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert env._spearman_corr([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert env._spearman_corr([1, 1, 1], [1, 2, 3]) == pytest.approx(0.0)
    assert env._spearman_corr([1], [1]) == pytest.approx(0.0)


def test_local_majority_dialect_none_without_same_species_neighbors():
    env = _make_test_env(overrides={"culture_range": 3, "n_initial_active_predators": 1, "n_initial_active_prey": 1})
    env.reset(seed=444)

    predator = next(a for a in env.agents if a.startswith("predator"))
    _place_agent(env, predator, (5, 5))
    assert env._local_majority_dialect(predator) is None


def test_local_majority_dialect_picks_most_common_neighbor_dialect():
    # _local_majority_dialect excludes the focal agent's own vote (the norm
    # to conform to is defined by *others*), so 4 agents are used here: 3
    # sharing dialect 2 and 1 with dialect 0, giving an unambiguous 2-vs-1
    # neighbor majority for every agent regardless of which one is focal.
    env = _make_test_env(
        overrides={
            "culture_range": 5,
            "n_initial_active_predators": 4,
            "n_initial_active_prey": 1,
        }
    )
    env.reset(seed=555)

    predators = [a for a in env.agents if a.startswith("predator")]
    for i, agent in enumerate(predators):
        _place_agent(env, agent, (5, 5 + i))
    env.agent_live_dialect[predators[0]] = 2
    env.agent_live_dialect[predators[1]] = 2
    env.agent_live_dialect[predators[2]] = 0
    env.agent_live_dialect[predators[3]] = 2

    assert env._local_majority_dialect(predators[2]) == 2
    assert env._local_majority_dialect(predators[0]) == 2


def test_cultural_learning_respects_check_interval_and_plasticity():
    env = _make_test_env(
        overrides={
            "culture_range": 5,
            "plasticity_check_interval": 4,
            "n_initial_active_predators": 2,
            "n_initial_active_prey": 1,
        }
    )
    env.reset(seed=666)

    predators = [a for a in env.agents if a.startswith("predator")]
    for i, agent in enumerate(predators):
        _place_agent(env, agent, (5, 5 + i))
    env.agent_genomes[predators[0]] = Genome(plasticity=1.0, dialect=0)  # always adopts
    env.agent_live_dialect[predators[0]] = 0
    env.agent_live_dialect[predators[1]] = 3

    env.current_step = 1  # not a multiple of the check interval (4): no-op
    env._apply_cultural_learning()
    assert env.agent_live_dialect[predators[0]] == 0

    env.current_step = 4  # a check step: predators[0] should adopt predators[1]'s dialect
    env._apply_cultural_learning()
    assert env.agent_live_dialect[predators[0]] == 3


def test_zero_plasticity_never_adopts_majority_dialect():
    env = _make_test_env(
        overrides={
            "culture_range": 5,
            "plasticity_check_interval": 1,
            "n_initial_active_predators": 2,
            "n_initial_active_prey": 1,
        }
    )
    env.reset(seed=777)

    predators = [a for a in env.agents if a.startswith("predator")]
    for i, agent in enumerate(predators):
        _place_agent(env, agent, (5, 5 + i))
    env.agent_genomes[predators[0]] = Genome(plasticity=0.0, dialect=0)  # never adopts
    env.agent_live_dialect[predators[0]] = 0
    env.agent_live_dialect[predators[1]] = 3

    env.current_step = 1
    for _ in range(20):
        env._apply_cultural_learning()

    assert env.agent_live_dialect[predators[0]] == 0


def test_dialect_match_grants_coordination_bonus_on_grass_energy_gain():
    bonus = 2.0
    env = _make_test_env(
        overrides={"coordination_bonus_multiplier": bonus, "culture_range": 5, "n_initial_active_prey": 2}
    )
    env.reset(seed=888)
    env.rewards = {}

    prey_agents = [a for a in env.agents if a.startswith("prey")]
    assert len(prey_agents) == 2

    prey = prey_agents[0]
    neighbor = prey_agents[1]
    _place_agent(env, prey, (3, 3))
    _place_agent(env, neighbor, (3, 4))
    env.agent_live_dialect[prey] = 1
    env.agent_live_dialect[neighbor] = 1  # matches -> majority is dialect 1

    grass_id = next(iter(env.grass_positions))
    env.grass_positions[grass_id] = (3, 3)
    env.grass_energies[grass_id] = 1.5
    env.grid_world_state[2, 3, 3] = 1.5

    env._apply_time_step_update()
    start_energy = env.agent_energies[prey]
    env._handle_prey_engagement(prey)

    assert env.agent_energies[prey] == pytest.approx(start_energy + 1.5 * bonus)


def test_dialect_mismatch_grants_no_coordination_bonus():
    bonus = 2.0
    env = _make_test_env(
        overrides={"coordination_bonus_multiplier": bonus, "culture_range": 5, "n_initial_active_prey": 2}
    )
    env.reset(seed=999)
    env.rewards = {}

    prey_agents = [a for a in env.agents if a.startswith("prey")]
    assert len(prey_agents) == 2

    prey = prey_agents[0]
    neighbor = prey_agents[1]
    _place_agent(env, prey, (3, 3))
    _place_agent(env, neighbor, (3, 4))
    env.agent_live_dialect[prey] = 0
    env.agent_live_dialect[neighbor] = 1  # neighbor's dialect is the (sole) majority -> mismatch for prey

    grass_id = next(iter(env.grass_positions))
    env.grass_positions[grass_id] = (3, 3)
    env.grass_energies[grass_id] = 1.5
    env.grid_world_state[2, 3, 3] = 1.5

    env._apply_time_step_update()
    start_energy = env.agent_energies[prey]
    env._handle_prey_engagement(prey)

    assert env.agent_energies[prey] == pytest.approx(start_energy + 1.5)
