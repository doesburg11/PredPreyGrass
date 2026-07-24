import copy

import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_metabolic_code.config.config_env_eco_evolutionary import config_env
from predpreygrass.evolutionary.eco_evolutionary_metabolic_code.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.evolutionary.eco_evolutionary_metabolic_code.utils.episode_return_callback import EpisodeReturn
from predpreygrass.evolutionary.eco_evolutionary_metabolic_code.utils.genome import (
    CORRECT,
    WRONG,
    PLASTIC,
    Genome,
    attempt_resolve,
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


ALL_CORRECT = tuple([CORRECT] * 10)
ALL_PLASTIC = tuple([PLASTIC] * 10)
ONE_WRONG = tuple([WRONG] + [CORRECT] * 9)


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
                    "predator_mean_wrong_loci": 2.1,
                    "predator_fraction_solved": 0.3,
                    "prey_mean_wrong_loci": 1.4,
                    "prey_fraction_solved": 0.5,
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
            "predator_mean_wrong_loci": 1.9,
            "prey_fraction_solved": 0.45,
        }


class _FakeVectorEnv:
    def __init__(self, envs):
        self.envs = envs


# ---- Callback tests ----


def test_episode_return_callback_logs_eco_evolution_metrics():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(episode=_FakeEpisode(), metrics_logger=logger)

    assert logger.values["eco_evolution/predator_mean_wrong_loci"] == pytest.approx(2.1)
    assert logger.values["eco_evolution/predator_fraction_solved"] == pytest.approx(0.3)
    assert logger.values["eco_evolution/prey_mean_wrong_loci"] == pytest.approx(1.4)
    assert logger.values["eco_evolution/prey_fraction_solved"] == pytest.approx(0.5)
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

    assert logger.values["eco_evolution/predator_mean_wrong_loci"] == pytest.approx(1.9)
    assert logger.values["eco_evolution/prey_fraction_solved"] == pytest.approx(0.45)


def test_episode_return_callback_logs_eco_metrics_from_vector_env_fallback():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(
        episode=_FakeEpisodeWithoutInfos(),
        metrics_logger=logger,
        env=_FakeVectorEnv([_FakeMetricsEnv()]),
        env_index=0,
    )

    assert logger.values["eco_evolution/predator_mean_wrong_loci"] == pytest.approx(1.9)
    assert logger.values["eco_evolution/prey_fraction_solved"] == pytest.approx(0.45)


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
    assert "predator_mean_wrong_loci" in metrics
    assert "prey_mean_wrong_loci" in metrics
    assert "predator_fraction_solved" in metrics
    assert "prey_fraction_solved" in metrics


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


def test_agent_pays_only_basal_cost_when_stationary():
    env = _make_test_env(
        overrides={
            "energy_loss_per_step_predator": 0.2,
            "movement_energy_cost_per_cell_predator": 0.05,
        }
    )
    env.reset(seed=1020)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    _place_agent(env, predator, (10, 10))
    start_energy = 10.0
    env.agent_energies[predator] = start_energy
    env.grid_world_state[0, *env.agent_positions[predator]] = start_energy

    env._apply_time_step_update()
    stay_action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (0, 0))
    env._process_agent_movements({predator: stay_action})

    assert env.agent_energies[predator] == pytest.approx(start_energy - 0.2)


def test_movement_cost_uses_actual_distance():
    env = _make_test_env(
        overrides={
            "energy_loss_per_step_predator": 0.2,
            "movement_energy_cost_per_cell_predator": 0.05,
        }
    )
    env.reset(seed=1030)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    _place_agent(env, predator, (10, 10))
    start_energy = 10.0
    env.agent_energies[predator] = start_energy
    env.grid_world_state[0, *env.agent_positions[predator]] = start_energy

    env._apply_time_step_update()
    action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (1, 0))
    env._process_agent_movements({predator: action})

    expected_movement_cost = 0.05 * 1.0
    assert env.agent_energies[predator] == pytest.approx(start_energy - 0.2 - expected_movement_cost)
    assert env._per_agent_step_deltas[predator]["move"] == pytest.approx(-expected_movement_cost)
    assert env.agent_stats_live[predator]["movement_energy_spent"] == pytest.approx(expected_movement_cost)


def test_insufficient_energy_blocks_reproduction_regardless_of_genome():
    env = _make_test_env(overrides={"predator_creation_energy_threshold": 10.0})
    env.reset(seed=88)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    env.agent_genomes[parent] = Genome(loci=ALL_CORRECT)
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


def test_parent_cannot_reproduce_again_until_energy_refills():
    threshold = 10.0
    env = _make_test_env(overrides={"predator_creation_energy_threshold": threshold})
    env.reset(seed=101)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    env.agent_energies[parent] = threshold  # just enough for one reproduction

    env._handle_predator_reproduction(parent)
    assert len(env.agent_live_offspring_ids[parent]) == 1
    assert env.agent_energies[parent] < threshold  # energy depleted below gate

    env._handle_predator_reproduction(parent)
    assert len(env.agent_live_offspring_ids[parent]) == 1  # second call blocked


def test_reproduction_threshold_uses_fixed_base_threshold():
    env = _make_test_env(overrides={"predator_creation_energy_threshold": 10.0})
    env.reset(seed=717)
    env.rewards = {}

    parent = next(agent for agent in env.agents if agent.startswith("predator"))
    env.agent_energies[parent] = 9.0
    env._handle_predator_reproduction(parent)
    assert env.agent_live_offspring_ids[parent] == []

    env.agent_energies[parent] = 10.0
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
    assert env.agent_event_log[child]["parent_id"] == parent
    assert env.agent_event_log[grandchild]["parent_id"] == child
    assert grandchild in env.agent_genomes


def test_child_genome_recorded_in_event_log_after_reproduction():
    env = _make_test_env(overrides={"predator_creation_energy_threshold": 10.0})
    env.reset(seed=121)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    env.agent_energies[parent] = 20.0

    env._handle_predator_reproduction(parent)

    child = env.agent_live_offspring_ids[parent][0]
    assert child in env.agent_genomes
    assert env.agent_event_log[child]["genome"] == env.agent_genomes[child].to_dict()
    assert env.agent_event_log[child]["parent_id"] == parent


# ---- Genome / combinatorial-trait tests ----


def test_founders_receive_genomes_in_event_logs():
    env = _make_test_env()
    env.reset(seed=515)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))

    assert predator in env.agent_genomes
    assert env.agent_stats_live[predator]["genome"] == env.agent_genomes[predator].to_dict()
    assert env.agent_event_log[predator]["genome"] == env.agent_genomes[predator].to_dict()
    assert len(env.agent_genomes[predator].loci) == 10


def test_genome_disabled_produces_no_genomes():
    env = _make_test_env(overrides={"genome_enabled": False})
    env.reset(seed=11)

    assert env.agent_genomes == {}
    for agent in env.agents:
        assert env.agent_event_log[agent]["genome"] is None
        assert env.agent_stats_live[agent]["genome"] is None


def test_founder_genome_all_wrong_when_forced():
    rng = np.random.default_rng(1)
    config = {"haystack_num_loci": 10, "haystack_founder_probs": {"predator": {"correct": 0.0, "wrong": 1.0, "plastic": 0.0}}}
    genome = founder_genome("predator", config, rng)
    assert genome.loci == tuple([WRONG] * 10)
    assert genome.has_wrong is True
    assert genome.num_wrong == 10


def test_founder_genome_all_correct_when_forced():
    rng = np.random.default_rng(2)
    config = {"haystack_num_loci": 10, "haystack_founder_probs": {"prey": {"correct": 1.0, "wrong": 0.0, "plastic": 0.0}}}
    genome = founder_genome("prey", config, rng)
    assert genome.loci == tuple([CORRECT] * 10)
    assert genome.has_wrong is False
    assert genome.num_correct == 10


def test_zero_mutation_rate_produces_exact_genome_copy():
    rng = np.random.default_rng(3)
    parent = Genome(loci=ONE_WRONG)
    config = {"haystack_mutation": {"rate": 0.0}}
    child = mutate_genome(parent, config, rng)
    assert child.loci == parent.loci


def test_mutation_always_stays_in_valid_state_space():
    rng = np.random.default_rng(4)
    genome = Genome(loci=ALL_PLASTIC)
    config = {"haystack_mutation": {"rate": 1.0}}
    for _ in range(200):
        genome = mutate_genome(genome, config, rng)
        assert all(locus in (CORRECT, WRONG, PLASTIC) for locus in genome.loci)
        assert len(genome.loci) == 10


def test_attempt_resolve_fails_with_any_wrong_locus():
    rng = np.random.default_rng(5)
    for _ in range(50):
        assert attempt_resolve(ONE_WRONG, rng) is False


def test_attempt_resolve_succeeds_with_no_wrong_and_no_plastic():
    rng = np.random.default_rng(6)
    assert attempt_resolve(ALL_CORRECT, rng) is True


def test_update_haystack_solving_marks_agent_solved_when_genome_is_all_correct():
    env = _make_test_env()
    env.reset(seed=606)

    predator = next(a for a in env.agents if a.startswith("predator"))
    env.agent_genomes[predator] = Genome(loci=ALL_CORRECT)
    env.agent_has_wrong[predator] = False
    env.agent_solved[predator] = False

    env._update_haystack_solving()

    assert env.agent_solved[predator] is True
    assert env.agent_solved_step[predator] == env.current_step


def test_agent_with_wrong_locus_never_solves_across_many_steps():
    env = _make_test_env()
    env.reset(seed=616)

    predator = next(a for a in env.agents if a.startswith("predator"))
    env.agent_genomes[predator] = Genome(loci=ONE_WRONG)
    env.agent_has_wrong[predator] = True
    env.agent_solved[predator] = False

    for _ in range(50):
        env._update_haystack_solving()
        env.current_step += 1

    assert env.agent_solved[predator] is False


def test_solved_agent_gets_bonus_multiplier_on_grass_energy_gain():
    bonus = 2.0
    env = _make_test_env(overrides={"haystack_bonus_multiplier": bonus})
    env.reset(seed=626)
    env.rewards = {}

    prey = next(a for a in env.agents if a.startswith("prey"))
    _place_agent(env, prey, (3, 3))
    grass_id = next(iter(env.grass_positions))
    env.grass_positions[grass_id] = (3, 3)
    env.grass_energies[grass_id] = 1.5
    env.grid_world_state[2, 3, 3] = 1.5

    # Populate _per_agent_step_deltas (normally done by _apply_time_step_update
    # earlier in step()) before calling the engagement handler directly.
    env._apply_time_step_update()
    start_energy = env.agent_energies[prey]
    env.agent_solved[prey] = True
    env._handle_prey_engagement(prey)

    assert env.agent_energies[prey] == pytest.approx(start_energy + 1.5 * bonus)


def test_unsolved_agent_gets_no_bonus_on_grass_energy_gain():
    env = _make_test_env(overrides={"haystack_bonus_multiplier": 2.0})
    env.reset(seed=636)
    env.rewards = {}

    prey = next(a for a in env.agents if a.startswith("prey"))
    _place_agent(env, prey, (3, 3))
    grass_id = next(iter(env.grass_positions))
    env.grass_positions[grass_id] = (3, 3)
    env.grass_energies[grass_id] = 1.5
    env.grid_world_state[2, 3, 3] = 1.5

    env._apply_time_step_update()
    start_energy = env.agent_energies[prey]
    env.agent_solved[prey] = False
    env._handle_prey_engagement(prey)

    assert env.agent_energies[prey] == pytest.approx(start_energy + 1.5)


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
    # Two very different genomes should not change the fixed investment fraction.
    env.agent_genomes[parent] = Genome(loci=ALL_CORRECT)

    env._handle_predator_reproduction(parent)
    child = env.agent_live_offspring_ids[parent][0]

    assert env.agent_energies[child] == pytest.approx(parent_energy * fraction)
    assert env.agent_energies[parent] == pytest.approx(parent_energy * (1 - fraction))


def test_neutral_drift_control_template_is_a_live_conspecific_not_necessarily_parent():
    env = _make_test_env(
        overrides={
            "predator_creation_energy_threshold": 10.0,
            "n_initial_active_predators": 2,
            "genome_neutral_drift_control": True,
            "haystack_mutation": {"rate": 0.0},
        }
    )
    env.reset(seed=656)
    env.rewards = {}

    predators = [a for a in env.agents if a.startswith("predator")]
    assert len(predators) == 2
    env.agent_genomes[predators[0]] = Genome(loci=ALL_CORRECT)
    env.agent_genomes[predators[1]] = Genome(loci=tuple([WRONG] * 10))

    env.agent_energies[predators[0]] = 20.0
    env._handle_predator_reproduction(predators[0])
    child = env.agent_live_offspring_ids[predators[0]][0]

    # With zero mutation, the child's genome must be an exact copy of ONE of the
    # two live conspecifics' genomes -- not necessarily the reproducing parent's,
    # since the template is a uniformly random live conspecific under the control.
    possible = {env.agent_genomes[predators[0]].loci, env.agent_genomes[predators[1]].loci}
    assert env.agent_genomes[child].loci in possible


def test_live_haystack_metrics_reflect_actual_genomes():
    env = _make_test_env()
    env.reset(seed=111)

    known_loci = tuple([WRONG, WRONG] + [CORRECT] * 3 + [PLASTIC] * 5)
    for agent in env.agents:
        env.agent_genomes[agent] = Genome(loci=known_loci)

    metrics = env._build_live_haystack_metrics()

    assert metrics["predator_mean_wrong_loci"] == pytest.approx(2.0)
    assert metrics["predator_mean_correct_loci"] == pytest.approx(3.0)
    assert metrics["predator_mean_plastic_loci"] == pytest.approx(5.0)
    assert metrics["prey_mean_wrong_loci"] == pytest.approx(2.0)
    assert metrics["predator_fraction_solved"] == pytest.approx(0.0)
