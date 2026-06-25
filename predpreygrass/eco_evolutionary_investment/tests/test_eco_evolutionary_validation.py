import copy

import numpy as np
import pytest

from predpreygrass.eco_evolutionary_investment.config.config_env_eco_evolutionary import config_env
from predpreygrass.eco_evolutionary_investment.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.eco_evolutionary_investment.utils.episode_return_callback import EpisodeReturn
from predpreygrass.eco_evolutionary_investment.utils.genome import Genome


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
                    "predator_investment_fraction_mean": 0.36,
                    "predator_investment_fraction_p50": 0.35,
                    "prey_investment_fraction_mean": 0.42,
                    "prey_investment_fraction_p50": 0.41,
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
            "predator_investment_fraction_mean": 0.37,
            "prey_investment_fraction_p50": 0.45,
        }


class _FakeVectorEnv:
    def __init__(self, envs):
        self.envs = envs


def test_episode_return_callback_logs_eco_evolution_metrics():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(episode=_FakeEpisode(), metrics_logger=logger)

    assert logger.values["eco_evolution/predator_investment_fraction_mean"] == pytest.approx(0.36)
    assert logger.values["eco_evolution/predator_investment_fraction_p50"] == pytest.approx(0.35)
    assert logger.values["eco_evolution/prey_investment_fraction_mean"] == pytest.approx(0.42)
    assert logger.values["eco_evolution/prey_investment_fraction_p50"] == pytest.approx(0.41)
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

    assert logger.values["eco_evolution/predator_investment_fraction_mean"] == pytest.approx(0.37)
    assert logger.values["eco_evolution/prey_investment_fraction_p50"] == pytest.approx(0.45)


def test_episode_return_callback_logs_eco_metrics_from_vector_env_fallback():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(
        episode=_FakeEpisodeWithoutInfos(),
        metrics_logger=logger,
        env=_FakeVectorEnv([_FakeMetricsEnv()]),
        env_index=0,
    )

    assert logger.values["eco_evolution/predator_investment_fraction_mean"] == pytest.approx(0.37)
    assert logger.values["eco_evolution/prey_investment_fraction_p50"] == pytest.approx(0.45)


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
    assert "predator_investment_fraction_mean" in metrics
    assert "prey_investment_fraction_mean" in metrics
    assert "predator_investment_fraction_p25" in metrics
    assert "predator_investment_fraction_p50" in metrics
    assert "predator_investment_fraction_p75" in metrics


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


def test_genome_disabled_produces_no_genomes():
    env = _make_test_env(overrides={"genome_enabled": False})
    env.reset(seed=11)

    assert env.agent_genomes == {}
    for agent in env.agents:
        assert env.agent_event_log[agent]["genome"] is None
        assert env.agent_stats_live[agent]["genome"] is None


def test_zero_mutation_rate_produces_exact_genome_copy():
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": 10.0,
        "genome_mutation": {"rate": 0.0, "std": 0.05},
    })
    env.reset(seed=22)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    parent_fraction = env.agent_genomes[parent].offspring_investment_fraction
    env.agent_energies[parent] = 20.0

    env._handle_predator_reproduction(parent)

    child = env.agent_live_offspring_ids[parent][0]
    assert env.agent_genomes[child].offspring_investment_fraction == pytest.approx(parent_fraction)


def test_zero_mutation_std_produces_exact_copy_even_at_full_rate():
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": 10.0,
        "genome_mutation": {"rate": 1.0, "std": 0.0},
    })
    env.reset(seed=33)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    parent_fraction = env.agent_genomes[parent].offspring_investment_fraction
    env.agent_energies[parent] = 20.0

    env._handle_predator_reproduction(parent)

    child = env.agent_live_offspring_ids[parent][0]
    assert env.agent_genomes[child].offspring_investment_fraction == pytest.approx(parent_fraction)


def test_mutation_never_violates_trait_bounds():
    from predpreygrass.eco_evolutionary_investment.utils.genome import mutate_genome

    rng = np.random.default_rng(44)
    config = {"genome_mutation": {"rate": 1.0, "std": 0.5}, "trait_bounds": {}}
    genome = Genome(offspring_investment_fraction=0.79)  # near upper bound

    for _ in range(500):
        genome = mutate_genome(genome, config, rng)
        f = genome.offspring_investment_fraction
        assert 0.10 <= f <= 0.80


def test_investment_fraction_determines_offspring_energy_and_parent_cost():
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": 10.0,
        "min_offspring_energy_predator": 0.1,
        "max_offspring_energy_predator": 100.0,
        "genome_mutation": {"rate": 0.0, "std": 0.0},
    })
    env.reset(seed=55)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    parent_energy = 20.0

    # High-fraction reproduction
    env.agent_energies[parent] = parent_energy
    env.agent_genomes[parent] = Genome(offspring_investment_fraction=0.60)
    env._handle_predator_reproduction(parent)
    child_high = env.agent_live_offspring_ids[parent][-1]
    energy_high = env.agent_energies[child_high]
    cost_high = parent_energy - env.agent_energies[parent]

    # Low-fraction reproduction (reset parent energy to same starting point)
    env.agent_energies[parent] = parent_energy
    env.agent_genomes[parent] = Genome(offspring_investment_fraction=0.30)
    env._handle_predator_reproduction(parent)
    child_low = env.agent_live_offspring_ids[parent][-1]
    energy_low = env.agent_energies[child_low]
    cost_low = parent_energy - env.agent_energies[parent]

    assert energy_high == pytest.approx(parent_energy * 0.60)
    assert energy_low == pytest.approx(parent_energy * 0.30)
    assert cost_high > cost_low


def test_genome_does_not_change_across_steps():
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": 999.0,
        "prey_creation_energy_threshold": 999.0,
    })
    env.reset(seed=66)

    predator = next(a for a in env.agents if a.startswith("predator"))
    fraction_at_birth = env.agent_genomes[predator].offspring_investment_fraction

    for _ in range(5):
        actions = {a: 0 for a in env.agents}
        env.step(actions)
        if predator in env.agent_genomes:
            assert env.agent_genomes[predator].offspring_investment_fraction == pytest.approx(fraction_at_birth)


def test_multi_generation_lineage_chain():
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": 10.0,
        "min_offspring_energy_predator": 1.0,
        "max_offspring_energy_predator": 100.0,
        "genome_mutation": {"rate": 0.0, "std": 0.0},
    })
    env.reset(seed=77)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))

    # generation 1: parent → child
    env.agent_energies[parent] = 20.0
    env._handle_predator_reproduction(parent)
    child = env.agent_live_offspring_ids[parent][0]

    # generation 2: child → grandchild
    env.agent_energies[child] = 20.0
    env._handle_predator_reproduction(child)
    grandchild = env.agent_live_offspring_ids[child][0]

    assert env.agent_parents[child] == parent
    assert env.agent_parents[grandchild] == child
    assert env.agent_event_log[child]["parent_id"] == parent
    assert env.agent_event_log[grandchild]["parent_id"] == child
    assert grandchild in env.agent_genomes


def test_insufficient_energy_blocks_reproduction_regardless_of_genome():
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": 10.0,
    })
    env.reset(seed=88)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    env.agent_genomes[parent] = Genome(offspring_investment_fraction=0.35)
    env.agent_energies[parent] = 9.99  # just below threshold

    env._handle_predator_reproduction(parent)

    assert env.agent_live_offspring_ids[parent] == []
    assert env.agent_offspring_counts[parent] == 0


def test_energy_exactly_at_threshold_triggers_one_offspring():
    threshold = 10.0
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": threshold,
        "min_offspring_energy_predator": 1.0,
        "max_offspring_energy_predator": 100.0,
    })
    env.reset(seed=99)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    env.agent_energies[parent] = threshold  # exactly at threshold

    env._handle_predator_reproduction(parent)

    assert len(env.agent_live_offspring_ids[parent]) == 1


def test_parent_cannot_reproduce_again_until_energy_refills():
    threshold = 10.0
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": threshold,
        "min_offspring_energy_predator": 1.0,
        "max_offspring_energy_predator": 100.0,
        "genome_mutation": {"rate": 0.0, "std": 0.0},
    })
    env.reset(seed=101)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    env.agent_energies[parent] = threshold  # just enough for one reproduction
    env.agent_genomes[parent] = Genome(offspring_investment_fraction=0.50)

    env._handle_predator_reproduction(parent)
    assert len(env.agent_live_offspring_ids[parent]) == 1
    assert env.agent_energies[parent] < threshold  # energy depleted below gate

    env._handle_predator_reproduction(parent)
    assert len(env.agent_live_offspring_ids[parent]) == 1  # second call blocked


def test_live_investment_metrics_reflect_actual_genomes():
    env = _make_test_env()
    env.reset(seed=111)

    known_fraction = 0.55
    for agent in env.agents:
        env.agent_genomes[agent] = Genome(offspring_investment_fraction=known_fraction)

    metrics = env._build_live_investment_metrics()

    assert metrics["predator_investment_fraction_mean"] == pytest.approx(known_fraction)
    assert metrics["prey_investment_fraction_mean"] == pytest.approx(known_fraction)
    assert metrics["predator_investment_fraction_p50"] == pytest.approx(known_fraction)
    assert metrics["prey_investment_fraction_p50"] == pytest.approx(known_fraction)
    assert metrics["predator_investment_fraction_std"] == pytest.approx(0.0)
    assert metrics["prey_investment_fraction_std"] == pytest.approx(0.0)


def test_child_genome_recorded_in_event_log_after_reproduction():
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": 10.0,
        "min_offspring_energy_predator": 1.0,
        "max_offspring_energy_predator": 100.0,
        "genome_mutation": {"rate": 1.0, "std": 0.01},
    })
    env.reset(seed=121)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    env.agent_energies[parent] = 20.0

    env._handle_predator_reproduction(parent)

    child = env.agent_live_offspring_ids[parent][0]
    assert child in env.agent_genomes
    assert env.agent_event_log[child]["genome"] == env.agent_genomes[child].to_dict()
    assert env.agent_event_log[child]["parent_id"] == parent


def test_offspring_inherits_mutated_parent_genome_and_invested_initial_energy():
    env = _make_test_env(
        overrides={
            "predator_creation_energy_threshold": 10.0,
            "min_offspring_energy_predator": 1.0,
            "max_offspring_energy_predator": 5.0,
            "founder_genome": {
                "predator": {
                    "offspring_investment_fraction_mean": 0.25,
                    "offspring_investment_fraction_std": 0.0,
                },
                "prey": {
                    "offspring_investment_fraction_mean": 0.25,
                    "offspring_investment_fraction_std": 0.0,
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
    env.agent_genomes[parent] = Genome(offspring_investment_fraction=0.25)

    env._handle_predator_reproduction(parent)

    children = env.agent_live_offspring_ids[parent]
    assert len(children) == 1
    child = children[0]
    expected_offspring_energy = 5.0
    assert child in env.agent_genomes
    assert env.agent_parents[child] == parent
    assert env.agent_energies[child] == pytest.approx(expected_offspring_energy)
    assert env.agent_energies[parent] == pytest.approx(parent_energy - expected_offspring_energy)
    assert env.agent_stats_live[child]["offspring_initial_energy"] == pytest.approx(expected_offspring_energy)
    assert env.agent_stats_live[parent]["reproduction_energy_invested_sum"] == pytest.approx(expected_offspring_energy)
    assert env.agent_genomes[child].to_dict() != env.agent_genomes[parent].to_dict()


def test_offspring_energy_uses_investment_fraction_without_clamping():
    env = _make_test_env(
        overrides={
            "predator_creation_energy_threshold": 10.0,
            "min_offspring_energy_predator": 1.0,
            "max_offspring_energy_predator": 8.0,
            "founder_genome": {
                "predator": {
                    "offspring_investment_fraction_mean": 0.30,
                    "offspring_investment_fraction_std": 0.0,
                },
            },
            "genome_mutation": {"rate": 0.0, "std": 0.0},
        }
    )
    env.reset(seed=617)
    env.rewards = {}

    parent = next(agent for agent in env.agents if agent.startswith("predator"))
    parent_energy = 20.0
    env.agent_energies[parent] = parent_energy
    env.agent_genomes[parent] = Genome(offspring_investment_fraction=0.30)

    env._handle_predator_reproduction(parent)

    children = env.agent_live_offspring_ids[parent]
    assert len(children) == 1
    child = children[0]
    assert env.agent_energies[child] == pytest.approx(6.0)
    assert env.agent_energies[parent] == pytest.approx(14.0)


def test_reproduction_threshold_uses_fixed_base_threshold():
    env = _make_test_env(
        overrides={
            "predator_creation_energy_threshold": 10.0,
            "founder_genome": {
                "predator": {
                    "offspring_investment_fraction_mean": 0.35,
                    "offspring_investment_fraction_std": 0.0,
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


def test_movement_cost_uses_actual_distance_without_genome_multiplier():
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
