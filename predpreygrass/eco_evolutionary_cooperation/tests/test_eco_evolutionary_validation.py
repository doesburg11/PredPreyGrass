import copy

import numpy as np
import pytest

from predpreygrass.eco_evolutionary_cooperation.config.config_env_eco_evolutionary import config_env
from predpreygrass.eco_evolutionary_cooperation.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.eco_evolutionary_cooperation.utils.episode_return_callback import EpisodeReturn
from predpreygrass.eco_evolutionary_cooperation.utils.genome import Genome, mutate_genome


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
                    "predator_cooperation_rate_mean": 0.30,
                    "predator_cooperation_rate_p50": 0.28,
                    "prey_cooperation_rate_mean": 0.10,
                    "prey_cooperation_rate_p50": 0.08,
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
            "predator_cooperation_rate_mean": 0.35,
            "prey_cooperation_rate_p50": 0.05,
        }


class _FakeVectorEnv:
    def __init__(self, envs):
        self.envs = envs


def test_episode_return_callback_logs_eco_evolution_metrics():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(episode=_FakeEpisode(), metrics_logger=logger)

    assert logger.values["eco_evolution/predator_cooperation_rate_mean"] == pytest.approx(0.30)
    assert logger.values["eco_evolution/predator_cooperation_rate_p50"] == pytest.approx(0.28)
    assert logger.values["eco_evolution/prey_cooperation_rate_mean"] == pytest.approx(0.10)
    assert logger.values["eco_evolution/prey_cooperation_rate_p50"] == pytest.approx(0.08)
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

    assert logger.values["eco_evolution/predator_cooperation_rate_mean"] == pytest.approx(0.35)
    assert logger.values["eco_evolution/prey_cooperation_rate_p50"] == pytest.approx(0.05)


def test_episode_return_callback_logs_eco_metrics_from_vector_env_fallback():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(
        episode=_FakeEpisodeWithoutInfos(),
        metrics_logger=logger,
        env=_FakeVectorEnv([_FakeMetricsEnv()]),
        env_index=0,
    )

    assert logger.values["eco_evolution/predator_cooperation_rate_mean"] == pytest.approx(0.35)
    assert logger.values["eco_evolution/prey_cooperation_rate_p50"] == pytest.approx(0.05)


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
    assert "predator_cooperation_rate_mean" in metrics
    assert "prey_cooperation_rate_mean" in metrics
    assert "predator_cooperation_rate_p25" in metrics
    assert "predator_cooperation_rate_p50" in metrics
    assert "predator_cooperation_rate_p75" in metrics


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
    parent_rate = env.agent_genomes[parent].cooperation_rate
    env.agent_energies[parent] = 20.0

    env._handle_predator_reproduction(parent)

    child = env.agent_live_offspring_ids[parent][0]
    assert env.agent_genomes[child].cooperation_rate == pytest.approx(parent_rate)


def test_zero_mutation_std_produces_exact_copy_even_at_full_rate():
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": 10.0,
        "genome_mutation": {"rate": 1.0, "std": 0.0},
    })
    env.reset(seed=33)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    parent_rate = env.agent_genomes[parent].cooperation_rate
    env.agent_energies[parent] = 20.0

    env._handle_predator_reproduction(parent)

    child = env.agent_live_offspring_ids[parent][0]
    assert env.agent_genomes[child].cooperation_rate == pytest.approx(parent_rate)


def test_mutation_never_violates_trait_bounds():
    rng = np.random.default_rng(44)
    config = {"genome_mutation": {"rate": 1.0, "std": 0.5}, "trait_bounds": {}}
    genome = Genome(cooperation_rate=0.9)  # near upper bound

    for _ in range(500):
        genome = mutate_genome(genome, config, rng)
        assert 0.0 <= genome.cooperation_rate <= 1.0


def test_genome_does_not_change_across_steps():
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": 999.0,
        "prey_creation_energy_threshold": 999.0,
    })
    env.reset(seed=66)

    predator = next(a for a in env.agents if a.startswith("predator"))
    rate_at_birth = env.agent_genomes[predator].cooperation_rate

    for _ in range(5):
        actions = {a: 0 for a in env.agents}
        env.step(actions)
        if predator in env.agent_genomes:
            assert env.agent_genomes[predator].cooperation_rate == pytest.approx(rate_at_birth)


def test_multi_generation_ancestry_chain():
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": 10.0,
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
    env.agent_genomes[parent] = Genome(cooperation_rate=0.3)
    env.agent_energies[parent] = 9.99  # just below threshold

    env._handle_predator_reproduction(parent)

    assert env.agent_live_offspring_ids[parent] == []
    assert env.agent_offspring_counts[parent] == 0


def test_energy_exactly_at_threshold_triggers_one_offspring():
    threshold = 10.0
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": threshold,
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
        "genome_mutation": {"rate": 0.0, "std": 0.0},
    })
    env.reset(seed=101)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    env.agent_energies[parent] = threshold  # just enough for one reproduction
    env.agent_genomes[parent] = Genome(cooperation_rate=0.5)

    env._handle_predator_reproduction(parent)
    assert len(env.agent_live_offspring_ids[parent]) == 1
    assert env.agent_energies[parent] < threshold  # energy depleted below gate

    env._handle_predator_reproduction(parent)
    assert len(env.agent_live_offspring_ids[parent]) == 1  # second call blocked


def test_live_genome_metrics_reflect_actual_genomes():
    env = _make_test_env()
    env.reset(seed=111)

    known_rate = 0.4
    for agent in env.agents:
        env.agent_genomes[agent] = Genome(cooperation_rate=known_rate)

    metrics = env._build_live_genome_metrics()

    assert metrics["predator_cooperation_rate_mean"] == pytest.approx(0.4)
    assert metrics["prey_cooperation_rate_mean"] == pytest.approx(0.4)
    assert metrics["predator_cooperation_rate_p50"] == pytest.approx(0.4)
    assert metrics["prey_cooperation_rate_p50"] == pytest.approx(0.4)
    assert metrics["predator_cooperation_rate_std"] == pytest.approx(0.0)
    assert metrics["prey_cooperation_rate_std"] == pytest.approx(0.0)


def test_child_genome_recorded_in_event_log_after_reproduction():
    env = _make_test_env(overrides={
        "predator_creation_energy_threshold": 10.0,
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
            "founder_genome": {
                "predator": {
                    "cooperation_rate_mean": 0.0,
                    "cooperation_rate_std": 0.0,
                },
                "prey": {
                    "cooperation_rate_mean": 0.0,
                    "cooperation_rate_std": 0.0,
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
    env.agent_genomes[parent] = Genome(cooperation_rate=0.3)

    env._handle_predator_reproduction(parent)

    children = env.agent_live_offspring_ids[parent]
    assert len(children) == 1
    child = children[0]
    expected_offspring_energy = env.initial_energy_predator
    assert child in env.agent_genomes
    assert env.agent_parents[child] == parent
    assert env.agent_energies[child] == pytest.approx(expected_offspring_energy)
    assert env.agent_energies[parent] == pytest.approx(parent_energy - expected_offspring_energy)
    assert env.agent_stats_live[child]["offspring_initial_energy"] == pytest.approx(expected_offspring_energy)
    assert env.agent_stats_live[parent]["reproduction_energy_invested_sum"] == pytest.approx(expected_offspring_energy)
    assert env.agent_genomes[child].to_dict() != env.agent_genomes[parent].to_dict()


def test_offspring_receives_fixed_initial_energy():
    env = _make_test_env(
        overrides={
            "predator_creation_energy_threshold": 10.0,
            "initial_energy_predator": 5.0,
            "genome_mutation": {"rate": 0.0, "std": 0.0},
        }
    )
    env.reset(seed=617)
    env.rewards = {}

    parent = next(agent for agent in env.agents if agent.startswith("predator"))
    env.agent_energies[parent] = 20.0
    env.agent_genomes[parent] = Genome(cooperation_rate=0.0)

    env._handle_predator_reproduction(parent)

    children = env.agent_live_offspring_ids[parent]
    assert len(children) == 1
    child = children[0]
    assert env.agent_energies[child] == pytest.approx(5.0)
    assert env.agent_energies[parent] == pytest.approx(15.0)


def test_reproduction_threshold_uses_fixed_base_threshold():
    env = _make_test_env(
        overrides={
            "predator_creation_energy_threshold": 10.0,
            "founder_genome": {
                "predator": {
                    "cooperation_rate_mean": 0.0,
                    "cooperation_rate_std": 0.0,
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
            "basal_energy_cost_predator": 0.2,
            "movement_energy_cost_per_cell_predator": 0.05,
        }
    )
    env.reset(seed=1020)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    env.agent_genomes[predator] = Genome(cooperation_rate=0.7)
    _place_agent(env, predator, (10, 10))
    start_energy = 10.0
    env.agent_energies[predator] = start_energy
    env.grid_world_state[0, *env.agent_positions[predator]] = start_energy

    env._apply_time_step_update()
    stay_action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (0, 0))
    env._process_agent_movements({predator: stay_action})

    # Basal decay is independent of cooperation_rate (cooperation only affects
    # this-step gain events, not the decay term).
    assert env.agent_energies[predator] == pytest.approx(start_energy - 0.2)


def test_movement_cost_uses_actual_distance_without_genome_multiplier():
    env = _make_test_env(
        overrides={
            "basal_energy_cost_predator": 0.2,
            "movement_energy_cost_per_cell_predator": 0.05,
        }
    )
    env.reset(seed=1030)

    predator = next(agent for agent in env.agents if agent.startswith("predator"))
    env.agent_genomes[predator] = Genome(cooperation_rate=0.7)
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


# ---- Cooperative-donation mechanism tests ----


def test_donation_shares_gain_with_same_species_neighbor_in_range():
    env = _make_test_env(overrides={"n_initial_active_predators": 2, "cooperation_range": 2})
    env.reset(seed=2001)
    env.rewards = {}
    env.terminations = {}

    donor, recipient = [a for a in env.agents if a.startswith("predator")]
    prey = next(a for a in env.agents if a.startswith("prey"))

    _place_agent(env, donor, (5, 5))
    _place_agent(env, recipient, (5, 6))  # Chebyshev distance 1, within range 2
    _place_agent(env, prey, (5, 5))  # co-located with donor -> gets caught

    env.agent_genomes[donor] = Genome(cooperation_rate=0.5)
    env.agent_energies[donor] = 3.0
    env.agent_energies[recipient] = 3.0
    env.agent_energies[prey] = 4.0

    donor_energy_before = env.agent_energies[donor]
    recipient_energy_before = env.agent_energies[recipient]

    env._handle_predator_engagement(donor)

    donation_total = 0.5 * 4.0  # cooperation_rate * prey_energy
    assert env.agent_energies[recipient] == pytest.approx(recipient_energy_before + donation_total)
    assert env.agent_energies[donor] == pytest.approx(donor_energy_before + (4.0 - donation_total))
    assert env.agent_stats_live[donor]["energy_donated"] == pytest.approx(donation_total)
    assert env.agent_stats_live[recipient]["energy_received"] == pytest.approx(donation_total)


def test_no_donation_without_eligible_neighbor():
    env = _make_test_env(overrides={"n_initial_active_predators": 1, "cooperation_range": 2})
    env.reset(seed=2002)
    env.rewards = {}
    env.terminations = {}

    donor = next(a for a in env.agents if a.startswith("predator"))
    prey = next(a for a in env.agents if a.startswith("prey"))
    _place_agent(env, donor, (5, 5))
    _place_agent(env, prey, (5, 5))

    env.agent_genomes[donor] = Genome(cooperation_rate=0.8)
    env.agent_energies[donor] = 3.0
    env.agent_energies[prey] = 4.0
    donor_energy_before = env.agent_energies[donor]

    env._handle_predator_engagement(donor)

    # No same-species neighbor within range -> donor keeps the full gain.
    assert env.agent_energies[donor] == pytest.approx(donor_energy_before + 4.0)
    assert env.agent_stats_live[donor]["energy_donated"] == pytest.approx(0.0)


def test_no_donation_to_other_species_neighbor():
    env = _make_test_env(overrides={"n_initial_active_predators": 1, "cooperation_range": 2})
    env.reset(seed=2003)
    env.rewards = {}
    env.terminations = {}

    donor = next(a for a in env.agents if a.startswith("predator"))
    prey = next(a for a in env.agents if a.startswith("prey"))
    _place_agent(env, donor, (5, 5))
    _place_agent(env, prey, (5, 5))

    env.agent_genomes[donor] = Genome(cooperation_rate=0.8)
    env.agent_energies[donor] = 3.0
    env.agent_energies[prey] = 4.0
    donor_energy_before = env.agent_energies[donor]

    # prey is the only other agent nearby, but it's a different species (and
    # about to be eaten) -- it must not be eligible to receive a donation.
    env._handle_predator_engagement(donor)

    assert env.agent_energies[donor] == pytest.approx(donor_energy_before + 4.0)


def test_zero_cooperation_rate_never_donates():
    env = _make_test_env(overrides={"n_initial_active_predators": 2, "cooperation_range": 2})
    env.reset(seed=2004)
    env.rewards = {}
    env.terminations = {}

    donor, recipient = [a for a in env.agents if a.startswith("predator")]
    prey = next(a for a in env.agents if a.startswith("prey"))
    _place_agent(env, donor, (5, 5))
    _place_agent(env, recipient, (5, 6))
    _place_agent(env, prey, (5, 5))

    env.agent_genomes[donor] = Genome(cooperation_rate=0.0)
    env.agent_energies[donor] = 3.0
    env.agent_energies[recipient] = 3.0
    env.agent_energies[prey] = 4.0
    recipient_energy_before = env.agent_energies[recipient]

    env._handle_predator_engagement(donor)

    assert env.agent_energies[recipient] == pytest.approx(recipient_energy_before)


def test_non_eating_step_never_triggers_donation():
    env = _make_test_env(overrides={"n_initial_active_predators": 2, "cooperation_range": 2})
    env.reset(seed=2005)
    env.rewards = {}
    env.terminations = {}

    donor, recipient = [a for a in env.agents if a.startswith("predator")]
    _place_agent(env, donor, (5, 5))
    _place_agent(env, recipient, (5, 6))

    env.agent_genomes[donor] = Genome(cooperation_rate=1.0)
    env.agent_energies[donor] = 3.0
    env.agent_energies[recipient] = 3.0
    recipient_energy_before = env.agent_energies[recipient]

    # No prey co-located with donor -> no catch this step -> nothing to share.
    env._handle_predator_engagement(donor)

    assert env.agent_energies[recipient] == pytest.approx(recipient_energy_before)


def test_is_kin_detects_parent_offspring_and_siblings():
    env = _make_test_env(overrides={"predator_creation_energy_threshold": 10.0})
    env.reset(seed=2006)
    env.rewards = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    env.agent_energies[parent] = 20.0
    env._handle_predator_reproduction(parent)
    child_a = env.agent_live_offspring_ids[parent][0]

    env.agent_energies[parent] = 20.0
    env._handle_predator_reproduction(parent)
    child_b = env.agent_live_offspring_ids[parent][1]

    assert env._is_kin(parent, child_a) is True
    assert env._is_kin(child_a, parent) is True
    assert env._is_kin(child_a, child_b) is True  # full siblings
    assert env._is_kin(parent, parent) is False  # trivial, not meaningful as "kin"


def test_local_relatedness_proxy_reflects_kin_vs_stranger_donations():
    env = _make_test_env(overrides={"n_initial_active_predators": 1, "cooperation_range": 2})
    env.reset(seed=2007)
    env.rewards = {}
    env.terminations = {}

    parent = next(a for a in env.agents if a.startswith("predator"))
    prey = next(a for a in env.agents if a.startswith("prey"))

    # Give parent enough energy to reproduce, spawning a kin neighbor nearby.
    env.agent_energies[parent] = env.predator_creation_energy_threshold
    env._handle_predator_reproduction(parent)
    child = env.agent_live_offspring_ids[parent][0]

    _place_agent(env, parent, (5, 5))
    _place_agent(env, child, (5, 6))
    _place_agent(env, prey, (5, 5))

    env.agent_genomes[parent] = Genome(cooperation_rate=0.5)
    env.agent_energies[parent] = 3.0
    env.agent_energies[prey] = 4.0

    env._handle_predator_engagement(parent)

    metrics = env._build_episode_training_metrics()
    # The only donation this episode went to genuine kin (the child).
    assert metrics["predator_local_relatedness_proxy"] == pytest.approx(1.0)
    assert metrics["predator_energy_donated_total"] > 0.0
