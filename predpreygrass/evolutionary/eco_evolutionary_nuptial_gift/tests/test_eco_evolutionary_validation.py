import copy

import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_nuptial_gift.config.config_env_eco_evolutionary import config_env
from predpreygrass.evolutionary.eco_evolutionary_nuptial_gift.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.evolutionary.eco_evolutionary_nuptial_gift.utils.episode_return_callback import EpisodeReturn
from predpreygrass.evolutionary.eco_evolutionary_nuptial_gift.utils.genome import Genome, mutate_genome


def _make_test_env(overrides=None):
    """Return a tiny eco_evolutionary_nuptial_gift env for deterministic unit tests."""
    cfg = copy.deepcopy(config_env)
    cfg.update(
        {
            "n_initial_active_predator_male": 1,
            "n_initial_active_predator_female": 1,
            "n_initial_active_prey": 1,
            "n_possible_predator_male": 8,
            "n_possible_predator_female": 8,
            "n_possible_prey": 8,
            "initial_num_grass": 4,
        }
    )
    if overrides:
        cfg.update(overrides)
    # Pin min == max so initial population is deterministic in tests.
    cfg["n_initial_active_predator_male_min"] = cfg["n_initial_active_predator_male"]
    cfg["n_initial_active_predator_female_min"] = cfg["n_initial_active_predator_female"]
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
    def get_return(self):
        return 2.0

    def env_steps(self):
        return 3

    @property
    def agent_episodes(self):
        return {
            "predator_male_0": _FakeAgentEpisode(1.0),
            "predator_female_0": _FakeAgentEpisode(2.0),
            "prey_0": _FakeAgentEpisode(-1.0),
        }

    def get_last_infos(self):
        return {
            "__all__": {
                "training_metrics": {
                    "predator_male_male_donation_rate_mean": 0.30,
                    "predator_male_male_donation_rate_p50": 0.28,
                    "predator_female_male_donation_rate_mean": 0.30,
                    "female_reproduction_gift_share_mean": 0.9,
                    "predator_male_movement_energy_spent_mean": 0.4,
                    "prey_offspring_count_mean": 2.0,
                }
            },
            "predator_female_0": {
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
            "predator_male_male_donation_rate_mean": 0.35,
            "female_reproduction_gift_share_mean": 0.05,
        }


class _FakeVectorEnv:
    def __init__(self, envs):
        self.envs = envs


def test_episode_return_callback_logs_eco_evolution_metrics():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(episode=_FakeEpisode(), metrics_logger=logger)

    assert logger.values["eco_evolution/predator_male_male_donation_rate_mean"] == pytest.approx(0.30)
    assert logger.values["eco_evolution/predator_male_male_donation_rate_p50"] == pytest.approx(0.28)
    assert logger.values["eco_evolution/female_reproduction_gift_share_mean"] == pytest.approx(0.9)
    assert logger.values["eco_evolution/predator_male_movement_energy_spent_mean"] == pytest.approx(0.4)
    assert logger.values["eco_evolution/prey_offspring_count_mean"] == pytest.approx(2.0)
    assert logger.values["predator_female_episode_return_p50"] == pytest.approx(2.0)
    assert logger.values["predator_female_return_per_lifetime_mean"] == pytest.approx(1.0)


def test_episode_return_callback_logs_eco_metrics_from_env_fallback():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(
        episode=_FakeEpisodeWithoutInfos(),
        metrics_logger=logger,
        env=_FakeMetricsEnv(),
    )

    assert logger.values["eco_evolution/predator_male_male_donation_rate_mean"] == pytest.approx(0.35)
    assert logger.values["eco_evolution/female_reproduction_gift_share_mean"] == pytest.approx(0.05)


def test_episode_return_callback_logs_eco_metrics_from_vector_env_fallback():
    callback = EpisodeReturn()
    logger = _FakeMetricsLogger()

    callback.on_episode_end(
        episode=_FakeEpisodeWithoutInfos(),
        metrics_logger=logger,
        env=_FakeVectorEnv([_FakeMetricsEnv()]),
        env_index=0,
    )

    assert logger.values["eco_evolution/predator_male_male_donation_rate_mean"] == pytest.approx(0.35)
    assert logger.values["eco_evolution/female_reproduction_gift_share_mean"] == pytest.approx(0.05)


def test_every_acted_agent_gets_next_or_final_observation():
    env = _make_test_env(
        {
            "seed": 456,
            "max_steps": 120,
            "grid_size": 12,
            "n_initial_active_predator_male": 3,
            "n_initial_active_predator_female": 3,
            "n_initial_active_prey": 6,
            "n_possible_predator_male": 60,
            "n_possible_predator_female": 60,
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
            "predator_female_creation_energy_threshold": 999.0,
            "prey_creation_energy_threshold": 999.0,
        }
    )
    env.reset(seed=124)

    male = next(agent for agent in env.agents if agent.startswith("predator_male"))
    prey = next(agent for agent in env.agents if agent.startswith("prey"))
    _place_agent(env, male, (5, 5))
    _place_agent(env, prey, (5, 5))

    stay_action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (0, 0))
    observations, rewards, terminations, truncations, infos = env.step(
        {agent: stay_action for agent in env.agents}
    )

    assert prey in observations
    assert male in observations
    assert rewards[prey] == pytest.approx(env._get_role_specific("penalty_prey_caught", prey))
    assert terminations[prey] is True
    assert truncations[prey] is False
    assert terminations[male] is True
    assert truncations[male] is False
    assert terminations["__all__"] is True
    assert truncations["__all__"] is False
    assert "final_cumulative_reward" in infos[male]
    assert env.agents == []


def test_terminal_reward_agent_is_not_returned_as_next_actor():
    env = _make_test_env(
        overrides={
            "n_initial_active_prey": 2,
            "predator_female_creation_energy_threshold": 999.0,
            "prey_creation_energy_threshold": 999.0,
        }
    )
    env.reset(seed=126)

    male = next(agent for agent in env.agents if agent.startswith("predator_male"))
    eaten_prey, survivor_prey = [agent for agent in env.agents if agent.startswith("prey")]
    _place_agent(env, male, (5, 5))
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
            "predator_female_creation_energy_threshold": 999.0,
            "prey_creation_energy_threshold": 999.0,
        }
    )
    env.reset(seed=125)

    male = next(agent for agent in env.agents if agent.startswith("predator_male"))
    female = next(agent for agent in env.agents if agent.startswith("predator_female"))
    prey = next(agent for agent in env.agents if agent.startswith("prey"))
    _place_agent(env, male, (1, 1))
    _place_agent(env, female, (2, 2))
    _place_agent(env, prey, (env.grid_size - 2, env.grid_size - 2))

    stay_action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (0, 0))
    observations, _, terminations, truncations, infos = env.step({agent: stay_action for agent in env.agents})

    n_ch = env._n_obs_channels()
    assert set(observations) == {male, female, prey}
    assert observations[male].shape == (n_ch, env.predator_obs_range, env.predator_obs_range)
    assert observations[prey].shape == (n_ch, env.prey_obs_range, env.prey_obs_range)
    assert terminations[male] is False
    assert terminations[prey] is False
    assert truncations[male] is True
    assert truncations[prey] is True
    assert terminations["__all__"] is False
    assert truncations["__all__"] is True
    assert env.agents == []
    assert "training_metrics" in infos["__all__"]
    metrics = infos["__all__"]["training_metrics"]
    assert "predator_male_male_donation_rate_mean" in metrics
    assert "predator_female_male_donation_rate_mean" in metrics
    assert "prey_male_donation_rate_mean" in metrics
    assert "predator_male_male_donation_rate_p25" in metrics
    assert "predator_male_male_donation_rate_p50" in metrics
    assert "predator_male_male_donation_rate_p75" in metrics
    assert "female_reproduction_gift_share_mean" in metrics


def test_founders_receive_genomes_in_event_logs():
    env = _make_test_env()
    env.reset(seed=515)

    for agent in env.agents:
        assert agent in env.agent_genomes
        assert env.agent_stats_live[agent]["genome"] == env.agent_genomes[agent].to_dict()
        assert env.agent_event_log[agent]["genome"] == env.agent_genomes[agent].to_dict()


def test_genome_disabled_produces_no_genomes():
    env = _make_test_env(overrides={"genome_enabled": False})
    env.reset(seed=11)

    assert env.agent_genomes == {}
    for agent in env.agents:
        assert env.agent_event_log[agent]["genome"] is None
        assert env.agent_stats_live[agent]["genome"] is None


def test_mutation_never_violates_trait_bounds():
    rng = np.random.default_rng(44)
    config = {"genome_mutation": {"rate": 1.0, "std": 0.5}, "trait_bounds": {}}
    genome = Genome(male_donation_rate=0.9)  # near upper bound

    for _ in range(500):
        genome = mutate_genome(genome, config, rng)
        assert 0.0 <= genome.male_donation_rate <= 1.0


def test_genome_does_not_change_across_steps():
    env = _make_test_env(overrides={
        "predator_female_creation_energy_threshold": 999.0,
        "prey_creation_energy_threshold": 999.0,
    })
    env.reset(seed=66)

    female = next(a for a in env.agents if a.startswith("predator_female"))
    rate_at_birth = env.agent_genomes[female].male_donation_rate

    for _ in range(5):
        actions = {a: 0 for a in env.agents}
        env.step(actions)
        if female in env.agent_genomes:
            assert env.agent_genomes[female].male_donation_rate == pytest.approx(rate_at_birth)


# ---- Sex-specific reproduction and taxonomy tests ----


def test_only_females_ever_reproduce():
    """Give every agent (male and female) ample energy; only females should spawn offspring."""
    env = _make_test_env(overrides={
        "predator_female_creation_energy_threshold": 10.0,
        "n_initial_active_predator_male": 2,
        "n_initial_active_predator_female": 2,
    })
    env.reset(seed=909)
    env.rewards = {}

    for agent in list(env.agents):
        if agent.startswith("predator"):
            env.agent_energies[agent] = 50.0

    stay_action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (0, 0))
    env.step({a: stay_action for a in env.agents})

    males_before = [a for a in env.agents if a.startswith("predator_male")]
    for m in males_before:
        assert env.agent_offspring_counts.get(m, 0) == 0
    females = [a for a in env.agents if a.startswith("predator_female")]
    assert any(env.agent_offspring_counts.get(f, 0) > 0 for f in females)


def test_female_reproduction_assigns_offspring_sex_roughly_evenly():
    env = _make_test_env(overrides={
        "predator_female_creation_energy_threshold": 10.0,
        "n_possible_predator_male": 150,
        "n_possible_predator_female": 150,
    })
    env.reset(seed=1234)
    env.rewards = {}

    female = next(a for a in env.agents if a.startswith("predator_female"))
    sexes = []
    # Directly exercise the sex-assignment coin flip many times via repeated
    # reproduction calls (refilling energy each time) to check it isn't biased.
    for _ in range(200):
        env.agent_energies[female] = env.predator_female_creation_energy_threshold
        before = list(env.agent_live_offspring_ids[female])
        env._handle_female_reproduction(female)
        after = env.agent_live_offspring_ids[female]
        if len(after) > len(before):
            child = after[-1]
            sexes.append("male" if "predator_male" in child else "female")
        else:
            break  # ID pool exhausted
        if len(sexes) >= 100:
            break

    assert len(sexes) >= 50
    male_fraction = sexes.count("male") / len(sexes)
    assert 0.3 < male_fraction < 0.7


def test_genome_inherits_from_mother_regardless_of_offspring_sex():
    env = _make_test_env(overrides={
        "predator_female_creation_energy_threshold": 10.0,
        "genome_mutation": {"rate": 0.0, "std": 0.0},
    })
    env.reset(seed=4242)
    env.rewards = {}

    mother = next(a for a in env.agents if a.startswith("predator_female"))
    env.agent_genomes[mother] = Genome(male_donation_rate=0.42)

    seen_sexes = set()
    for _ in range(20):
        env.agent_energies[mother] = env.predator_female_creation_energy_threshold
        before = list(env.agent_live_offspring_ids[mother])
        env._handle_female_reproduction(mother)
        after = env.agent_live_offspring_ids[mother]
        if len(after) == len(before):
            break
        child = after[-1]
        assert env.agent_genomes[child].male_donation_rate == pytest.approx(0.42)
        seen_sexes.add(env._policy_group(child))
        if seen_sexes == {"predator_male", "predator_female"}:
            break

    assert len(seen_sexes) >= 1  # at least verified inheritance held for whichever sex(es) appeared


def test_male_hunts_prey_female_never_does():
    env = _make_test_env(overrides={"n_initial_active_predator_female": 1, "n_initial_active_predator_male": 0})
    env.reset(seed=55)
    env.rewards = {}
    env.terminations = {}

    female = next(a for a in env.agents if a.startswith("predator_female"))
    prey = next(a for a in env.agents if a.startswith("prey"))
    _place_agent(env, female, (7, 7))
    _place_agent(env, prey, (7, 7))
    env.agent_energies[prey] = 4.0
    female_energy_before = env.agent_energies[female]

    # Females never hunt: co-locating with prey and calling grazing must not
    # consume the prey or grant prey-sized energy.
    env._handle_female_grazing(female)

    assert prey in env.prey_positions  # prey untouched, still alive
    assert env.agent_energies[female] <= female_energy_before + env.female_grass_energy_gain_cap + 1e-9


def test_female_grazing_is_capped_below_basal_decay():
    env = _make_test_env()
    env.reset(seed=56)
    env.rewards = {}
    env.terminations = {}

    female = next(a for a in env.agents if a.startswith("predator_female"))
    # Move the female onto an existing grass patch and force it fully grown.
    grass_id, grass_pos = next(iter(env.grass_positions.items()))
    _place_agent(env, female, grass_pos)
    env.grass_energies[grass_id] = env.max_energy_grass
    energy_before = env.agent_energies[female]

    env._handle_female_grazing(female)

    gained = env.agent_energies[female] - energy_before
    assert gained == pytest.approx(env.female_grass_energy_gain_cap)
    assert gained < env.basal_energy_cost_predator_female


def test_female_cannot_reach_threshold_from_grazing_alone_over_many_steps():
    """Mechanical guarantee check: with no gifts, a grazing-only female's energy
    must trend non-increasing over time (cap is below basal decay)."""
    env = _make_test_env(overrides={
        "n_initial_active_predator_male": 0,
        "n_initial_active_predator_female": 1,
        "predator_female_creation_energy_threshold": 10.0,
    })
    env.reset(seed=57)

    female = next(a for a in env.agents if a.startswith("predator_female"))
    start_energy = env.agent_energies[female]

    stay_action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (0, 0))
    for _ in range(30):
        if female not in env.agents:
            break
        env.step({a: stay_action for a in env.agents if a in env.agents})

    if female in env.agent_energies:
        assert env.agent_energies[female] <= start_energy + 1e-6
    assert female not in env.agent_genomes or env.agent_offspring_counts.get(female, 0) == 0


# ---- Nuptial-gift donation mechanism tests ----


def test_gift_flows_from_male_to_nearby_female():
    env = _make_test_env(overrides={
        "n_initial_active_predator_male": 1,
        "n_initial_active_predator_female": 1,
        "cooperation_range": 2,
    })
    env.reset(seed=2001)
    env.rewards = {}
    env.terminations = {}

    male = next(a for a in env.agents if a.startswith("predator_male"))
    female = next(a for a in env.agents if a.startswith("predator_female"))
    prey = next(a for a in env.agents if a.startswith("prey"))

    _place_agent(env, male, (5, 5))
    _place_agent(env, female, (5, 6))  # Chebyshev distance 1, within range 2
    _place_agent(env, prey, (5, 5))  # co-located with male -> gets caught

    env.agent_genomes[male] = Genome(male_donation_rate=0.5)
    env.agent_energies[male] = 3.0
    env.agent_energies[female] = 3.0
    env.agent_energies[prey] = 4.0

    male_energy_before = env.agent_energies[male]
    female_energy_before = env.agent_energies[female]

    env._handle_male_hunting(male)

    donation_total = 0.5 * 4.0  # male_donation_rate * prey_energy
    assert env.agent_energies[female] == pytest.approx(female_energy_before + donation_total)
    assert env.agent_energies[male] == pytest.approx(male_energy_before + (4.0 - donation_total))
    assert env.agent_stats_live[male]["energy_donated"] == pytest.approx(donation_total)
    assert env.agent_stats_live[female]["energy_received"] == pytest.approx(donation_total)
    assert env.agent_stats_live[female]["lifetime_energy_from_gifts"] == pytest.approx(donation_total)


def test_no_gift_flows_female_to_male():
    """Females never hunt and never donate -- confirm no mechanism exists to
    move energy from a female to a nearby male."""
    env = _make_test_env(overrides={
        "n_initial_active_predator_male": 1,
        "n_initial_active_predator_female": 1,
    })
    env.reset(seed=2002)
    env.rewards = {}
    env.terminations = {}

    male = next(a for a in env.agents if a.startswith("predator_male"))
    female = next(a for a in env.agents if a.startswith("predator_female"))
    grass_id, grass_pos = next(iter(env.grass_positions.items()))
    _place_agent(env, male, (5, 5))
    _place_agent(env, female, grass_pos)
    env.grass_energies[grass_id] = env.max_energy_grass

    male_energy_before = env.agent_energies[male]

    env._handle_female_grazing(female)

    # Female grazing never touches the male's energy at all.
    assert env.agent_energies[male] == pytest.approx(male_energy_before)


def test_no_gift_between_two_males():
    env = _make_test_env(overrides={
        "n_initial_active_predator_male": 2,
        "n_initial_active_predator_female": 0,
        "cooperation_range": 2,
    })
    env.reset(seed=2003)
    env.rewards = {}
    env.terminations = {}

    donor, other_male = [a for a in env.agents if a.startswith("predator_male")]
    prey = next(a for a in env.agents if a.startswith("prey"))
    _place_agent(env, donor, (5, 5))
    _place_agent(env, other_male, (5, 6))
    _place_agent(env, prey, (5, 5))

    env.agent_genomes[donor] = Genome(male_donation_rate=1.0)
    env.agent_energies[donor] = 3.0
    env.agent_energies[other_male] = 3.0
    env.agent_energies[prey] = 4.0
    other_male_energy_before = env.agent_energies[other_male]

    env._handle_male_hunting(donor)

    assert env.agent_energies[other_male] == pytest.approx(other_male_energy_before)


def test_no_gift_without_eligible_female_neighbor():
    env = _make_test_env(overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 0})
    env.reset(seed=2004)
    env.rewards = {}
    env.terminations = {}

    male = next(a for a in env.agents if a.startswith("predator_male"))
    prey = next(a for a in env.agents if a.startswith("prey"))
    _place_agent(env, male, (5, 5))
    _place_agent(env, prey, (5, 5))

    env.agent_genomes[male] = Genome(male_donation_rate=0.8)
    env.agent_energies[male] = 3.0
    env.agent_energies[prey] = 4.0
    male_energy_before = env.agent_energies[male]

    env._handle_male_hunting(male)

    assert env.agent_energies[male] == pytest.approx(male_energy_before + 4.0)
    assert env.agent_stats_live[male]["energy_donated"] == pytest.approx(0.0)


def test_zero_donation_rate_never_donates():
    env = _make_test_env(overrides={
        "n_initial_active_predator_male": 1,
        "n_initial_active_predator_female": 1,
        "cooperation_range": 2,
    })
    env.reset(seed=2005)
    env.rewards = {}
    env.terminations = {}

    male = next(a for a in env.agents if a.startswith("predator_male"))
    female = next(a for a in env.agents if a.startswith("predator_female"))
    prey = next(a for a in env.agents if a.startswith("prey"))
    _place_agent(env, male, (5, 5))
    _place_agent(env, female, (5, 6))
    _place_agent(env, prey, (5, 5))

    env.agent_genomes[male] = Genome(male_donation_rate=0.0)
    env.agent_energies[male] = 3.0
    env.agent_energies[female] = 3.0
    env.agent_energies[prey] = 4.0
    female_energy_before = env.agent_energies[female]

    env._handle_male_hunting(male)

    assert env.agent_energies[female] == pytest.approx(female_energy_before)


def test_fixed_donation_rate_mode_applies_uniformly_without_genomes():
    """genome_enabled=False should use a fixed, non-inherited donation rate for
    every predator_male, driven by founder_genome's mean -- the fixed-genome
    fitness-sweep mode used by run_fixed_genome_sweep.sh."""
    env = _make_test_env(overrides={
        "genome_enabled": False,
        "founder_genome": {
            "predator_male": {"male_donation_rate_mean": 0.5, "male_donation_rate_std": 0.0},
            "predator_female": {"male_donation_rate_mean": 0.5, "male_donation_rate_std": 0.0},
            "prey": {"male_donation_rate_mean": 0.0, "male_donation_rate_std": 0.0},
        },
        "n_initial_active_predator_male": 1,
        "n_initial_active_predator_female": 1,
        "cooperation_range": 2,
    })
    env.reset(seed=5001)
    env.rewards = {}
    env.terminations = {}

    male = next(a for a in env.agents if a.startswith("predator_male"))
    female = next(a for a in env.agents if a.startswith("predator_female"))
    prey = next(a for a in env.agents if a.startswith("prey"))
    _place_agent(env, male, (5, 5))
    _place_agent(env, female, (5, 6))
    _place_agent(env, prey, (5, 5))

    assert env.agent_genomes == {}  # no genome objects at all in fixed mode
    env.agent_energies[male] = 3.0
    env.agent_energies[female] = 3.0
    env.agent_energies[prey] = 4.0
    female_energy_before = env.agent_energies[female]

    env._handle_male_hunting(male)

    donation_total = 0.5 * 4.0  # fixed founder mean * prey_energy
    assert env.agent_energies[female] == pytest.approx(female_energy_before + donation_total)


def test_fixed_donation_rate_zero_never_donates():
    env = _make_test_env(overrides={
        "genome_enabled": False,
        "founder_genome": {
            "predator_male": {"male_donation_rate_mean": 0.0, "male_donation_rate_std": 0.0},
            "predator_female": {"male_donation_rate_mean": 0.0, "male_donation_rate_std": 0.0},
            "prey": {"male_donation_rate_mean": 0.0, "male_donation_rate_std": 0.0},
        },
        "n_initial_active_predator_male": 1,
        "n_initial_active_predator_female": 1,
        "cooperation_range": 2,
    })
    env.reset(seed=5002)
    env.rewards = {}
    env.terminations = {}

    male = next(a for a in env.agents if a.startswith("predator_male"))
    female = next(a for a in env.agents if a.startswith("predator_female"))
    prey = next(a for a in env.agents if a.startswith("prey"))
    _place_agent(env, male, (5, 5))
    _place_agent(env, female, (5, 6))
    _place_agent(env, prey, (5, 5))

    env.agent_energies[male] = 3.0
    env.agent_energies[female] = 3.0
    env.agent_energies[prey] = 4.0
    female_energy_before = env.agent_energies[female]

    env._handle_male_hunting(male)

    assert env.agent_energies[female] == pytest.approx(female_energy_before)


def test_is_kin_detects_mother_and_siblings():
    env = _make_test_env(overrides={"predator_female_creation_energy_threshold": 10.0})
    env.reset(seed=2006)
    env.rewards = {}

    mother = next(a for a in env.agents if a.startswith("predator_female"))
    env.agent_energies[mother] = 20.0
    env._handle_female_reproduction(mother)
    child_a = env.agent_live_offspring_ids[mother][0]

    env.agent_energies[mother] = 20.0
    env._handle_female_reproduction(mother)
    child_b = env.agent_live_offspring_ids[mother][1]

    assert env._is_kin(mother, child_a) is True
    assert env._is_kin(child_a, mother) is True
    assert env._is_kin(child_a, child_b) is True  # full siblings, regardless of their sexes
    assert env._is_kin(mother, mother) is False  # trivial, not meaningful as "kin"


# ---- Three-way episode termination tests ----


def test_episode_terminates_when_prey_extinct():
    env = _make_test_env(overrides={"n_initial_active_prey": 1})
    env.reset(seed=3001)
    env.rewards = {}
    env.terminations = {}

    male = next(a for a in env.agents if a.startswith("predator_male"))
    prey = next(a for a in env.agents if a.startswith("prey"))
    _place_agent(env, male, (5, 5))
    _place_agent(env, prey, (5, 5))
    env.agent_energies[prey] = 4.0

    stay_action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (0, 0))
    _, _, terminations, _, _ = env.step({a: stay_action for a in env.agents})

    assert terminations["__all__"] is True


def test_episode_terminates_when_predator_male_extinct():
    env = _make_test_env(overrides={
        "n_initial_active_predator_male": 1,
        "n_initial_active_predator_female": 1,
        "basal_energy_cost_predator_male": 999.0,
    })
    env.reset(seed=3002)

    stay_action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (0, 0))
    _, _, terminations, _, _ = env.step({a: stay_action for a in env.agents})

    assert terminations["__all__"] is True


def test_episode_terminates_when_predator_female_extinct():
    env = _make_test_env(overrides={
        "n_initial_active_predator_male": 1,
        "n_initial_active_predator_female": 1,
        "basal_energy_cost_predator_female": 999.0,
    })
    env.reset(seed=3003)

    stay_action = next(i for i, move in env.action_to_move_tuple_agents.items() if move == (0, 0))
    _, _, terminations, _, _ = env.step({a: stay_action for a in env.agents})

    assert terminations["__all__"] is True


def test_live_genome_metrics_reflect_actual_genomes():
    env = _make_test_env()
    env.reset(seed=111)

    known_rate = 0.4
    for agent in env.agents:
        env.agent_genomes[agent] = Genome(male_donation_rate=known_rate)

    metrics = env._build_live_genome_metrics()

    assert metrics["predator_male_male_donation_rate_mean"] == pytest.approx(0.4)
    assert metrics["predator_female_male_donation_rate_mean"] == pytest.approx(0.4)
    assert metrics["predator_male_male_donation_rate_std"] == pytest.approx(0.0)
