"""Tests of the environment: only predators are agents, the catch and hunt
rules, births and episode ends."""

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import MOVES, OBSERVED, StagHuntRulePrey
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.geometry import Vector

STOP = MOVES - 1  # speed 0


def stopped(env, join=None):
    return {a: np.array([STOP, (join or {}).get(a, 0)]) for a in env.active}


def world(**config):
    """Hunters and mammoths far apart, no food, no metabolic costs."""
    settings = dict(
        predator_count=4,
        rabbit_count=1,
        mammoth_count=2,
        grass_count=0,
        steppe_count=0,
        energy_predator_resting_metabolic_cost=0.0,
        energy_predator_speed_cost=0.0,
        energy_mammoth_resting_metabolic_cost=0.0,
        energy_mammoth_speed_cost=0.0,
        energy_rabbit_resting_metabolic_cost=0.0,
        energy_rabbit_speed_cost=0.0,
        rabbit_flee_distance=0.0,
        mammoth_flee_distance=0.0,
    )
    settings.update(config)
    env = StagHuntRulePrey(settings)
    env.reset(seed=0)
    for i, predator in enumerate(env.predators):
        predator.position = Vector(100 + 150 * i, 700)
        predator.velocity = Vector(0, 0)
    for i, mammoth in enumerate(env.mammoths):
        mammoth.position = Vector(200 + 400 * i, 100)
        mammoth.velocity = Vector(0, 0)
    for rabbit in env.rabbits:
        rabbit.position = Vector(400, 400)
        rabbit.velocity = Vector(0, 0)
    return env


def put_on_mammoth(env, *numbers):
    """Puts those predators touching mammoth_0 for the rest of a decision."""
    mammoth = env.mammoths[0]
    for number in numbers:
        env.predators[number].position = Vector(mammoth.position.x + 10 * number, mammoth.position.y + 30)
    # The rule moves prey slowly; the decision is a single physics step.
    env.action_repeat = 1


def test_only_predators_are_agents():
    env = StagHuntRulePrey()
    obs, infos = env.reset(seed=0)
    assert sorted(obs) == sorted(f"predator_{i}" for i in range(10)) == sorted(infos)
    assert set(env.possible_agents) == {f"predator_{i}" for i in range(env.pool["predator"])}
    assert env.observation_spaces["predator_0"].contains(obs["predator_0"])
    assert env.action_spaces["predator_0"].contains(np.array([0, 1]))
    assert env.obs_size == 64


def test_step_reports_only_predators_through_an_episode_with_prey_deaths_and_births():
    env = StagHuntRulePrey(dict(truncate_on_prey_extinction=False, energy_rabbit_initial=1.0))
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    rabbits_seen = len(env.rabbits)
    for _ in range(300):
        obs, rewards, terminateds, truncateds, infos = env.step({a: np.array([rng.integers(MOVES), rng.integers(2)]) for a in env.active})
        for mapping in (obs, rewards, infos, terminateds, truncateds):
            assert all(k == "__all__" or k.startswith("predator") for k in mapping)
        for agent, observation in obs.items():
            assert observation.shape == (env.obs_size,)
        rabbits_seen = max(rabbits_seen, len(env.rabbits))
        if terminateds["__all__"] or truncateds["__all__"]:
            break
    assert rabbits_seen > 8


def test_missing_and_invalid_actions_are_refused():
    env = world()
    with pytest.raises(ValueError, match="no action"):
        env.step({})
    actions = stopped(env)
    actions["predator_0"] = np.array([MOVES, 0])
    with pytest.raises(ValueError, match="out of range"):
        env.step(actions)
    actions["predator_0"] = np.array([0, 2])
    with pytest.raises(ValueError, match="join"):
        env.step(actions)


def test_a_rabbit_is_caught_on_contact_for_half_its_energy():
    env = world()
    rabbit, hunter = env.rabbits[0], env.predators[0]
    rabbit.position = Vector(hunter.position.x + 5, hunter.position.y)
    rabbit.energy = 6.0
    before = hunter.energy
    env.action_repeat = 1
    _, _, terminateds, _, _ = env.step(stopped(env))
    assert not env.rabbits and hunter.energy == pytest.approx(before + 3.0)
    assert env.hunt_stats["rabbits_caught"] == 1 and "rabbit_0" not in terminateds


@pytest.mark.parametrize(
    "energies, mammoth, ratio, minimum, killed",
    [
        ((4.0, 6.0), 15.0, 0.6, 2, True),
        ((4.0, 6.0), 17.0, 0.6, 2, False),
        ((30.0,), 8.0, 0.6, 2, False),
        ((30.0,), 8.0, 0.6, 1, True),
        ((4.0, 6.0), 9.0, 1.0, 2, True),
        ((4.0, 6.0), 10.0, 1.0, 2, False),
    ],
)
def test_the_hunt_rule(energies, mammoth, ratio, minimum, killed):
    env = world(hunt_energy_ratio=ratio, hunt_min_party=minimum, energy_predator_max=80.0, reproduction_predator_threshold=70.0)
    env.mammoths[0].energy = mammoth
    numbers = range(len(energies))
    for number, energy in zip(numbers, energies):
        env.predators[number].energy = energy
    put_on_mammoth(env, *numbers)
    env.step(stopped(env, {f"predator_{n}": 1 for n in numbers}))
    assert (env.mammoths[0].id != "mammoth_0") == killed
    assert env.hunt_stats[f"kills_party_{len(energies)}"] == killed
    assert env.hunt_stats[f"hunts_party_{len(energies)}"] == 1


def test_a_failed_solo_hunt_costs_join_struggle_and_a_cooldown():
    env = world()
    env.mammoths[0].energy = 30.0
    hunter = env.predators[0]
    hunter.energy = 5.0
    put_on_mammoth(env, 0)
    env.step(stopped(env, {"predator_0": 1}))
    assert hunter.energy == pytest.approx(5.0 - 0.02 - 0.64)
    assert hunter.ready_at == env.decision + 2


def test_a_success_shares_the_carcass_with_bystanders():
    env = world()
    env.mammoths[0].energy = 4.0
    for number in (0, 1, 2):
        env.predators[number].energy = 3.0
    put_on_mammoth(env, 0, 1)
    env.predators[2].position = Vector(env.mammoths[0].position.x, env.mammoths[0].position.y + 90)
    env.step(stopped(env, {"predator_0": 1, "predator_1": 1}))
    assert env.predators[0].energy == pytest.approx(3.0 - 0.02 + 48 * 0.8 / 2)
    assert env.predators[2].energy == pytest.approx(3.0 + 48 * 0.2)


def test_prey_births_do_not_cut_a_decision_short():
    env = world(energy_rabbit_initial=3.0)
    env.rabbits[0].energy = 20.0
    env.max_energy["rabbit"] = 40.0
    start = env.time_step
    _, rewards, _, _, _ = env.step(stopped(env))
    assert env.births["rabbit"] >= 1 and env.time_step == start + env.action_repeat
    assert all(reward == 0.0 for reward in rewards.values())


def test_a_predator_birth_is_rewarded_and_ends_the_decision_with_the_child_active():
    env = world()
    parent = env.predators[0]
    parent.energy = 13.0
    start = env.time_step
    obs, rewards, _, _, infos = env.step(stopped(env))
    assert env.time_step == start + 1
    assert rewards["predator_0"] == 10.0
    assert "predator_4" in env.active and "predator_4" in obs and infos["predator_4"]["born"]
    assert parent.energy == pytest.approx(13.0 - 5.0)


def test_alive_reward_is_paid_per_physics_step_to_each_living_hunter():
    env = world(alive_reward=0.5, reproduction_predator_reward=0.0)
    _, rewards, _, _, _ = env.step(stopped(env))
    assert all(rewards[f"predator_{i}"] == pytest.approx(0.5 * env.action_repeat) for i in range(4))


def test_alive_reward_stops_when_the_hunter_starves():
    env = world(alive_reward=0.5, energy_predator_resting_metabolic_cost=1.0)
    env.predators[0].energy = 1.5
    env.action_repeat = 2
    _, rewards, terminateds, _, _ = env.step(stopped(env))
    assert terminateds["predator_0"] and rewards["predator_0"] == pytest.approx(0.5)
    assert rewards["predator_1"] == pytest.approx(1.0)


def test_shared_alive_reward_scales_with_living_hunters_and_skips_the_newborn_step():
    env = world(alive_reward=0.5, alive_reward_shared=True, reproduction_predator_reward=0.0)
    env.action_repeat = 1
    env.predators[0].energy = 13.0
    _, rewards, _, _, _ = env.step(stopped(env))
    assert rewards["predator_0"] == pytest.approx(0.5)
    assert rewards["predator_4"] == 0.0
    _, rewards, _, _, _ = env.step(stopped(env))
    assert rewards["predator_1"] == rewards["predator_4"] == pytest.approx(0.5 * 5 / 4)


def test_shared_alive_reward_falls_when_a_hunter_starves():
    env = world(alive_reward=0.5, alive_reward_shared=True, energy_predator_resting_metabolic_cost=1.0)
    env.predators[0].energy = 0.5
    env.action_repeat = 1
    _, rewards, _, _, _ = env.step(stopped(env))
    assert rewards["predator_1"] == pytest.approx(0.5 * 3 / 4)
    assert rewards["predator_0"] == 0.0


def test_time_limit_survivors_are_paid_on_the_last_step():
    env = world(alive_reward=0.5, max_time_steps=4, action_repeat=8)
    _, rewards, _, truncateds, _ = env.step(stopped(env))
    assert truncateds["__all__"] and rewards["predator_0"] == pytest.approx(2.0)


@pytest.mark.parametrize("settings", [{"alive_reward": -1.0}, {"alive_reward_shared": 1}])
def test_alive_reward_settings_are_validated(settings):
    with pytest.raises(ValueError):
        StagHuntRulePrey(settings)


def test_alive_arms_set_reward_episodes_and_gamma():
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.launchers.calibration import ARMS, PPO_ARMS

    for arm, shared in (("alive_individual", False), ("alive_shared", True)):
        assert ARMS[arm]["truncate_on_prey_extinction"] is False and ARMS[arm]["reproduction_predator_reward"] == 0.0
        assert ARMS[arm]["alive_reward"] > 0 and ARMS[arm].get("alive_reward_shared", False) is shared
        assert PPO_ARMS[arm]["gamma"] == 0.999


def test_starving_prey_are_removed_without_agent_flags():
    env = world(energy_rabbit_resting_metabolic_cost=2.0)
    env.rabbits[0].energy = 0.5
    env.action_repeat = 1
    _, _, terminateds, truncateds, _ = env.step(stopped(env))
    assert not env.rabbits and "rabbit_0" not in terminateds and "rabbit_0" not in truncateds


@pytest.mark.parametrize("truncate, ended", [(True, True), (False, False)])
def test_losing_one_prey_species_ends_an_episode_only_when_asked(truncate, ended):
    env = world(truncate_on_prey_extinction=truncate)
    rabbit, hunter = env.rabbits[0], env.predators[0]
    rabbit.position = Vector(hunter.position.x + 5, hunter.position.y)
    env.action_repeat = 1
    _, _, terminateds, truncateds, _ = env.step(stopped(env))
    assert truncateds["__all__"] == ended and not terminateds["__all__"]
    assert env.extinct_at["rabbit"] == env.time_step


def test_the_episode_ends_when_the_last_hunter_starves():
    env = world(predator_count=1, energy_predator_resting_metabolic_cost=10.0, truncate_on_prey_extinction=False)
    _, _, terminateds, truncateds, _ = env.step(stopped(env))
    assert terminateds["predator_0"] and (terminateds["__all__"] or truncateds["__all__"])
    assert env.episode_hunt_stats is not None


def test_the_episode_ends_at_max_time_steps():
    env = world(max_time_steps=4, action_repeat=8)
    _, _, _, truncateds, _ = env.step(stopped(env))
    assert truncateds["__all__"] and env.time_step == 4


def test_a_predators_observation_has_the_mammoth_and_partner_energies():
    env = world()
    hunter, partner, mammoth = env.predators[0], env.predators[1], env.mammoths[0]
    hunter.orientation = 90
    partner.position = Vector(hunter.position.x, hunter.position.y - 50)
    mammoth.position = Vector(hunter.position.x, hunter.position.y - 100)
    mammoth.energy, partner.energy = 20.0, 12.0
    obs = env._observe(hunter)
    tail = obs[-(env.observe["mammoth"] + env.observe["predator"]) :]
    assert tail[0] == pytest.approx(20.0 / env.max_energy["mammoth"])
    assert tail[env.observe["mammoth"]] == pytest.approx(12.0 / env.max_energy["predator"])
    assert list(OBSERVED) == ["predator", "mammoth", "rabbit"]


@pytest.mark.parametrize("key, value", [("hunt_min_party", 0), ("hunt_energy_ratio", 0), ("rabbit_flee_distance", -1.0), ("hunt_cooldown", -1)])
def test_settings_are_checked(key, value):
    with pytest.raises(ValueError, match=key):
        StagHuntRulePrey({key: value})


def test_unknown_settings_are_refused():
    with pytest.raises(ValueError, match="unknown"):
        StagHuntRulePrey({"burrow_count": 3})
