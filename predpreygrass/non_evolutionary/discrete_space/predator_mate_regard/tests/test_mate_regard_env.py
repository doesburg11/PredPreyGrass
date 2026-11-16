import copy

import numpy as np
import pytest

from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning import analyze_compound_gift as acg
from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.compound_gift_env import (
    CompoundGiftPredPreyGrass,
)
from predpreygrass.non_evolutionary.discrete_space.predator_mate_regard.config_env import config_env as BASE
from predpreygrass.non_evolutionary.discrete_space.predator_mate_regard.mate_regard_env import (
    MateRegardPredPreyGrass,
    mate_regard_env_creator,
)


def small(**kw):
    c = copy.deepcopy(BASE)
    c.update({"grid_size": 12, "initial_num_grass": 5, "initial_num_fruit": 5,
              "n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1,
              "n_initial_active_prey": 1, "max_steps": 30, "reward_predator_per_energy": 0.5,
              "reward_predator_catch_prey": 0.0, "reward_predator_gather_fruit": 0.0})
    c.update(kw)
    return c


def make(w, cls=MateRegardPredPreyGrass):
    env = cls(small(mate_regard_weight=w))
    env.reset(seed=1)
    return env


def put(env, agent, pos):
    env.grid_world_state[1, *env.agent_positions[agent]] = 0
    env.agent_positions[agent] = env.predator_positions[agent] = pos
    env.grid_world_state[1, *pos] = env.agent_energies[agent]


def pair(env, female_pos=(5, 6)):
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    env.agent_mate[male], env.agent_mate[female] = female, male
    env.agent_energies[male], env.agent_energies[female] = 8.0, 5.0
    put(env, male, (5, 5))
    put(env, female, female_pos)
    # keep food away from both predators so only the gift changes energy intake
    for fruit, pos in list(env.fruit_positions.items()):
        env.grid_world_state[4, *pos] = 0
        env.fruit_energies[fruit] = 0
    return male, female


def acts(env, male, female, m, f=4):
    d = {a: env.noop_action_id for a in env.agents}
    d[male], d[female] = m, f
    return d


def test_gift_rewards_the_giver_by_weight_times_energy_reward():
    rewards, intake = {}, {}
    for w in (0.0, 0.5):
        env = make(w)
        male, female = pair(env)
        _, r, _, _, _ = env.step(acts(env, male, female, 9 + 4))  # stay put + gift
        assert env.learned_gifts_this_step == [male]
        rewards[w], intake[w] = r[male], env.predator_energy_intake[female]
    assert intake[0.0] == intake[0.5] >= 2.0  # the gift (plus any regrown fruit the female stands on)
    assert rewards[0.5] - rewards[0.0] == pytest.approx(0.5 * 0.5 * intake[0.5])


def test_mate_food_intake_rewards_partner():
    env = make(0.5)
    male, female = pair(env)
    fruit = next(iter(env.fruit_positions))
    env.grid_world_state[4, *env.fruit_positions[fruit]] = 0
    env.fruit_positions[fruit] = (5, 7)
    env.fruit_energies[fruit] = 3.0
    env.grid_world_state[4, 5, 7] = 3.0
    _, r, _, _, _ = env.step(acts(env, male, female, 4, 5))  # female moves (0, +1) onto (5, 7)
    assert env.predator_energy_intake[female] == pytest.approx(min(3.0 + env.energy_gain_per_step_fruit, env.initial_energy_fruit))
    assert male not in env.predator_energy_intake
    assert env.mate_regard_reward_total == pytest.approx(0.5 * 0.5 * env.predator_energy_intake[female])
    assert r[male] == pytest.approx(env.reward_predator_step + env.mate_regard_reward_total)


def test_no_bonus_without_a_valid_transfer_or_mate():
    env = make(0.5)
    male, female = pair(env, female_pos=(5, 10))  # out of gift range
    _, r, _, _, _ = env.step(acts(env, male, female, 9 + 4))
    assert env.learned_gifts_this_step == []
    assert env.mate_regard_reward_total == 0.0
    env2 = make(0.5)
    male2 = next(a for a in env2.agents if "predator_male" in a)
    env2.agent_mate.clear()
    env2.step({a: 4 for a in env2.agents})
    assert env2.mate_regard_reward_total == 0.0


def test_weight_zero_reproduces_compound_env_exactly():
    a, b = make(0.0), CompoundGiftPredPreyGrass(small())
    b.reset(seed=1)
    rng = np.random.default_rng(5)
    for _ in range(30):
        act = {ag: int(rng.integers(a.action_spaces[ag].n)) for ag in a.agents}
        ra, rb = a.step(act), b.step(dict(act))
        assert ra[1] == rb[1] and ra[2] == rb[2]
        if ra[2].get("__all__") or ra[3].get("__all__"):
            break


def test_negative_weight_rejected_and_creator_builds_requested_weight():
    with pytest.raises(ValueError):
        MateRegardPredPreyGrass(small(mate_regard_weight=-0.1))
    env = mate_regard_env_creator({"mate_regard_weight": 0.5})
    assert isinstance(env, MateRegardPredPreyGrass) and env.mate_regard_weight == 0.5
    assert env.action_spaces["predator_male_0"].n == 18


def test_analysis_eligibility_charges_homeostatic_cost():
    env = make(0.0)
    male, female = pair(env)
    assert acg.eligible(env, male)
    env.agent_energies[male] = env.learned_gift_amount + 0.05  # above the gift, below gift + homeostatic cost
    assert not acg.eligible(env, male)
    env.agent_energies[male] = 8.0
    env.agent_energies[female] = 0.05
    assert not acg.eligible(env, male)


def test_truncation_call_pays_no_stale_mate_bonus():
    env = make(0.5)
    male, female = pair(env)
    env.current_step = env.max_steps - 1
    env.step(acts(env, male, female, 9 + 4))  # last real step: gift transfers, bonus paid
    paid = env.mate_regard_reward_total
    assert paid > 0
    _, r, _, trunc, _ = env.step(acts(env, male, female, 4))  # truncation call
    assert trunc["__all__"]
    assert all(v == 0.0 for v in r.values())
    assert env.mate_regard_reward_total == paid
