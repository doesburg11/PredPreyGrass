import copy

import numpy as np
import pytest

from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.compound_gift_env import (
    CompoundGiftPredPreyGrass,
)
from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.config_env import config_env as BASE


def make_env(mode="on"):
    config = copy.deepcopy(BASE)
    config.update(
        {
            "compound_gift": mode,
            "grid_size": 12,
            "initial_num_grass": 5,
            "initial_num_fruit": 5,
            "n_initial_active_predator_male": 1,
            "n_initial_active_predator_female": 1,
            "n_initial_active_prey": 1,
            "max_steps": 30,
        }
    )
    env = CompoundGiftPredPreyGrass(config)
    env.reset(seed=1)
    return env


def put(env, agent, pos):
    old = env.agent_positions[agent]
    env.grid_world_state[1, *old] = 0
    env.agent_positions[agent] = env.predator_positions[agent] = pos
    env.grid_world_state[1, *pos] = env.agent_energies[agent]


def pair(env):
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    env.agent_mate[male] = female
    env.agent_mate[female] = male
    put(env, male, (5, 5))
    put(env, female, (5, 6))
    env.agent_energies[male] = 8.0
    env.agent_energies[female] = 5.0
    env.grid_world_state[1, 5, 5] = 8.0
    env.grid_world_state[1, 5, 6] = 5.0
    return male, female


def actions(env, male, female, male_action, female_action=4):
    d = {a: env.noop_action_id for a in env.agents}
    d[male], d[female] = male_action, female_action
    return d


def test_compound_spaces_are_matched_and_prey_stays_nine_actions():
    env = make_env()
    assert env.action_spaces["predator_male_0"].n == 18
    assert env.action_spaces["predator_female_0"].n == 18
    assert env.action_spaces["prey_0"].n == 9
    assert env.num_actions == 9 and env.num_predator_actions == 18


@pytest.mark.parametrize("move", range(9))
def test_give_and_nongive_variants_have_identical_movement_and_cost(move):
    a, b = make_env(), make_env()
    ma, fa = pair(a)
    mb, fb = pair(b)
    a.step(actions(a, ma, fa, move))
    b.step(actions(b, mb, fb, move + 9))
    assert a.agent_positions[ma] == b.agent_positions[mb]
    # A real transfer is the only energy difference: the mover pays exactly
    # the same homeostatic and movement cost in both variants.
    assert a.agent_energies[ma] - b.agent_energies[mb] == pytest.approx(2.0)


def test_give_bit_transfers_after_its_movement():
    env = make_env("on")
    male, female = pair(env)
    put(env, female, (5, 7))
    # East: action 5; give-while-east: 14. Male moves beside mate then transfers.
    env.step(actions(env, male, female, 14))
    assert env.agent_positions[male] == (5, 6)
    assert env.learned_gift_events == 1
    assert env.agent_energies[female] == pytest.approx(7.0 - env.homeostatic_energy_cost_per_step_predator)


def test_inert_has_identical_motion_but_no_transfer():
    env = make_env("inert")
    male, female = pair(env)
    put(env, female, (5, 7))
    env.step(actions(env, male, female, 14))
    assert env.agent_positions[male] == (5, 6)
    assert env.learned_gift_attempts == 1
    assert env.learned_gift_events == 0
    assert env.agent_energies[female] == pytest.approx(5.0 - env.homeostatic_energy_cost_per_step_predator)


def test_give_bit_is_not_a_dedicated_stay_action():
    env = make_env()
    male, female = pair(env)
    # Action 9 is give-while-moving northwest, not the no-op give used in Iteration 19.
    env.step(actions(env, male, female, 9))
    assert env.agent_positions[male] == (4, 4)
    assert env.learned_gift_events == 1


def test_invalid_compound_mode_rejected():
    with pytest.raises(ValueError, match="compound_gift"):
        make_env("no")


def test_random_rollout_actions_match_each_agents_space():
    env = make_env()
    rng = np.random.default_rng(0)
    for _ in range(30):
        step = {a: int(rng.integers(env.action_spaces[a].n)) for a in env.agents}
        _, _, term, trunc, _ = env.step(step)
        if term.get("__all__") or trunc.get("__all__"):
            break
