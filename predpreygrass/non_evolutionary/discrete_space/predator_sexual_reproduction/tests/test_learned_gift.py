"""
Tests for the optional learned gift action (config "learned_gift", RESULTS.md Iteration 19).

Run explicitly:
    pytest predpreygrass/non_evolutionary/discrete_space/predator_sexual_reproduction/tests/test_learned_gift.py -v
"""
import copy

import numpy as np
import pytest

from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.config_env import config_env as _base_config_env
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.fixed_predator_density_env import (
    FixedPredatorDensityEnv,
)
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.predpreygrass_rllib_env import PredPreyGrass

M, F, M2, F2 = "predator_male_0", "predator_female_0", "predator_male_1", "predator_female_1"


def _make_env(mode="on", overrides=None, cls=PredPreyGrass):
    config = copy.deepcopy(_base_config_env)
    config.update(
        {
            "grid_size": 12,
            "n_initial_active_predator_male": 2,
            "n_initial_active_predator_female": 2,
            "n_initial_active_prey": 2,
            "initial_num_grass": 5,
            "initial_num_fruit": 5,
            "max_steps": 50,
            "learned_gift": mode,
            "male_gift_donation_rate": 0.0,
        }
    )
    if overrides:
        config.update(overrides)
    env = cls(config)
    env.reset(seed=42)
    return env


def _move(env, agent, pos):
    layer = 1 if "predator" in agent else 2
    old = env.agent_positions[agent]
    env.grid_world_state[layer, *old] = 0
    env.agent_positions[agent] = pos
    if "predator" in agent:
        env.predator_positions[agent] = pos
    else:
        env.prey_positions[agent] = pos
    env.grid_world_state[layer, *pos] = env.agent_energies[agent]


def _set_energy(env, agent, e):
    env.agent_energies[agent] = e
    env.grid_world_state[1, *env.agent_positions[agent]] = e


def _clear_food(env):
    """Remove prey, grass and fruit from play so no foraging changes energies during a test step."""
    for prey in [a for a in env.agents if "prey" in a]:
        _move(env, prey, (0, 0) if prey.endswith("_0") else (0, 1))
    for name, pos in list(env.fruit_positions.items()):
        env.grid_world_state[4, *pos] = 0
        env.fruit_energies[name] = 0
    for name, pos in list(env.grass_positions.items()):
        env.grid_world_state[3, *pos] = 0
        env.grass_energies[name] = 0


def _setup_pair(env, m_pos=(6, 6), f_pos=(6, 8), m_energy=8.0, f_energy=8.0):
    _clear_food(env)
    _move(env, M2, (11, 11))
    _move(env, F2, (11, 9))
    _move(env, M, m_pos)
    _move(env, F, f_pos)
    _set_energy(env, M, m_energy)
    _set_energy(env, F, f_energy)
    env.agent_mate[M] = F
    env.agent_mate[F] = M


def _step_pair(env, giver, give=True):
    """Step with every agent staying put; `giver` chooses give (or the plain noop when give=False)."""
    actions = {a: env.noop_action_id for a in env.agents}
    actions[giver] = env.give_action_id if give else env.noop_action_id
    env.step(actions)


def _diff(env, giver):
    """Energy differences (give minus noop) for M and F after one identical step from a copied state."""
    a, b = copy.deepcopy(env), copy.deepcopy(env)
    _step_pair(a, giver, give=True)
    _step_pair(b, giver, give=False)
    return (
        a.agent_energies[M] - b.agent_energies[M],
        a.agent_energies[F] - b.agent_energies[F],
        a,
    )


def test_default_off_keeps_nine_actions():
    config = copy.deepcopy(_base_config_env)
    assert config["learned_gift"] is False
    env = _make_env(mode=False)
    assert env.give_action_id is None
    assert env.num_predator_actions == 9
    assert all(space.n == 9 for space in env.action_spaces.values())


@pytest.mark.parametrize("mode", ["on", True, "inert"])
def test_on_and_inert_add_a_tenth_predator_action_only(mode):
    env = _make_env(mode=mode)
    assert env.give_action_id == 9
    assert env.num_predator_actions == 10
    for agent, space in env.action_spaces.items():
        assert space.n == (10 if "predator" in agent else 9)


@pytest.mark.parametrize("bad", ["yes", 1, None, "blank"])
def test_rejects_unknown_modes(bad):
    with pytest.raises(ValueError):
        _make_env(mode=bad)


@pytest.mark.parametrize("amount", [0.0, -1.0, float("inf")])
def test_rejects_bad_amounts(amount):
    with pytest.raises(ValueError):
        _make_env(overrides={"learned_gift_amount": amount})


def test_male_gift_to_mate_in_range_transfers_amount_and_stays_put():
    env = _make_env()
    _setup_pair(env)
    dm, df, after = _diff(env, M)
    assert dm == pytest.approx(-2.0)
    assert df == pytest.approx(2.0)
    assert after.agent_positions[M] == (6, 6)
    assert after.learned_gift_events == 1
    assert after.learned_gift_events_by_sex["predator_male"] == 1
    assert after.learned_gift_energy_total == pytest.approx(2.0)


def test_give_costs_no_move_energy_beyond_homeostasis():
    env = _make_env()
    _setup_pair(env, m_pos=(6, 6), f_pos=(6, 10))  # distance 4, out of range: a wasted turn
    before = env.agent_energies[M]
    _step_pair(env, M, give=True)
    assert env.agent_energies[M] == pytest.approx(before - env.homeostatic_energy_cost_per_step_predator)


def test_female_can_give_to_male():
    env = _make_env()
    _setup_pair(env)
    dm, df, after = _diff(env, F)
    assert dm == pytest.approx(2.0)
    assert df == pytest.approx(-2.0)
    assert after.learned_gift_events_by_sex["predator_female"] == 1


def test_custom_amount():
    env = _make_env(overrides={"learned_gift_amount": 3.5})
    _setup_pair(env)
    dm, df, _ = _diff(env, M)
    assert (dm, df) == (pytest.approx(-3.5), pytest.approx(3.5))


@pytest.mark.parametrize("f_pos", [(6, 10), (10, 6), (2, 2)])
def test_out_of_range_is_wasted(f_pos):
    env = _make_env()
    _setup_pair(env, f_pos=f_pos)
    dm, df, after = _diff(env, M)
    assert (dm, df) == (0.0, 0.0)
    assert after.learned_gift_events == 0
    assert after.learned_gift_attempts == 1


def test_edge_of_range_transfers():
    env = _make_env()
    _setup_pair(env, f_pos=(9, 9))  # Chebyshev distance 3 == predator_gift_range
    dm, df, _ = _diff(env, M)
    assert (dm, df) == (pytest.approx(-2.0), pytest.approx(2.0))


def test_distance_uses_post_move_positions():
    env = _make_env()
    _setup_pair(env, f_pos=(6, 10))  # distance 4 before the step
    actions = {a: env.noop_action_id for a in env.agents}
    actions[M] = env.give_action_id
    move_left = next(k for k, v in env.action_to_move_tuple.items() if v == (0, -1))
    actions[F] = move_left  # female moves to (6, 9): distance 3 after movement
    env.step(actions)
    assert env.learned_gift_events == 1


def test_no_mate_is_wasted():
    env = _make_env()
    _setup_pair(env)
    env.agent_mate.clear()
    dm, df, after = _diff(env, M)
    assert (dm, df) == (0.0, 0.0)
    assert after.learned_gift_events == 0


def test_non_mate_in_range_gets_nothing():
    env = _make_env()
    _setup_pair(env, f_pos=(9, 9))
    _move(env, F2, (6, 7))  # another female right next to him, not his mate
    f2_before = copy.deepcopy(env)
    _step_pair(f2_before, M, give=True)
    noop = copy.deepcopy(env)
    _step_pair(noop, M, give=False)
    assert f2_before.agent_energies[F2] == pytest.approx(noop.agent_energies[F2])
    assert f2_before.agent_energies[F] - noop.agent_energies[F] == pytest.approx(2.0)


def test_giver_must_keep_positive_energy():
    env = _make_env()
    # after homeostasis (0.1) he has 2.0; giving 2.0 would leave 0 -> refused
    _setup_pair(env, m_energy=2.1)
    dm, df, after = _diff(env, M)
    assert (dm, df) == (0.0, 0.0)
    env2 = _make_env()
    _setup_pair(env2, m_energy=2.2)
    dm2, df2, _ = _diff(env2, M)
    assert (dm2, df2) == (pytest.approx(-2.0), pytest.approx(2.0))


def test_starving_mate_is_not_rescued():
    env = _make_env()
    _setup_pair(env, f_energy=0.05)  # homeostasis takes her to <= 0 before gifts
    a = copy.deepcopy(env)
    _step_pair(a, M, give=True)
    assert F not in a.agent_positions
    assert a.learned_gift_events == 0


def test_dead_mate_is_wasted():
    env = _make_env()
    _setup_pair(env)
    env._remove_agent(F) if hasattr(env, "_remove_agent") else None
    env.predator_positions.pop(F, None)
    env.agent_positions.pop(F, None)
    env.agent_energies.pop(F, None)
    env.agents = [a for a in env.agents if a != F]
    actions = {a: env.noop_action_id for a in env.agents}
    actions[M] = env.give_action_id
    before = env.agent_energies[M]
    env.step(actions)
    assert env.learned_gift_events == 0
    assert env.agent_energies[M] == pytest.approx(before - env.homeostatic_energy_cost_per_step_predator)


def test_inert_never_transfers_but_counts_attempts():
    env = _make_env(mode="inert")
    _setup_pair(env)
    dm, df, after = _diff(env, M)
    assert (dm, df) == (0.0, 0.0)
    assert after.learned_gift_attempts == 1
    assert after.learned_gift_events == 0
    assert after.agent_positions[M] == (6, 6)


def test_both_partners_giving_same_step():
    env = _make_env()
    _setup_pair(env)
    before_m, before_f = env.agent_energies[M], env.agent_energies[F]
    actions = {a: env.noop_action_id for a in env.agents}
    actions[M] = actions[F] = env.give_action_id
    env.step(actions)
    h = env.homeostatic_energy_cost_per_step_predator
    assert env.agent_energies[M] == pytest.approx(before_m - h)
    assert env.agent_energies[F] == pytest.approx(before_f - h)
    assert env.learned_gift_events == 2


def test_grid_layer_tracks_energies_after_gift():
    env = _make_env()
    _setup_pair(env)
    _step_pair(env, M, give=True)
    for a in (M, F):
        assert env.grid_world_state[1, *env.agent_positions[a]] == pytest.approx(env.agent_energies[a])


def test_reset_clears_counters_and_episode_metrics_report_them():
    env = _make_env()
    _setup_pair(env)
    _step_pair(env, M, give=True)
    metrics = env._build_episode_training_metrics()
    assert metrics["learned_gift_events"] == 1.0
    assert metrics["learned_gift_events_male"] == 1.0
    assert metrics["learned_gift_energy_total"] == pytest.approx(2.0)
    env.reset(seed=1)
    assert env.learned_gift_events == 0 and env.learned_gift_attempts == 0
    assert env.learned_gift_energy_total == 0.0


@pytest.mark.parametrize("cls", [PredPreyGrass, FixedPredatorDensityEnv])
def test_random_rollout_conserves_gift_energy(cls):
    overrides = {"grid_size": 15, "initial_num_grass": 15, "initial_num_fruit": 15, "max_steps": 300}
    if cls is FixedPredatorDensityEnv:
        overrides["predator_density_target"] = 4
    env = _make_env(overrides=overrides, cls=cls)
    rng = np.random.default_rng(0)
    for _ in range(300):
        actions = {a: int(rng.integers(env.action_spaces[a].n)) for a in env.agents}
        _, _, term, trunc, _ = env.step(actions)
        if trunc.get("__all__") or term.get("__all__"):
            break
    assert env.learned_gift_attempts >= env.learned_gift_events
    assert env.learned_gift_energy_total == pytest.approx(2.0 * env.learned_gift_events)
    assert env.learned_gift_events == sum(env.learned_gift_events_by_sex.values())


@pytest.mark.parametrize("female_first", [False, True])
def test_simultaneous_gifts_are_order_independent(female_first):
    """He cannot afford a gift (1.6 - 0.1 homeostasis = 1.5 < 2); she can. Applied sequentially with her
    gift first, his would become affordable. Simultaneous resolution must refuse his in both orders."""
    env = _make_env()
    _setup_pair(env, m_energy=1.6, f_energy=8.0)
    actions = {a: env.noop_action_id for a in env.agents}
    order = [F, M] if female_first else [M, F]
    ordered = {a: env.give_action_id for a in order}
    ordered.update({a: v for a, v in actions.items() if a not in ordered})
    env.step(ordered)
    h = env.homeostatic_energy_cost_per_step_predator
    assert env.learned_gift_events == 1
    assert env.learned_gifts_this_step == [F]
    assert env.agent_energies[M] == pytest.approx(1.6 - h + 2.0)
    assert env.agent_energies[F] == pytest.approx(8.0 - h - 2.0)


def test_gifts_this_step_resets():
    env = _make_env()
    _setup_pair(env)
    _step_pair(env, M, give=True)
    assert env.learned_gifts_this_step == [M]
    _step_pair(env, M, give=False)
    assert env.learned_gifts_this_step == []


def test_fixed_prey_density_env_rollout():
    from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.fixed_prey_density_env import (
        FixedPreyDensityEnv,
    )
    overrides = {"grid_size": 15, "initial_num_grass": 15, "initial_num_fruit": 15, "max_steps": 200,
                 "prey_density_floor": 2}
    env = _make_env(overrides=overrides, cls=FixedPreyDensityEnv)
    rng = np.random.default_rng(1)
    for _ in range(200):
        actions = {a: int(rng.integers(env.action_spaces[a].n)) for a in env.agents}
        _, _, term, trunc, _ = env.step(actions)
        if trunc.get("__all__") or term.get("__all__"):
            break
    assert env.learned_gift_energy_total == pytest.approx(2.0 * env.learned_gift_events)
