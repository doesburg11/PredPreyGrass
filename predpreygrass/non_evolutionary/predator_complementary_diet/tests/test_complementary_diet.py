"""Tests for the complementary-diet mechanics added in predator_complementary_diet: the fruit/meat stores, the
fixed-share cost draw, death and reproduction gating, the female fruit gift, type-preserving parental care, the
store observation channel, and snapshots. The inherited mechanics are covered by
test_predator_complementary_diet_validation.py (run with the diet requirement off)."""
import copy

import numpy as np
import pytest

from predpreygrass.non_evolutionary.predator_complementary_diet.config_env import config_env as _base
from predpreygrass.non_evolutionary.predator_complementary_diet.predpreygrass_rllib_env import PredPreyGrass


def _env(**overrides):
    config = copy.deepcopy(_base)
    config.update(
        {
            "grid_size": 10,
            "n_initial_active_predator_male": 1,
            "n_initial_active_predator_female": 1,
            "n_initial_active_prey": 2,
            "initial_num_grass": 5,
            "initial_num_fruit": 5,
            "max_steps": 50,
            "male_gift_donation_rate": 0.0,
            "female_gift_donation_rate": 0.0,
            "parent_offspring_share_rate": 0.0,
        }
    )
    config.update(overrides)
    env = PredPreyGrass(config)
    env.reset(seed=1)
    return env


def _agents(env):
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    return male, female


def _place(env, agent, pos):
    old = env.agent_positions.get(agent)
    if old is not None:
        env.grid_world_state[1, *old] = 0
    env.agent_positions[agent] = pos
    env.predator_positions[agent] = pos
    env.grid_world_state[1, *pos] = env.agent_energies.get(agent, 0.0)


def _noop(env):
    return {a: env.noop_action_id for a in env.agents}


def _meat(env, a):
    return env.agent_energies[a] - env.agent_fruit_store[a]


def test_reset_splits_initial_energy_between_stores():
    env = _env()
    male, female = _agents(env)
    for a in (male, female):
        assert env.agent_fruit_store[a] == pytest.approx(env.agent_energies[a] * env.diet_initial_fruit_share)
        assert _meat(env, a) == pytest.approx(env.agent_energies[a] * (1 - env.diet_initial_fruit_share))


def test_debit_draws_from_stores_in_fixed_shares():
    env = _env(diet_meat_cost_share=0.25)
    male, _ = _agents(env)
    e0, f0, m0 = env.agent_energies[male], env.agent_fruit_store[male], _meat(env, male)
    env._debit(male, 1.0)
    assert env.agent_energies[male] == pytest.approx(e0 - 1.0)
    assert env.agent_fruit_store[male] == pytest.approx(f0 - 0.75)
    assert _meat(env, male) == pytest.approx(m0 - 0.25)


def test_step_costs_use_the_same_shares():
    env = _env()
    male, female = _agents(env)
    f0, m0 = env.agent_fruit_store[female], _meat(env, female)
    env.step(_noop(env))  # homeostatic cost only (noop is free)
    c = env.homeostatic_energy_cost_per_step_predator
    assert env.agent_fruit_store[female] == pytest.approx(f0 - 0.75 * c)
    assert _meat(env, female) == pytest.approx(m0 - 0.25 * c)


def test_fruit_raises_fruit_store_and_meat_raises_meat_store():
    env = _env()
    male, female = _agents(env)
    fruit = next(iter(env.fruit_positions))
    prey = next(a for a in env.agents if a.startswith("prey"))
    _place(env, female, (5, 5))
    env.fruit_positions[fruit] = (5, 5)
    env.fruit_energies[fruit] = 2.0
    f0, m0 = env.agent_fruit_store[female], _meat(env, female)
    env.step(_noop(env))
    c = env.homeostatic_energy_cost_per_step_predator
    assert env.agent_fruit_store[female] == pytest.approx(f0 - 0.75 * c + 2.0)
    assert _meat(env, female) == pytest.approx(m0 - 0.25 * c)

    # meat: male on a prey, forced hunting success
    env2 = _env()
    male2, _ = _agents(env2)
    prey2 = next(a for a in env2.agents if a.startswith("prey"))
    _place(env2, male2, (4, 4))
    env2.agent_positions[prey2] = (4, 4)
    env2.prey_positions[prey2] = (4, 4)
    env2.rng = type("R", (), {"random": lambda self: 0.0, "__getattr__": lambda self, n: getattr(env2._real, n)})()
    env2._real = np.random.default_rng(0)
    f0, m0 = env2.agent_fruit_store[male2], _meat(env2, male2)
    gained = env2.agent_energies[prey2] - env2.homeostatic_energy_cost_per_step_prey
    env2.step(_noop(env2))
    assert env2.agent_fruit_store[male2] == pytest.approx(f0 - 0.75 * c)
    assert _meat(env2, male2) == pytest.approx(m0 - 0.25 * c + gained)


@pytest.mark.parametrize("store,cause", [("fruit", "fruit"), ("meat", "meat")])
def test_predator_dies_when_a_store_is_exhausted_despite_positive_energy(store, cause):
    env = _env()
    _, female = _agents(env)
    env.agent_energies[female] = 8.0
    env.agent_fruit_store[female] = 0.01 if store == "fruit" else 7.99  # meat store = 0.01
    env.step(_noop(env))
    assert female not in env.agent_positions
    assert env.diet_deaths["predator_female"][cause] == 1
    assert env.episode_deaths["predator_female"] == 1


def test_no_diet_death_when_not_required():
    env = _env(diet_required=False)
    _, female = _agents(env)
    env.agent_energies[female] = 8.0
    env.agent_fruit_store[female] = 0.01
    env.step(_noop(env))
    assert female in env.agent_positions
    assert env.diet_deaths["predator_female"] == {"fruit": 0, "meat": 0}


def test_reproduction_requires_both_stores():
    def run(f_share):
        env = _env(mate_search_radius=3)
        male, female = _agents(env)
        _place(env, male, (5, 5))
        _place(env, female, (5, 6))
        for a in (male, female):
            env.agent_energies[a] = 13.0
            env.agent_fruit_store[a] = 13.0 * f_share
        before = len([a for a in env.agents if "predator" in a])
        env.step(_noop(env))
        return len([a for a in env.agents if "predator" in a]) - before, env, male, female

    born, env, male, female = run(0.5)
    assert born == 1
    child = next(a for a in env.agents if a in env.agent_parents)
    assert env.agent_fruit_store[child] == pytest.approx(env.agent_energies[child] * env.diet_initial_fruit_share)
    for f_share in (0.0, 1.0):  # all-meat / all-fruit: energy is above the threshold but one store is empty
        born, *_ = run(f_share)
        assert born == 0
    # with the requirement off the same states do reproduce
    env = _env(diet_required=False)
    male, female = _agents(env)
    _place(env, male, (5, 5))
    _place(env, female, (5, 6))
    for a in (male, female):
        env.agent_energies[a] = 13.0
        env.agent_fruit_store[a] = 0.0
    env.step(_noop(env))
    assert env.episode_births["predator_male"] + env.episode_births["predator_female"] == 1


def test_birth_cost_is_drawn_from_both_stores():
    env = _env()
    male, female = _agents(env)
    _place(env, male, (5, 5))
    _place(env, female, (5, 6))
    for a in (male, female):
        env.agent_energies[a] = 13.0
        env.agent_fruit_store[a] = 6.5
    f0, m0 = env.agent_fruit_store[female], _meat(env, female)
    env.step(_noop(env))
    child = next(a for a in env.agent_parents)
    cost = env.agent_energies[child] * env.predator_birth_cost_share_female
    c = env.homeostatic_energy_cost_per_step_predator
    assert env.agent_fruit_store[female] == pytest.approx(f0 - 0.75 * (c + cost))
    assert _meat(env, female) == pytest.approx(m0 - 0.25 * (c + cost))


def test_female_gift_moves_fruit_energy_to_recorded_mate_only_in_range():
    def run(rate=0.3, mate_pos=(5, 6), recorded=True):
        env = _env(female_gift_donation_rate=rate)
        male, female = _agents(env)
        fruit = next(iter(env.fruit_positions))
        _place(env, female, (5, 5))
        _place(env, male, mate_pos)
        if recorded:
            env.agent_mate[male] = female
            env.agent_mate[female] = male
        env.fruit_positions[fruit] = (5, 5)
        env.fruit_energies[fruit] = 2.0
        before = {a: (env.agent_energies[a], env.agent_fruit_store[a], _meat(env, a)) for a in (male, female)}
        env.step(_noop(env))
        return env, male, female, before

    c = 0.1
    env, male, female, b = run()
    donation = 0.3 * 2.0
    assert env.agent_fruit_store[male] == pytest.approx(b[male][1] - 0.75 * c + donation)
    assert env.agent_energies[male] == pytest.approx(b[male][0] - c + donation)
    assert _meat(env, male) == pytest.approx(b[male][2] - 0.25 * c)  # the gift is fruit-type only
    assert env.agent_fruit_store[female] == pytest.approx(b[female][1] - 0.75 * c + 2.0 - donation)
    assert _meat(env, female) == pytest.approx(b[female][2] - 0.25 * c)
    assert env.female_gift_events == 1 and env.female_gift_energy_total == pytest.approx(donation)

    for kwargs in ({"rate": 0.0}, {"mate_pos": (5, 9)}, {"recorded": False}):
        env, male, female, b = run(**kwargs)
        assert env.female_gift_events == 0
        assert env.agent_energies[male] == pytest.approx(b[male][0] - c)


def test_gift_rates_plus_care_share_must_not_exceed_one():
    with pytest.raises(ValueError):
        _env(female_gift_donation_rate=0.9, parent_offspring_share_rate=0.2)
    with pytest.raises(ValueError):
        _env(female_gift_donation_rate=1.5)
    with pytest.raises(ValueError):
        _env(diet_meat_cost_share=-0.1)


def test_parental_care_of_fruit_moves_fruit_store():
    env = _env(parent_offspring_share_rate=0.2)
    male, female = _agents(env)
    child = "predator_female_test_child"
    _place(env, female, (5, 5))
    _place(env, child, (5, 6))
    env.agent_energies[child] = 5.0
    env.agent_fruit_store[child] = 2.5
    env.agent_parents[child] = (male, female)
    env.cumulative_rewards[child] = 0
    env.agents.append(child)
    fruit = next(iter(env.fruit_positions))
    env.fruit_positions[fruit] = (5, 5)
    env.fruit_energies[fruit] = 2.0
    cf0, cm0 = env.agent_fruit_store[child], _meat(env, child)
    env.step(_noop(env))
    c = env.homeostatic_energy_cost_per_step_predator
    share = 0.2 * 2.0
    assert env.agent_fruit_store[child] == pytest.approx(cf0 - 0.75 * c + share)
    assert _meat(env, child) == pytest.approx(cm0 - 0.25 * c)


def test_observation_has_store_channel_with_own_and_neighbor_store():
    env = _env()
    male, female = _agents(env)
    _place(env, female, (5, 5))
    _place(env, male, (5, 7))
    env.agent_fruit_store[female] = 3.0
    env.agent_fruit_store[male] = 1.5
    obs = env._get_observation(female)
    off = (env.predator_obs_range - 1) // 2
    assert obs.shape == (env.num_obs_channels, env.predator_obs_range, env.predator_obs_range)
    assert env.num_obs_channels == 6
    assert obs[5, off, off] == 3.0  # own store at the center
    assert obs[5, off, off + 2] == 1.5  # neighbor's store at its relative cell
    assert (obs[5] > 0).sum() == 2
    # sex is still not observable: the predator layer holds only energies
    assert obs[1, off, off + 2] == env.agent_energies[male]


def test_snapshot_round_trips_the_stores():
    env = _env()
    male, female = _agents(env)
    snap = env.get_state_snapshot()
    env.step(_noop(env))
    assert env.agent_fruit_store[female] != snap["agent_fruit_store"][female]
    env.restore_state_snapshot(snap)
    assert env.agent_fruit_store[female] == snap["agent_fruit_store"][female]


def test_random_policy_episode_runs_and_reports_diet_metrics():
    env = _env(n_initial_active_predator_male=3, n_initial_active_predator_female=3, initial_num_fruit=30, max_steps=200)
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs, r, t, tr, _ = env.step({a: int(rng.integers(env.num_actions)) for a in obs})
        for a, o in obs.items():
            assert o.shape[0] == 6 and np.isfinite(o).all()
        if t.get("__all__") or tr.get("__all__"):
            break
    m = env._build_episode_training_metrics()
    for k in ("female_gift_events", "diet_deaths_male_fruit", "diet_deaths_male_meat",
              "diet_deaths_female_fruit", "diet_deaths_female_meat"):
        assert k in m


def _breeding_pair(env, female_fruit, female_energy=13.0):
    male, female = _agents(env)
    _place(env, male, (5, 5))
    _place(env, female, (5, 6))
    env.agent_energies[male] = 13.0
    env.agent_fruit_store[male] = 6.5
    env.agent_energies[female] = female_energy
    env.agent_fruit_store[female] = female_fruit
    return male, female


def test_no_suicidal_birth_when_a_store_cannot_cover_the_birth_cost():
    # female pays 0.9 * 5 = 4.5 of birth cost: 3.375 from fruit, 1.125 from meat. The store floor alone (3.0) is smaller.
    env = _env()
    _breeding_pair(env, female_fruit=3.2)  # fruit store above the 3.0 floor but below the 3.375 the birth needs
    env.step(_noop(env))
    assert env.episode_births["predator_male"] + env.episode_births["predator_female"] == 0

    env = _env()
    male, female = _breeding_pair(env, female_fruit=6.5)
    env.step(_noop(env))
    assert env.episode_births["predator_male"] + env.episode_births["predator_female"] == 1
    # no parent is left in a diet-deficiency state by the birth
    for a in (male, female):
        assert env._diet_death_cause(a) is None


def test_gift_does_not_rescue_a_recipient_already_doomed_by_a_store():
    env = _env(male_gift_donation_rate=0.3)
    male, female = _agents(env)
    prey = next(a for a in env.agents if a.startswith("prey"))
    _place(env, male, (5, 5))
    _place(env, female, (5, 6))
    env.agent_mate[male] = female
    env.agent_mate[female] = male
    env.agent_energies[female] = 8.0
    env.agent_fruit_store[female] = 7.999  # meat store 0.001: dead after this step's cost
    env.agent_positions[prey] = (5, 5)
    env.prey_positions[prey] = (5, 5)
    env.rng = type("R", (), {"random": lambda self: 0.0})()  # hunting succeeds
    env.step(_noop(env))
    assert env.mate_gift_events == 0
    assert female not in env.agent_positions


def test_snapshot_restores_counters_and_removal_drops_the_store():
    env = _env()
    _, female = _agents(env)
    snap = env.get_state_snapshot()
    env.agent_energies[female] = 8.0
    env.agent_fruit_store[female] = 0.01
    env.step(_noop(env))
    assert env.diet_deaths["predator_female"]["fruit"] == 1
    assert female not in env.agent_fruit_store  # removal drops the store
    env.female_gift_events = 5
    env.restore_state_snapshot(snap)
    assert env.diet_deaths["predator_female"] == {"fruit": 0, "meat": 0}
    assert env.female_gift_events == 0
    assert female in env.agent_fruit_store


@pytest.mark.parametrize("corner", [(0, 0), (9, 9), (0, 9), (9, 0)])
def test_store_channel_at_grid_corners(corner):
    env = _env()
    male, female = _agents(env)
    _place(env, female, corner)
    other = (min(corner[0] + 1, 9) if corner[0] == 0 else corner[0] - 1, corner[1])
    _place(env, male, other)
    env.agent_fruit_store[female] = 2.0
    env.agent_fruit_store[male] = 1.0
    obs = env._get_observation(female)
    off = (env.predator_obs_range - 1) // 2
    assert obs[5, off, off] == 2.0
    assert obs[5, off + other[0] - corner[0], off + other[1] - corner[1]] == 1.0
    assert obs[0, off, off] == 0  # own cell is not border


# ---- scripted prey ----------------------------------------------------------------------------------------------
def _scripted_env(**kw):
    return _env(scripted_prey=True, **kw)


def test_scripted_prey_are_not_learning_agents_but_still_live_in_the_world():
    env = _scripted_env(n_initial_active_prey=4)
    assert not any(a.startswith("prey") for a in env.agents)
    assert not any(a.startswith("prey") for a in env.possible_agents)
    assert not any(a.startswith("prey") for a in env.observation_spaces)
    assert len(env._scripted_prey_ids) == 4 and env.current_num_prey == 4
    obs, _, t, tr, _ = env.step(_noop(env))
    assert all("prey" not in a for a in obs)  # RLlib only ever sees predators


def test_scripted_prey_pay_homeostatic_cost_and_can_graze():
    env = _scripted_env(n_initial_active_prey=1)
    prey = env._scripted_prey_ids[0]
    grass = next(iter(env.grass_positions))
    env.grid_world_state[2, *env.agent_positions[prey]] = 0
    pos = (3, 3)
    env.agent_positions[prey] = pos
    env.prey_positions[prey] = pos
    env.grid_world_state[2, *pos] = env.agent_energies[prey]
    env.grass_positions[grass] = pos
    env.grass_energies[grass] = 2.0
    env.grid_world_state[3, *pos] = 2.0
    male, female = _agents(env)
    _place(env, male, (9, 9))  # keep the predators out of the prey's flee radius
    _place(env, female, (9, 8))
    e0 = env.agent_energies[prey]
    env.step(_noop(env))
    # noop on grass (the rule stays put on grass): pays homeostasis, eats the (regrown to cap) grass energy
    assert env.agent_energies[prey] == pytest.approx(e0 - env.homeostatic_energy_cost_per_step_prey + 2.0)


def test_scripted_prey_flee_and_walk_to_grass():
    env = _scripted_env(n_initial_active_prey=1, initial_num_grass=1)
    prey = env._scripted_prey_ids[0]
    male, female = _agents(env)
    pos = (5, 5)
    env.grid_world_state[2, *env.agent_positions[prey]] = 0
    env.agent_positions[prey] = pos
    env.prey_positions[prey] = pos
    env.grid_world_state[2, *pos] = 3.0
    _place(env, male, (5, 6))  # predator adjacent to the east
    _place(env, female, (9, 9))
    dx, dy = env.action_to_move_tuple[env._scripted_prey_action(prey)]
    assert dy < 0 or dx != 0  # moves away from the predator (not toward or staying next to it)
    new = (5 + dx, 5 + dy)
    assert max(abs(new[0] - 5), abs(new[1] - 6)) >= 2
    # no predator near: heads for the only grass in view
    _place(env, male, (0, 0))
    grass = next(iter(env.grass_positions))
    env.grid_world_state[3, *env.grass_positions[grass]] = 0
    env.grass_positions[grass] = (5, 8)
    env.grass_energies[grass] = 2.0
    env.grid_world_state[3, 5, 8] = 2.0
    _place(env, female, (9, 0))
    dx, dy = env.action_to_move_tuple[env._scripted_prey_action(prey)]
    assert (dx, dy) in {(0, 1), (-1, 1), (1, 1)}


def test_scripted_prey_reproduce_and_get_hunted_without_errors():
    env = _scripted_env(grid_size=20, n_initial_active_predator_male=3, n_initial_active_predator_female=3, n_initial_active_prey=10,
                        initial_num_grass=60, initial_num_fruit=30, max_steps=300)
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(1)
    for _ in range(300):
        obs, r, t, tr, _ = env.step({a: int(rng.integers(env.num_actions)) for a in obs})
        assert all("prey" not in a for a in obs)
        assert set(env._scripted_prey_ids) <= set(env.agent_positions)
        if t.get("__all__") or tr.get("__all__"):
            break
    assert env.episode_births["prey"] > 0
    assert env.current_num_prey == len([a for a in env.agent_positions if a.startswith("prey")])


def test_scripted_prey_snapshot_round_trip():
    env = _scripted_env(n_initial_active_prey=3)
    snap = env.get_state_snapshot()
    ids = list(env._scripted_prey_ids)
    env.step(_noop(env))
    env._scripted_prey_ids = []
    env.restore_state_snapshot(snap)
    assert env._scripted_prey_ids == ids
