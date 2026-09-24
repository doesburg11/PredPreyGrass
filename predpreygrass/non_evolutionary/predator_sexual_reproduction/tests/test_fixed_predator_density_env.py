"""
Validation tests for FixedPredatorDensityEnv (../fixed_predator_density_env.py): the exact-density
predator intervention (cull overflow uniformly at random, replenish shortfall), which -- unlike
predator_population_cap -- never blocks reproduction. Modeled on test_fixed_prey_density_env.py's
_make_env + narrowly-scoped test_* pattern.

Run explicitly (not auto-discovered by the repo's pytest testpaths):
    pytest predpreygrass/non_evolutionary/predator_sexual_reproduction/tests/ -v
"""
import copy

import numpy as np
import pytest

from predpreygrass.non_evolutionary.predator_sexual_reproduction.config_env import config_env as _base_config_env
from predpreygrass.non_evolutionary.predator_sexual_reproduction.fixed_predator_density_env import (
    FixedPredatorDensityEnv,
)


def _make_env(overrides=None):
    config = copy.deepcopy(_base_config_env)
    config.update(
        {
            "grid_size": 10,
            "n_initial_active_predator_male": 2,
            "n_initial_active_predator_female": 2,
            "n_initial_active_prey": 10,
            "initial_num_grass": 5,
            "initial_num_fruit": 5,
            "max_steps": 50,
            "prey_density_floor": 5,
        }
    )
    if overrides:
        config.update(overrides)
    env = FixedPredatorDensityEnv(config)
    env.reset(seed=42)
    return env


def _noop_actions(env):
    return {agent: env.noop_action_id for agent in env.agents}


def _total_predators(env):
    return env.current_num_predator_male + env.current_num_predator_female


def test_culls_down_to_the_target_when_above_it():
    env = _make_env(overrides={"predator_density_target": 3})  # 4 initial predators, target 3
    assert _total_predators(env) == 4

    env.step(_noop_actions(env))

    assert _total_predators(env) == 3


def test_replenishes_up_to_the_target_when_below_it():
    env = _make_env(overrides={"predator_density_target": 6})  # 4 initial predators, target 6
    assert _total_predators(env) == 4

    env.step(_noop_actions(env))

    assert _total_predators(env) == 6


def test_no_change_when_already_at_the_target():
    env = _make_env(overrides={"predator_density_target": 4})
    assert _total_predators(env) == 4

    env.step(_noop_actions(env))

    assert _total_predators(env) == 4


def test_none_target_disables_the_mechanism_entirely():
    env = _make_env(overrides={"predator_density_target": None})
    assert _total_predators(env) == 4

    env.step(_noop_actions(env))

    assert _total_predators(env) == 4  # untouched -- no cull, no replenish


def test_culled_predator_pays_no_extra_penalty_and_is_removed_next_step():
    env = _make_env(overrides={"predator_density_target": 3})
    actions = _noop_actions(env)
    deaths_before = dict(env.episode_deaths)
    male_before, female_before = env.current_num_predator_male, env.current_num_predator_female
    # Noop actions don't move anyone, so each predator's pre-step position is still its position at
    # the moment it's culled -- captured here so the grid cell can be checked after removal, since
    # the victim's own position entry is gone from agent_positions by the time step() returns.
    positions_before = {a: pos for a, pos in env.agent_positions.items() if "predator" in a}

    observations, rewards, terminations, truncations, infos = env.step(actions)

    culled = [a for a in terminations if terminations[a] and "predator" in a and truncations.get(a) is False]
    assert len(culled) == 1
    victim = culled[0]
    # No extra penalty beyond whatever reward it already earned this step (here: 0, noop, no forage).
    assert rewards[victim] == 0.0
    assert victim in env._pending_removal

    # Full bookkeeping parity with every other death path (Codex review: the first version of this
    # test only checked agent_positions and _pending_removal, which would have let a regression in
    # any of the other removed-agent bookkeeping escape unnoticed).
    assert victim not in env.agent_positions
    assert victim not in env.predator_positions
    assert victim not in env.agent_energies
    assert env.grid_world_state[1, *positions_before[victim]] == 0
    sex = "predator_male" if "predator_male" in victim else "predator_female"
    if sex == "predator_male":
        assert env.current_num_predator_male == male_before - 1
    else:
        assert env.current_num_predator_female == female_before - 1
    assert env.episode_deaths[sex] == deaths_before[sex] + 1

    # The base class's own pending-removal cleanup (which every death path relies on) runs at the
    # START of the next step() call -- confirm it actually takes effect there.
    assert victim in env.agents
    env.step(_noop_actions(env))
    assert victim not in env.agents


def test_reproduction_is_never_blocked_even_at_the_target():
    """The defining difference from predator_population_cap: this class has no reproduction-blocking
    condition at all -- a birth always happens, pays its normal cost, and forms the normal mate bond,
    regardless of the target. (Target set to exactly match the post-birth population, 3, so
    NEITHER cull nor replenish triggers this step, keeping the parents' own agent_energies entries
    intact to check -- cull-vs-birth interaction is covered separately by the cull-specific tests;
    a target below 3 would risk the cull removing one of the two parents being asserted on, since it
    chooses uniformly among all three post-birth predators.)"""
    env = _make_env(
        overrides={
            "mate_search_radius": 2,
            "n_initial_active_predator_male": 1,
            "n_initial_active_predator_female": 1,
            "predator_density_target": 3,
        }
    )
    assert _total_predators(env) == 2
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)

    env.agent_energies[male] = env.predator_creation_energy_threshold + 1.0
    env.agent_energies[female] = env.predator_creation_energy_threshold + 1.0
    env.agent_positions[male] = (5, 5)
    env.predator_positions[male] = (5, 5)
    env.agent_positions[female] = (6, 6)
    env.predator_positions[female] = (6, 6)
    male_energy_before = env.agent_energies[male]
    female_energy_before = env.agent_energies[female]

    _, rewards, _, _, _ = env.step(_noop_actions(env))

    # Codex review: deriving "a birth happened" from env.agents membership is ambiguous here, since a
    # newborn immediately culled would still be in env.agents (pending removal only takes effect at
    # the top of the NEXT step()). episode_births and agent_parents are unambiguous regardless of
    # whether the newborn itself is the one later culled.
    assert env.episode_births["predator_male"] + env.episode_births["predator_female"] == 1
    child = next(a for a, parents in env.agent_parents.items() if parents == (male, female))
    assert child is not None
    child_energy = (
        env.initial_energy_predator_male if "predator_male" in child else env.initial_energy_predator_female
    )
    # Normal birth costs were actually deducted -- not skipped or refused.
    assert env.agent_energies[male] == pytest.approx(
        male_energy_before - env.homeostatic_energy_cost_per_step_predator - child_energy * env.predator_birth_cost_share_male
    )
    assert env.agent_energies[female] == pytest.approx(
        female_energy_before - env.homeostatic_energy_cost_per_step_predator - child_energy * env.predator_birth_cost_share_female
    )
    assert rewards[male] >= env.reproduction_reward_predator  # normal reproduction reward was paid
    assert rewards[female] >= env.reproduction_reward_predator
    assert env.agent_mate[male] == female  # normal mate bond was formed
    # Post-birth population (3) exactly matches the target -- neither cull nor replenish triggers.
    assert _total_predators(env) == 3


def test_max_steps_truncation_is_untouched_no_cull_or_replenish_attempted():
    env = _make_env(overrides={"predator_density_target": 3, "max_steps": 1})
    env.step(_noop_actions(env))  # current_step 0 -> 1; not yet truncated, culls normally
    total_before_truncation = _total_predators(env)

    _, _, terminations, truncations, _ = env.step(_noop_actions(env))  # current_step 1 >= max_steps 1

    assert truncations.get("__all__") is True
    assert _total_predators(env) == total_before_truncation  # override returned early


def test_predator_extinction_still_ends_the_episode_no_cull_or_replenish_attempted():
    env = _make_env(
        overrides={
            "n_initial_active_predator_male": 1,
            "n_initial_active_predator_female": 2,
            "predator_density_target": 6,  # would otherwise trigger replenishment
        }
    )
    male = next(a for a in env.agents if "predator_male" in a)
    env.agent_energies[male] = 0.0  # starves this step (Step 1's homeostatic cost makes it <= 0)

    _, _, terminations, _, _ = env.step(_noop_actions(env))

    assert env.current_num_predator_male == 0
    assert terminations.get("__all__") is True  # predator extinction ends the episode, not masked
    assert env.current_num_predator_female == 2  # guarded: no replenishment once a sex is extinct


@pytest.mark.parametrize("bad", [-1, 3.9, True, "5"])
def test_rejects_invalid_predator_density_target(bad):
    with pytest.raises(ValueError, match="predator_density_target"):
        _make_env(overrides={"predator_density_target": bad})


def test_rejects_predator_population_cap_combined_with_density_target():
    """Codex review: predator_population_cap lives in the shared base class and would silently
    still block reproduction here too if both were configured, contradicting this class's whole
    premise. Must be rejected outright, not merely discouraged in a CLI help string."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _make_env(overrides={"predator_density_target": 4, "predator_population_cap": 4})


def test_replenishment_falls_back_to_the_sex_with_remaining_room():
    """Codex review: choosing a sex uniformly BEFORE checking its pool would abort the whole
    replenishment loop on a single exhausted-sex coin-flip, even when the other sex's pool still has
    plenty of room. With the male pool already exhausted, replenishment must still reach the target
    using only females."""
    env = _make_env(
        overrides={
            "n_initial_active_predator_male": 1,
            "n_initial_active_predator_female": 1,
            "n_possible_predator_male": 1,  # exhausted: _next_predator_male_idx (1) >= this
            "predator_density_target": 6,
        }
    )
    assert env._next_predator_male_idx >= env.n_possible_predator_male  # confirm the setup

    env.step(_noop_actions(env))

    assert _total_predators(env) == 6  # reached despite the exhausted male pool
    assert env.current_num_predator_male == 1  # unchanged -- no room to grow
    assert env.current_num_predator_female == 5  # absorbed the entire deficit


def test_refreshes_every_live_agents_returned_observation_after_a_cull():
    """Same regression class as FixedPreyDensityEnv's observation-refresh test: the base class
    generates final observations (Step 6) BEFORE this override's cull/replenish runs, so without a
    refresh, every other agent's returned observation would be stale."""
    env = _make_env(
        overrides={
            "grid_size": 4,
            "n_initial_active_predator_male": 2,
            "n_initial_active_predator_female": 2,
            "n_initial_active_prey": 1,
            "initial_num_grass": 2,
            "initial_num_fruit": 2,
            "predator_density_target": 3,
        }
    )
    survivor_candidates = list(env.agents)

    observations, _, terminations, _, _ = env.step(_noop_actions(env))

    survivors = [a for a in survivor_candidates if not terminations.get(a, False) and a in env.agent_positions]
    assert survivors  # at least one predator (or the prey) must have survived the cull
    agent = survivors[0]
    fresh = env._get_observation(agent)
    assert np.array_equal(observations[agent], fresh)
