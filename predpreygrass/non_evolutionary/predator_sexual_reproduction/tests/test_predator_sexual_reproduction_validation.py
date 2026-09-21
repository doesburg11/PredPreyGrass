"""
Validation tests for the predator_sexual_reproduction environment: mate-finding
correctness, per-sex foraging roles, energy-split reproduction cost, ID
allocation, and extinction conditions. Modeled on
eco_evolutionary_nuptial_gift/tests/test_eco_evolutionary_validation.py's
_make_test_env + narrowly-scoped test_* pattern.

Run explicitly (not auto-discovered by the repo's pytest testpaths):
    pytest predpreygrass/non_evolutionary/predator_sexual_reproduction/tests/ -v
"""
import copy

import pytest

from predpreygrass.non_evolutionary.predator_sexual_reproduction.config_env import config_env as _base_config_env
from predpreygrass.non_evolutionary.predator_sexual_reproduction.predpreygrass_rllib_env import PredPreyGrass


class _FixedRandom:
    """Wraps a real np.random.Generator, forcing .random() to a fixed value
    while forwarding everything else (e.g. .integers(), used by
    _find_available_spawn_position's fallback) to the real generator.

    Needed because np.random.Generator is a C-extension type whose methods
    are read-only -- `env.rng.random = lambda: ...` raises AttributeError
    directly on the instance. Reassigning `env.rng` itself works fine, since
    that's a plain Python attribute on the env object, not on the Generator.
    """

    def __init__(self, real_rng, fixed_value):
        self._real_rng = real_rng
        self._fixed_value = fixed_value

    def random(self, *args, **kwargs):
        return self._fixed_value

    def __getattr__(self, name):
        return getattr(self._real_rng, name)


def _force_next_random(env, value):
    env.rng = _FixedRandom(env.rng, value)


def _make_test_env(overrides=None):
    config = copy.deepcopy(_base_config_env)
    config.update(
        {
            "grid_size": 10,
            "n_initial_active_predator_male": 2,
            "n_initial_active_predator_female": 2,
            "n_initial_active_prey": 2,
            "initial_num_grass": 5,
            "initial_num_fruit": 5,
            "max_steps": 50,
        }
    )
    if overrides:
        config.update(overrides)
    env = PredPreyGrass(config)
    env.reset(seed=42)
    return env


def _noop_actions(env):
    return {agent: env.noop_action_id for agent in env.agents}


def test_reset_creates_expected_agents():
    env = _make_test_env()
    male_ids = [a for a in env.agents if "predator_male" in a]
    female_ids = [a for a in env.agents if "predator_female" in a]
    prey_ids = [a for a in env.agents if "prey" in a and "predator" not in a]
    assert len(male_ids) == 2
    assert len(female_ids) == 2
    assert len(prey_ids) == 2


def test_predator_male_eats_fruit_and_hunts_same_step():
    """Both sexes can gather fruit; hunting is now a probabilistic contest for
    both sexes too (not a deterministic, male-only catch). Force the RNG into
    the success band so this remains a deterministic test of "gains energy
    from both engagements in the same step" rather than a flaky one.
    Disables male provisioning (unrelated to what this test checks) so a
    same-grid female doesn't add an untested energy deduction to the math."""
    env = _make_test_env(overrides={"male_gift_donation_rate": 0.0})
    male = next(a for a in env.agents if "predator_male" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))
    fruit = next(iter(env.fruit_positions))

    shared_pos = (5, 5)
    env.agent_positions[male] = shared_pos
    env.predator_positions[male] = shared_pos
    env.agent_positions[prey] = shared_pos
    env.prey_positions[prey] = shared_pos
    env.fruit_positions[fruit] = shared_pos
    env.fruit_energies[fruit] = 2.0
    prey_energy_before = env.agent_energies[prey]
    male_energy_before = env.agent_energies[male]

    _force_next_random(env, 0.0)  # forces the hunting-attempt success band
    actions = _noop_actions(env)
    obs, rewards, terms, truncs, infos = env.step(actions)

    assert prey not in env.agent_positions  # hunted
    # Gained the fruit's energy, and the prey's CURRENT energy at the moment
    # it's eaten -- which already reflects the prey's own Step-1 homeostatic
    # deduction (homeostatic cost is applied to every agent before any
    # engagement is resolved), not prey_energy_before.
    expected = (
        male_energy_before
        - env.homeostatic_energy_cost_per_step_predator
        + (prey_energy_before - env.homeostatic_energy_cost_per_step_prey)
        + 2.0
    )
    assert env.agent_energies[male] == expected


def test_reproduction_requires_both_sexes_eligible_and_nearby():
    """A lone eligible male with no eligible/nearby female must not reproduce."""
    env = _make_test_env(overrides={"mate_search_radius": 1})
    male = next(a for a in env.agents if "predator_male" in a)
    females = [a for a in env.agents if "predator_female" in a]

    env.agent_energies[male] = env.predator_creation_energy_threshold
    for female in females:
        env.agent_energies[female] = 0.5  # below threshold: ineligible
        # Push every female far away too, to double-guard against radius.
        env.agent_positions[female] = (9, 9)
        env.predator_positions[female] = (9, 9)
    env.agent_positions[male] = (0, 0)
    env.predator_positions[male] = (0, 0)

    n_predators_before = sum(1 for a in env.agents if "predator" in a)
    actions = _noop_actions(env)
    env.step(actions)

    n_predators_after = sum(1 for a in env.agents if "predator" in a)
    assert n_predators_after == n_predators_before  # no new predator born


def test_reproduction_succeeds_when_both_eligible_and_within_radius():
    # n_initial_active_predator_female=1 so the only female present is the
    # one being manipulated -- otherwise a second, untouched female (at her
    # default reset energy/position) would need to be separately excluded
    # from "new_predators" below.
    env = _make_test_env(
        overrides={"mate_search_radius": 2, "n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1}
    )
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    initial_agent_ids = set(env.agents)

    # A buffer above threshold: Step 1's homeostatic cost is deducted before
    # Step 5b's eligibility check runs within the same step() call, so energy
    # set exactly at threshold would drop below it before being checked.
    env.agent_energies[male] = env.predator_creation_energy_threshold + 1.0
    env.agent_energies[female] = env.predator_creation_energy_threshold + 1.0
    env.agent_positions[male] = (5, 5)
    env.predator_positions[male] = (5, 5)
    env.agent_positions[female] = (6, 6)  # Chebyshev distance 1 <= radius 2
    env.predator_positions[female] = (6, 6)

    male_energy_before = env.agent_energies[male]
    female_energy_before = env.agent_energies[female]

    actions = _noop_actions(env)
    obs, rewards, terms, truncs, infos = env.step(actions)

    new_predators = [a for a in env.agents if "predator" in a and a not in initial_agent_ids]
    assert len(new_predators) == 1
    child = new_predators[0]
    child_energy = env.agent_energies[child]

    # Birth cost is split asymmetrically: female pays the larger share.
    assert (
        env.agent_energies[male]
        == male_energy_before - env.homeostatic_energy_cost_per_step_predator - child_energy * env.predator_birth_cost_share_male
    )
    assert (
        env.agent_energies[female]
        == female_energy_before
        - env.homeostatic_energy_cost_per_step_predator
        - child_energy * env.predator_birth_cost_share_female
    )
    assert rewards[male] >= env.reproduction_reward_predator
    assert rewards[female] >= env.reproduction_reward_predator
    # A successful reproduction event records the pair bond both ways, used
    # by _apply_male_gift to restrict provisioning to this specific mate.
    assert env.agent_mate[male] == female
    assert env.agent_mate[female] == male


def test_a_male_cannot_be_paired_twice_in_one_step():
    """With two eligible females both in radius of one eligible male, only
    one pairing (one offspring) should happen, not two. Uses exactly one
    male so there's no untouched second male that could make this pass
    vacuously (a false-positive "birth" from unrelated pre-existing state)."""
    env = _make_test_env(
        overrides={"mate_search_radius": 3, "n_initial_active_predator_male": 1, "n_initial_active_predator_female": 2}
    )
    male = next(a for a in env.agents if "predator_male" in a)
    females = [a for a in env.agents if "predator_female" in a]
    assert len(females) == 2
    initial_agent_ids = set(env.agents)

    env.agent_energies[male] = env.predator_creation_energy_threshold + 1.0
    env.agent_positions[male] = (5, 5)
    env.predator_positions[male] = (5, 5)
    for i, female in enumerate(females):
        env.agent_energies[female] = env.predator_creation_energy_threshold + 1.0
        pos = (5, 6 + i)
        env.agent_positions[female] = pos
        env.predator_positions[female] = pos

    actions = _noop_actions(env)
    env.step(actions)

    new_predators = [a for a in env.agents if "predator" in a and a not in initial_agent_ids]
    assert len(new_predators) == 1
    # Exactly one of the two females paid the reproduction cost; the other
    # (unmatched) female's energy is untouched by reproduction this step.
    paid_count = sum(
        1
        for female in females
        if env.agent_energies[female] < env.predator_creation_energy_threshold + 1.0 - env.homeostatic_energy_cost_per_step_predator
    )
    assert paid_count == 1


def test_full_grid_declines_birth_instead_of_crashing():
    """When every cell is occupied, _find_available_spawn_position returns
    None. Both the prey (Step 5a) and predator (Step 5b) reproduction paths
    must decline the birth cleanly (no ID/energy/agents-list mutation, no
    crash) rather than assigning None as a position."""
    # 2x2 grid = 4 cells; active counts must fit within reset()'s placement.
    env = _make_test_env(
        overrides={
            "grid_size": 2,
            "initial_num_grass": 0,
            "initial_num_fruit": 0,
            "n_initial_active_predator_male": 1,
            "n_initial_active_predator_female": 1,
            "n_initial_active_prey": 2,
        }
    )
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))

    # Occupy every one of the 4 cells so no spawn position can ever be free.
    all_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env.agent_positions[male] = all_cells[0]
    env.predator_positions[male] = all_cells[0]
    env.agent_positions[female] = all_cells[1]
    env.predator_positions[female] = all_cells[1]
    env.agent_positions[prey] = all_cells[2]
    env.prey_positions[prey] = all_cells[2]
    # A second prey occupies the last free cell so the grid is fully packed.
    other_prey = next(a for a in env.agents if a.startswith("prey") and a != prey)
    env.agent_positions[other_prey] = all_cells[3]
    env.prey_positions[other_prey] = all_cells[3]

    env.agent_energies[male] = env.predator_creation_energy_threshold + 1.0
    env.agent_energies[female] = env.predator_creation_energy_threshold + 1.0
    env.agent_energies[prey] = env.prey_creation_energy_threshold + 1.0

    n_agents_before = len(env.agents)
    actions = _noop_actions(env)
    obs, rewards, terms, truncs, infos = env.step(actions)  # must not raise

    assert len(env.agents) == n_agents_before  # no births actually landed


def test_prey_reproduction_is_unchanged_asexual():
    """Prey still reproduces solo once past its energy threshold -- no mate
    search involved."""
    env = _make_test_env()
    prey = next(a for a in env.agents if a.startswith("prey"))
    # Buffer above threshold: Step 1's homeostatic cost is deducted before
    # Step 5a's eligibility check runs within the same step() call.
    env.agent_energies[prey] = env.prey_creation_energy_threshold + 1.0

    n_prey_before = sum(1 for a in env.agents if a.startswith("prey"))
    actions = _noop_actions(env)
    env.step(actions)
    n_prey_after = sum(1 for a in env.agents if a.startswith("prey"))
    assert n_prey_after == n_prey_before + 1


def test_episode_terminates_when_either_predator_sex_extinct():
    env = _make_test_env()
    for female in [a for a in env.agents if "predator_female" in a]:
        env.agent_energies[female] = -1.0  # force starvation this step

    actions = _noop_actions(env)
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert terms["__all__"] is True


def test_no_agent_id_reused_within_episode():
    env = _make_test_env(overrides={"mate_search_radius": 2})
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    env.agent_energies[male] = env.predator_creation_energy_threshold + 1.0
    env.agent_energies[female] = env.predator_creation_energy_threshold + 1.0
    env.agent_positions[male] = (5, 5)
    env.predator_positions[male] = (5, 5)
    env.agent_positions[female] = (5, 6)
    env.predator_positions[female] = (5, 6)

    seen_ids = set(env.agents)
    actions = _noop_actions(env)
    env.step(actions)
    new_ids = set(env.agents) - seen_ids
    assert len(new_ids) == 1
    assert not (new_ids & seen_ids)


def test_resolve_hunting_attempt_outcome_bands():
    """Pins the three outcome bands of _resolve_hunting_attempt directly,
    documenting the RNG-forcing mechanism as the recommended pattern for
    testing this stochastic branch, decoupled from full step() plumbing."""
    env = _make_test_env()
    success_prob, death_prob = 0.5, 0.3  # bands: [0,0.5) success, [0.5,0.8) death, [0.8,1) failure

    def make_fake_pair(tag):
        predator_id = f"predator_male_test_{tag}"
        prey_id = f"prey_test_{tag}"
        pos = (0, 0)
        env.agent_positions[predator_id] = pos
        env.predator_positions[predator_id] = pos
        env.agent_energies[predator_id] = 5.0
        env.cumulative_rewards[predator_id] = 0
        env.agent_positions[prey_id] = pos
        env.prey_positions[prey_id] = pos
        env.agent_energies[prey_id] = 3.0
        env.cumulative_rewards[prey_id] = 0
        return predator_id, prey_id, pos

    predator_id, prey_id, pos = make_fake_pair("success")
    _force_next_random(env, 0.0)
    outcome, _, _ = env._resolve_hunting_attempt(predator_id, pos, prey_id, success_prob, death_prob, {}, {}, {}, {})
    assert outcome == "success"
    assert prey_id not in env.agent_positions

    predator_id, prey_id, pos = make_fake_pair("death")
    _force_next_random(env, 0.6)
    outcome, _, _ = env._resolve_hunting_attempt(predator_id, pos, prey_id, success_prob, death_prob, {}, {}, {}, {})
    assert outcome == "predator_dies"
    assert predator_id not in env.agent_positions
    assert prey_id in env.agent_positions  # untouched

    predator_id, prey_id, pos = make_fake_pair("failure")
    _force_next_random(env, 0.9)
    outcome, _, _ = env._resolve_hunting_attempt(predator_id, pos, prey_id, success_prob, death_prob, {}, {}, {}, {})
    assert outcome == "failure"
    assert predator_id in env.agent_positions
    assert prey_id in env.agent_positions


def test_predator_female_can_successfully_hunt():
    """predator_female can now attempt to hunt (reversing the old structural
    incapability) -- force the RNG into the success band and confirm she eats
    the prey exactly like a male would."""
    env = _make_test_env()
    female = next(a for a in env.agents if "predator_female" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))

    shared_pos = (5, 5)
    env.agent_positions[female] = shared_pos
    env.predator_positions[female] = shared_pos
    env.agent_positions[prey] = shared_pos
    env.prey_positions[prey] = shared_pos

    _force_next_random(env, 0.0)  # forces the success band for any success_prob > 0
    actions = _noop_actions(env)
    obs, rewards, terms, truncs, infos = env.step(actions)

    assert prey not in env.agent_positions
    assert rewards[female] >= env.reward_predator_catch_prey


def test_predator_female_dies_from_failed_hunt_and_prey_survives():
    env = _make_test_env()
    female = next(a for a in env.agents if "predator_female" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))

    shared_pos = (5, 5)
    env.agent_positions[female] = shared_pos
    env.predator_positions[female] = shared_pos
    env.agent_positions[prey] = shared_pos
    env.prey_positions[prey] = shared_pos
    prey_energy_before = env.agent_energies[prey]

    # Lands in the death band: success_prob <= roll < success_prob + death_prob.
    _force_next_random(env, env.prey_vs_predator_female_success_prob + 1e-6)
    actions = _noop_actions(env)
    obs, rewards, terms, truncs, infos = env.step(actions)

    assert female not in env.agent_positions
    assert terms.get(female) is True
    # Prey is untouched by the combat outcome -- only its ordinary Step-1
    # homeostatic cost applies, same as any agent that step.
    assert prey in env.agent_positions
    assert not terms.get(prey, False)
    assert env.agent_energies[prey] == prey_energy_before - env.homeostatic_energy_cost_per_step_prey


def test_predator_male_dies_from_failed_hunt_and_prey_survives():
    env = _make_test_env()
    male = next(a for a in env.agents if "predator_male" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))

    shared_pos = (5, 5)
    env.agent_positions[male] = shared_pos
    env.predator_positions[male] = shared_pos
    env.agent_positions[prey] = shared_pos
    env.prey_positions[prey] = shared_pos
    prey_energy_before = env.agent_energies[prey]

    _force_next_random(env, env.prey_vs_predator_male_success_prob + 1e-6)
    actions = _noop_actions(env)
    obs, rewards, terms, truncs, infos = env.step(actions)

    assert male not in env.agent_positions
    assert terms.get(male) is True
    assert prey in env.agent_positions
    assert not terms.get(prey, False)
    assert env.agent_energies[prey] == prey_energy_before - env.homeostatic_energy_cost_per_step_prey


def test_birth_cost_split_uses_configured_shares():
    """Confirms the birth-cost split is actually read from config, not a
    hardcoded literal -- override to non-default shares and verify."""
    env = _make_test_env(
        overrides={
            "mate_search_radius": 2,
            "n_initial_active_predator_male": 1,
            "n_initial_active_predator_female": 1,
            "predator_birth_cost_share_female": 0.7,
            "predator_birth_cost_share_male": 0.3,
        }
    )
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    initial_agent_ids = set(env.agents)

    env.agent_energies[male] = env.predator_creation_energy_threshold + 1.0
    env.agent_energies[female] = env.predator_creation_energy_threshold + 1.0
    env.agent_positions[male] = (5, 5)
    env.predator_positions[male] = (5, 5)
    env.agent_positions[female] = (6, 6)
    env.predator_positions[female] = (6, 6)

    male_energy_before = env.agent_energies[male]
    female_energy_before = env.agent_energies[female]

    actions = _noop_actions(env)
    env.step(actions)

    new_predators = [a for a in env.agents if "predator" in a and a not in initial_agent_ids]
    assert len(new_predators) == 1
    child_energy = env.agent_energies[new_predators[0]]

    assert env.agent_energies[male] == male_energy_before - env.homeostatic_energy_cost_per_step_predator - child_energy * 0.3
    assert (
        env.agent_energies[female]
        == female_energy_before - env.homeostatic_energy_cost_per_step_predator - child_energy * 0.7
    )


def test_offspring_spawns_near_female_not_male():
    """Offspring spawns adjacent to the FEMALE (mate), not the male --
    reversing the module's original male-anchored spawn position."""
    env = _make_test_env(
        overrides={
            "grid_size": 20,
            "mate_search_radius": 3,
            "n_initial_active_predator_male": 1,
            "n_initial_active_predator_female": 1,
        }
    )
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    initial_agent_ids = set(env.agents)

    env.agent_energies[male] = env.predator_creation_energy_threshold + 1.0
    env.agent_energies[female] = env.predator_creation_energy_threshold + 1.0
    male_pos = (2, 2)
    # Chebyshev distance 3 <= radius 3, but far enough apart that "adjacent
    # to A" and "adjacent to B" can never both be true.
    female_pos = (2, 5)
    env.agent_positions[male] = male_pos
    env.predator_positions[male] = male_pos
    env.agent_positions[female] = female_pos
    env.predator_positions[female] = female_pos

    actions = _noop_actions(env)
    env.step(actions)

    new_predators = [a for a in env.agents if "predator" in a and a not in initial_agent_ids]
    assert len(new_predators) == 1
    child_pos = env.agent_positions[new_predators[0]]

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    assert manhattan(child_pos, female_pos) == 1
    assert manhattan(child_pos, male_pos) != 1


def test_male_hunting_success_donates_to_nearby_female():
    """A predator_male's successful hunt donates male_gift_donation_rate of
    the energy gained to HIS RECORDED MATE -- offsets her post-birth energy
    deficit (see README's "Male provisioning" section). Provisioning is
    exclusive/pair-bonded: it only reaches an agent recorded in
    env.agent_mate, not just any nearby female (see
    test_donation_goes_only_to_recorded_mate_not_other_nearby_females)."""
    env = _make_test_env(
        overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1}
    )
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))
    env.agent_mate[male] = female
    env.agent_mate[female] = male

    male_pos = (5, 5)
    env.agent_positions[male] = male_pos
    env.predator_positions[male] = male_pos
    env.agent_positions[prey] = male_pos
    env.prey_positions[prey] = male_pos
    female_pos = (5, 6)  # Chebyshev distance 1 <= default predator_gift_range (3)
    env.agent_positions[female] = female_pos
    env.predator_positions[female] = female_pos

    prey_energy_before = env.agent_energies[prey]
    male_energy_before = env.agent_energies[male]
    female_energy_before = env.agent_energies[female]

    _force_next_random(env, 0.0)  # forces the hunting-attempt success band
    actions = _noop_actions(env)
    env.step(actions)

    # Prey's energy at the moment of the catch already reflects its own
    # Step-1 homeostatic deduction (applied to every agent before any
    # engagement is resolved).
    energy_gained = prey_energy_before - env.homeostatic_energy_cost_per_step_prey
    donation = env.male_gift_donation_rate * energy_gained

    assert (
        env.agent_energies[male]
        == male_energy_before - env.homeostatic_energy_cost_per_step_predator + energy_gained - donation
    )
    assert (
        env.agent_energies[female] == female_energy_before - env.homeostatic_energy_cost_per_step_predator + donation
    )


def test_no_donation_before_first_reproduction():
    """A male with no recorded mate yet (never successfully reproduced)
    gives no gift, even to a female right next to him."""
    env = _make_test_env(
        overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1}
    )
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))
    assert male not in env.agent_mate  # no reproduction has happened yet

    male_pos = (5, 5)
    env.agent_positions[male] = male_pos
    env.predator_positions[male] = male_pos
    env.agent_positions[prey] = male_pos
    env.prey_positions[prey] = male_pos
    female_pos = (5, 6)  # would be well within default predator_gift_range (3)
    env.agent_positions[female] = female_pos
    env.predator_positions[female] = female_pos

    prey_energy_before = env.agent_energies[prey]
    male_energy_before = env.agent_energies[male]
    female_energy_before = env.agent_energies[female]

    _force_next_random(env, 0.0)
    actions = _noop_actions(env)
    env.step(actions)

    energy_gained = prey_energy_before - env.homeostatic_energy_cost_per_step_prey
    assert env.agent_energies[male] == male_energy_before - env.homeostatic_energy_cost_per_step_predator + energy_gained
    # Female's energy only reflects her own ordinary homeostatic cost -- no gift.
    assert env.agent_energies[female] == female_energy_before - env.homeostatic_energy_cost_per_step_predator


def test_no_donation_when_mate_out_of_range():
    """His recorded mate is out of predator_gift_range this step -- the
    physical-proximity requirement still applies even to an established
    pair bond. He keeps the full gain."""
    env = _make_test_env(
        overrides={
            "n_initial_active_predator_male": 1,
            "n_initial_active_predator_female": 1,
            "predator_gift_range": 1,
        }
    )
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))
    env.agent_mate[male] = female
    env.agent_mate[female] = male

    male_pos = (5, 5)
    env.agent_positions[male] = male_pos
    env.predator_positions[male] = male_pos
    env.agent_positions[prey] = male_pos
    env.prey_positions[prey] = male_pos
    female_pos = (5, 8)  # Chebyshev distance 3 > predator_gift_range (1)
    env.agent_positions[female] = female_pos
    env.predator_positions[female] = female_pos

    prey_energy_before = env.agent_energies[prey]
    male_energy_before = env.agent_energies[male]
    female_energy_before = env.agent_energies[female]

    _force_next_random(env, 0.0)
    actions = _noop_actions(env)
    env.step(actions)

    energy_gained = prey_energy_before - env.homeostatic_energy_cost_per_step_prey
    assert env.agent_energies[male] == male_energy_before - env.homeostatic_energy_cost_per_step_predator + energy_gained
    assert env.agent_energies[female] == female_energy_before - env.homeostatic_energy_cost_per_step_predator


def test_donation_goes_only_to_recorded_mate_not_other_nearby_females():
    """Two predator_females are equally nearby, but only one is his recorded
    mate -- she gets the full donation (not split), the other stranger
    female gets nothing. Pins the exclusivity behavior the user specifically
    asked for over the module's earlier broadcast-to-any-nearby-female
    design."""
    env = _make_test_env(
        overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 2}
    )
    male = next(a for a in env.agents if "predator_male" in a)
    females = [a for a in env.agents if "predator_female" in a]
    assert len(females) == 2
    mate, stranger = females[0], females[1]
    env.agent_mate[male] = mate
    env.agent_mate[mate] = male
    prey = next(a for a in env.agents if a.startswith("prey"))

    male_pos = (5, 5)
    env.agent_positions[male] = male_pos
    env.predator_positions[male] = male_pos
    env.agent_positions[prey] = male_pos
    env.prey_positions[prey] = male_pos
    female_positions = {mate: (5, 6), stranger: (6, 5)}  # both Chebyshev distance 1
    female_energies_before = {}
    for female, pos in female_positions.items():
        env.agent_positions[female] = pos
        env.predator_positions[female] = pos
        female_energies_before[female] = env.agent_energies[female]

    prey_energy_before = env.agent_energies[prey]

    _force_next_random(env, 0.0)
    actions = _noop_actions(env)
    env.step(actions)

    energy_gained = prey_energy_before - env.homeostatic_energy_cost_per_step_prey
    donation = env.male_gift_donation_rate * energy_gained

    assert (
        env.agent_energies[mate] == female_energies_before[mate] - env.homeostatic_energy_cost_per_step_predator + donation
    )
    # The stranger receives nothing, despite being equally nearby.
    assert env.agent_energies[stranger] == female_energies_before[stranger] - env.homeostatic_energy_cost_per_step_predator


def test_female_successful_hunt_does_not_trigger_donation():
    """Donation is unidirectional (male -> female only). A female's own
    successful hunt must not affect a nearby male's energy at all."""
    env = _make_test_env(
        overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1}
    )
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))

    female_pos = (5, 5)
    env.agent_positions[female] = female_pos
    env.predator_positions[female] = female_pos
    env.agent_positions[prey] = female_pos
    env.prey_positions[prey] = female_pos
    male_pos = (5, 6)  # nearby, within default predator_gift_range
    env.agent_positions[male] = male_pos
    env.predator_positions[male] = male_pos

    male_energy_before = env.agent_energies[male]

    _force_next_random(env, 0.0)  # forces the female's own success band
    actions = _noop_actions(env)
    env.step(actions)

    assert prey not in env.agent_positions  # female's hunt succeeded
    # Male is untouched -- only his own ordinary homeostatic cost applies.
    assert env.agent_energies[male] == male_energy_before - env.homeostatic_energy_cost_per_step_predator


def test_no_donation_to_already_starving_female():
    """A female whose energy is already <= 0 this step (about to starve) must
    not receive a gift that rescues her. Regression test for an
    order-dependent bug: on the very first step of an episode self.agents
    isn't sorted yet (males precede females), so a male's turn -- and his
    gift -- could run before a female's own starvation check, letting an
    unfiltered gift push her energy back above zero and make her survive
    purely as an accident of iteration order."""
    env = _make_test_env(
        overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1}
    )
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))
    env.agent_mate[male] = female
    env.agent_mate[female] = male

    male_pos = (5, 5)
    env.agent_positions[male] = male_pos
    env.predator_positions[male] = male_pos
    env.agent_positions[prey] = male_pos
    env.prey_positions[prey] = male_pos
    female_pos = (5, 6)
    env.agent_positions[female] = female_pos
    env.predator_positions[female] = female_pos
    # Guarantees her Step-1 homeostatic hit alone drops her to <= 0.
    env.agent_energies[female] = env.homeostatic_energy_cost_per_step_predator / 2.0

    # Force male-before-female iteration order this step, mimicking the
    # unsorted post-reset agent list on an episode's first step.
    env.agents = [male, female, prey]

    _force_next_random(env, 0.0)  # forces the male's hunting success band
    actions = _noop_actions(env)
    obs, rewards, terms, truncs, infos = env.step(actions)

    assert female not in env.agent_positions  # she starved as expected
    assert terms.get(female) is True


def test_remating_severs_stale_reverse_mate_pointer():
    """Regression test: if a female was previously bonded to one male and
    later remates with a different male, the first male's stale one-way
    pointer to her must be removed -- otherwise he'd keep donating gifts to
    an ex indefinitely, and she'd be receiving from two "mates" at once,
    breaking the exclusivity _apply_male_gift is supposed to guarantee."""
    env = _make_test_env(
        overrides={"mate_search_radius": 2, "n_initial_active_predator_male": 2, "n_initial_active_predator_female": 1}
    )
    males = [a for a in env.agents if "predator_male" in a]
    assert len(males) == 2
    female = next(a for a in env.agents if "predator_female" in a)
    m1, m2 = males

    # Simulate a prior bond from an earlier reproduction event.
    env.agent_mate[m1] = female
    env.agent_mate[female] = m1
    env.has_reproduced.add(m1)
    env.has_reproduced.add(female)

    # Keep m1 far away and energy-ineligible so only m2+female pair this step.
    env.agent_positions[m1] = (0, 0)
    env.predator_positions[m1] = (0, 0)
    env.agent_energies[m1] = 1.0

    env.agent_energies[m2] = env.predator_creation_energy_threshold + 1.0
    env.agent_energies[female] = env.predator_creation_energy_threshold + 1.0
    env.agent_positions[m2] = (5, 5)
    env.predator_positions[m2] = (5, 5)
    env.agent_positions[female] = (5, 6)
    env.predator_positions[female] = (5, 6)

    actions = _noop_actions(env)
    env.step(actions)

    assert env.agent_mate[female] == m2
    assert env.agent_mate[m2] == female
    # m1's stale pointer to the female must be gone -- either removed
    # entirely or (if present for any other reason) not pointing at her.
    assert env.agent_mate.get(m1) != female
    # has_reproduced is permanent history, unlike agent_mate's mutable
    # current-pair bookkeeping -- remating must NOT un-mark m1 (or the
    # female) as having reproduced before.
    assert m1 in env.has_reproduced
    assert female in env.has_reproduced


def test_reproduction_records_parentage():
    """A successful reproduction event records agent_parents[child] =
    (father, mother), used by _share_energy_with_offspring."""
    env = _make_test_env(
        overrides={"mate_search_radius": 2, "n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1}
    )
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    initial_agent_ids = set(env.agents)

    env.agent_energies[male] = env.predator_creation_energy_threshold + 1.0
    env.agent_energies[female] = env.predator_creation_energy_threshold + 1.0
    env.agent_positions[male] = (5, 5)
    env.predator_positions[male] = (5, 5)
    env.agent_positions[female] = (6, 6)
    env.predator_positions[female] = (6, 6)

    actions = _noop_actions(env)
    env.step(actions)

    new_predators = [a for a in env.agents if "predator" in a and a not in initial_agent_ids]
    assert len(new_predators) == 1
    child = new_predators[0]
    assert env.agent_parents[child] == (male, female)
    # Both parents are now breeding adults -- see has_reproduced.
    assert male in env.has_reproduced
    assert female in env.has_reproduced
    assert child not in env.has_reproduced  # the newborn itself hasn't


def test_male_hunt_success_shares_with_nearby_child():
    """A father's successful hunt shares parent_offspring_share_rate of the
    gain with his own nearby child, independent of (and on top of) the
    mate-gift mechanic."""
    env = _make_test_env(
        overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1}
    )
    male = next(a for a in env.agents if "predator_male" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))

    male_pos = (5, 5)
    env.agent_positions[male] = male_pos
    env.predator_positions[male] = male_pos
    env.agent_positions[prey] = male_pos
    env.prey_positions[prey] = male_pos

    child = "predator_female_test_child"
    child_pos = (5, 6)
    env.agent_positions[child] = child_pos
    env.predator_positions[child] = child_pos
    env.agent_energies[child] = 2.0
    env.cumulative_rewards[child] = 0
    env.agent_parents[child] = (male, "predator_female_0")
    env.agents.append(child)

    prey_energy_before = env.agent_energies[prey]
    male_energy_before = env.agent_energies[male]
    child_energy_before = env.agent_energies[child]

    _force_next_random(env, 0.0)  # forces the hunting-attempt success band
    actions = _noop_actions(env)
    env.step(actions)

    energy_gained = prey_energy_before - env.homeostatic_energy_cost_per_step_prey
    donation = env.parent_offspring_share_rate * energy_gained

    assert (
        env.agent_energies[male]
        == male_energy_before - env.homeostatic_energy_cost_per_step_predator + energy_gained - donation
    )
    assert env.agent_energies[child] == child_energy_before - env.homeostatic_energy_cost_per_step_predator + donation


def test_female_fruit_success_shares_with_nearby_child():
    """A mother's successful fruit-gather shares parent_offspring_share_rate
    of the gain with her own nearby child -- parental care applies to
    either sex's foraging, not just hunting."""
    env = _make_test_env(
        overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1}
    )
    female = next(a for a in env.agents if "predator_female" in a)
    fruit = next(iter(env.fruit_positions))

    female_pos = (5, 5)
    env.agent_positions[female] = female_pos
    env.predator_positions[female] = female_pos
    env.fruit_positions[fruit] = female_pos
    env.fruit_energies[fruit] = 2.0

    child = "predator_male_test_child"
    child_pos = (5, 6)
    env.agent_positions[child] = child_pos
    env.predator_positions[child] = child_pos
    env.agent_energies[child] = 2.0
    env.cumulative_rewards[child] = 0
    env.agent_parents[child] = ("predator_male_0", female)
    env.agents.append(child)

    female_energy_before = env.agent_energies[female]
    child_energy_before = env.agent_energies[child]

    actions = _noop_actions(env)
    env.step(actions)

    donation = env.parent_offspring_share_rate * 2.0  # fruit's full energy, before her homeostatic cost

    assert (
        env.agent_energies[female]
        == female_energy_before - env.homeostatic_energy_cost_per_step_predator + 2.0 - donation
    )
    assert env.agent_energies[child] == child_energy_before - env.homeostatic_energy_cost_per_step_predator + donation


def test_no_share_with_unrelated_nearby_predator():
    """A nearby predator that is NOT this agent's recorded offspring
    receives nothing, even at the same distance a real child would."""
    env = _make_test_env(
        overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1}
    )
    male = next(a for a in env.agents if "predator_male" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))

    male_pos = (5, 5)
    env.agent_positions[male] = male_pos
    env.predator_positions[male] = male_pos
    env.agent_positions[prey] = male_pos
    env.prey_positions[prey] = male_pos

    stranger = "predator_female_test_stranger"
    stranger_pos = (5, 6)
    env.agent_positions[stranger] = stranger_pos
    env.predator_positions[stranger] = stranger_pos
    env.agent_energies[stranger] = 2.0
    env.cumulative_rewards[stranger] = 0
    # No agent_parents entry recorded for `stranger` at all -- not this
    # male's child (or anyone's).
    env.agents.append(stranger)

    prey_energy_before = env.agent_energies[prey]
    male_energy_before = env.agent_energies[male]
    stranger_energy_before = env.agent_energies[stranger]

    _force_next_random(env, 0.0)
    actions = _noop_actions(env)
    env.step(actions)

    energy_gained = prey_energy_before - env.homeostatic_energy_cost_per_step_prey
    assert env.agent_energies[male] == male_energy_before - env.homeostatic_energy_cost_per_step_predator + energy_gained
    assert env.agent_energies[stranger] == stranger_energy_before - env.homeostatic_energy_cost_per_step_predator


def test_share_split_among_multiple_children():
    """Two of the forager's own children, both nearby, split the donation
    evenly (unlike the exclusive, single-recipient mate gift)."""
    env = _make_test_env(
        overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1}
    )
    male = next(a for a in env.agents if "predator_male" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))

    male_pos = (5, 5)
    env.agent_positions[male] = male_pos
    env.predator_positions[male] = male_pos
    env.agent_positions[prey] = male_pos
    env.prey_positions[prey] = male_pos

    children = ["predator_female_test_child_a", "predator_male_test_child_b"]
    child_positions = [(5, 6), (6, 5)]
    child_energies_before = {}
    for child, pos in zip(children, child_positions):
        env.agent_positions[child] = pos
        env.predator_positions[child] = pos
        env.agent_energies[child] = 2.0
        env.cumulative_rewards[child] = 0
        env.agent_parents[child] = (male, "predator_female_0")
        env.agents.append(child)
        child_energies_before[child] = env.agent_energies[child]

    prey_energy_before = env.agent_energies[prey]

    _force_next_random(env, 0.0)
    actions = _noop_actions(env)
    env.step(actions)

    energy_gained = prey_energy_before - env.homeostatic_energy_cost_per_step_prey
    donation_total = env.parent_offspring_share_rate * energy_gained
    share = donation_total / 2

    for child in children:
        assert (
            env.agent_energies[child]
            == child_energies_before[child] - env.homeostatic_energy_cost_per_step_predator + share
        )


def test_male_fruit_success_shares_with_nearby_child():
    """A father's successful fruit-gather (not just his hunts) shares with
    his nearby child too."""
    env = _make_test_env(
        overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1}
    )
    male = next(a for a in env.agents if "predator_male" in a)
    fruit = next(iter(env.fruit_positions))

    male_pos = (5, 5)
    env.agent_positions[male] = male_pos
    env.predator_positions[male] = male_pos
    env.fruit_positions[fruit] = male_pos
    env.fruit_energies[fruit] = 2.0

    child = "predator_female_test_child"
    child_pos = (5, 6)
    env.agent_positions[child] = child_pos
    env.predator_positions[child] = child_pos
    env.agent_energies[child] = 2.0
    env.cumulative_rewards[child] = 0
    env.agent_parents[child] = (male, "predator_female_0")
    env.agents.append(child)

    male_energy_before = env.agent_energies[male]
    child_energy_before = env.agent_energies[child]

    actions = _noop_actions(env)
    env.step(actions)

    donation = env.parent_offspring_share_rate * 2.0

    assert env.agent_energies[male] == male_energy_before - env.homeostatic_energy_cost_per_step_predator + 2.0 - donation
    assert env.agent_energies[child] == child_energy_before - env.homeostatic_energy_cost_per_step_predator + donation


def test_female_hunt_success_shares_with_nearby_child():
    """A mother's successful hunt (rare, but possible) shares with her
    nearby child too -- not just her fruit-gathers."""
    env = _make_test_env(
        overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1}
    )
    female = next(a for a in env.agents if "predator_female" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))

    female_pos = (5, 5)
    env.agent_positions[female] = female_pos
    env.predator_positions[female] = female_pos
    env.agent_positions[prey] = female_pos
    env.prey_positions[prey] = female_pos

    child = "predator_male_test_child"
    child_pos = (5, 6)
    env.agent_positions[child] = child_pos
    env.predator_positions[child] = child_pos
    env.agent_energies[child] = 2.0
    env.cumulative_rewards[child] = 0
    env.agent_parents[child] = ("predator_male_0", female)
    env.agents.append(child)

    prey_energy_before = env.agent_energies[prey]
    female_energy_before = env.agent_energies[female]
    child_energy_before = env.agent_energies[child]

    _force_next_random(env, 0.0)  # forces the female's own hunting success band
    actions = _noop_actions(env)
    env.step(actions)

    energy_gained = prey_energy_before - env.homeostatic_energy_cost_per_step_prey
    donation = env.parent_offspring_share_rate * energy_gained

    assert (
        env.agent_energies[female]
        == female_energy_before - env.homeostatic_energy_cost_per_step_predator + energy_gained - donation
    )
    assert env.agent_energies[child] == child_energy_before - env.homeostatic_energy_cost_per_step_predator + donation


def test_combined_donation_rates_exceeding_one_raises_error():
    """male_gift_donation_rate and parent_offspring_share_rate both apply to
    the same gross hunting gain (not sequentially off a shrinking
    remainder) -- an unchecked sum above 1.0 could deduct more energy than
    a hunt actually gained, so __init__ must reject it."""
    with pytest.raises(ValueError):
        _make_test_env(overrides={"male_gift_donation_rate": 0.7, "parent_offspring_share_rate": 0.4})


def test_no_share_with_offspring_that_has_reproduced():
    """A reproduction-based independence cutoff: an offspring that has
    already reproduced itself is a breeding adult, not a dependent
    juvenile, so it stops receiving parental care even if it's recorded as
    this forager's child and stands right next to it."""
    env = _make_test_env(
        overrides={"n_initial_active_predator_male": 1, "n_initial_active_predator_female": 1}
    )
    male = next(a for a in env.agents if "predator_male" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))

    male_pos = (5, 5)
    env.agent_positions[male] = male_pos
    env.predator_positions[male] = male_pos
    env.agent_positions[prey] = male_pos
    env.prey_positions[prey] = male_pos

    grown_child = "predator_female_test_grown_child"
    child_pos = (5, 6)
    env.agent_positions[grown_child] = child_pos
    env.predator_positions[grown_child] = child_pos
    env.agent_energies[grown_child] = 5.0
    env.cumulative_rewards[grown_child] = 0
    env.agent_parents[grown_child] = (male, "predator_female_0")
    env.has_reproduced.add(grown_child)  # already a breeding adult itself
    env.agents.append(grown_child)

    prey_energy_before = env.agent_energies[prey]
    male_energy_before = env.agent_energies[male]
    grown_child_energy_before = env.agent_energies[grown_child]

    _force_next_random(env, 0.0)  # forces the hunting-attempt success band
    actions = _noop_actions(env)
    env.step(actions)

    energy_gained = prey_energy_before - env.homeostatic_energy_cost_per_step_prey
    # Male keeps the whole gain (minus any mate gift, but he has no
    # recorded mate here) -- no offspring-share deduction at all.
    assert env.agent_energies[male] == male_energy_before - env.homeostatic_energy_cost_per_step_predator + energy_gained
    # The grown child gets nothing beyond its own ordinary homeostatic cost.
    assert (
        env.agent_energies[grown_child]
        == grown_child_energy_before - env.homeostatic_energy_cost_per_step_predator
    )


def test_training_metrics_track_hunting_and_provisioning():
    """_build_episode_training_metrics reports hunting attempts/successes/
    deaths by sex and mate-gift/parental-care event+energy totals -- the
    observability needed to tell 'never attempts hunting' apart from
    'attempts but fails' or 'never gets the chance', and to see whether the
    provisioning mechanics are actually firing during training."""
    env = _make_test_env(
        overrides={
            "n_initial_active_predator_male": 1,
            "n_initial_active_predator_female": 1,
            "n_initial_active_prey": 1,  # avoid a second prey coincidentally landing on the child's cell
        }
    )
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))

    # A successful male hunt with the female as his recorded mate nearby,
    # so both the hunting counters and the mate-gift counters fire.
    env.agent_mate[male] = female
    env.agent_mate[female] = male
    male_pos = (5, 5)
    env.agent_positions[male] = male_pos
    env.predator_positions[male] = male_pos
    env.agent_positions[prey] = male_pos
    env.prey_positions[prey] = male_pos
    female_pos = (5, 6)
    env.agent_positions[female] = female_pos
    env.predator_positions[female] = female_pos

    child = "predator_male_test_child"
    child_pos = (5, 4)
    env.agent_positions[child] = child_pos
    env.predator_positions[child] = child_pos
    env.agent_energies[child] = 2.0
    env.cumulative_rewards[child] = 0
    env.agent_parents[child] = (male, female)
    env.agents.append(child)

    _force_next_random(env, 0.0)  # forces the male's hunting success band
    actions = _noop_actions(env)
    env.step(actions)

    metrics = env._build_episode_training_metrics()

    assert metrics["hunting_attempts_predator_male"] == 1
    assert metrics["hunting_successes_predator_male"] == 1
    assert metrics["hunting_success_rate_predator_male"] == 1.0
    assert metrics["hunting_deaths_predator_male"] == 0
    assert metrics["hunting_attempts_predator_female"] == 0
    assert metrics["mate_gift_events"] == 1
    assert metrics["mate_gift_energy_total"] > 0.0
    assert metrics["parental_care_events"] == 1
    assert metrics["parental_care_energy_total"] > 0.0


# ---------------------------------------------------------------------------
# Energy-proportional forage reward (reward_predator_per_energy)
# ---------------------------------------------------------------------------


def _isolate_predator_on_cell(env, predator, pos, prey=None, fruit=None):
    """Place `predator` (and optionally one prey and/or one fruit) on `pos`, and move every
    other prey and fruit far away so nothing else can engage the predator this step."""
    env.agent_positions[predator] = pos
    env.predator_positions[predator] = pos
    far = 0
    for other in [a for a in env.agents if a.startswith("prey")]:
        target = pos if other == prey else (9, far % 3)
        far += 1
        env.agent_positions[other] = target
        env.prey_positions[other] = target
    far = 0
    for f in list(env.fruit_positions):
        target = pos if f == fruit else (0, 5 + far % 4)
        far += 1
        env.fruit_positions[f] = target
    assert sum(1 for p in env.fruit_positions.values() if p == pos) == (1 if fruit else 0)


def _per_energy_env(k, **overrides):
    cfg = {
        "reward_predator_per_energy": k,
        "reward_predator_catch_prey": 0.0,
        "reward_predator_gather_fruit": 0.0,
        "male_gift_donation_rate": 0.0,
        "parent_offspring_share_rate": 0.0,
    }
    cfg.update(overrides)
    return _make_test_env(overrides=cfg)


def test_per_energy_reward_defaults_to_off():
    env = _make_test_env()
    assert env.reward_predator_per_energy == 0.0


def test_per_energy_reward_paid_for_fruit_in_proportion_to_energy_gained():
    env = _per_energy_env(0.2)
    male = next(a for a in env.agents if "predator_male" in a)
    fruit = next(iter(env.fruit_positions))
    _isolate_predator_on_cell(env, male, (5, 5), fruit=fruit)
    env.fruit_energies[fruit] = 2.0
    energy_before = env.agent_energies[male]

    _, rewards, _, _, _ = env.step(_noop_actions(env))

    gain = env.agent_energies[male] - (energy_before - env.homeostatic_energy_cost_per_step_predator)
    assert gain == pytest.approx(2.0)
    assert rewards[male] == pytest.approx(0.2 * 2.0)


def test_per_energy_reward_pays_little_for_a_nearly_empty_fruit():
    """The point of the change: a barely regrown fruit pays far less than a full one."""
    env = _per_energy_env(0.2)
    male = next(a for a in env.agents if "predator_male" in a)
    fruit = next(iter(env.fruit_positions))
    _isolate_predator_on_cell(env, male, (5, 5), fruit=fruit)
    env.fruit_energies[fruit] = 0.0  # just eaten; regrows by energy_gain_per_step_fruit before the check

    _, rewards, _, _, _ = env.step(_noop_actions(env))

    assert rewards[male] == pytest.approx(0.2 * env.energy_gain_per_step_fruit)
    assert rewards[male] < 0.05


@pytest.mark.parametrize("sex", ["predator_male", "predator_female"])
def test_per_energy_reward_paid_for_prey_in_proportion_to_energy_gained(sex):
    env = _per_energy_env(0.2)
    predator = next(a for a in env.agents if sex in a)
    prey = next(a for a in env.agents if a.startswith("prey"))
    _isolate_predator_on_cell(env, predator, (5, 5), prey=prey)
    energy_before = env.agent_energies[predator]

    _force_next_random(env, 0.0)  # success band
    _, rewards, _, _, _ = env.step(_noop_actions(env))

    assert prey not in env.agent_positions
    gain = env.agent_energies[predator] - (energy_before - env.homeostatic_energy_cost_per_step_predator)
    assert gain > 0.0
    assert rewards[predator] == pytest.approx(0.2 * gain)


def test_per_energy_reward_is_added_to_the_flat_per_event_rewards():
    env = _per_energy_env(0.2, reward_predator_catch_prey=1.0, reward_predator_gather_fruit=0.5)
    male = next(a for a in env.agents if "predator_male" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))
    fruit = next(iter(env.fruit_positions))
    _isolate_predator_on_cell(env, male, (5, 5), prey=prey, fruit=fruit)
    env.fruit_energies[fruit] = 2.0
    energy_before = env.agent_energies[male]

    _force_next_random(env, 0.0)
    _, rewards, _, _, _ = env.step(_noop_actions(env))

    total_gain = env.agent_energies[male] - (energy_before - env.homeostatic_energy_cost_per_step_predator)
    assert rewards[male] == pytest.approx(1.0 + 0.5 + 0.2 * total_gain)


def test_flat_rewards_unchanged_when_per_energy_is_zero():
    env = _per_energy_env(0.0, reward_predator_catch_prey=1.0, reward_predator_gather_fruit=0.5)
    male = next(a for a in env.agents if "predator_male" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))
    fruit = next(iter(env.fruit_positions))
    _isolate_predator_on_cell(env, male, (5, 5), prey=prey, fruit=fruit)
    env.fruit_energies[fruit] = 2.0

    _force_next_random(env, 0.0)
    _, rewards, _, _, _ = env.step(_noop_actions(env))

    assert rewards[male] == pytest.approx(1.5)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_per_energy_reward_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="reward_predator_per_energy"):
        _make_test_env(overrides={"reward_predator_per_energy": bad})


def test_hunting_a_prey_that_starved_this_step_gives_no_negative_energy_or_reward():
    """A prey at energy <= 0 after Step 1 can still be caught by a predator processed before it in
    Step 3; the catch must be worth 0 energy and 0 proportional reward, never negative."""
    env = _per_energy_env(0.2)
    male = next(a for a in env.agents if "predator_male" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))
    _isolate_predator_on_cell(env, male, (5, 5), prey=prey)
    env.agent_energies[prey] = 0.0  # goes negative after the homeostatic deduction
    energy_before = env.agent_energies[male]

    _force_next_random(env, 0.0)
    _, rewards, _, _, _ = env.step(_noop_actions(env))

    assert env.hunting_successes["predator_male"] == 1  # the catch really happened
    assert env.agent_energies[male] == pytest.approx(energy_before - env.homeostatic_energy_cost_per_step_predator)
    assert rewards[male] == pytest.approx(0.0)


@pytest.mark.parametrize("sex", ["predator_male", "predator_female"])
def test_per_energy_reward_fruit_by_sex_and_cumulative_rewards(sex):
    env = _per_energy_env(0.2)
    predator = next(a for a in env.agents if sex in a)
    fruit = next(iter(env.fruit_positions))
    _isolate_predator_on_cell(env, predator, (5, 5), fruit=fruit)
    env.fruit_energies[fruit] = 2.0

    _, rewards, _, _, _ = env.step(_noop_actions(env))

    assert rewards[predator] == pytest.approx(0.4)
    assert env.cumulative_rewards[predator] == pytest.approx(0.4)  # paid exactly once


def test_per_energy_reward_cumulative_after_successful_hunt():
    env = _per_energy_env(0.2)
    male = next(a for a in env.agents if "predator_male" in a)
    prey = next(a for a in env.agents if a.startswith("prey"))
    _isolate_predator_on_cell(env, male, (5, 5), prey=prey)
    energy_before = env.agent_energies[male]

    _force_next_random(env, 0.0)
    _, rewards, _, _, _ = env.step(_noop_actions(env))

    gain = env.agent_energies[male] - (energy_before - env.homeostatic_energy_cost_per_step_predator)
    assert env.cumulative_rewards[male] == pytest.approx(0.2 * gain)


def test_per_energy_reward_not_paid_on_failed_hunt_or_combat_death():
    # failure band: success_prob <= roll < success_prob + death_prob is death; above that is failure
    for roll_offset, expect_dead in ((0.5, False), (None, True)):
        env = _per_energy_env(0.2, penalty_predator_death_in_combat=-0.3)
        female = next(a for a in env.agents if "predator_female" in a)
        prey = next(a for a in env.agents if a.startswith("prey"))
        _isolate_predator_on_cell(env, female, (5, 5), prey=prey)
        if expect_dead:
            _force_next_random(env, env.prey_vs_predator_female_success_prob + 1e-6)  # death band
        else:
            _force_next_random(env, env.prey_vs_predator_female_success_prob + env.prey_vs_predator_female_death_prob + roll_offset * 0.1)  # failure band
        _, rewards, terms, _, _ = env.step(_noop_actions(env))

        assert prey in env.agent_positions  # prey untouched in both cases
        if expect_dead:
            assert terms.get(female) is True
            assert rewards[female] == pytest.approx(-0.3)
            assert env.cumulative_rewards[female] == pytest.approx(-0.3)
        else:
            assert rewards[female] == pytest.approx(0.0)
            assert env.cumulative_rewards[female] == pytest.approx(0.0)
