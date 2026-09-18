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
    from both engagements in the same step" rather than a flaky one."""
    env = _make_test_env()
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
    outcome, _ = env._resolve_hunting_attempt(predator_id, pos, prey_id, success_prob, death_prob, {}, {}, {}, {})
    assert outcome == "success"
    assert prey_id not in env.agent_positions

    predator_id, prey_id, pos = make_fake_pair("death")
    _force_next_random(env, 0.6)
    outcome, _ = env._resolve_hunting_attempt(predator_id, pos, prey_id, success_prob, death_prob, {}, {}, {}, {})
    assert outcome == "predator_dies"
    assert predator_id not in env.agent_positions
    assert prey_id in env.agent_positions  # untouched

    predator_id, prey_id, pos = make_fake_pair("failure")
    _force_next_random(env, 0.9)
    outcome, _ = env._resolve_hunting_attempt(predator_id, pos, prey_id, success_prob, death_prob, {}, {}, {}, {})
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
