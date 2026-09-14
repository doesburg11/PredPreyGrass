"""Direct tests of Trial13Driver._select_predator_action: lazy registration of
new predator ids, and that the reward used to update the shared policy is the
real net energy change from the PREVIOUS step's action relative to the
BASELINE of doing nothing (energy_change + energy_loss_per_step_predator, not
raw energy change -- see driver.py's _select_predator_action docstring for why
the baseline subtraction matters), further corrected for reproduction's own
energy cost when a predator reproduced on that step."""

import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.driver import PredatorTrainingState, Trial13Driver

ENERGY_LOSS_PER_STEP_PREDATOR = 0.15
INITIAL_ENERGY_PREDATOR = 5.0
REPRODUCTION_REWARD_PREDATOR = 10.0


class _FakeEnv:
    def __init__(self, energies, obs_range=7):
        self.agent_energies = dict(energies)
        self.predator_creation_energy_threshold = 12.0
        self.energy_loss_per_step_predator = ENERGY_LOSS_PER_STEP_PREDATOR
        self.initial_energy_predator = INITIAL_ENERGY_PREDATOR
        self.reproduction_reward_predator = REPRODUCTION_REWARD_PREDATOR
        self._obs_range = obs_range

    def _get_observation(self, agent_id):
        return np.zeros((4, self._obs_range, self._obs_range))


class _SpyPredatorPolicy:
    """Records every act()/update() call instead of doing real REINFORCE math,
    so a test can assert exactly what the driver fed it."""

    def __init__(self):
        self.act_calls = []
        self.update_calls = []
        self.next_action = 4  # arbitrary fixed action to return from act()

    def act(self, features, rng):
        self.act_calls.append(features.copy())
        return self.next_action

    def update(self, features, action, reinforcement):
        self.update_calls.append((features.copy(), action, reinforcement))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_new_predator_is_registered_lazily_on_first_call(rng):
    env = _FakeEnv({"predator_0": 5.0})
    policy = _SpyPredatorPolicy()
    driver = Trial13Driver(env=env, predator_policy=policy, cfg={}, rng=rng)

    assert "predator_0" not in driver.predator_registry
    driver._select_predator_action("predator_0")
    assert "predator_0" in driver.predator_registry


def test_no_update_on_the_very_first_call(rng):
    """A predator's first-ever action has no PREVIOUS transition to attribute a
    reward to -- update() must not fire yet."""
    env = _FakeEnv({"predator_0": 5.0})
    policy = _SpyPredatorPolicy()
    driver = Trial13Driver(env=env, predator_policy=policy, cfg={}, rng=rng)

    driver._select_predator_action("predator_0")
    assert policy.update_calls == []
    assert len(policy.act_calls) == 1


def test_second_call_updates_using_baseline_subtracted_energy_change(rng):
    env = _FakeEnv({"predator_0": 5.0})
    policy = _SpyPredatorPolicy()
    driver = Trial13Driver(env=env, predator_policy=policy, cfg={}, rng=rng)

    driver._select_predator_action("predator_0")  # energy=5.0, no update yet
    env.agent_energies["predator_0"] = 8.5  # simulates a catch happening in between
    driver._select_predator_action("predator_0")

    assert len(policy.update_calls) == 1
    features_used, action_used, reinforcement = policy.update_calls[0]
    # raw energy change (3.5) PLUS the ambient per-step drain baseline (0.15)
    assert reinforcement == pytest.approx(3.5 + ENERGY_LOSS_PER_STEP_PREDATOR)
    assert action_used == policy.next_action


def test_ordinary_no_catch_step_nets_to_near_zero_reinforcement(rng):
    """The whole point of the baseline subtraction: a step where a predator
    just loses the normal ambient energy (no catch) should net to ~0
    reinforcement, not the raw (systematically negative) energy delta."""
    env = _FakeEnv({"predator_0": 5.0})
    policy = _SpyPredatorPolicy()
    driver = Trial13Driver(env=env, predator_policy=policy, cfg={}, rng=rng)

    driver._select_predator_action("predator_0")  # energy=5.0
    env.agent_energies["predator_0"] = 5.0 - ENERGY_LOSS_PER_STEP_PREDATOR  # ordinary drain, no catch
    driver._select_predator_action("predator_0")

    assert len(policy.update_calls) == 1
    _, _, reinforcement = policy.update_calls[0]
    assert reinforcement == pytest.approx(0.0, abs=1e-9)


def test_reproduction_cost_is_excluded_from_reinforcement(rng):
    """A predator that reproduces loses initial_energy_predator on top of the
    ambient drain -- that cost must NOT be blamed on the action taken that
    step (reproduction reflects past hunting success, not a failure of this
    step's action). driver.step() sets _predators_reproduced_last_step from
    the env's own reproduction_reward_predator signal; simulate that directly
    since this test calls _select_predator_action without going through
    step()."""
    env = _FakeEnv({"predator_0": 5.0})
    policy = _SpyPredatorPolicy()
    driver = Trial13Driver(env=env, predator_policy=policy, cfg={}, rng=rng)

    driver._select_predator_action("predator_0")  # energy=5.0
    # Simulate: predator caught nothing net-new, but reproduced -- energy drops
    # by ambient drain (0.15) AND the reproduction cost (5.0).
    env.agent_energies["predator_0"] = 5.0 - ENERGY_LOSS_PER_STEP_PREDATOR - INITIAL_ENERGY_PREDATOR
    driver._predators_reproduced_last_step = {"predator_0"}
    driver._select_predator_action("predator_0")

    assert len(policy.update_calls) == 1
    _, _, reinforcement = policy.update_calls[0]
    # Without the correction this would be a large NEGATIVE number (~-5.0);
    # with it, the reproduction cost is excluded and it nets to ~0, same as an
    # ordinary no-catch step.
    assert reinforcement == pytest.approx(0.0, abs=1e-9)


def test_reproduction_correction_only_applies_to_the_reproducing_agent(rng):
    """predator_0 reproduces (large raw energy drop, should be corrected to
    ~0); predator_1 catches prey instead (large raw energy GAIN, should stay a
    clear positive signal, untouched by the correction) -- distinguishes the
    two rather than both coincidentally netting to the same value."""
    env = _FakeEnv({"predator_0": 5.0, "predator_1": 5.0})
    policy = _SpyPredatorPolicy()
    driver = Trial13Driver(env=env, predator_policy=policy, cfg={}, rng=rng)

    driver._select_predator_action("predator_0")
    driver._select_predator_action("predator_1")
    env.agent_energies["predator_0"] = 5.0 - ENERGY_LOSS_PER_STEP_PREDATOR - INITIAL_ENERGY_PREDATOR
    env.agent_energies["predator_1"] = 5.0 + 4.0  # caught prey worth 4.0 energy
    driver._predators_reproduced_last_step = {"predator_0"}  # only predator_0 reproduced

    driver._select_predator_action("predator_0")
    driver._select_predator_action("predator_1")

    calls_by_order = policy.update_calls
    assert calls_by_order[0][2] == pytest.approx(0.0, abs=1e-9)  # predator_0: reproduction cost excluded
    assert calls_by_order[1][2] == pytest.approx(4.0 + ENERGY_LOSS_PER_STEP_PREDATOR)  # predator_1: real catch, untouched


def test_predator_state_tracks_previous_transition(rng):
    env = _FakeEnv({"predator_0": 5.0})
    policy = _SpyPredatorPolicy()
    driver = Trial13Driver(env=env, predator_policy=policy, cfg={}, rng=rng)

    driver._select_predator_action("predator_0")
    state = driver.predator_registry["predator_0"]
    assert isinstance(state, PredatorTrainingState)
    assert state.prev_energy == 5.0
    assert state.prev_action == policy.next_action
    assert state.prev_features is not None


def test_death_removes_predator_from_registry(rng):
    env = _FakeEnv({"predator_0": 5.0})
    policy = _SpyPredatorPolicy()
    driver = Trial13Driver(env=env, predator_policy=policy, cfg={}, rng=rng)
    driver._select_predator_action("predator_0")

    driver._handle_deaths({"predator_0": True})
    assert "predator_0" not in driver.predator_registry
