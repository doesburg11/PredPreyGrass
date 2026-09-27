"""Tests for band_ostracism (default off): a predator that repeatedly counts as a defender (within threat_defense_radius of
an attacked band-mate) without ever being genuinely exposed (within distance 1 of the threat itself) is ostracized --
excluded from counting as anyone's defender and from receiving band shares. See predpreygrass_rllib_env.py's __init__ and
_update_reputation for the full rationale."""
import copy

import pytest

from predpreygrass.non_evolutionary.predator_bands.config_env import config_env as _base
from predpreygrass.non_evolutionary.predator_bands.predpreygrass_rllib_env import PredPreyGrass


class _FixedRandom:
    """Real generator with random() forced to a value (integers/choice/etc. stay real)."""

    def __init__(self, real, value):
        self._real, self._value = real, value

    def random(self, *a, **k):
        return self._value

    def __getattr__(self, name):
        return getattr(self._real, name)


def _members(env, band):
    return [a for a, b in env.agent_band.items() if b == band]


def _place(env, agent, pos):
    env.grid_world_state[1, *env.agent_positions[agent]] = 0
    env.agent_positions[agent] = pos
    env.predator_positions[agent] = pos
    env.grid_world_state[1, *pos] = env.agent_energies[agent]


def _park_others(env, keep, start=(24, 0)):
    i = 0
    for a in list(env.predator_positions):
        if a not in keep:
            _place(env, a, (24 - i // 8, 16 + i % 8))
            i += 1


def _env(**kw):
    c = copy.deepcopy(_base)
    c.update({"max_steps": 200, "diet_required": False, "num_threats": 1, "band_ostracism": True, "ostracism_min_opportunities": 3})
    c.update(kw)
    env = PredPreyGrass(c)
    env.reset(seed=1)
    return env


def _fattest_band(env):
    return max(range(env.num_bands), key=lambda b: len(_members(env, b)))


def _free_rider_setup(**kw):
    """target at (10, 10), adjacent to a threat at (9, 10); free_rider at (10, 12): within threat_defense_radius (2) of
    target but distance 3 from the threat itself -- always credited as a defender, never genuinely at risk. Default
    threat_defenders_to_repel (3) is never reached by a single defender, so every call resolves as a kill attempt."""
    env = _env(**kw)
    band = sorted(_members(env, _fattest_band(env)))
    target, free_rider = band[0], band[1]
    _park_others(env, {target, free_rider})
    _place(env, target, (10, 10))
    _place(env, free_rider, (10, 12))
    env.threat_positions = {"threat_0": (9, 10)}
    for a in env.predator_positions:
        env.agent_energies[a] = 5.0
    env.rng = _FixedRandom(env.rng, 0.99)  # kill prob 0.5 * (1 - 1/3) = 0.333: never kills, target survives every call
    return env, target, free_rider


def test_a_pure_free_rider_is_ostracized_after_enough_opportunities():
    env, target, free_rider = _free_rider_setup(ostracism_min_opportunities=3)
    for _ in range(2):
        env._threats_act()
        assert not env._is_ostracized(free_rider)
    assert env.defense_opportunities[free_rider] == 2 and env.defense_exposures.get(free_rider, 0) == 0
    env._threats_act()  # 3rd opportunity, 0 exposures: ratio 0.0 < 0.34
    assert env._is_ostracized(free_rider)
    assert env.ostracized_until[free_rider] == env.current_step + env.ostracism_duration
    assert env.ostracism_events == 1
    # counts reset so judgment restarts fresh once the ostracism ends
    assert env.defense_opportunities[free_rider] == 0 and env.defense_exposures[free_rider] == 0


def test_ostracized_predator_no_longer_counts_as_a_defender():
    env, target, free_rider = _free_rider_setup(ostracism_min_opportunities=3)
    for _ in range(3):
        env._threats_act()
    assert env._is_ostracized(free_rider)
    assert env._threat_defenders(target) == 0
    assert free_rider not in env._defender_ids(target)


def test_ostracized_predator_receives_no_band_share():
    env, target, free_rider = _free_rider_setup(ostracism_min_opportunities=3, band_meat_share_rate=0.5)
    for _ in range(3):
        env._threats_act()
    assert env._is_ostracized(free_rider)
    e_before = env.agent_energies[free_rider]
    env._apply_band_share(target, gained=4.0, is_fruit=False)
    assert env.agent_energies[free_rider] == e_before  # excluded despite being in range


def test_a_genuinely_exposed_defender_is_never_ostracized():
    # Same setup, but the "free rider" instead sits adjacent to the threat too (distance 1): fully exposed every time.
    env, target, exposed = _free_rider_setup(ostracism_min_opportunities=3, threat_attack_all_adjacent=True)
    _place(env, exposed, (10, 9))  # distance 1 from the threat at (9, 10)
    for _ in range(6):
        env._threats_act()
        if env.agent_energies[target] <= 0 or env.agent_energies[exposed] <= 0:
            break  # a kill roll could in principle land now that `exposed` is itself attacked too; not the point of this test
    assert not env._is_ostracized(exposed)


def test_ostracism_is_off_by_default_even_with_a_pure_free_rider():
    env, target, free_rider = _free_rider_setup(band_ostracism=False)
    for _ in range(10):
        env._threats_act()
    assert not env._is_ostracized(free_rider)
    assert env.ostracism_events == 0
    assert env._threat_defenders(target) == 1  # still counted: nothing excludes it


def test_snapshot_round_trips_ostracism_state():
    env, target, free_rider = _free_rider_setup(ostracism_min_opportunities=3)
    for _ in range(3):
        env._threats_act()
    assert env._is_ostracized(free_rider)
    snap = env.get_state_snapshot()
    env.ostracized_until, env.defense_opportunities, env.defense_exposures, env.ostracism_events = {}, {}, {}, 0
    env.restore_state_snapshot(snap)
    assert env._is_ostracized(free_rider)
    assert env.ostracism_events == 1


def test_newly_ostracized_predator_still_defends_a_later_target_in_the_same_phase():
    # Codex-caught bug: ostracism must be frozen at the start of the phase (like `viable`), so a predator who crosses the
    # threshold defending target_a still counts as target_b's defender in the SAME _threats_act() call, matching the
    # function's own documented "no order dependence" guarantee. target_a and target_b are both adjacent to the threat and
    # (being within threat_defense_radius of each other) already mutual defenders; free_rider is in range of both, so it
    # gets 2 opportunities per call (one per target) -- 2 after the 1st call, crossing the threshold of 3 partway through
    # the 2nd (while resolving target_a, processed first).
    env = _env(ostracism_min_opportunities=3, threat_attack_all_adjacent=True)
    band = sorted(_members(env, _fattest_band(env)))
    target_a, target_b, free_rider = band[0], band[1], band[2]
    _park_others(env, {target_a, target_b, free_rider})
    _place(env, target_a, (10, 10))
    _place(env, target_b, (10, 11))  # adjacent to target_a and to the threat: mutual defender for both, every call
    _place(env, free_rider, (10, 12))  # within radius 2 of both targets, distance 2 from the threat: never exposed
    env.threat_positions = {"threat_0": (9, 10)}
    for a in env.predator_positions:
        env.agent_energies[a] = 5.0
    env.rng = _FixedRandom(env.rng, 0.99)  # never kills, whatever the defender count
    env._threats_act()
    assert env.defense_opportunities[free_rider] == 2 and not env._is_ostracized(free_rider)
    env._threats_act()  # target_a (processed first) pushes free_rider's count to 3: ostracized mid-phase
    assert env._is_ostracized(free_rider)
    assert env.ostracism_events == 1
    # target_b (processed 2nd, same phase) must still have counted free_rider as a defender when IT was resolved: its own
    # _update_reputation call increments free_rider's (just-reset-to-0) opportunity count to 1. Without the phase-start
    # freeze, target_b's _defender_ids would have already excluded the now-ostracized free_rider, leaving this at 0.
    assert env.defense_opportunities.get(free_rider, 0) == 1
    assert env.threat_encounters == 4  # 2 targets x 2 calls: target_b's own attack was never skipped


def test_ostracism_expires_and_a_repeat_offender_is_ostracized_again():
    env, target, free_rider = _free_rider_setup(ostracism_min_opportunities=3, ostracism_duration=50)
    for _ in range(3):
        env._threats_act()
    assert env._is_ostracized(free_rider)
    until = env.ostracized_until[free_rider]
    env.current_step = until - 1
    assert env._is_ostracized(free_rider)
    env.current_step = until
    assert not env._is_ostracized(free_rider)  # expired exactly at the boundary
    for _ in range(3):
        env._threats_act()  # still a pure free rider: re-offends and is judged again from a clean slate
    assert env._is_ostracized(free_rider)
    assert env.ostracism_events == 2


def test_ostracism_duration_zero_records_the_event_but_never_actually_excludes():
    env, target, free_rider = _free_rider_setup(ostracism_min_opportunities=3, ostracism_duration=0)
    for _ in range(3):
        env._threats_act()
    assert env.ostracism_events == 1
    assert not env._is_ostracized(free_rider)  # current_step < current_step + 0 is false: never excluded


def test_invalid_ostracism_settings_are_rejected():
    with pytest.raises(ValueError):
        _env(ostracism_min_opportunities=0)
    with pytest.raises(ValueError):
        _env(ostracism_exposure_threshold=1.5)
    with pytest.raises(ValueError):
        _env(ostracism_exposure_threshold=-0.1)
    with pytest.raises(ValueError):
        _env(ostracism_duration=-1)
