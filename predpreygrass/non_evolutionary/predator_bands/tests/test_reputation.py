"""Tests for band_reputation (default off): a continuous, per-predator reputation score in [0, 1] that scales (rather than
switches) how much a predator's presence counts toward repelling a threat / lowering the kill probability, and how much of a
band-mate's shared forage it receives -- an image-scoring-style free-rider punishment. Supersedes an earlier binary
"ostracism" design from the same session (never used in a run). See predpreygrass_rllib_env.py's __init__, _update_reputation
and _defender_weight for the full rationale."""
import copy

import numpy as np
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
    c.update({"max_steps": 200, "diet_required": False, "num_threats": 1, "band_reputation": True})
    c.update(kw)
    env = PredPreyGrass(c)
    env.reset(seed=1)
    return env


def _fattest_band(env):
    return max(range(env.num_bands), key=lambda b: len(_members(env, b)))


def _free_rider_setup(**kw):
    """target at (10, 10), adjacent to a threat at (9, 10); free_rider at (10, 12): within threat_defense_radius (2) of
    target but distance 3 from the threat itself -- always credited as a defender, never genuinely at risk. Default
    threat_defenders_to_repel (3) is never reached by a single defender at full reputation, so every call resolves as a
    kill attempt."""
    env = _env(**kw)
    band = sorted(_members(env, _fattest_band(env)))
    target, free_rider = band[0], band[1]
    _park_others(env, {target, free_rider})
    _place(env, target, (10, 10))
    _place(env, free_rider, (10, 12))
    env.threat_positions = {"threat_0": (9, 10)}
    for a in env.predator_positions:
        env.agent_energies[a] = 5.0
    env.rng = _FixedRandom(env.rng, 0.99)  # kill prob at reputation 1.0 is 0.5*(1-1/3)=0.333: never kills initially
    return env, target, free_rider


def test_default_reputation_is_full_and_off_is_a_true_no_op():
    env, target, free_rider = _free_rider_setup(band_reputation=False)
    assert env._reputation_weight(free_rider) == 1.0
    for _ in range(10):
        env._threats_act()
    assert env._reputation_weight(free_rider) == 1.0
    assert free_rider not in env.reputation  # never even tracked when the feature is off
    assert env._threat_defenders(target) == 1


def test_a_pure_free_rider_loses_reputation_by_ema_and_a_genuine_defender_keeps_it():
    env, target, free_rider = _free_rider_setup(reputation_ema_alpha=0.1)
    assert env._reputation_weight(free_rider) == 1.0  # unseen: full benefit of the doubt
    prev = 1.0
    for _ in range(5):
        env._threats_act()
        cur = env._reputation_weight(free_rider)
        assert cur == pytest.approx((1 - 0.1) * prev)  # EMA toward 0.0: never exposed
        assert 0.0 < cur < prev
        prev = cur


def test_reputation_scales_the_defender_weight_not_a_hard_cutoff():
    env, target, free_rider = _free_rider_setup(reputation_ema_alpha=0.2)
    for _ in range(20):  # drive reputation well down, but an EMA never reaches exactly 0
        env._threats_act()
    rep = env._reputation_weight(free_rider)
    assert 0.0 < rep < 0.05
    # still a real (if tiny) contribution: not excluded, just diluted
    assert env._defender_weight(env._defender_ids(target)) == pytest.approx(rep)
    assert env._threat_defenders(target) == 1  # the plain headcount is unaffected by reputation


def test_reputation_recovers_when_a_predator_becomes_genuinely_exposed():
    env, target, free_rider = _free_rider_setup(reputation_ema_alpha=0.2)
    for _ in range(5):
        env._threats_act()
    low = env._reputation_weight(free_rider)
    assert low < 1.0
    _place(env, free_rider, (10, 9))  # now distance 1 from the threat: genuinely exposed every call
    for _ in range(10):
        env._threats_act()
    assert env._reputation_weight(free_rider) > low


def test_band_share_is_scaled_by_reputation_not_all_or_nothing():
    env, target, free_rider = _free_rider_setup(reputation_ema_alpha=0.2, band_meat_share_rate=0.5)
    for _ in range(10):
        env._threats_act()
    rep = env._reputation_weight(free_rider)
    assert 0.0 < rep < 1.0
    e_before = env.agent_energies[free_rider]
    env._apply_band_share(target, gained=4.0, is_fruit=False)
    received = env.agent_energies[free_rider] - e_before
    assert received == pytest.approx(0.5 * 4.0 * rep)  # the full share, scaled down by reputation -- not zeroed


def test_no_order_dependence_an_updated_reputation_does_not_change_a_later_resolved_targets_repel_decision():
    # Discriminating regression test (the earlier version of this test could not tell a frozen implementation from a live
    # one -- see the Codex review this session). Geometry: threat at (9, 10); target_a=(9, 9) and target_b=(9, 11), both
    # adjacent to the threat and (distance 2 apart, threat_defense_radius=2) mutual defenders of each other. defender_x=
    # (11, 10) is within defense radius of BOTH targets but distance 2 from the threat itself (never exposed). helper_b=
    # (9, 12) is within defense radius of target_b only (defends only it, so its own reputation update -- also toward
    # 0.0, since it is never exposed either -- only ever happens using target_b's already-resolved, frozen weight; it
    # stays at 1.0 for the whole of this test's assertions, which only look at what happened up to and including
    # target_b's resolution).
    #   target_a's defenders (processed first): target_b (1.0) + defender_x (1.0, phase-start) = 2.0 < repel threshold
    #   (3) -> not repelled -> proceeds to a kill roll, whose _update_reputation call sees defender_x NOT exposed and
    #   (alpha=1.0) sets its LIVE reputation to exactly 0.0.
    #   target_b's defenders (processed second): target_a (1.0) + defender_x + helper_b (1.0). Using the correct,
    #   phase-start value for defender_x (1.0) totals 3.0 -- reaches the repel threshold, so target_b is repelled and
    #   SURVIVES. Using the (buggy) live value (0.0) totals only 2.0 -- misses the threshold, falls through to a kill
    #   roll instead, and with rng fixed at 0.0 target_b would be killed. The two implementations disagree on whether
    #   target_b lives, which is exactly what this test checks.
    env = _env(reputation_ema_alpha=1.0, threat_attack_all_adjacent=True)
    band = sorted(_members(env, _fattest_band(env)))
    target_a, target_b, defender_x, helper_b = band[0], band[1], band[2], band[3]
    _park_others(env, {target_a, target_b, defender_x, helper_b})
    _place(env, target_a, (9, 9))
    _place(env, target_b, (9, 11))
    _place(env, defender_x, (11, 10))
    _place(env, helper_b, (9, 12))
    env.threat_positions = {"threat_0": (9, 10)}
    for a in env.predator_positions:
        env.agent_energies[a] = 5.0
    assert env._defender_weight(env._defender_ids(target_a)) == 2.0
    assert env._defender_weight(env._defender_ids(target_b)) == 3.0
    env.rng = _FixedRandom(env.rng, 0.0)  # guarantees a kill whenever a kill roll is actually reached
    env._threats_act()
    assert env.reputation[defender_x] == 0.0  # dragged down by target_a's (correctly non-order-affecting) processing
    assert env.threat_repelled == 1
    assert env.agent_energies[target_b] == 5.0  # repelled using the frozen (3.0), not the live (2.0), weight
    assert env.threat_encounters == 2


def test_reputation_is_dropped_on_removal():
    env, target, free_rider = _free_rider_setup(reputation_ema_alpha=0.2)
    for _ in range(5):
        env._threats_act()
    assert free_rider in env.reputation
    env._remove_agent(free_rider)
    assert free_rider not in env.reputation


def test_snapshot_round_trips_reputation():
    env, target, free_rider = _free_rider_setup(reputation_ema_alpha=0.2)
    for _ in range(5):
        env._threats_act()
    rep = env.reputation[free_rider]
    snap = env.get_state_snapshot()
    env.reputation = {}
    env.restore_state_snapshot(snap)
    assert env.reputation[free_rider] == rep


def test_metrics_average_reputation_over_the_whole_population_not_just_observed_predators():
    env, target, free_rider = _free_rider_setup(reputation_ema_alpha=0.2)
    for _ in range(10):
        env._threats_act()
    rep = env.reputation[free_rider]
    assert 0.0 < rep < 1.0
    n = len(env.predator_positions)
    metrics = env._build_episode_training_metrics()
    # only free_rider has been observed; everyone else defaults to 1.0 and must still count toward the average
    expected_mean = (rep + (n - 1) * 1.0) / n
    assert metrics["mean_reputation"] == pytest.approx(expected_mean)
    assert metrics["min_reputation"] == pytest.approx(rep)


def test_invalid_reputation_ema_alpha_is_rejected():
    with pytest.raises(ValueError):
        _env(reputation_ema_alpha=0.0)
    with pytest.raises(ValueError):
        _env(reputation_ema_alpha=1.5)
    with pytest.raises(ValueError):
        _env(reputation_ema_alpha=-0.1)


# ---- observation channel: reputation is now perceivable, not just mechanical --------------------------------------------
def test_band_reputation_is_off_by_default_and_adds_one_channel_when_on():
    c = copy.deepcopy(_base)
    env = PredPreyGrass(c)
    env.reset(seed=1)
    assert env.band_reputation is False and env.num_obs_channels == 8
    assert _env(num_threats=0).num_obs_channels == 9  # band_reputation=True via _env(); no threats/mammoths/compass
    assert _env().num_obs_channels == 10  # _env()'s own default num_threats=1: +1 threat, +1 reputation
    assert _env(band_compass=True).num_obs_channels == 8 + 4 + 1 + 1  # compass + threat (num_threats=1) + reputation


def test_reputation_channel_shows_every_visible_predators_own_weight_including_the_observer():
    env, target, free_rider = _free_rider_setup(reputation_ema_alpha=0.2)
    for _ in range(10):
        env._threats_act()
    rep = env.reputation[free_rider]
    assert 0.0 < rep < 1.0
    obs = env._get_observation(target)
    off = (env.predator_obs_range - 1) // 2
    ch = env.num_obs_channels - 1  # last channel: _free_rider_setup uses num_threats=1 and no compass/mammoths, so
    # this is 8 (base) + 0 (compass) + 1 (threat) + 0 (mammoths) = channel 9, the reputation channel
    tx, ty = env.agent_positions[target]
    fx, fy = env.agent_positions[free_rider]
    assert obs[ch, off, off] == pytest.approx(1.0)  # the observer itself: never touched, still at 1.0
    assert obs[ch, off + (fx - tx), off + (fy - ty)] == pytest.approx(rep)


def test_reputation_channel_is_all_zero_when_band_reputation_is_off():
    env_off = _env(band_reputation=False)
    env_on = _env(band_reputation=True)
    assert env_on.num_obs_channels == env_off.num_obs_channels + 1  # exactly one extra channel, nothing else changes
    a = _members(env_off, _fattest_band(env_off))[0]
    obs_off = env_off._get_observation(a)
    assert obs_off.shape[0] == env_off.num_obs_channels
    obs_on = env_on._get_observation(a)  # same seed/layout (both from _env(), same reset(seed=1)): only channel 9 differs
    assert np.array_equal(obs_off, obs_on[: env_off.num_obs_channels])
    assert not (obs_on[env_off.num_obs_channels] == 0.0).all()  # the extra channel is genuinely populated (own weight, >=1 predator)


def test_reputation_channel_is_never_populated_for_prey():
    env = _env()
    prey = next(iter(env.prey_positions))  # scripted prey: not in env.agents (not policy-controlled)
    obs = env._get_observation(prey)
    assert obs.shape[0] == env.num_obs_channels
    ch = env.num_obs_channels - 1
    assert (obs[ch] == 0.0).all()
