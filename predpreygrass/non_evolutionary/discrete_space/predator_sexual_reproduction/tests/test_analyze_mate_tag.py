"""Sign checks for analyze_mate_tag.py with synthetic policies (no checkpoints needed)."""
import copy

import numpy as np
import torch

from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction import analyze_mate_tag as amt
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.config_env import config_env as _base

MOVES = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)])


class TagFollower:
    """Puts its probability on moves toward the marked cell (sign +1) or away from it (sign -1); uniform if no mark."""

    def __init__(self, sign=1.0, strength=3.0):
        self.sign, self.strength = sign, strength

    def _forward_inference(self, batch):
        x = batch["obs"].numpy()
        logits = np.zeros((len(x), 9))
        for i, o in enumerate(x):
            if o.shape[0] > amt.MATE_CH:
                tag = np.argwhere(o[amt.MATE_CH] > 0)
                if len(tag):
                    c = (o.shape[1] - 1) // 2
                    d = np.sqrt((((c + MOVES) - tag[0]) ** 2).sum(1))
                    logits[i] = -self.sign * self.strength * d
        return {"action_dist_inputs": torch.as_tensor(logits, dtype=torch.float32)}


class Uniform:
    def _forward_inference(self, batch):
        return {"action_dist_inputs": torch.zeros((len(batch["obs"]), 9))}


def _config(mate_channel=True):
    c = copy.deepcopy(_base)
    c.update(
        {
            "grid_size": 10,
            "initial_num_grass": 15,
            "initial_num_fruit": 15,
            "n_initial_active_predator_male": 5,
            "n_initial_active_predator_female": 5,
            "n_initial_active_prey": 6,
            "predator_creation_energy_threshold": 5.5,
            "initial_energy_predator_male": 6.0,
            "initial_energy_predator_female": 6.0,
            "max_steps": 150,
            "mate_obs_channel": mate_channel,
        }
    )
    return c


def _modules(pred):
    return {"predator_male_policy": pred, "predator_female_policy": pred, "prey_policy": Uniform()}


def _bank(pred, mate_channel=True):
    bank, outcomes = amt.rollout(_config(mate_channel), _modules(pred), 3, 11, 1, 5000, mate_channel)
    return bank, outcomes


def test_follower_shows_positive_mate_and_decoy_effects():
    bank, _ = _bank(TagFollower(+1))
    assert len(bank) > 30
    eff = amt.tag_effects(bank, TagFollower(+1))
    assert np.nanmean(eff["mate"]) > 0.3
    assert np.nanmean(eff["decoy"]) > 0.3
    assert (eff["tv"] > 0).all()


def test_avoider_shows_negative_effects():
    bank, _ = _bank(TagFollower(+1))
    eff = amt.tag_effects(bank, TagFollower(-1))
    assert np.nanmean(eff["mate"]) < -0.3 and np.nanmean(eff["decoy"]) < -0.3


def test_tag_blind_policy_has_exactly_zero_effects():
    bank, _ = _bank(TagFollower(+1))
    eff = amt.tag_effects(bank, Uniform())
    for k in ("mate", "decoy", "food_prey", "food_fruit", "tv"):
        v = eff[k][~np.isnan(eff[k])]
        assert np.allclose(v, 0.0), k


def test_bank_tags_match_env_mates_and_decoys_exclude_mate():
    bank, _ = _bank(TagFollower(+1))
    for b in bank:
        assert b["obs"][amt.MATE_CH].sum() == 1.0
        tag = tuple(np.argwhere(b["obs"][amt.MATE_CH] > 0)[0])
        assert b["obs"][amt.PRED_CH][tag] > 0
        assert all(c != tag and c != (3, 3) and b["obs"][amt.PRED_CH][c] > 0 for c, _ in b["decoys"])


def test_follower_keeps_pairs_closer_than_uniform_in_part_b():
    _, out_f = _bank(TagFollower(+1))
    _, out_u = _bank(Uniform())
    near_f = np.nanmean([o["pair_near_frac"] for o in out_f])
    near_u = np.nanmean([o["pair_near_frac"] for o in out_u])
    assert near_f > near_u
    for o in out_f + out_u:
        assert o["births"] >= 0 and o["steps"] > 0
        assert np.isnan(o["repeat_birth_frac"]) or 0.0 <= o["repeat_birth_frac"] <= 1.0


def test_part_b_runs_on_a_five_channel_run():
    _, outcomes = amt.rollout(_config(False), _modules(Uniform()), 2, 3, 1, 100, False)
    assert len(outcomes) == 2 and all(o["steps"] > 0 for o in outcomes)


def test_part_b_samples_ten_action_predator_policies():
    """Part B must not use env.num_actions (nine movement actions) for a learned-gift policy."""
    c = _config(True)
    c["learned_gift"] = "on"
    c["male_gift_donation_rate"] = 0.0
    class TenActionUniform:
        def _forward_inference(self, batch):
            return {"action_dist_inputs": torch.zeros((len(batch["obs"]), 10))}
    bank, outcomes = amt.rollout(
        c,
        {"predator_male_policy": TenActionUniform(), "predator_female_policy": TenActionUniform(), "prey_policy": Uniform()},
        2, 3, 1, 100, False,
    )
    assert bank == [] and len(outcomes) == 2 and all(o["steps"] > 0 for o in outcomes)


class _FakeEnv:
    def __init__(self):
        self.predator_positions = {}
        self.agent_mate = {}
        self.agent_parents = {}
        self.mate_gift_events = 0


def test_pair_tracker_bookkeeping_on_a_known_sequence():
    env = _FakeEnv()
    t = amt.PairTracker(gift_range=3, window=3)
    m, f, m2 = "predator_male_0", "predator_female_0", "predator_male_1"
    env.predator_positions = {m: (0, 0), f: (0, 2), m2: (9, 9)}
    t.on_decision(env)  # step 0: no bonds yet
    # birth 1 by (m, f)
    env.agent_mate = {m: f, f: m}
    env.agent_parents = {"predator_male_2": (m, f)}
    env.predator_positions["predator_male_2"] = (1, 0)
    t.after_step(env)
    t.on_decision(env)  # step 1: pair at distance 2 -> near; post-birth window 1/3 near
    env.agent_parents["predator_female_1"] = (m, f)  # birth 2 by the same pair -> repeat
    env.predator_positions["predator_female_1"] = (1, 1)
    t.after_step(env)
    env.predator_positions[f] = (0, 6)  # distance 6 -> far
    t.on_decision(env)  # step 2: windows: w1 2/3 (far), w2 1/3 (far)
    del env.predator_positions[f]  # f dies; stale bond stays in agent_mate
    t.after_step(env)
    env.mate_gift_events = 4
    t.on_decision(env)  # step 3: w1 3/3 (dead -> not near), w2 2/3 (not near)
    t.after_step(env)
    t.on_decision(env)  # step 4: w2 3/3 (not near); w1 closed
    t.after_step(env)
    r = t.result()
    assert r["steps"] == 5 and r["births"] == 2
    assert r["repeat_birth_frac"] == 0.5
    # pred steps: 3 + 4 + 5 + 4 + 4 = 20; male steps: 2 + 3 + 3 + 3 + 3 = 14
    assert r["births_per_1000_pred_steps"] == 1000 * 2 / 20
    assert r["gifts_per_1000_male_steps"] == 1000 * 4 / 14
    # paired steps: step1 m,f near (2); step2 m,f far (2); steps 3-4: f dead -> 0
    assert r["paired_step_frac"] == 4 / 20
    assert r["pair_near_frac"] == 0.5
    assert r["pair_dist"] == (2 + 2 + 6 + 6) / 4
    # post-birth: both windows complete (steps 1-3 and 2-4), 6 window-steps, 1 near
    assert r["post_birth_near_frac"] == 1 / 6
    assert r["post_birth_windows"] == 2 and r["post_birth_windows_censored"] == 0
    assert r["mean_predators"] == 20 / 5


def test_pair_tracker_undefined_ratios_are_nan():
    env = _FakeEnv()
    t = amt.PairTracker(gift_range=3, window=5)
    env.predator_positions = {"predator_male_0": (0, 0)}
    t.on_decision(env)
    t.after_step(env)
    r = t.result()
    assert r["births"] == 0
    assert np.isnan(r["repeat_birth_frac"]) and np.isnan(r["post_birth_near_frac"]) and np.isnan(r["pair_near_frac"])


def test_episode_mean_and_pooled_estimands_differ_as_intended():
    rng = np.random.default_rng(0)
    # 5 episodes; episode 0 has 10 states of value 1, others one state of value 0
    values = np.array([1.0] * 10 + [0.0] * 4)
    episodes = np.array([0] * 10 + [1, 2, 3, 4])
    eps5 = np.concatenate([episodes, [5]])
    vals5 = np.concatenate([values, [0.0]])
    pooled = amt.cluster_mean(vals5, eps5, rng)
    per_ep = amt.cluster_mean(vals5, eps5, rng, per_episode=True)
    assert np.isclose(pooled[0], 10 / 15)
    assert np.isclose(per_ep[0], 1 / 6)


def test_pair_tracker_censors_windows_cut_off_by_episode_end():
    env = _FakeEnv()
    t = amt.PairTracker(gift_range=3, window=3)
    m, f = "predator_male_0", "predator_female_0"
    env.predator_positions = {m: (0, 0), f: (0, 1)}
    t.on_decision(env)
    env.agent_mate = {m: f, f: m}
    env.agent_parents = {"predator_male_1": (m, f)}
    t.after_step(env)
    t.on_decision(env)  # only 1 of 3 window steps before the episode ends; near
    t.after_step(env)
    r = t.result()
    assert r["births"] == 1
    assert r["post_birth_windows"] == 0 and r["post_birth_windows_censored"] == 1
    assert np.isnan(r["post_birth_near_frac"])
