"""Tests for analyze_counterfactual_crowding.py: observation edits, blocking-aware distance, placement, bootstrap."""
import numpy as np

from predpreygrass.non_evolutionary.predator_sexual_reproduction import analyze_counterfactual_crowding as cf
from predpreygrass.non_evolutionary.predator_sexual_reproduction.analysis_env import make_env
from predpreygrass.non_evolutionary.predator_sexual_reproduction.config_env import config_env

OFFSET = 3
W = 2 * OFFSET + 1


def _obs():
    o = np.zeros((5, W, W))
    o[cf.PRED_CH, OFFSET, OFFSET] = 6.0  # own energy
    return o


def test_remove_neighbors_keeps_only_center_and_does_not_mutate():
    o = _obs()
    o[cf.PRED_CH, 1, 1] = 5.0
    o[cf.PRED_CH, 5, 2] = 9.0
    before = o.copy()
    out = cf.remove_neighbors(o, OFFSET)
    assert np.array_equal(o, before)
    assert out[cf.PRED_CH, OFFSET, OFFSET] == 6.0
    assert (out[cf.PRED_CH] > 0).sum() == 1


def test_placements_respect_distance_border_and_occupancy_and_allow_food_cells():
    o = _obs()
    o[0, 0, :] = 1  # a border row
    o[cf.PRED_CH, OFFSET + 1, OFFSET] = 4.0  # occupied cell at distance 1
    o[cf.FRUIT_CH, OFFSET, OFFSET + 1] = 2.0  # food cell at distance 1 stays legal
    cells = cf.placements(o, OFFSET, 1)
    assert (OFFSET + 1, OFFSET) not in cells
    assert (OFFSET, OFFSET + 1) in cells
    assert len(cells) == 7
    assert all(max(abs(x - OFFSET), abs(y - OFFSET)) == 1 for x, y in cells)
    assert all(x != 0 for x, _ in cf.placements(o, OFFSET, 3))  # border row excluded at distance 3


def test_with_predator_does_not_mutate_and_sets_energy():
    o = _obs()
    out = cf.with_predator(o, (OFFSET + 2, OFFSET), 8.0)
    assert o[cf.PRED_CH].sum() == 6.0
    assert out[cf.PRED_CH, OFFSET + 2, OFFSET] == 8.0


def test_blocked_actions_and_edited_d_after_use_stay_distance():
    moves = np.array([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)])
    o = _obs()
    o[cf.PRED_CH, OFFSET + 1, OFFSET] = 4.0
    blocked = cf.blocked_actions(o, OFFSET, moves)
    assert blocked.tolist() == [False, True, False, False, False]
    d_free = np.array([3.0, 1.0, 5.0, 3.5, 3.5])
    d = cf.edited_d_after(d_free, 3.0, o, OFFSET, moves)
    assert d[1] == 3.0 and d[0] == 3.0 and d[2] == 5.0


def test_obs_axis_convention_matches_positions_in_real_env():
    """A predator at pos+(dx,dy) appears at obs[PRED, offset+dx, offset+dy] (what blocked_actions assumes)."""
    env = make_env({**config_env, "seed": 3})
    env.reset(seed=3)
    agents = [a for a in env.predator_positions]
    a, b = agents[0], agents[1]
    xa, ya = env.agent_positions[a]
    off = (env.predator_obs_range - 1) // 2
    obs = env._get_observation(a)
    xb, yb = env.agent_positions[b]
    if max(abs(xb - xa), abs(yb - ya)) <= off:
        assert obs[cf.PRED_CH, off + xb - xa, off + yb - ya] == env.agent_energies[b]
    else:  # move b next to a in the state and re-observe
        env.grid_world_state[1, xb, yb] = 0
        nx, ny = min(xa + 1, env.grid_size - 1), ya
        env.grid_world_state[1, nx, ny] = 7.0
        obs = env._get_observation(a)
        assert obs[cf.PRED_CH, off + nx - xa, off + ny - ya] == 7.0


def test_approach_bias_sign_and_uniform_policy_is_zero():
    d = np.array([3.0, 1.0, 2.0, 4.0])
    assert abs(cf.approach_bias(np.full(4, 0.25), d)) < 1e-12
    assert cf.approach_bias(np.array([0, 1.0, 0, 0]), d) > 0  # picks the closest-after action


class _FakeModule:
    """Policy that goes to action 1 iff a predator sits at obs[PRED, OFFSET+1, OFFSET]; used through module_probs."""


def test_evaluate_matches_hand_computation_and_blocks_moves(monkeypatch):
    moves = np.array([(0, 0), (1, 0), (-1, 0)])
    o = _obs()
    d_free, d_stay = np.array([2.0, 1.0, 3.0]), 2.0
    bank = [{"obs": o, "episode": 0, "near": False, "geo": {"prey": (d_free, d_stay), "fruit": None}}]

    def fake_probs(module, obs_list):
        return np.array([[0.0, 1.0, 0.0] for _ in obs_list])  # always tries to step to (+1,0)

    monkeypatch.setattr(cf, "module_probs", fake_probs)
    bias = cf.evaluate(bank, OFFSET, moves, None, 8.0)
    # base/alone: nothing blocks, policy picks d=1, uniform mean 2.0 -> bias 1.0
    assert abs(bias["base"]["prey"][0] - 1.0) < 1e-12
    assert abs(bias["alone"]["prey"][0] - 1.0) < 1e-12
    # +d1 averaged over all 8 placements. Predator at (+1,0) blocks the chosen step (d -> d_stay=2, uniform mean 7/3,
    # bias 1/3); predator at (-1,0) blocks the other move (uniform mean 5/3, chosen d stays 1, bias 2/3); the other 6
    # placements block nothing (bias 1.0).
    assert abs(bias["+d1"]["prey"][0] - (6 * 1.0 + 1 / 3 + 2 / 3) / 8) < 1e-12
    # energy variants share the same placements -> identical here because the fake policy ignores energy
    assert abs(bias["+d2 e3"]["prey"][0] - bias["+d2"]["prey"][0]) < 1e-12
    assert np.isnan(bias["base"]["fruit"][0])


def test_common_mask_intersects_conditions():
    bias = {"a": {"x": np.array([1.0, np.nan, 3.0])}, "b": {"x": np.array([1.0, 2.0, np.nan])}}
    assert cf.common_mask(bias, "x").tolist() == [True, False, False]


def test_cluster_stat_requires_min_episodes_and_recovers_constant():
    rng = np.random.default_rng(0)
    x = np.arange(10, dtype=float)
    assert cf._cluster_stat(lambda w: cf.wmean(x, w), np.zeros(10, dtype=int), rng) is None
    ep = np.repeat(np.arange(5), 2)
    point, lo, hi, n = cf._cluster_stat(lambda w: cf.wmean(np.ones(10), w), ep, rng)
    assert (point, lo, hi, n) == (1.0, 1.0, 1.0, 5)


def test_cluster_stat_resamples_whole_episodes():
    rng = np.random.default_rng(1)
    ep = np.repeat(np.arange(6), 3)
    vals = ep.astype(float) * 10  # constant within an episode: bootstrap spread comes only from episode resampling
    point, lo, hi, _ = cf._cluster_stat(lambda w: cf.wmean(vals, w), ep, rng)
    assert lo < point < hi and hi - lo > 5


def test_blocking_false_isolates_policy_response(monkeypatch):
    """An observation-invariant policy has zero policy-only crowding effect but a nonzero total (mechanical) one."""
    moves = np.array([(0, 0), (1, 0), (-1, 0)])
    bank = [{"obs": _obs(), "episode": 0, "near": False, "geo": {"prey": (np.array([2.0, 1.0, 3.0]), 2.0), "fruit": None}}]
    monkeypatch.setattr(cf, "module_probs", lambda module, obs_list: np.array([[0.0, 1.0, 0.0] for _ in obs_list]))
    total = cf.evaluate(bank, OFFSET, moves, None, 8.0, blocking=True)
    policy = cf.evaluate(bank, OFFSET, moves, None, 8.0, blocking=False)
    assert abs(policy["+d1"]["prey"][0] - policy["alone"]["prey"][0]) < 1e-12
    assert abs(total["+d1"]["prey"][0] - total["alone"]["prey"][0]) > 1e-3


def _bank(episodes, near, n_per=10):
    bank = []
    for e, nr in zip(episodes, near):
        for _ in range(n_per):
            bank.append({"obs": _obs(), "episode": e, "near": nr, "geo": {"prey": (np.ones(3), 1.0), "fruit": None}})
    return bank


def test_summarize_reports_na_when_an_arm_has_too_few_episodes(monkeypatch):
    rng = np.random.default_rng(0)
    bank = _bank(list(range(6)), [True, False, False, False, False, False])  # one near episode only
    n = len(bank)
    mk = lambda: {c: {"prey": np.linspace(0, 1, n), "fruit": np.full(n, np.nan)} for c, _, _ in cf.conditions(8.0)}
    lines = cf.summarize("x", "female", bank, mk(), mk(), rng)
    assert any("observational near-away=n/a" in l for l in lines)
    assert any("(near 1, away 5)" in l for l in lines)
    assert any("fruit" in l and "skipped" in l for l in lines)
