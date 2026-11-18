import numpy as np
import torch

from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning import analyze_compound_gift as acg
from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning import (
    analyze_gift_decomposition as agd,
)
from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.compound_gift_env import (
    CompoundGiftPredPreyGrass,
)
from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.tests.test_analyze_compound_gift import (
    GiveWhenTagged,
    NeverGive,
    config,
    modules,
)


class GiveWhenMateHungry:
    """Gift bit only when the tagged mate's cell shows energy below 9 (the test threshold), else never."""

    def _forward_inference(self, batch):
        obs = batch["obs"].numpy()
        logits = np.full((len(obs), 18), 0.0, dtype=np.float32)
        logits[:, 9:] = -100.0
        for i, row in enumerate(obs):
            cells = np.argwhere(row[5] == 1)
            if len(cells) and row[1][tuple(cells[0])] < 9.0:
                logits[i, 9:] = 100.0
                logits[i, :9] = -100.0
        return {"action_dist_inputs": torch.as_tensor(logits)}


def test_in_need_bucket_is_exactly_the_eligible_state():
    env = CompoundGiftPredPreyGrass(config())
    rng = np.random.default_rng(1)
    seen = set()
    for episode in range(4):
        obs, _ = env.reset(seed=10 + episode)
        active = list(obs)
        for _ in range(40):
            for a in active:
                if "predator" in a:
                    b = agd.bucket_of(env, a)
                    seen.add(b)
                    assert (b == "in_need") == acg.eligible(env, a)
            actions = {a: int(rng.integers(0, 18 if "predator" in a else 9)) for a in active}
            obs, _, term, trunc, _ = env.step(actions)
            active = [a for a in obs if not term.get(a, False) and not trunc.get(a, False)]
            if term.get("__all__") or trunc.get("__all__"):
                break
    assert {"in_need", "no_mate"} <= seen


def test_shares_sum_to_one_and_never_give_has_zero_contrasts():
    rows = agd.rollout(config(), modules(NeverGive()), 2, 3)
    for r in rows:
        for sex in acg.SEXES:
            assert np.isclose(sum(r[f"{sex}_share_{b}"] for b in agd.BUCKETS), 1.0)
            for k in ("need_in", "range_need"):
                assert np.isnan(r[f"{sex}_{k}"]) or abs(r[f"{sex}_{k}"]) < 1e-6


def test_need_tracking_policy_shows_need_contrast():
    rows = agd.rollout(config(), modules(GiveWhenMateHungry()), 6, 3)
    vals = [r["predator_male_need_in"] for r in rows] + [r["predator_female_need_in"] for r in rows]
    vals = [v for v in vals if not np.isnan(v)]
    assert vals and np.mean(vals) > 0.9


def test_range_tracking_policy_shows_range_contrast():
    rows = agd.rollout(config(), modules(GiveWhenTagged()), 4, 3)
    vals = [r["predator_male_range_need"] for r in rows if not np.isnan(r["predator_male_range_need"])]
    assert vals and np.mean(vals) > 0.0


class StubEnv:
    homeostatic_energy_cost_per_step_predator = 0.1
    learned_gift_amount = 2.0
    predator_creation_energy_threshold = 12.0
    predator_gift_range = 3

    def __init__(self, giver_e, mate_e, mate_pos, has_mate=True, mate_alive=True):
        self.agent_mate = {"predator_male_0": "predator_female_0"} if has_mate else {}
        self.predator_positions = {"predator_male_0": (5, 5)}
        if mate_alive:
            self.predator_positions["predator_female_0"] = mate_pos
        self.agent_energies = {"predator_male_0": giver_e, "predator_female_0": mate_e}


import pytest


@pytest.mark.parametrize(
    "giver_e, mate_e, mate_pos, kw, expected",
    [
        (8.0, 5.0, (5, 5), {"has_mate": False}, "no_mate"),
        (8.0, 5.0, (5, 5), {"mate_alive": False}, "no_mate"),
        (8.0, 0.1, (6, 6), {}, "mate_dying"),          # 0.1 - 0.1 <= 0
        (2.1, 5.0, (6, 6), {}, "no_afford"),          # 2.1 - 0.1 - 2.0 <= 0
        (8.0, 11.99, (8, 8), {}, "in_need"),          # Chebyshev 3, just below threshold
        (8.0, 12.0, (8, 8), {}, "in_full"),           # at threshold counts as full
        (8.0, 5.0, (9, 5), {}, "out_need"),           # Chebyshev 4
        (8.0, 15.0, (5, 9), {}, "out_full"),
    ],
)
def test_bucket_boundaries(giver_e, mate_e, mate_pos, kw, expected):
    env = StubEnv(giver_e, mate_e, mate_pos, **kw)
    assert agd.bucket_of(env, "predator_male_0") == expected
    assert (expected == "in_need") == acg.eligible(env, "predator_male_0")


def test_episode_row_handles_empty_cells():
    stats = {sex: {b: [0.0, 0] for b in agd.BUCKETS} for sex in acg.SEXES}
    stats["predator_male"]["in_need"] = [3.0, 4]   # mean 0.75
    stats["predator_male"]["out_need"] = [1.0, 4]  # mean 0.25
    row = agd.episode_row(stats)
    assert np.isclose(row["predator_male_range_need"], 0.5)
    assert np.isnan(row["predator_male_need_in"])            # in_full empty
    assert np.isnan(row["predator_female_need_in"])          # no female data at all
    assert np.isnan(row["predator_female_share_in_need"])
    assert np.isclose(row["predator_male_share_in_need"], 0.5)


def rows_with(key, values):
    return [{key: v} for v in values]


def test_pair_test_requires_min_episodes_and_diff_interval():
    rng = np.random.default_rng(0)
    k = "predator_male_need_in"
    few = rows_with(k, [0.5] * 9 + [np.nan] * 11)
    ok_inert = rows_with(k, [0.0] * 20)
    assert agd.pair_test(few, ok_inert, k, rng)[0] is False                  # 9 < MIN_EPISODES
    clear = rows_with(k, list(np.linspace(0.08, 0.12, 20)))
    assert agd.pair_test(clear, ok_inert, k, rng)[0] is True
    noisy_inert = rows_with(k, list(np.linspace(-0.5, 0.7, 20)))           # mean 0.1, wide
    assert agd.pair_test(clear, noisy_inert, k, rng)[0] is False             # diff interval spans 0
