"""RESULTS.md section 3 (DESIGN_3.md): learning deer in train.py, a
learning-deer checkpoint's own deer, the cross-play checks, and the
verdict's statistics (launchers/verdict_3.py)."""

import json
from itertools import combinations

import numpy as np
import pytest

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import checkpoints, train
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_env import config_env
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers import cross_play, verdict_3


# --- settings: exactly one deer mode -------------------------------------------------


@pytest.fixture
def ppo(monkeypatch, tmp_path):
    def set_(**kwargs):
        base = dict(train.config_ppo, checkpoint_dir=str(tmp_path / "r" / "checkpoint"), tensorboard_dir=None)
        base.update(kwargs)
        monkeypatch.setattr(train, "config_ppo", base)
        return train.load_settings()

    return set_


def test_learn_mode_is_the_default_and_valid(ppo):
    _, settings = ppo()
    assert settings.deer_learn is True and settings.deer_schedule is None and settings.deer_checkpoint is None


@pytest.mark.parametrize("kwargs", [
    {"deer_learn": False},  # no mode
    {"deer_learn": True, "deer_schedule": str(train.DEER_RUNS_DIR / "touch_1")},  # two modes
    {"deer_learn": 1},  # not a boolean
    {"vf_share_layers": True},  # the deer network must be RLlib's default
])
def test_settings_refuse_other_deer_modes(ppo, kwargs):
    with pytest.raises(ValueError):
        ppo(**kwargs)


def test_learn_mode_requires_deer_stag_hunts_touch_world(monkeypatch, ppo):
    monkeypatch.setattr(train, "config_env", dict(config_env, deer_catch_rule="party"))
    with pytest.raises(ValueError):
        ppo()


def test_check_both_learned():
    a = {"w": np.zeros(3)}
    b = {"w": np.ones(3)}
    train.check_both_learned({"deer_policy": a, "predator_policy": a}, {"deer_policy": b, "predator_policy": b})
    for frozen in ("deer_policy", "predator_policy"):
        after = {"deer_policy": b, "predator_policy": b, frozen: a}
        with pytest.raises(RuntimeError):
            train.check_both_learned({"deer_policy": a, "predator_policy": a}, after)


# --- a learning-deer checkpoint acts with its own deer --------------------------------


def fake_run(root, name, learn=True, env=None, module=True):
    run = root / name
    (run / "source_code").mkdir(parents=True)
    ppo = {"arm": "L", "beta": 0.0, "iterations": 500, "deer_learn": learn, "deer_schedule": None, "deer_checkpoint": None}
    (run / "source_code" / "settings.json").write_text(json.dumps({"config_env": dict(env or config_env), "config_ppo": ppo}))
    checkpoint = run / "checkpoint" / "iter_000250"
    deer = checkpoint / "learner_group" / "learner" / "rl_module" / "deer_policy"
    deer.mkdir(parents=True)
    if module:
        (deer / "module_state.pkl").write_bytes(b"x")
    return run, checkpoint, deer


def test_learning_checkpoint_deer_is_inside_the_checkpoint(tmp_path):
    _, checkpoint, deer = fake_run(tmp_path, "a")
    assert checkpoints.checkpoint_deer(checkpoint) == deer


def test_learning_checkpoint_without_its_deer_is_refused(tmp_path):
    _, checkpoint, _ = fake_run(tmp_path, "a", module=False)
    with pytest.raises(ValueError):
        checkpoints.checkpoint_deer(checkpoint)


def test_cross_play_refuses_a_non_learning_run_or_another_world(tmp_path):
    _, a, _ = fake_run(tmp_path, "a")
    _, b, _ = fake_run(tmp_path, "b", learn=False)
    with pytest.raises(ValueError):
        cross_play.check_pair(a, b)
    _, c, _ = fake_run(tmp_path, "c", env=dict(config_env, energy_grass_gain=config_env["energy_grass_gain"] + 1.0))
    with pytest.raises(ValueError):
        cross_play.check_pair(a, c)
    _, d, _ = fake_run(tmp_path, "d", env=dict(config_env, gene_mutation_std=0.2))
    cross_play.check_pair(a, d)  # gene settings aside


# --- the verdict's statistics ---------------------------------------------------------


def test_permutation_all_b_above_is_the_smallest_p():
    diff, p = verdict_3.permutation_p([7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6])
    assert diff == pytest.approx(6.0) and p == pytest.approx(1 / 924)
    assert verdict_3.permutation_p([1] * 6, [1] * 6)[1] == 1.0  # ties: no evidence


def test_permutation_p_is_the_share_of_labellings():
    bs, ls = [3, 5, 6, 8, 9, 12], [1, 2, 4, 7, 10, 11]
    diff, p = verdict_3.permutation_p(bs, ls)
    pooled = bs + ls
    diffs = [np.mean([pooled[i] for i in c]) - np.mean([pooled[i] for i in range(12) if i not in c]) for c in combinations(range(12), 6)]
    assert p == pytest.approx(sum(d >= diff - 1e-12 for d in diffs) / 924)


def cell(caught, failures, episodes=50):
    return {"episodes": episodes, "totals": {"adults_caught": caught, "adult_failures": failures}}


def test_panel_values_rows_columns_and_pooled_s():
    panel = {("h", "x", 500): cell(50, 50), ("h", "y", 500): cell(100, 900), ("x", "h", 500): cell(10, 0)}
    hunters = verdict_3.panel_values(panel, "h")
    assert hunters["A"] == pytest.approx((1 + 2) / 2) and hunters["S"] == pytest.approx(150 / 1100) and hunters["cells"] == 2
    deer = verdict_3.panel_values(panel, "h", "deer")
    assert deer["A"] == pytest.approx(0.2) and deer["cells"] == 1


def test_load_panel_leaves_out_the_diagonal(tmp_path, monkeypatch):
    monkeypatch.setattr(verdict_3, "RESULTS", tmp_path)
    for h in "ab":
        for d in "ab":
            (tmp_path / f"{h}_iter500_vs_{d}_eval.json").write_text(json.dumps({"summary": cell(1, 1)}))
    panel = verdict_3.load_panel(["a", "b"], marks=(500,), check=False)
    assert sorted(panel) == [("a", "b", 500), ("b", "a", 500)]


def test_selection_value_counts_undefined_episodes():
    rows = [[{"stats": {"gene_a_adult_selection": 0.2}}, {"stats": {}}], [{"stats": {"gene_a_adult_selection": 0.4}}, {"stats": None}]]
    assert verdict_3.selection_value(rows, "a_adult") == (pytest.approx(0.3), 2, 2)
    assert verdict_3.selection_value([[{"stats": {}}]], "a_adult") == (None, 0, 1)


def arms(b_a, b_s, coexist=(150,) * 6, sel_b=0.1, sel_l=0.0):
    return (
        {"B": [{"A": a, "S": s} for a, s in zip(b_a, b_s)], "L": [{"A": 1.0, "S": 0.1}] * 6},
        {"B": list(coexist), "L": [150] * 6},
        {"B": [{"a_hunter": sel_b, "a_adult": sel_b}] * 6, "L": [{"a_hunter": sel_l + 0.001 * k, "a_adult": sel_l} for k in range(6)]},
    )


def test_judge_needs_both_measures_and_coexistence():
    tests, _ = verdict_3.judge(*arms([2.0] * 6, [0.2] * 6))
    assert tests["1_panel_B_over_L"] and tests["2_B_coexists"]
    tests, _ = verdict_3.judge(*arms([2.0] * 6, [0.05] * 6))  # A met, S not
    assert not tests["1_panel_B_over_L"]
    tests, details = verdict_3.judge(*arms([2.0] * 6, [0.2] * 6, coexist=(150, 150, 150, 150, 134, 134)))
    assert not tests["2_B_coexists"] and details["coexisting_B_runs"] == 4
    tests, _ = verdict_3.judge(*arms([2.0] * 6, [0.2] * 6, coexist=(150, 150, 150, 150, 135, 0)))
    assert tests["2_B_coexists"]  # the sum, at least 135 of 150, in 5 of 6


def test_judge_selection_alpha_and_missing_values():
    tests, details = verdict_3.judge(*arms([2.0] * 6, [0.2] * 6))
    assert tests["3_selection_a_hunter"] and tests["3_selection_a_adult"]
    panel, coexist, selection = arms([2.0] * 6, [0.2] * 6)
    selection["B"] = [{"a_hunter": None, "a_adult": 0.1}] + selection["B"][1:]
    tests, details = verdict_3.judge(panel, coexist, selection)
    assert not tests["3_selection_a_hunter"] and details["selection_a_hunter"]["p"] is None
