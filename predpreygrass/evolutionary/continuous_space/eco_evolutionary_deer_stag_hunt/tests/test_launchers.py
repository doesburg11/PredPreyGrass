"""verdict.py's statistics and evolve.py's budget split."""

import json
from itertools import combinations

import pytest

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import evolve
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers import verdict


def test_mann_whitney_exact():
    assert verdict.mann_whitney_greater([6, 7, 8, 9, 10], [1, 2, 3, 4, 5]) == (25.0, 1 / 252)
    assert verdict.mann_whitney_greater([5, 7, 8, 9, 10], [1, 2, 3, 4, 6]) == (24.0, 2 / 252)
    u, p = verdict.mann_whitney_greater([1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
    assert u == 12.5 and p == 1.0  # all ties: no evidence
    # p is the share of splits with U at least the observed one.
    xs, ys = [3, 5, 6, 8, 9], [1, 2, 4, 7, 10]
    u, p = verdict.mann_whitney_greater(xs, ys)
    pooled = xs + ys
    def U(a, b):
        return sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    splits = [U([pooled[i] for i in c], [pooled[i] for i in range(10) if i not in c]) for c in combinations(range(10), 5)]
    assert p == sum(v >= u for v in splits) / len(splits)


def test_design_tolerance_four_pairs_out_of_order():
    # 21 of 25 pairs in order: p = 0.048 < 0.05 (DESIGN.md); 20 of 25: not.
    assert verdict.mann_whitney_greater([5, 6, 8, 9, 10], [1, 2, 3, 4, 7])[1] < 0.05
    _, p = verdict.mann_whitney_greater([2, 6, 8, 9, 10], [1, 3, 4, 5, 7])
    assert p >= 0.05


def write(tmp_path, run, mark, caught, failures, coexisting=50, ratio=1.2):
    totals = {"adults_caught": caught, "adult_failures": failures}
    (tmp_path / f"{run}_iter{mark}_eval.json").write_text(json.dumps({"summary": {"episodes": 50, "coexisting": coexisting, "totals": totals}}))
    (tmp_path / f"{run}_iter{mark}_coop.json").write_text(json.dumps({"summary": {"episodes": 20, "adult_over_chance": ratio}}))


def test_run_value_pools_s_over_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(verdict, "RESULTS", tmp_path)
    write(tmp_path, "B_1", 500, 1, 1)  # 50%, few attempts
    write(tmp_path, "B_1", 750, 10, 90)  # 10%
    write(tmp_path, "B_1", 1000, 0, 0, coexisting=44, ratio=None)  # no attempts
    value = verdict.run_value("B_1")
    assert value["S"] == pytest.approx(11 / 102)  # pooled, not the mean of ratios
    assert value["A"] == pytest.approx((1 + 10 + 0) / 3 / 50)
    assert value["C"] == pytest.approx(1.2)  # checkpoints with a value
    assert value["coexisting"] == pytest.approx((50 + 50 + 44) / 3)


def test_run_value_without_attempts_is_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(verdict, "RESULTS", tmp_path)
    for mark in verdict.CHECKPOINTS:
        write(tmp_path, "E_1", mark, 0, 0)
    assert verdict.run_value("E_1")["S"] == 0.0


def test_worker_targets_split_exactly():
    shares = evolve.worker_targets({250: 1003, 500: 2000}, 14)
    assert sum(s[250] for s in shares) == 1003 and sum(s[500] for s in shares) == 2000
    assert max(s[250] for s in shares) - min(s[250] for s in shares) <= 1
