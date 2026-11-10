"""RESULTS.md section 2's verdict and pilot rules (launchers/verdict_2.py,
launchers/pilot_check_2.py)."""

import pytest

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.pilot_check_2 import judge as pilot
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.verdict_2 import judge, sign_flip_p


def test_sign_flip_all_positive_is_the_smallest_p():
    mean, p = sign_flip_p([1, 2, 3, 4, 5, 6])
    assert mean == pytest.approx(3.5) and p == pytest.approx(1 / 64)


def test_sign_flip_one_small_negative_still_passes_and_a_large_one_does_not():
    assert sign_flip_p([1, 2, 3, 4, 5, -0.5])[1] < 0.05
    assert sign_flip_p([1, 2, 3, 4, 5, -6])[1] >= 0.05
    assert sign_flip_p([0, 0, 0, 0, 0, 0])[1] == 1.0


def values(b_a, l_a, b_s=None, l_s=None, coexist=(50,) * 6):
    b_s = b_s or [0.5] * 6
    l_s = l_s or [0.4] * 6
    return {
        "B": [{"A": a, "S": s, "coexisting": c} for a, s, c in zip(b_a, b_s, coexist)],
        "L": [{"A": a, "S": s, "coexisting": 50} for a, s in zip(l_a, l_s)],
    }


def test_judge_needs_both_measures_and_coexistence():
    tests, _ = judge(values([2] * 6, [1] * 6))
    assert tests == {"1_B_over_L": True, "2_B_coexists": True}
    tests, _ = judge(values([2] * 6, [1] * 6, b_s=[0.4] * 6))
    assert tests["1_B_over_L"] is False  # S tied
    tests, _ = judge(values([2] * 6, [1] * 6, coexist=(50, 50, 50, 50, 44, 44)))
    assert tests["2_B_coexists"] is False
    tests, _ = judge(values([2] * 6, [1] * 6, coexist=(50, 50, 50, 50, 50, 44)))
    assert tests["2_B_coexists"] is True


def evaluation(coexisting, births):
    return {"episodes": 50, "coexisting": coexisting, "totals": {"births_predator": births * 50}}


def test_pilot_bar():
    ok = {("r", 250): evaluation(40, 1), ("r", 500): evaluation(40, 5)}
    assert pilot(ok)["r"][0] is True
    assert pilot({**ok, ("r", 250): evaluation(39, 9)})["r"][0] is False
    assert pilot({**ok, ("r", 500): evaluation(50, 4.98)})["r"][0] is False
