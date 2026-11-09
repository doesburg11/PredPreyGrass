"""RESULTS.md section 5's verdict rules (launchers/control_verdict.py)."""

import pytest

from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.launchers.control_verdict import classify, verdict_of


def summary(a=(0.01, 0.05), b=(0.01, 0.03), c=(0.6, 0.9)):
    def entry(iv):
        return None if iv is None else sum(iv) / 2
    return {"episodes": 40, "trend": entry(a), "trend_interval": a, "top_minus_chance": entry(b),
            "top_minus_chance_interval": b, "joint_ratio": entry(c), "joint_ratio_interval": c}


PATTERN = summary()
ABSENT = summary(b=(-0.02, 0.01), c=(0.1, 0.4))


def test_validity_is_at_least_45_of_50():
    assert classify(45, PATTERN)[0] and not classify(44, PATTERN)[0]


@pytest.mark.parametrize("c, absent", [((0.1, 0.49), True), ((0.1, 0.5), False), (None, False)])
def test_absent_needs_c_upper_bound_strictly_below_half(c, absent):
    assert classify(50, summary(b=(-0.02, 0.01), c=c))[2] is absent


def test_absent_needs_b_not_met():
    assert classify(50, summary(b=(0.001, 0.02), c=(0.1, 0.4)))[2] is False


def test_all_met_is_section_2_1s_tests():
    assert classify(50, PATTERN)[1] is True
    assert classify(50, summary(c=(0.5, 0.9)))[1] is False


def rows(*items):
    return [classify(coexisting, s)[:3] for coexisting, s in items]


def test_verdicts():
    assert verdict_of(rows((50, ABSENT), (50, ABSENT), (50, PATTERN))) == "payoff supported"
    assert verdict_of(rows((50, PATTERN), (50, PATTERN), (50, ABSENT))) == "payoff not supported"
    assert verdict_of(rows((50, PATTERN), (50, ABSENT), (50, summary(c=(0.4, 0.8))))) == "inconclusive"
    # invalid runs do not count, either way
    assert verdict_of(rows((50, ABSENT), (44, ABSENT), (50, PATTERN))) == "inconclusive"
    assert verdict_of(rows((50, PATTERN), (44, PATTERN), (44, PATTERN))).startswith("inconclusive (fewer than 2")
