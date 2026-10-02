import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_hebbian.uniform_alpha_control import (
    _opposite,
    run_probe,
)


@pytest.mark.parametrize("direction, expected", [(0, 1), (1, 0), (2, 3), (3, 2)])
def test_opposite_direction(direction, expected):
    assert _opposite(direction) == expected


def test_probe_is_reproducible_and_uses_common_decision_count():
    first = run_probe(41, "uniform_positive", 100)
    second = run_probe(41, "uniform_positive", 100)

    assert first == second
    assert first["steps"] == 100
    assert 0 <= first["away_actions"] <= 100
    assert first["away_fraction"] == first["away_actions"] / 100


def test_probe_rejects_unknown_routing_mode():
    with pytest.raises(ValueError, match="unknown action_alpha_mode"):
        run_probe(41, "invalid", 10)
