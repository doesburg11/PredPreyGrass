"""Fast deterministic tests: python -m pytest .../test_social_bias.py"""
import numpy as np

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.environment import SELF_SIZE
from .social_bias import social_scores, biased_logits, probabilities


def observation(x=0, y=-0.5, seen=True):
    obs = np.zeros(SELF_SIZE + 6, dtype=float)
    obs[SELF_SIZE:SELF_SIZE+4] = [float(seen), x, y, np.hypot(x, y)]
    return obs


def test_neutral_is_exactly_unchanged():
    original = np.arange(48, dtype=float) / 10
    np.testing.assert_array_equal(biased_logits(original, observation(), 0, 2, 1), original)


def test_no_visible_conspecific_no_effect():
    original = np.arange(48, dtype=float) / 10
    np.testing.assert_array_equal(biased_logits(original, observation(seen=False), 1, 2, 1), original)


def test_attraction_avoidance_opposite():
    score = social_scores(observation(), 1)
    assert score[0] > 0.99  # direction 0 = up
    assert score[8] < -0.99  # direction 8 = down
    assert np.all(score[32:] == 0)  # stopped
    attraction = probabilities(biased_logits(np.zeros(48), observation(), 1, 2, 1))
    avoidance = probabilities(biased_logits(np.zeros(48), observation(), -1, 2, 1))
    assert attraction[0] > attraction[8]
    assert avoidance[0] < avoidance[8]


def test_invalid_inputs():
    import pytest
    with pytest.raises(ValueError):
        biased_logits(np.zeros(48), observation(), 2, 1, 1)
