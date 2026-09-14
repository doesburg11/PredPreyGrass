import numpy as np
import pytest

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import CH_PREY
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.rule_based_predator import RuleBasedPredatorPolicy

# Matches predpreygrass_rllib_env.py's action_to_move_tuple exactly.
ACTION_TO_MOVE = {
    0: (-1, -1), 1: (-1, 0), 2: (-1, 1),
    3: (0, -1),  4: (0, 0),  5: (0, 1),
    6: (1, -1),  7: (1, 0),  8: (1, 1),
}


def _obs_with_prey_at(offset_row: int, offset_col: int, obs_range: int = 7) -> np.ndarray:
    obs = np.zeros((4, obs_range, obs_range))
    half = obs_range // 2
    obs[CH_PREY, half + offset_row, half + offset_col] = 3.0
    return obs


def test_moves_toward_prey_directly_below():
    policy = RuleBasedPredatorPolicy(ACTION_TO_MOVE, seed=0)
    obs = _obs_with_prey_at(2, 0)  # prey straight in the +row direction
    assert policy.act(obs) == 7  # (1, 0)


def test_moves_toward_prey_diagonally():
    policy = RuleBasedPredatorPolicy(ACTION_TO_MOVE, seed=0)
    obs = _obs_with_prey_at(-2, 2)  # prey up and to the right
    assert policy.act(obs) == 2  # (-1, 1)


def test_moves_toward_nearest_of_multiple_prey():
    policy = RuleBasedPredatorPolicy(ACTION_TO_MOVE, seed=0)
    obs = np.zeros((4, 7, 7))
    half = 3
    obs[CH_PREY, half + 1, half + 0] = 1.0  # nearer, straight down
    obs[CH_PREY, half + 3, half + 0] = 1.0  # farther, same direction
    assert policy.act(obs) == 7  # (1, 0) -- chases the nearer one


def test_explores_when_no_prey_visible():
    policy = RuleBasedPredatorPolicy(ACTION_TO_MOVE, seed=0)
    obs = np.zeros((4, 7, 7))
    action = policy.act(obs)
    assert ACTION_TO_MOVE[action] != (0, 0)  # never just sits still while searching


def test_exploration_is_reproducible_with_same_seed():
    obs = np.zeros((4, 7, 7))
    policy_a = RuleBasedPredatorPolicy(ACTION_TO_MOVE, seed=42)
    policy_b = RuleBasedPredatorPolicy(ACTION_TO_MOVE, seed=42)
    actions_a = [policy_a.act(obs) for _ in range(20)]
    actions_b = [policy_b.act(obs) for _ in range(20)]
    assert actions_a == actions_b


def test_exploration_differs_with_different_seeds():
    obs = np.zeros((4, 7, 7))
    policy_a = RuleBasedPredatorPolicy(ACTION_TO_MOVE, seed=1)
    policy_b = RuleBasedPredatorPolicy(ACTION_TO_MOVE, seed=2)
    actions_a = [policy_a.act(obs) for _ in range(20)]
    actions_b = [policy_b.act(obs) for _ in range(20)]
    assert actions_a != actions_b
