import pytest
from predpreygrass.rllib.numpy_vectorization.np_vec_env import PredPreyGrassEnv

# RLlib's check_env utility (works for MultiAgentEnv)
try:
    from ray.rllib.env import check_env
    HAS_RLLIB = True
except ImportError:
    HAS_RLLIB = False

# Gymnasium's check_env utility (works for single-agent Gymnasium envs)
try:
    from gymnasium.utils.env_checker import check_env as gym_check_env
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

def test_rllib_env_compliance():
    if not HAS_RLLIB:
        pytest.skip("RLlib not installed; skipping RLlib check_env test.")
    env = PredPreyGrassEnv(
        grid_shape=(5, 5),
        num_possible_predators=2,
        num_possible_prey=2,
        initial_num_predators=1,
        initial_num_prey=1,
        initial_num_grass=0,
        seed=42,
        max_episode_steps=10,
    )
    check_env(env)

# Optionally, if you want to check Gymnasium compliance for a single-agent wrapper:
def test_gymnasium_env_compliance():
    if not HAS_GYM:
        pytest.skip("Gymnasium not installed; skipping gymnasium check_env test.")
    # This will fail unless you have a single-agent wrapper for your env
    # env = YourSingleAgentWrapper(PredPreyGrassEnv(...))
    # gym_check_env(env)
    pass  # No-op for now
