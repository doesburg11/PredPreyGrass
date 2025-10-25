import pytest
from predpreygrass.rllib.numpy_vectorization.np_vec_env import PredPreyGrassEnv


@pytest.mark.parametrize("wipe_group", ["predators", "prey"])
def test_extinction_terminates_episode(wipe_group):
    # Minimal env: 1 predator, 1 prey, no grass dynamics needed
    env = PredPreyGrassEnv(
        grid_shape=(5, 5),
        num_possible_predators=2,
        num_possible_prey=2,
        initial_num_predators=1,
        initial_num_prey=1,
        initial_num_grass=0,
        seed=123,
        max_episode_steps=50,
    )
    env.reset(seed=123)

    # Wipe one group to trigger extinction condition
    if wipe_group == "predators":
        env.active[env.is_pred] = False
    else:
        env.active[env.is_prey] = False
    env._active_dirty = True  # ensure env.agents cache updates

    # Step once; with one group extinct, __all__ should be terminated
    obs, rewards, terminations, truncations, infos = env.step({})
    assert terminations.get('__all__', False) is True
    # When terminated via extinction, it's not a truncation
    assert truncations.get('__all__', False) is False
