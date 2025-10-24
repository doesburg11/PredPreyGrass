import time
import numpy as np
from predpreygrass.rllib.numpy_vectorization.np_vec_env import PredPreyGrassEnv

if __name__ == "__main__":
    render_sleep = 0.05
    SEED = 42
    np.random.seed(SEED)
    env = PredPreyGrassEnv(max_episode_steps=1000, seed=SEED)
    obs, infos = env.reset(seed=SEED)
    done_all = False
    step = 0
    while not done_all:
        # Take random actions for all active agents
        action_dict = {aid: np.random.randint(0, 5) for aid in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(action_dict)
        env.render()
        time.sleep(render_sleep)
        terminated = terminations.get('__all__', False)
        truncated = truncations.get('__all__', False)
        done_all = terminated or truncated
        step += 1
    if truncated:
        print(f"Episode finished due to truncation at step {step}.")
    if terminated:
        print(f"Episode finished due to termination at step {step}.")
    