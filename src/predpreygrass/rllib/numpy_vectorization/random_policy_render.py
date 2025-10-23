import time
import numpy as np
from predpreygrass.rllib.numpy_vectorization.np_vec_env import PredPreyGrassEnv

if __name__ == "__main__":
    num_steps = 1000
    render_sleep = 0.05
    env = PredPreyGrassEnv()
    obs, infos = env.reset()
    for step in range(num_steps):
        # Take random actions for all active agents
        action_dict = {aid: np.random.randint(0, 5) for aid in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(action_dict)
        env.render()
        time.sleep(render_sleep)
        if terminations.get("__all__", False) or truncations.get("__all__", False):
            print(f"Episode ended at step {step}")
            break
