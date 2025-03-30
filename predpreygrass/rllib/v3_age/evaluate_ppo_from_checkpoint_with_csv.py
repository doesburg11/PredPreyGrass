
import os
import csv
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from predpreygrass_rllib_env import PredPreyGrass
from renderer import Renderer

# --- Setup ---
def env_creator(config):
    return PredPreyGrass(config)

register_env("PredPreyGrass", env_creator)

checkpoint_path = "path_to_your_checkpoint"
config_path = "path_to_your_config"

# Modify this depending on how you save/load the config
algo = Algorithm.from_checkpoint(checkpoint_path)

env_config = algo.config["env_config"]
env = PredPreyGrass(env_config)
renderer = Renderer(env.grid_size)

obs = env.reset(seed=42)
done = {"__all__": False}

# --- CSV Setup ---
csv_filename = "evaluation_results.csv"
csv_fields = ["timestep", "agent_id", "reward", "cumulative_reward", "x", "y"]

with open(csv_filename, mode="w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    writer.writeheader()

    t = 0
    while not done["__all__"] and t < 500:
        actions = {}
        for agent_id, observation in obs.items():
            policy_id = algo.config["policy_mapping_fn"](agent_id, None, None)
            action = algo.compute_single_action(observation, policy_id=policy_id)
            actions[agent_id] = action

        obs, rewards, dones, truncs, infos = env.step(actions)

        renderer.record(env)

        for agent_id, reward in rewards.items():
            x, y = env.agent_positions.get(agent_id, (-1, -1))
            row = {
                "timestep": t,
                "agent_id": agent_id,
                "reward": reward,
                "cumulative_reward": env.cumulative_rewards.get(agent_id, 0.0),
                "x": x,
                "y": y
            }
            writer.writerow(row)

        t += 1

renderer.render()
renderer.save_gif("evaluation.gif")
