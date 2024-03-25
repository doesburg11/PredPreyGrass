# parallel predpreygrass environment using rllib configuration

from typing import Dict

from environments.predpreygrass_env import PredPreyGrassEnv
from config.config_rllib import configuration

import numpy as np
import time

num_games = 1
if num_games > 1:
    configuration["render_mode"]="None"
else:
    configuration["render_mode"]="human"

env = PredPreyGrassEnv(configuration=configuration) 

def average(rewards):
    N = len(rewards)
    avg_rewards = sum(rewards) / N
    return avg_rewards

def std_dev(rewards, avg_reward):
    N = len(rewards.values())
    variance_rewards = 0
    for _agent in rewards:
        variance_rewards += pow(rewards[_agent]-avg_reward,2)
    variance_rewards = 1/(N-1)*variance_rewards
    std_rewards = pow(variance_rewards,0.5)
    return std_rewards

def generate_random_actions(agent_names):
    return {agent_name: np.random.randint(0, 4) for agent_name in agent_names}

def is_all_terminated(terminations: Dict) -> bool:
    for key, val in terminations.items():
        if not val:
            return False
    return True



avg_rewards = [0 for _ in range(num_games)]
avg_cycles = [0 for _ in range(num_games)]
std_rewards = [0 for _ in range(num_games)]

agent_names = env.agents


for i in range(num_games):
    obs, _ = env.reset(seed=i)
    cumulative_rewards = {agent: 0.0 for agent in env.possible_agents}
    stop_loop = False
    n_cycles = 0
    while not stop_loop:
        actions = {agent: env.action_space[agent].sample() for agent in env.agents}
        observations, rewards, terminations, truncations, info = env.step(actions)
        for agent in env.agents:
            cumulative_rewards[agent] += rewards[agent]
        env.render()
        terminated = env.terminateds["__all__"]
        truncated = truncations["__all__"]
        stop_loop = terminated or truncated
        n_cycles += 1
        #time.sleep(0.5)

    avg_rewards[i]= average(cumulative_rewards.values()) # type: ignore
    avg_cycles[i]= n_cycles
    std_rewards[i]= std_dev(cumulative_rewards, avg_rewards[i])
    print(f"Cycles = {n_cycles}", f"Avg = {round(avg_rewards[i],1)}", 
          f"Std = {round(std_rewards[i],1)}",end=" ")
    print()
env.close()
print(f"Average of Avg = {round(average(avg_rewards),1)}")
print(f"Average of Cycles = {round(average(avg_cycles),1)}")
  