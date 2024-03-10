# parallel predpreygrass environment using rllib configuration

from typing import Dict

from environments.predpreygrass_random_env import PredPreyGrassEnv
from config.config_rllib import config

#from agents.discrete_agent import DiscreteAgent
import numpy as np

num_games = 1
if num_games > 1:
    config["render_mode"]="None"
else:
    config["render_mode"]="human"

PredPreyGrassEnv = PredPreyGrassEnv(env_config=config) 
#PredPreyGrassEnv = PredPreyGrassEnv() 

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

agent_names = PredPreyGrassEnv.agents


for i in range(num_games):
    PredPreyGrassEnv.reset(seed=i)
    cumulative_rewards = {agent: 0.0 for agent in PredPreyGrassEnv.possible_agents}
    stop_loop = False
    n_cycles = 0
    while not stop_loop:
        actions = {agent: PredPreyGrassEnv.action_space(agent).sample() for agent in PredPreyGrassEnv.agents}
        observations, rewards, terminations, truncations, info = PredPreyGrassEnv.step(actions)
        for agent in PredPreyGrassEnv.agents:
            cumulative_rewards[agent] += rewards[agent]
        PredPreyGrassEnv.render()
        terminated = is_all_terminated(terminations)
        truncated = truncations[PredPreyGrassEnv.agents[0]]
        stop_loop = terminated or truncated
        n_cycles += 1

    avg_rewards[i]= average(cumulative_rewards.values()) # type: ignore
    avg_cycles[i]= n_cycles
    std_rewards[i]= std_dev(cumulative_rewards, avg_rewards[i])
    print(f"Cycles = {n_cycles}", f"Avg = {round(avg_rewards[i],1)}", 
          f"Std = {round(std_rewards[i],1)}",end=" ")
    print()
PredPreyGrassEnv.close()
print(f"Average of Avg = {round(average(avg_rewards),1)}")
print(f"Average of Cycles = {round(average(avg_cycles),1)}")
  