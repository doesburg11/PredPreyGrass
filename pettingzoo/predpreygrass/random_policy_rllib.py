# parallel predpreygrass environment using rllib configuration

from typing import Dict

from environments.predpreygrass_rllib import PredPreyGrassEnv as pred_prey_grass_env
from config.parameters_rllib import env_kwargs

#from agents.discrete_agent import DiscreteAgent
import numpy as np

num_games = 1
if num_games > 1:
    env_kwargs["render_mode"]="None"
else:
    env_kwargs["render_mode"]="human"

pred_prey_grass_env = pred_prey_grass_env(**env_kwargs) 

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

agent_names = pred_prey_grass_env.agents


for i in range(num_games):
    pred_prey_grass_env.reset(seed=i)
    cumulative_rewards = {agent: 0.0 for agent in pred_prey_grass_env.possible_agents}
    stop_loop = False
    n_cycles = 0
    while not stop_loop:
        actions = {agent: pred_prey_grass_env.action_space(agent).sample() for agent in pred_prey_grass_env.agents}
        observations, rewards, terminations, truncations, info = pred_prey_grass_env.step(actions)
        for agent in pred_prey_grass_env.agents:
            cumulative_rewards[agent] += rewards[agent]
        pred_prey_grass_env.render()
        stop_loop = is_all_terminated(terminations)
        n_cycles += 1

    avg_rewards[i]= average(cumulative_rewards.values()) # type: ignore
    avg_cycles[i]= n_cycles
    std_rewards[i]= std_dev(cumulative_rewards, avg_rewards[i])
    print(f"Cycles = {n_cycles}", f"Avg = {round(avg_rewards[i],1)}", 
          f"Std = {round(std_rewards[i],1)}",end=" ")
    print()
pred_prey_grass_env.close()
print(f"Average of Avg = {round(average(avg_rewards),1)}")
print(f"Average of Cycles = {round(average(avg_cycles),1)}")
  