# implement the creation of predators and prey when above certain energy level
# implement the recreation of grass when eaten after certain time steps

# AEC pettingzoo predpreygrass environment using random policy
from environments.predpreygrass_create_agents import raw_env
from config.config_pettingzoo_create_agents import env_kwargs

import numpy as np
from pettingzoo.utils import agent_selector

num_games = 1
if num_games > 1: 
    env_kwargs["render_mode"]="None"
else:
    env_kwargs["render_mode"]="human"

raw_env = raw_env(**env_kwargs) 

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

avg_rewards = [0 for _ in range(num_games)]
avg_cycles = [0 for _ in range(num_games)]
std_rewards = [0 for _ in range(num_games)]

agent_selector = agent_selector(agent_order=raw_env.agents)

for i in range(num_games):
    raw_env.reset(seed=i)
    agent_selector.reset()
    cumulative_rewards = {agent: 0.0 for agent in raw_env.possible_agents}
    n_aec_cycles = 0
    for agent in raw_env.agent_iter():

        observation, reward, termination, truncation, info = raw_env.last()
        cumulative_rewards[agent] += reward
        if termination or truncation:
            action = None
        else:
            action = raw_env.action_space(agent).sample()
            #print(f"Agent {agent} steps, action =", end=" ")
            #print(action)
            """
            0: [-1, 0], # move left
            1: [0, -1], # move up
            2: [0, 0], # stay
            3: [0, 1], # move down
            4: [1, 0], # move right
            """
        #print()
        #print(f"Observation = ")
        #print(np.transpose(observation))
        raw_env.step(action)
        if agent_selector.is_last(): # called at end of cycle
            n_aec_cycles += 1
            #
            # ({key : round(cumulative_rewards[key], 2) for key in cumulative_rewards}) # DON'T REMOVE
            #print(f'Cumulative rewards = {cumulative_rewards}')


        agent_selector.next()   # called at end of cycle

    avg_rewards[i]= average(cumulative_rewards.values()) # type: ignore
    avg_cycles[i]= n_aec_cycles
    std_rewards[i]= std_dev(cumulative_rewards, avg_rewards[i])
    print(f"Cycles = {n_aec_cycles}", f"Avg = {round(avg_rewards[i],1)}", 
          f"Std = {round(std_rewards[i],1)}",end=" ")
    print()
raw_env.close()
print(f"Average of Avg = {round(average(avg_rewards),1)}")
print(f"Average of Cycles = {round(average(avg_cycles),1)}")
  