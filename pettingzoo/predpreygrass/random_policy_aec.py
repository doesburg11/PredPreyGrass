# AEC pettingzoo predpreygrass environment using random policy
from environments.predpreygrass_available_energy_transfer import raw_env
from config.config_pettingzoo import env_kwargs

from pettingzoo.utils import agent_selector

from statistics import mean, stdev

import numpy as np

num_episodes = 1
env_kwargs["render_mode"] = "human" if num_episodes == 1 else "None"

raw_env = raw_env(**env_kwargs)


avg_cum_rewards = [0 for _ in range(num_episodes)]
avg_cycles = [0 for _ in range(num_episodes)]
std_cum_rewards = [0 for _ in range(num_episodes)]

agent_selector = agent_selector(agent_order=raw_env.agents)

for i in range(num_episodes):
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
            """
            0: [-1, 0], # move left
            1: [0, -1], # move up
            2: [0, 0], # stay
            3: [0, 1], # move down
            4: [1, 0], # move right
            """
        raw_env.step(action)
        if agent_selector.is_last():  
            n_aec_cycles += 1
            # print({key : round(cumulative_rewards[key], 2) for key in cumulative_rewards}) # DON'T REMOVE
        agent_selector.next()

    avg_cum_rewards[i] = mean(cumulative_rewards.values())  # type: ignore
    avg_cycles[i] = n_aec_cycles
    std_cum_rewards[i] = stdev(cumulative_rewards.values())
    print(
        f"Cycles = {n_aec_cycles}",
        f"Avg = {round(avg_cum_rewards[i],1)}",
        f"Std = {round(std_cum_rewards[i],1)}",
        end=" ",
    )
    print()
raw_env.close()
print(f"Average of Avg = {round(mean(avg_cum_rewards),1)}")
print(f"Average of Cycles = {round(mean(avg_cycles),1)}")
