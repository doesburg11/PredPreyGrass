from predprey import raw_env
from pettingzoo.utils import agent_selector
import numpy as np


env_kwargs = dict(
    render_mode="human", 
    max_cycles=10000, 
    x_grid_size=16, 
    y_grid_size=16, 
    n_predator=6,
    n_prey=6,
    n_grass=30,
    max_observation_range=7, # must be odd
    obs_range_predator=5, # must be odd    
    obs_range_prey=7, # must be odd
    action_range=3, # must be odd
    moore_neighborhood_actions=False,
    energy_loss_per_step_predator = -0.3,
    energy_loss_per_step_prey = -0.1,     
    pixel_scale=40,
    initial_energy_predator = 12.0,
    initial_energy_prey = 8.0
)

num_games = 1
if num_games > 1:
    env_kwargs["render_mode"]="None"

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
        """
        prey_name_list = raw_env.pred_prey_env.create_agent_name_list_from_instance_list(
            raw_env.pred_prey_env.prey_instance_list

        )
        predator_name_list = raw_env.pred_prey_env.create_agent_name_list_from_instance_list(
            raw_env.pred_prey_env.predator_instance_list

        )
        agent_name_list = predator_name_list + prey_name_list
        print("agent_name_list ",agent_name_list)
        
        print(agent)
        """
        observation, reward, termination, truncation, info = raw_env.last()
        """
        print("observation predators")
        print(np.transpose(np.transpose(observation)[1]))
        print("observation prey")
        print(np.transpose(np.transpose(observation)[2]))
        print("observation grass")
        print(np.transpose(np.transpose(observation)[3]))
        print("reward")
        print(reward)
        print()
        """
        cumulative_rewards[agent] += reward
        if termination or truncation:
            action = None
        else:
            action = raw_env.action_space(agent).sample()
            #print(agent," takes action ", action)
        raw_env.step(action)
        if agent_selector.is_last(): # called at end of cycle
            n_aec_cycles += 1
            #
            # ({key : round(cumulative_rewards[key], 2) for key in cumulative_rewards}) # DON'T REMOVE

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
  