import predprey

import os

import supersuit as ss
from stable_baselines3 import PPO


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):

    # Evaluate a trained agent vs a random agent
    raw_env = env_fn.raw_env(render_mode=render_mode, **env_kwargs)
    model = PPO.load(loaded_policy)
    cumulative_rewards = {agent: 0 for agent in raw_env.possible_agents}
    
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

    average_rewards_predator = [0 for _ in range(num_games)]
    std_rewards_predator = [0 for _ in range(num_games)]
    average_rewards_prey = [0 for _ in range(num_games)]
    std_rewards_prey = [0 for _ in range(num_games)]

    from pettingzoo.utils import agent_selector # on top of file gives error unbound(?)
    agent_selector = agent_selector(agent_order=raw_env.agents)


    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent

    for i in range(num_games):
        n_aec_cycles = 0
        raw_env.reset()
        raw_env._agent_selector.reset()
        predator_name_list = raw_env.pred_prey_env.predator_name_list
        prey_name_list = raw_env.pred_prey_env.prey_name_list
        agent_name_list = raw_env.pred_prey_env.agent_name_list
        agent_selector.reset()

        cumulative_rewards = {agent: 0 for agent in agent_name_list}
        cumulative_rewards_predator = {agent: 0 for agent in predator_name_list}
        cumulative_rewards_prey = {agent: 0 for agent in prey_name_list}

        for agent in raw_env.agent_iter():
            observation, reward, termination, truncation, info = raw_env.last()

            cumulative_rewards[agent] += reward
            if agent in predator_name_list:
                cumulative_rewards_predator[agent] += reward
            elif agent in prey_name_list:
                cumulative_rewards_prey[agent] += reward

            if termination or truncation:
                action = None
            else:
                action = model.predict(observation, deterministic=False)[0]
            raw_env.step(action)
            if agent_selector.is_last(): # called at end of cycle
                n_aec_cycles += 1
                #print({key : round(cumulative_rewards[key], 2) for key in cumulative_rewards}) # DON'T REMOVE
            agent_selector.next()   # called at end of cycle

        avg_rewards[i]= average(cumulative_rewards.values()) # type: ignore
        avg_cycles[i]= raw_env.pred_prey_env.n_aec_cycles
        std_rewards[i]= std_dev(cumulative_rewards, avg_rewards[i])
        average_rewards_predator[i]= average(cumulative_rewards_predator.values()) # type: ignore
        std_rewards_predator[i]= std_dev(cumulative_rewards_predator, average_rewards_predator[i])
        average_rewards_prey[i]= average(cumulative_rewards_prey.values()) # type: ignore
        std_rewards_prey[i]= std_dev(cumulative_rewards_prey, average_rewards_prey[i])
        print(f"Cycles = {raw_env.pred_prey_env.n_aec_cycles}", f"Avg = {round(avg_rewards[i],1)}", 
              f"Std = {round(std_rewards[i],1)}",end=" ")
        print()
    raw_env.close()
    print(f"Average of Avg = {round(average(avg_rewards),1)}")
    print(f"Average of Avg_predators = {round(average(average_rewards_predator),1)}")
    print(f"Average of Avg_prey = {round(average(average_rewards_prey),1)}")
    print(f"Average of Cycles = {round(average(avg_cycles),1)}")
    # end evaluation

if __name__ == "__main__":
    env_fn = predprey

    eval_model = False
    eval_and_watch_model = True
    training_steps_string = "10_000_000"
    training_steps = int(training_steps_string)

    # output file name
    start_time = "2024-01-15_17:37"
    environment_name = "predprey"
    file_name = f"{environment_name}_steps_{training_steps_string}"

    # Define the destination directory for the sourse code
    destination_directory_source_code = os.path.join('/home/doesburg/Dropbox/02_marl_results/predpreygras_results', start_time)
    loaded_policy = destination_directory_source_code +"/output/"+file_name

    env_kwargs = dict(
        max_cycles=10000, 
        x_grid_size=16,
        y_grid_size=16, 
        n_initial_predator=6,
        n_initial_prey=8,
        n_initial_grass=30,
        max_observation_range=7, # must be odd and not smaller than any obs_range
        obs_range_predator=5, # must be odd    
        obs_range_prey=7, # must be odd
        action_range=3, # must be odd
        moore_neighborhood_actions=False,
        energy_loss_per_step_predator = -0.1,
        energy_loss_per_step_prey = -0.05,     
        initial_energy_predator = 5.0,
        initial_energy_prey = 5.0,  
        catch_grass_reward = 3.0,
        catch_prey_reward = 3.0,      
        # visualization parameters
        cell_scale=40,
        x_pygame_window=0,
        y_pygame_window=0,
    )
         
    if eval_model:
        # Evaluate games 
        eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    if eval_and_watch_model:
        # Evaluate and watch games
        eval(env_fn, num_games=5, render_mode="human", **env_kwargs)
