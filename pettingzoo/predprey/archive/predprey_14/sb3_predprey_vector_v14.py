import predprey

import glob
import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from pettingzoo.utils.conversions import parallel_wrapper_fn

def train(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):

    parallel_env = parallel_wrapper_fn(env_fn.raw_env)
       
    # Train a single model to play as each agent in a parallel environment
    raw_parallel_env = parallel_env(**env_kwargs)
    raw_parallel_env.reset(seed=seed)

    print(f"Starting training on {str(raw_parallel_env.metadata['name'])}.")

    raw_parallel_env = ss.pettingzoo_env_to_vec_env_v1(raw_parallel_env)
    raw_parallel_env = ss.concat_vec_envs_v1(raw_parallel_env, 8, num_cpus=8, base_class="stable_baselines3")

    model = PPO(
        MlpPolicy,
        raw_parallel_env,
        verbose=3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)
    model.save(saved_directory_and_model_file_name)
    print("saved path: ",saved_directory_and_model_file_name)
    print("Model has been saved.")
    print(f"Finished training on {str(raw_parallel_env.unwrapped.metadata['name'])}.")

    raw_parallel_env.close()

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

    avg_rewards1 = [0 for _ in range(num_games)]
    std_rewards1 = [0 for _ in range(num_games)]
    avg_rewards2 = [0 for _ in range(num_games)]
    std_rewards2 = [0 for _ in range(num_games)]

    from pettingzoo.utils import agent_selector # on top of file gives error unbound
    agent_selector = agent_selector(agent_order=raw_env.agents)

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        raw_env.reset(seed=i)
        prey1_name_list = raw_env.pred_prey_env.prey1_name_list
        prey2_name_list = raw_env.pred_prey_env.prey2_name_list
        agent_selector.reset()
        cumulative_rewards = {agent: 0 for agent in raw_env.possible_agents}
        cumulative_rewards1 = {agent: 0 for agent in prey1_name_list}
        cumulative_rewards2 = {agent: 0 for agent in prey2_name_list}
        n_cycles = 0
        for agent in raw_env.agent_iter():
            observation, reward, termination, truncation, info = raw_env.last()
            cumulative_rewards[agent] += reward
            if agent in prey1_name_list:
                cumulative_rewards1[agent] += reward
            elif agent in prey2_name_list:
                cumulative_rewards2[agent] += reward
            if termination or truncation:
                break
            else:
                action = model.predict(observation, deterministic=False)[0]
            raw_env.step(action)
            if agent_selector.is_last(): # called at end of cycle
                n_cycles += 1
                #print("cycle ",n_cycles," ",{key : round(cumulative_rewards[key], 1) for key in cumulative_rewards})
            agent_selector.next()   # called at end of cycle
        #print("cumulative_rewards.values() ",cumulative_rewards.values())
        avg_rewards[i]= average(cumulative_rewards.values()) # type: ignore
        avg_cycles[i]= n_cycles
        std_rewards[i]= std_dev(cumulative_rewards, avg_rewards[i])
        avg_rewards1[i]= average(cumulative_rewards1.values()) # type: ignore
        std_rewards1[i]= std_dev(cumulative_rewards1, avg_rewards1[i])
        avg_rewards2[i]= average(cumulative_rewards2.values()) # type: ignore
        std_rewards2[i]= std_dev(cumulative_rewards2, avg_rewards2[i])
        print(f"Cycles = {n_cycles}", f"Avg = {round(avg_rewards[i],1)}", f"Std = {round(std_rewards[i],1)}",end=" ")
        print()
    raw_env.close()
    print(f"Average of Avg = {round(average(avg_rewards),1)}")
    print(f"Average of Avg1 = {round(average(avg_rewards1),1)}")
    print(f"Average of Avg2 = {round(average(avg_rewards2),1)}")
    print(f"Average of Cycles = {round(average(avg_cycles),1)}")
    # end evaluation

if __name__ == "__main__":
    env_fn = predprey

    train_model = True  # True evaluates latest policy, False evaluates at predefined loaded policy
    eval_model = True
    eval_and_watch_model = True
    training_steps_string = "1_000_000"
    training_steps = int(training_steps_string)
    #loaded_policy = "./trained_models/pursuit/conclusions/1/pursuit_2023-10-24 11:04/pursuit_steps_5_000_000.zip"
    loaded_policy = "./trained_models/predprey/predprey_2023-12-06 22:42/predprey_steps_5_000_000"
    #loaded_policy = "./trained_models/predprey/conclusions/freeze_grasses/steps_1_000_000/predprey_steps_1_000_000"
    #loaded_policy = "./trained_models/predprey/conclusions/moore/time_step_vatiation/50_000_000/predprey_steps_50_000_000"
    env_kwargs = dict(
        max_cycles=100000, 
        x_size=16, 
        y_size=16, 
        n_grasses=30,
        n_prey1=4,
        n_prey2=4,
        max_observation_range=7, # influences number of calculations; make as small as possible
        obs_range_prey1=7,     
        obs_range_prey2=3, 
        freeze_grasses=True,
        catch_reward=5.0,
        urgency_reward=-0.1,
        moore_neighborhood_prey=False,
        moore_neighborhood_grasses=False,
        pixel_scale=40
        )

    if train_model:
        # Save the trained model in specified directory
        start_time = str(time.strftime('%Y-%m-%d %H:%M'))
        environment_name = "predprey"
        file_name = f"{environment_name}_steps_{training_steps_string}"
        directory = "./trained_models/predprey/"+f"{environment_name}_{start_time}"
        os.makedirs(directory, exist_ok=True)
        saved_directory_and_model_file_name = os.path.join(directory, file_name)

        #save parameters to file
        saved_directory_and_parameter_file_name = os.path.join(directory, "parameters.txt")
        file = open(saved_directory_and_parameter_file_name, "w")
        file.write("parameters:\n")
        file.write("=========================\n")
        for item in env_kwargs:
            file.write(str(item)+" = "+str(env_kwargs[item])+"\n")
        file.write("=========================\n")
        file.close()

        train(env_fn, steps=training_steps, seed=0, **env_kwargs)
        # load latest policy
        loaded_policy = max(
            glob.glob(os.path.join(directory,f"{environment_name}*.zip")), key=os.path.getctime
            )
            
    if eval_model:
        # Evaluate games 
        eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    if eval_and_watch_model:
        # Evaluate and watch games
        eval(env_fn, num_games=1, render_mode="human", **env_kwargs)
