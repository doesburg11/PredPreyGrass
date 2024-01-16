import predprey
from parameters import env_kwargs, training_steps_string 

import glob
import os
import time
import sys
import shutil

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
        verbose=0, # 0 for no output, 1 for info messages, 2 for debug messages, 3 deafult
        batch_size=256,
        tensorboard_log=directory+"/ppo_predprey_tensorboard/"
    )

    model.learn(total_timesteps=steps,progress_bar=True)
    model.save(saved_directory_and_model_file_name)
    print("saved path: ",saved_directory_and_model_file_name)
    print("Model has been saved.")
    print(f"Finished training on {str(raw_parallel_env.unwrapped.metadata['name'])}.")

    raw_parallel_env.close()

def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):

    # Evaluate a trained agent vs a random agent
    raw_env = env_fn.raw_env(render_mode=render_mode, **env_kwargs)
    model = PPO.load(loaded_policy)
    if not save_model:
        shutil.rmtree(output_project)
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
    Average_of_Avg = round(average(avg_rewards),1)
    Average_of_Avg_predators = round(average(average_rewards_predator),1)
    Average_of_Avg_prey = round(average(average_rewards_prey),1)
    Average_of_Cycles = round(average(avg_cycles),1)
    return Average_of_Avg, Average_of_Avg_predators, Average_of_Avg_prey, Average_of_Cycles
    # end evaluation

if __name__ == "__main__":
    env_fn = predprey
    save_model = True # True saves the model and output externally
    watch_model = False
    #training_steps_string = "10_000_000"
    training_steps = int(training_steps_string)
    num_games = 100
    avg_rewards = [0 for _ in range(num_games)]
    avg_cycles = [0 for _ in range(num_games)]
    std_rewards = [0 for _ in range(num_games)]

    # global variables for evaluation
    average_rewards_predator = [0 for _ in range(num_games)]
    std_rewards_predator = [0 for _ in range(num_games)]
    average_rewards_prey = [0 for _ in range(num_games)]
    std_rewards_prey = [0 for _ in range(num_games)]
    Average_of_Avg = 0
    Average_of_Avg_predators = 0
    Average_of_Avg_prey = 0
    Average_of_Cycles = 0

    # output file name
    start_time = str(time.strftime('%Y-%m-%d_%H:%M'))
    environment_name = "predprey"
    file_name = f"{environment_name}_steps_{training_steps_string}"

    # Define the destination directory for the sourse code
    destination_directory_source_code = os.path.join('/home/doesburg/Dropbox/02_marl_results/predpreygras_results', start_time)
    output_project = destination_directory_source_code+"/output/"
    loaded_policy = destination_directory_source_code+"/output/"+file_name

    env_kwargs = dict(
        max_cycles=10000, 
        x_grid_size=16,
        y_grid_size=16, 
        n_initial_predator=6,
        n_initial_prey=8,
        n_initial_grass=30,
        max_observation_range=7, # must be odd and not smaller than any obs_range
        obs_range_predator=7, # must be odd    
        obs_range_prey=3, # must be odd
        action_range=3, # must be odd
        moore_neighborhood_actions=True,
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

    if save_model:
        # save the source code
        python_file_name = os.path.basename(sys.argv[0])
        python_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
        file_names_in_directory = os.listdir(python_directory)

        # Create the destination directory for the soursce code
        os.makedirs(destination_directory_source_code, exist_ok=True)

        # Copy all files and directories in the current directory to the new directory
        for item_name in file_names_in_directory:
            source_item = os.path.join(python_directory, item_name)
            destination_item = os.path.join(destination_directory_source_code, item_name)
            
            if os.path.isfile(source_item):
                shutil.copy2(source_item, destination_item)
            elif os.path.isdir(source_item):
                shutil.copytree(source_item, destination_item)

    if not save_model:
    # Save the trained model temporarily in repository
        output_project = "./trained_models/predprey/"+f"{environment_name}_{start_time}"
    directory = output_project
    os.makedirs(directory, exist_ok=True)
    saved_directory_and_model_file_name = os.path.join(directory, file_name)

    if save_model:
        #save parameters to file
        saved_directory_and_parameter_file_name = os.path.join(directory, "parameters.txt")
        file = open(saved_directory_and_parameter_file_name, "w")
        file.write("model: PredPreyGrass\n")
        file.write("parameters:\n")
        file.write("training steps: "+training_steps_string+"\n")
        file.write("=========================\n")
        for item in env_kwargs:
            file.write(str(item)+" = "+str(env_kwargs[item])+"\n")
        file.write("=========================\n")
        start_training_time = time.time()
        train(env_fn, steps=training_steps, seed=0, **env_kwargs)
        end_training_time = time.time()
        training_time = end_training_time - start_training_time
        if training_time<3600:
            file.write("training time (min): " + str(round(training_time/60,1))+"\n")
        else:
            file.write("training time (hours): " + str(round(training_time/3600,1))+"\n")

    # load latest policy
    if not save_model:
        train(env_fn, steps=training_steps, seed=0, **env_kwargs)
        loaded_policy = max(
            glob.glob(os.path.join(directory,f"{environment_name}*.zip")), key=os.path.getctime
            )
            
    if watch_model:
        # Evaluate and watch games
        Average_of_Avg, Average_of_Avg_predators, Average_of_Avg_prey, Average_of_Cycles = eval(
            env_fn, 
            num_games=num_games, 
            render_mode="human", 
            **env_kwargs
            )
    else:
        # Evaluate games 
        Average_of_Avg, Average_of_Avg_predators, Average_of_Avg_prey, Average_of_Cycles = eval(
            env_fn, 
            num_games=num_games, 
            render_mode=None, 
            **env_kwargs
            )
    

    if save_model:
        file.write("=========================\n")
        file.write(f"Number of games = {num_games}"+"\n")
        file.write(f"Average of Avg = {Average_of_Avg}"+"\n")
        file.write(f"Average of Avg_predators = {Average_of_Avg_predators}"+"\n")
        file.write(f"Average of Avg_prey = {Average_of_Avg_prey}"+"\n")
        file.write(f"Average of Cycles = {Average_of_Cycles}"+"\n")
        file.close()
        print(f"Average of Avg = {Average_of_Avg}")
        print(f"Average of Avg_predators = {Average_of_Avg_predators}")
        print(f"Average of Avg_prey = {Average_of_Avg_prey}")
        print(f"Average of Cycles = {Average_of_Cycles}")
    else:
        file.close()
        print(f"Average of Avg = {Average_of_Avg}")
        print(f"Average of Avg_predators = {Average_of_Avg_predators}")
        print(f"Average of Avg_prey = {Average_of_Avg_prey}")
        print(f"Average of Cycles = {Average_of_Cycles}")





