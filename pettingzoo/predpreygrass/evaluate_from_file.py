"""
- navigate with linux mint file-explorer to your defined local directory in 
 "config/config_pettingzoo.py".
- note that all necessary files/directories are copied to the defined local directory
- go to the directory with the appropriate time stamp of training
- right mouseclick "evaluate_from_file.py" and:
- select "Open with"
- select "Visual Studio Code" (or default VS Code for .py files)
- select "Run" (in taskbar of Visual Studio Code)
- select "Run without debugging"
- note that ajusting the configuration of the trained model is done 
  in the defined local directory (and not in your cloned directory!)
"""
# Agent Enironment Cycle (AEC) pettingzoo predpreygrass environment
import environments.predpreygrass as predpreygrass
# make sure this configuration is consistent with the training configuration in "train_sb3_vector_ppo.py"
from config.config_pettingzoo import env_kwargs, training_steps_string, local_output_directory

# displaying the population of predators and prey
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator # for integer ticks

import os
from statistics import mean, stdev

import supersuit as ss
from stable_baselines3 import PPO


def eval(env_fn, num_episodes: int = 100, render_mode: str | None = None, **env_kwargs):

    # Evaluate a trained agent vs a random agent
    raw_env = env_fn.raw_env(render_mode=render_mode, **env_kwargs)
    model = PPO.load(loaded_policy)
    cumulative_rewards = {agent: 0 for agent in raw_env.possible_agents}
    

    from pettingzoo.utils import agent_selector # on top of file gives error unbound(?)
    agent_selector = agent_selector(agent_order=raw_env.agents)


    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    print("Start evaluation.")
    for i in range(num_episodes):
        raw_env.reset()
        agent_selector.reset()
        raw_env._agent_selector.reset()
        predator_name_list = raw_env.pred_prey_env.predator_name_list
        prey_name_list = raw_env.pred_prey_env.prey_name_list
        agent_name_list = raw_env.pred_prey_env.agent_name_list
        cumulative_rewards = {agent: 0 for agent in agent_name_list}
        cumulative_rewards_predator = {agent: 0 for agent in predator_name_list}
        cumulative_rewards_prey = {agent: 0 for agent in prey_name_list}
        n_aec_cycles = 0
        for agent in raw_env.agent_iter():
            observation, reward, termination, truncation, info = raw_env.last()
            cumulative_rewards[agent] += reward
            if agent in predator_name_list:
                cumulative_rewards_predator[agent] += reward
            elif agent in prey_name_list:
                cumulative_rewards_prey[agent] += reward

            if termination or truncation:
                action = None
                if raw_env.pred_prey_env.is_no_predator:
                    predator_extinct_at_termination[i] = 1
            else:
                action = model.predict(observation, deterministic=False)[0]
            raw_env.step(action)
            if agent_selector.is_last(): # called at end of cycle
                n_aec_cycles += 1
                #print({key : round(cumulative_rewards[key], 2) for key in cumulative_rewards}) # DON'T REMOVE
            agent_selector.next()   # called at end of cycle

        # plot population of Predators and Prey
        plt.clf()
        plt.plot(raw_env.pred_prey_env.n_active_predator_list, 'r')
        plt.plot(raw_env.pred_prey_env.n_active_prey_list, 'b')
        plt.title('Predator and Prey Population', weight='bold')
        plt.xlabel('Time steps', weight='bold')
        ax = plt.gca()
        # Set x and y limits
        ax.set_xlim([0, raw_env.pred_prey_env.n_aec_cycles])  
        ax.set_ylim([0, max(raw_env.pred_prey_env.n_active_predator_list + raw_env.pred_prey_env.n_active_prey_list)])  
        # Display only whole numbers on the y-axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # Remove box/spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # Make axes thicker
        plt.axhline(0, color='black', linewidth=4)
        plt.axvline(0, color='black', linewidth=4)
        # Make tick marks thicker
        plt.tick_params(width=2)


        population_dir = output_directory + 'population_charts/'
        os.makedirs(population_dir, exist_ok=True) 
        file_name = population_dir+'/PredPreyPopulation_episode_'+str(i)+'.pdf'
        plt.savefig(file_name)


        episode_length[i]= n_aec_cycles
        mean_cumulative_rewards[i]= mean(cumulative_rewards.values())
        mean_cumulative_rewards_predator[i]= mean(cumulative_rewards_predator.values())
        mean_cumulative_rewards_prey[i]= mean(cumulative_rewards_prey.values())
        std_cumulative_rewards[i]= stdev(cumulative_rewards.values())
        std_cumulative_rewards_predator[i]= stdev(cumulative_rewards_predator.values())
        std_cumulative_rewards_prey[i]= stdev(cumulative_rewards_prey.values())
        print(f"Episode {i}", f"Episode length = {n_aec_cycles}", f"Mean cumulative rewards agents = {round(mean_cumulative_rewards[i],1)}", 
            f"(Std = {round(std_cumulative_rewards[i],1)})")
        if eval_model_only:
            file.write(f"Episode {i}, Episode length = {n_aec_cycles}, Mean cumulative rewards agents = {round(mean_cumulative_rewards[i],1)}, (Std = {round(std_cumulative_rewards[i],1)})\n")
    print("Finish evaluation.")
    raw_env.close()
    predator_extinct_at_termination_count = sum(predator_extinct_at_termination)
    episode_mean_of_mean_cumulative_rewards = round(mean(mean_cumulative_rewards),1)
    episode_mean_of_mean_cumulative_rewards_predators = round(mean(mean_cumulative_rewards_predator),1)
    episode_mean_of_mean_cumulative_rewards_prey = round(mean(mean_cumulative_rewards_prey),1)
    mean_episode_length = round(mean(episode_length),1)
    std_episode_length = round(stdev(episode_length),1)
    return episode_mean_of_mean_cumulative_rewards, episode_mean_of_mean_cumulative_rewards_predators, episode_mean_of_mean_cumulative_rewards_prey, mean_episode_length, std_episode_length, predator_extinct_at_termination_count
    # end evaluation

if __name__ == "__main__":
    env_fn = predpreygrass
    model_file_name = "predprey_steps_"+training_steps_string
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_directory = script_directory+"/output/"
    loaded_policy = output_directory + model_file_name
    eval_model_only = True
    watch_grid_model = not eval_model_only
    #save parameters to file
    if eval_model_only:
        saved_directory_and_evaluation_file_name = os.path.join(output_directory, "evaluation.txt")
        file = open(saved_directory_and_evaluation_file_name, "w")
        file.write("model: PredPreyGrass\n")
        file.write("evaluation:\n")
        file.write("training steps: "+training_steps_string+"\n")
        file.write("--------------------------\n")
    print()
    print("loaded_policy:",loaded_policy)
    print()
    print("model: PredPreyGrass")
    print("evaluation:")
    print("training steps: "+training_steps_string)
    print("--------------------------")

    eval_model_only = True
    watch_grid_model = not eval_model_only
    training_steps = int(training_steps_string)
    num_episodes = 100

    # global variables for evaluation used in eval function
    episode_length = [0 for _ in range(num_episodes)]
    predator_extinct_at_termination = [0 for _ in range(num_episodes)]
    mean_cumulative_rewards = [0 for _ in range(num_episodes)]
    mean_cumulative_rewards_predator = [0 for _ in range(num_episodes)]
    mean_cumulative_rewards_prey = [0 for _ in range(num_episodes)]
    std_cumulative_rewards = [0 for _ in range(num_episodes)]
    std_cumulative_rewards_predator = [0 for _ in range(num_episodes)]
    std_cumulative_rewards_prey = [0 for _ in range(num_episodes)]
    episode_mean_of_mean_cumulative_rewards = 0
    episode_mean_of_mean_cumulative_rewards_predators = 0
    episode_mean_of_mean_cumulative_rewards_prey = 0
    predator_extinct_at_termination_count = 0
    mean_episode_length = 0 
    std_episode_length = 0

      
    if eval_model_only:
        # Evaluate episodes 
        episode_mean_of_mean_cumulative_rewards, episode_mean_of_mean_cumulative_rewards_predators, episode_mean_of_mean_cumulative_rewards_prey, mean_episode_length, std_episode_length, predator_extinct_at_termination_count = eval(
            env_fn, 
            num_episodes=num_episodes, 
            render_mode=None, 
            **env_kwargs
        )
        # save evaluation results to file
        file.write("--------------------------\n")
        file.write(f"Number of episodes = {num_episodes}"+"\n")
        file.write(f"Mean episode length = {mean_episode_length}"+"\n")
        file.write(f"Standard deviation episode length = {std_episode_length}"+"\n")
        file.write(f"Per episode mean of per agent mean cumulative reward = {episode_mean_of_mean_cumulative_rewards}"+"\n")
        file.write(f"Per episode mean of per Predator mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_predators}"+"\n")
        file.write(f"Per episode mean of per Prey mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_prey}"+"\n")
        file.write(f"% Predator extinct at termination = {round(predator_extinct_at_termination_count/num_episodes*100,1)}"+"\n")
        file.write("--------------------------\n")
        file.write("Evaluation parameters:\n")
        for item in env_kwargs:
            file.write(str(item)+" = "+str(env_kwargs[item])+"\n")
        file.write("--------------------------\n")
        file.close()
        # and print to console
        print("--------------------------")
        print(f"Number of episodes = {num_episodes}")
        print(f"Mean episode length = {mean_episode_length}")
        print(f"Standard deviation episode length = {std_episode_length}")
        print(f"Per episode mean of per agent mean cumulative reward = {episode_mean_of_mean_cumulative_rewards}")
        print(f"Per episode mean of per Predator mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_predators}")
        print(f"Per episode mean of per Prey mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_prey}")
        print(f"% Predator extinct at termination = {round(predator_extinct_at_termination_count/num_episodes*100,1)}")


    if watch_grid_model:
        # Evaluate and watch games
        episode_mean_of_mean_cumulative_rewards, episode_mean_of_mean_cumulative_rewards_predators, episode_mean_of_mean_cumulative_rewards_prey, mean_episode_length, std_episode_length, predator_extinct_at_termination_count = eval(
            env_fn, 
            num_episodes=5, 
            render_mode="human", 
            **env_kwargs
        )
        # print to console
        print("--------------------------")
        print(f"Number of episodes = {num_episodes}")
        print(f"Mean episode length = {mean_episode_length}")
        print(f"Standard deviation episode length = {std_episode_length}")
        print(f"Per episode mean of per agent mean cumulative reward = {episode_mean_of_mean_cumulative_rewards}")
        print(f"Per episode mean of per Predator mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_predators}")
        print(f"Per episode mean of per Prey mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_prey}")
