# AEC pettingzoo predpreygrass environment using random policy
from predpreygrass.envs import so_predpreygrass_v0
from predpreygrass.envs._so_predpreygrass_v0.config.so_config_predpreygrass import (
    env_kwargs,
    local_output_directory,
)


# displaying the population of predators and prey
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator  # for integer ticks

import os
import time
import sys
import shutil
from os.path import dirname as up

from statistics import mean, stdev

import numpy as np

# evaluation options
WATCH_GRID_MODEL = False # if false only evaluation is done
NUM_EPISODES = 100

num_episodes = NUM_EPISODES
env_kwargs["render_mode"] = "human" if WATCH_GRID_MODEL else "None"

def eval(env_fn, num_episodes: int = 100, render_mode: str | None = None, **env_kwargs):

    raw_env = env_fn.raw_env(render_mode=render_mode, **env_kwargs)
    cumulative_rewards = {agent: 0 for agent in raw_env.possible_agents}

    from pettingzoo.utils import agent_selector  # on top of file gives error unbound(?)

    agent_selector = agent_selector(agent_order=raw_env.agents)

    print("Start evaluation.")
    if parameter_variation:
        print(
            "Parameter variation " + parameter_variation_parameter_string + ": ",
            env_kwargs[parameter_variation_parameter_string],
        )
    print("--------------------------")

    # age lists over all episodes
    total_predator_age_list = []
    total_prey_age_list = []

    for i in range(num_episodes):
        raw_env.reset()
        agent_selector.reset()
        raw_env._agent_selector.reset()
        predator_name_list = raw_env._env.possible_predator_name_list
        prey_name_list = raw_env._env.possible_prey_name_list
        agent_name_list = raw_env._env.possible_agent_name_list
        cumulative_rewards = {agent: 0 for agent in agent_name_list}
        cumulative_rewards_predator = {agent: 0 for agent in predator_name_list}
        cumulative_rewards_prey = {agent: 0 for agent in prey_name_list}
        n_aec_cycles = 0
        for agent in raw_env.agent_iter():
            _ , reward, termination, truncation, _ = raw_env.last()
            cumulative_rewards[agent] += reward
            if agent in predator_name_list:
                cumulative_rewards_predator[agent] += reward
            elif agent in prey_name_list:
                cumulative_rewards_prey[agent] += reward

            if termination or truncation:
                action = None
                if raw_env._env.is_no_predator:
                    predator_extinct_at_termination[i] = 1
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

        # plot population of Predators and Prey
        plt.clf()
        plt.plot(raw_env._env.n_active_predator_list, "r")
        plt.plot(raw_env._env.n_active_prey_list, "b")
        plt.title("Predator and Prey Population", weight="bold")
        plt.xlabel("Time steps", weight="bold")
        ax = plt.gca()
        # Set x and y limits
        ax.set_xlim([0, raw_env._env.n_aec_cycles])
        ax.set_ylim(
            [
                0,
                max(
                    raw_env._env.n_active_predator_list
                    + raw_env._env.n_active_prey_list
                ),
            ]
        )
        # Display only whole numbers on the y-axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # Remove box/spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # Make axes thicker
        plt.axhline(0, color="black", linewidth=4)
        plt.axvline(0, color="black", linewidth=4)
        # Make tick marks thicker
        plt.tick_params(width=2)

        population_dir = output_directory + "population_charts/"
        os.makedirs(population_dir, exist_ok=True)
        file_name = population_dir + "/PredPreyPopulation_episode_" + str(i) + ".pdf"
        plt.savefig(file_name)


        # plot toal energy of Predator agents, Prey agents and Grass agents
        plt.clf()
        plt.plot(raw_env._env.total_energy_predator_list, "r")
        plt.plot(raw_env._env.total_energy_prey_list, "b")
        plt.plot(raw_env._env.total_energy_grass_list, "g")
        plt.plot(raw_env._env.total_energy_learning_agents_list, "k")
        plt.title("Total energy", weight="bold")
        plt.xlabel("Time steps", weight="bold")
        ax = plt.gca()
        # Set x and y limits
        ax.set_xlim([0, raw_env._env.n_aec_cycles])
        ax.set_ylim(
            [
                0,
                max(
                    raw_env._env.total_energy_predator_list
                    + raw_env._env.total_energy_prey_list
                    + raw_env._env.total_energy_grass_list
                    + raw_env._env.total_energy_learning_agents_list
                ),
            ]
        )
        # Display only whole numbers on the y-axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # Remove box/spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # Make axes thicker
        plt.axhline(0, color="black", linewidth=4)
        plt.axvline(0, color="black", linewidth=4)
        # Make tick marks thicker
        plt.tick_params(width=2)

        total_energy_dir = output_directory + "total_energy_charts/"
        os.makedirs(total_energy_dir, exist_ok=True)
        file_name = total_energy_dir + "/PredPreyGrassTotalEnergy_episode_" + str(i) + ".pdf"
        plt.savefig(file_name)


        episode_length[i] = n_aec_cycles
        n_starved_predator_per_cycle[i] = (
            raw_env._env.n_starved_predator / n_aec_cycles
        )
        n_starved_prey_per_cycle[i] = (
            raw_env._env.n_starved_prey / n_aec_cycles
        )
        n_eaten_prey_per_cycle[i] = raw_env._env.n_eaten_prey / n_aec_cycles
        n_eaten_grass_per_cycle[i] = raw_env._env.n_eaten_grass / n_aec_cycles
        n_born_predator_per_cycle[i] = (
            raw_env._env.n_born_predator / n_aec_cycles
        )
        n_born_prey_per_cycle[i] = raw_env._env.n_born_prey / n_aec_cycles
        episode_predator_age_list = raw_env._env.predator_age_list
        episode_prey_age_list = raw_env._env.prey_age_list
        mean_age_predator[i] = (
            mean(episode_predator_age_list) if episode_predator_age_list else 0
        )
        mean_age_prey[i] = mean(episode_prey_age_list) if episode_prey_age_list else 0
        total_predator_age_list += episode_predator_age_list
        total_prey_age_list += episode_prey_age_list
        mean_cumulative_rewards[i] = mean(cumulative_rewards.values())
        mean_cumulative_rewards_predator[i] = mean(cumulative_rewards_predator.values())
        mean_cumulative_rewards_prey[i] = mean(cumulative_rewards_prey.values())
        std_cumulative_rewards[i] = stdev(cumulative_rewards.values())
        std_cumulative_rewards_predator[i] = stdev(cumulative_rewards_predator.values())
        std_cumulative_rewards_prey[i] = stdev(cumulative_rewards_prey.values())
        print(
            f"Eps {i}",
            f"Lngth = {n_aec_cycles}",
            f"Strv Prd/cycl = {round(n_starved_predator_per_cycle[i],3)}",
            f"Strv Pry/cycl = {round(n_starved_prey_per_cycle[i],3)}",
            f"Eatn Pry/cycl = {round(n_eaten_prey_per_cycle[i],3)}",
            f"Eatn Gra/cycl = {round(n_eaten_grass_per_cycle[i],3)}",
            f"Brn Prd/cycl = {round(n_born_predator_per_cycle[i],3)}",
            f"Brn Pry/cycle = {round(n_born_prey_per_cycle[i],3)}",
            f"Mn age Prd = {round(mean_age_predator[i],1)}",
            f"Mn age Pry = {round(mean_age_prey[i],1)}",
        )
        if eval_model_only:
            file.write(f"Eps {i} ")
            file.write(f"Lngth = {n_aec_cycles} ")
            file.write(f"Strv Prd/cycl = {round(n_starved_predator_per_cycle[i],3)} ")
            file.write(f"Strv Pry/cycl = {round(n_starved_prey_per_cycle[i],3)} ")
            file.write(f"Eeatn Pry/cycl = {round(n_eaten_prey_per_cycle[i],3)} ")
            file.write(f"Eeatn Gra/cycl = {round(n_eaten_grass_per_cycle[i],3)} ")
            file.write(f"Brn Prd/cycl = {round(n_born_predator_per_cycle[i],3)} ")
            file.write(f"Brn Pry/cycl = {round(n_born_prey_per_cycle[i],3)} ")
            file.write(f"Mn age Prd = {round(mean_age_predator[i],1)} ")
            file.write(f"Mn age Pry = {round(mean_age_prey[i],1)}\n")
    print("--------------------------")
    print("Finish evaluation.")
    raw_env.close()
    predator_extinct_at_termination_count = sum(predator_extinct_at_termination)
    episode_mean_of_mean_cumulative_rewards = round(mean(mean_cumulative_rewards), 1)
    episode_mean_of_mean_cumulative_rewards_predators = round(
        mean(mean_cumulative_rewards_predator), 1
    )
    episode_mean_of_mean_cumulative_rewards_prey = round(
        mean(mean_cumulative_rewards_prey), 1
    )
    mean_episode_length = round(mean(episode_length), 1)
    std_episode_length = round(stdev(episode_length), 1)
    episode_mean_of_n_starved_predator_per_cycle = round(
        mean(n_starved_predator_per_cycle), 3
    )
    episode_mean_of_n_starved_prey_per_cycle = round(mean(n_starved_prey_per_cycle), 3)
    episode_mean_of_n_eaten_prey_per_cycle = round(mean(n_eaten_prey_per_cycle), 3)
    episode_mean_of_n_eaten_grass_per_cycle = round(mean(n_eaten_grass_per_cycle), 3)
    episode_mean_of_n_born_predator_per_cycle = round(
        mean(n_born_predator_per_cycle), 3
    )
    episode_mean_of_n_born_prey_per_cycle = round(mean(n_born_prey_per_cycle), 3)
    episode_mean_of_mean_age_predator = round(
        sum(total_predator_age_list) / len(total_predator_age_list), 1
    )
    episode_mean_of_mean_age_prey = round(
        sum(total_prey_age_list) / len(total_prey_age_list), 1
    )
    return (
        episode_mean_of_mean_cumulative_rewards,
        episode_mean_of_mean_cumulative_rewards_predators,
        episode_mean_of_mean_cumulative_rewards_prey,
        mean_episode_length,
        std_episode_length,
        predator_extinct_at_termination_count,
        episode_mean_of_n_starved_predator_per_cycle,
        episode_mean_of_n_starved_prey_per_cycle,
        episode_mean_of_n_eaten_prey_per_cycle,
        episode_mean_of_n_eaten_grass_per_cycle,
        episode_mean_of_n_born_predator_per_cycle,
        episode_mean_of_n_born_prey_per_cycle,
        episode_mean_of_mean_age_predator,
        episode_mean_of_mean_age_prey,
    )
    # end evaluation


if __name__ == "__main__":
    # Train
    env_fn = so_predpreygrass_v0
    policy = "random"
    environment_name = "predpreygrass_v0"
    parameter_variation = True
    parameter_variation_parameter_string = "predator_creation_energy_threshold"
    if parameter_variation:
        parameter_variation_scenarios = [10]
    else:
        parameter_variation_scenarios = [
            env_kwargs[parameter_variation_parameter_string]
        ]  # default value, must be iterable
    # output file name
    start_time = str(time.strftime("%Y-%m-%d_%H:%M:%S"))  # add seconds
    file_name = f"{environment_name}_{policy}"
    if parameter_variation:
        root_destination_directory_source_code = (
            local_output_directory + "parameter_variation/" + parameter_variation_parameter_string + "_" + start_time
        )
    else:
        root_destination_directory_source_code = local_output_directory

    for parameter_variation_parameter in parameter_variation_scenarios:
        if parameter_variation:
            env_kwargs[
                parameter_variation_parameter_string
            ] = parameter_variation_parameter
            # define the destination directory for the source code
            destination_directory_source_code = (
                root_destination_directory_source_code
                + "/"
                + str(parameter_variation_parameter) 
            )
            output_directory = destination_directory_source_code + "/output/"
            print("output_directory:", output_directory)
        else:
            # define the destination directory for the source code
            destination_directory_source_code = os.path.join(
                local_output_directory, start_time
            )
            output_directory = destination_directory_source_code + "/output/"
            print("output_directory:", output_directory)

        # save the source code locally
        project_directory = up(up(up(up(__file__)))) # up 4 levels in directory tree
        # copy the project code to the local directory
        shutil.copytree(project_directory, destination_directory_source_code)
        # Create the output directory
        output_directory = destination_directory_source_code + "/output/"
        os.makedirs(output_directory, exist_ok=True)
 
       
        # Copy all files and directories in the current directory to the local directory
        # for safekeeping experiment scenarios
        for item_name in file_names_in_directory:
            source_item = os.path.join(python_directory, item_name)
            destination_item = os.path.join(
                destination_directory_source_code, item_name
            )

            if os.path.isfile(source_item):
                shutil.copy2(source_item, destination_item)
            elif os.path.isdir(source_item):
                shutil.copytree(source_item, destination_item)

        if parameter_variation:
            # overwrite config file locally
            # Start of the code string
            code = "local_output_directory = '{}'\n".format(local_output_directory)
            code += "training_steps_string = 0\n"
            code += "env_kwargs = dict(\n"
            # Add each item from env_kwargs to the code string
            for key, value in env_kwargs.items():
                code += f"    {key}={value},\n"

            # Close the dict in the code string
            code += ")\n"
            config_file_name = "config_pettingzoo.py"
            config_file_directory = destination_directory_source_code + "/config/"
            print("destination_directory_source_code:", destination_directory_source_code)
            print("config_file_directory:", config_file_directory)

            with open(config_file_directory + config_file_name, "w") as config_file:
                config_file.write(code)
            config_file.close()
        # Create the output directory
        os.makedirs(output_directory, exist_ok=True)
        saved_directory_and_model_file_name = os.path.join(output_directory, file_name)
        print("saved_directory_and_model_file_name:", saved_directory_and_model_file_name)

        # save parameters to file
        saved_directory_and_parameter_file_name = os.path.join(
            output_directory, "train_parameters.txt"
        )
        print("saved_directory_and_parameter_file_name:", saved_directory_and_parameter_file_name)
        file = open(saved_directory_and_parameter_file_name, "w")
        file.write("model:" + environment_name + "\n")
        file.write("parameters:\n")
        file.write("training steps: 0\n")
        file.write("policy: " + policy + "\n")
        file.write("------------------------\n")
        for item in env_kwargs:
            file.write(str(item) + " = " + str(env_kwargs[item]) + "\n")
        file.write("------------------------\n")
        file.close()

    # Evaluation
    for parameter_variation_parameter in parameter_variation_scenarios:

        env_kwargs[parameter_variation_parameter_string] = parameter_variation_parameter
        script_directory = (
            root_destination_directory_source_code
            + "/"
            + str(parameter_variation_parameter)
        )
        #output_directory = script_directory + "/output/"
        eval_model_only = True
        watch_grid_model = not eval_model_only
        # save parameters to file
        if eval_model_only:
            saved_directory_and_evaluation_file_name = os.path.join(
                output_directory, "evaluation.txt"
            )
            file = open(saved_directory_and_evaluation_file_name, "w")
            file.write("model: PredPreyGrass\n")
            file.write("evaluation: " + saved_directory_and_evaluation_file_name + "\n")
            file.write("policy: " + policy + "\n")
            file.write("--------------------------\n")
        print()
        print("policy:", policy)
        print("model: PredPreyGrass")
        print("evaluation: " + output_directory)
        print("--------------------------")

        #training_steps = int(training_steps_string)

        # global variables for evaluation used in eval function
        episode_length = [0 for _ in range(num_episodes)]
        predator_extinct_at_termination = [0 for _ in range(num_episodes)]
        n_starved_predator_per_cycle = [0 for _ in range(num_episodes)]
        n_starved_prey_per_cycle = [0 for _ in range(num_episodes)]
        n_eaten_prey_per_cycle = [0 for _ in range(num_episodes)]
        n_eaten_grass_per_cycle = [0 for _ in range(num_episodes)]
        n_born_predator_per_cycle = [0 for _ in range(num_episodes)]
        n_born_prey_per_cycle = [0 for _ in range(num_episodes)]
        mean_cumulative_rewards = [0 for _ in range(num_episodes)]
        mean_cumulative_rewards_predator = [0 for _ in range(num_episodes)]
        mean_cumulative_rewards_prey = [0 for _ in range(num_episodes)]
        std_cumulative_rewards = [0 for _ in range(num_episodes)]
        std_cumulative_rewards_predator = [0 for _ in range(num_episodes)]
        std_cumulative_rewards_prey = [0 for _ in range(num_episodes)]
        mean_age_predator = [0 for _ in range(num_episodes)]
        mean_age_prey = [0 for _ in range(num_episodes)]
        episode_mean_of_mean_cumulative_rewards = 0
        episode_mean_of_mean_cumulative_rewards_predators = 0
        episode_mean_of_mean_cumulative_rewards_prey = 0
        predator_extinct_at_termination_count = 0
        mean_episode_length = 0
        std_episode_length = 0

        if eval_model_only:
            # Evaluate episodes
            (
                episode_mean_of_mean_cumulative_rewards,
                episode_mean_of_mean_cumulative_rewards_predators,
                episode_mean_of_mean_cumulative_rewards_prey,
                mean_episode_length,
                std_episode_length,
                predator_extinct_at_termination_count,
                episode_mean_of_n_starved_predator_per_cycle,
                episode_mean_of_n_starved_prey_per_cycle,
                episode_mean_of_n_eaten_prey_per_cycle,
                episode_mean_of_n_eaten_grass_per_cycle,
                episode_mean_of_n_born_predator_per_cycle,
                episode_mean_of_n_born_prey_per_cycle,
                episode_mean_of_mean_age_predator,
                episode_mean_of_mean_age_prey,
            ) = eval(env_fn, num_episodes=num_episodes, **env_kwargs)
            
            # save evaluation results to file
            file.write("--------------------------\n")
            file.write(f"Number of episodes = {num_episodes}" + "\n")
            file.write(f"Mean episode length = {mean_episode_length}" + "\n")
            file.write(
                f"Standard deviation episode length = {std_episode_length}" + "\n"
            )
            file.write(
                f"Per episode mean of per agent mean cumulative reward = {episode_mean_of_mean_cumulative_rewards}"
                + "\n"
            )
            file.write(
                f"Per episode mean of per Predator mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_predators}"
                + "\n"
            )
            file.write(
                f"Per episode mean of per Prey mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_prey}"
                + "\n"
            )
            file.write(
                f"% Predator extinct at termination = {round(predator_extinct_at_termination_count/num_episodes*100,1)}"
                + "\n"
            )
            file.write(
                f"Per episode mean of starved Predator/cycle = {round(episode_mean_of_n_starved_predator_per_cycle,3)}"
                + "\n"
            )
            file.write(
                f"Per episode mean of starved_Prey/cycle = {round(episode_mean_of_n_starved_prey_per_cycle,3)}"
                + "\n"
            )
            file.write(
                f"Per episode mean of eaten Prey/cycle = {round(episode_mean_of_n_eaten_prey_per_cycle,3)}"
                + "\n"
            )
            file.write(
                f"Per episode mean of eaten Grass/cycle = {round(episode_mean_of_n_eaten_grass_per_cycle,3)}"
                + "\n"
            )
            file.write(
                f"Per episode mean of born Predator/cycle = {round(episode_mean_of_n_born_predator_per_cycle,3)}"
                + "\n"
            )
            file.write(
                f"Per episode_mean_of_born Prey/cycle = {round(episode_mean_of_n_born_prey_per_cycle,3)}"
                + "\n"
            )
            file.write(
                f"Per episode mean of mean age Predator = {round(episode_mean_of_mean_age_predator,1)}"
                + "\n"
            )
            file.write(
                f"Per episode_mean_of_mean age Prey = {round(episode_mean_of_mean_age_prey,1)}"
                + "\n"
            )
            file.write("--------------------------\n")
            file.write("Evaluation parameters:\n")
            for item in env_kwargs:
                file.write(str(item) + " = " + str(env_kwargs[item]) + "\n")
            file.write("--------------------------\n")
            file.close()
            # and print to console
            print("--------------------------")
            print(f"Number of episodes = {num_episodes}")
            print(f"Mean episode length = {mean_episode_length}")
            print(f"Standard deviation episode length = {std_episode_length}")
            print(
                f"Per episode mean of per agent mean cumulative reward = {episode_mean_of_mean_cumulative_rewards}"
            )
            print(
                f"Per episode mean of per Predator mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_predators}"
            )
            print(
                f"Per episode mean of per Prey mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_prey}"
            )
            print(
                f"% Predator extinct at termination = {round(predator_extinct_at_termination_count/num_episodes*100,1)}"
            )
            print(
                f"Per episode mean of starved predator/cycle = {round(episode_mean_of_n_starved_predator_per_cycle,3)}"
            )
            print(
                f"Per episode mean of starved_prey/cycle = {round(episode_mean_of_n_starved_prey_per_cycle,3)}"
            )
            print(
                f"Per episode mean of eaten prey/cycle = {round(episode_mean_of_n_eaten_prey_per_cycle,3)}"
            )
            print(
                f"Per episode mean of eaten grass/cycle = {round(episode_mean_of_n_eaten_grass_per_cycle,3)}"
            )
            print(
                f"Per episode mean of born predator/cycle = {round(episode_mean_of_n_born_predator_per_cycle,3)}"
            )
            print(
                f"Per episode_mean_of_born prey/cycle = {round(episode_mean_of_n_born_prey_per_cycle,3)}"
            )
            print(
                f"Per episode mean of mean age Predator = {round(episode_mean_of_mean_age_predator,1)}"
            )
            print(
                f"Per episode_mean_of_mean age Prey = {round(episode_mean_of_mean_age_prey,1)}"
            )

        if watch_grid_model:
            # Evaluate and watch games
            (
                episode_mean_of_mean_cumulative_rewards,
                episode_mean_of_mean_cumulative_rewards_predators,
                episode_mean_of_mean_cumulative_rewards_prey,
                mean_episode_length,
                std_episode_length,
                predator_extinct_at_termination_count,
                episode_mean_of_n_starved_predator_per_cycle,
                episode_mean_of_n_starved_prey_per_cycle,
                episode_mean_of_n_eaten_prey_per_cycle,
                episode_mean_of_n_eaten_grass_per_cycle,
                episode_mean_of_n_born_predator_per_cycle,
                episode_mean_of_n_born_prey_per_cycle,
                episode_mean_of_mean_age_predator,
                episode_mean_of_mean_age_prey,
            ) = eval(env_fn, num_episodes=5, render_mode="human", **env_kwargs)
            # print to console
            print("--------------------------")
            print(f"Number of episodes = {num_episodes}")
            print(f"Mean episode length = {mean_episode_length}")
            print(f"Standard deviation episode length = {std_episode_length}")
            print(
                f"Per episode mean of per agent mean cumulative reward = {episode_mean_of_mean_cumulative_rewards}"
            )
            print(
                f"Per episode mean of per Predator mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_predators}"
            )
            print(
                f"Per episode mean of per Prey mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_prey}"
            )
            print(
                f"% Predator extinct at termination = {round(predator_extinct_at_termination_count/num_episodes*100,1)}"
            )
            print(
                f"Per episode mean of starved predator/cycle = {round(episode_mean_of_n_starved_predator_per_cycle,3)}"
            )
            print(
                f"Per episode mean of starved_prey/cycle = {round(episode_mean_of_n_starved_prey_per_cycle,3)}"
            )
            print(
                f"Per episode mean of eaten prey/cycle = {round(episode_mean_of_n_eaten_prey_per_cycle,3)}"
            )
            print(
                f"Per episode mean of eaten grass/cycle = {round(episode_mean_of_n_eaten_grass_per_cycle,3)}"
            )
            print(
                f"Per episode mean of born predator/cycle = {round(episode_mean_of_n_born_predator_per_cycle,3)}"
            )
            print(
                f"Per episode_mean_of_born prey/cycle = {round(episode_mean_of_n_born_prey_per_cycle,3)}"
            )
            print(
                f"Per episode mean of mean age Predator = {round(episode_mean_of_mean_age_predator,1)}"
            )
            print(
                f"Per episode_mean_of_mean age Prey = {round(episode_mean_of_mean_age_prey,1)}"
            )
