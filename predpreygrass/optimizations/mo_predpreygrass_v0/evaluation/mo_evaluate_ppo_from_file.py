"""
- to evaluate a ppo trained and saved model
- evaluation can be done with or without watching the grid
"""
"""
instructions:
- navigate with linux mint file-explorer to your local directory 
  (definedd in "env/_predpreygraas_v0/config/config_predpreygrass.py").
- note that the entire files/directories of the project are copied 
  to the defined local directory
- go to the directory with the appropriate time stamp of training
- right mouseclick "evaluate_from_file.py" and:
- select "Open with"
- select "Visual Studio Code" (or default VS Code for .py files)
- select "Run" (in taskbar of Visual Studio Code)
- select "Run without debugging"
- note that adjusting the configuration of the trained model is done 
  in the defined local directory (and *not* in your original directory!)
"""
# discretionary libraries
from predpreygrass.envs import mo_predpreygrass_v0
from predpreygrass.envs._mo_predpreygrass_v0.config.mo_config_predpreygrass import (
    env_kwargs,
    training_steps_string,
)
from predpreygrass.optimizations.mo_predpreygrass_v0.evaluation.utils.evaluator import Evaluator


# displaying the population of predators and prey
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator  # for integer ticks

from momaland.utils.aec_wrappers import LinearizeReward
import os
from statistics import mean, stdev
from typing import List
from os.path import dirname as up


from stable_baselines3 import PPO

def eval(env_fn, num_episodes: int = 100, render_mode: str | None = None, **env_kwargs):
    weights = {}

    # Define the number of predators and prey
    num_predators = env_kwargs["n_possible_predator"]
    num_prey = env_kwargs["n_possible_prey"]

    # Populate the weights dictionary for predators
    for i in range(num_predators):
        weights[f"predator_{i}"] = [0.5, 0.5]

    # Populate the weights dictionary for prey
    for i in range(num_prey):
        weights[f"prey_{i + num_predators}"] = [0.5, 0.5]

    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    env = LinearizeReward(env, weights)
    model = PPO.load(loaded_policy)
    cumulative_rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    print("Start evaluation.")
    # age lists over all episodes
    total_predator_age_list = []
    total_prey_age_list = []
    for i in range(num_episodes):
        env.reset()
        predator_name_list = env.predpreygrass.possible_predator_name_list
        prey_name_list = env.predpreygrass.possible_prey_name_list
        agent_name_list = env.predpreygrass.possible_agent_name_list
        cumulative_rewards = {agent: 0 for agent in agent_name_list}
        cumulative_rewards_predator = {agent: 0 for agent in predator_name_list}
        cumulative_rewards_prey = {agent: 0 for agent in prey_name_list}
        n_aec_cycles = 0
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            #print(f"Agent: {agent} rew={reward}")
            #print("cumulative_rewards", cumulative_rewards)
            cumulative_rewards[agent] += reward
            if agent in predator_name_list:
                cumulative_rewards_predator[agent] += reward
            elif agent in prey_name_list:
                cumulative_rewards_prey[agent] += reward

            if termination or truncation:
                action = None
                if env.predpreygrass.is_no_predator:
                    predator_extinct_at_termination[i] = 1
                    break
            else:
                action = model.predict(observation, deterministic=False)[0]
            env.step(action)
        n_aec_cycles = env.predpreygrass.n_aec_cycles
        # plot population of Predators and Prey
        plt.clf()
        plt.plot(env.predpreygrass.n_active_predator_list, "r")
        plt.plot(env.predpreygrass.n_active_prey_list, "b")
        plt.title("Predator and Prey Population", weight="bold")
        plt.xlabel("Time steps", weight="bold")
        ax = plt.gca()
        # Set x and y limits
        ax.set_xlim([0, env.predpreygrass.n_aec_cycles])
        ax.set_ylim(
            [
                0,
                max(
                    env.predpreygrass.n_active_predator_list
                    + env.predpreygrass.n_active_prey_list
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


        # plot energy of Predators, Prey and Grass
        plt.clf()
        plt.plot(env.predpreygrass.total_energy_predator_list, "r")
        plt.plot(env.predpreygrass.total_energy_prey_list, "b")
        plt.plot(env.predpreygrass.total_energy_grass_list, "g")
        plt.plot(env.predpreygrass.total_energy_learning_agents_list, "k")
        plt.title("Total energy", weight="bold")
        plt.xlabel("Time steps", weight="bold")
        ax = plt.gca()
        # Set x and y limits
        ax.set_xlim([0, env.predpreygrass.n_aec_cycles])
        ax.set_ylim(
            [
                0,
                max(
                    env.predpreygrass.total_energy_predator_list
                    + env.predpreygrass.total_energy_prey_list
                    + env.predpreygrass.total_energy_grass_list
                    + env.predpreygrass.total_energy_learning_agents_list
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
            env.predpreygrass.n_starved_predator / n_aec_cycles
        )
        n_starved_prey_per_cycle[i] = (
            env.predpreygrass.n_starved_prey / n_aec_cycles
        )
        n_eaten_prey_per_cycle[i] = env.predpreygrass.n_eaten_prey / n_aec_cycles
        n_born_predator_per_cycle[i] = (
            env.predpreygrass.n_born_predator / n_aec_cycles
        )
        n_eaten_grass_per_cycle[i] = env.predpreygrass.n_eaten_grass / n_aec_cycles
        n_born_predator_per_cycle[i] = (
            env.predpreygrass.n_born_predator / n_aec_cycles
        )
        n_born_prey_per_cycle[i] = env.predpreygrass.n_born_prey / n_aec_cycles
        episode_predator_age_list = env.predpreygrass.predator_age_list
        episode_prey_age_list = env.predpreygrass.prey_age_list
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
    print("Finish evaluation.")
    env.close()
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
    env_fn = mo_predpreygrass_v0
    environment_name = str(env_fn.raw_env.metadata['name'])
    model_file_name = f"{environment_name}_steps_{training_steps_string}"
    evaluation_directory = os.path.dirname(os.path.abspath(__file__))
    destination_source_code_dir = up(up(up(up(__file__))))  # up 4 levels in directory tree
    output_directory = destination_source_code_dir +"/output/"
    loaded_policy = output_directory + model_file_name
    # input from so_config_predpreygrass.py
    watch_grid_model = env_kwargs["watch_grid_model"]
    eval_model_only = not watch_grid_model
    num_episodes = env_kwargs["num_episodes"] 
    training_steps = int(training_steps_string)

    render_mode = "human" if watch_grid_model else None

    # Create an instance of the Evaluator class
    evaluator = Evaluator(
        env_fn,
        output_directory, # destination_output_dir,
        loaded_policy,
        destination_source_code_dir, # destination_root_dir,
        render_mode,
        training_steps_string,
        destination_source_code_dir, # destination_source_code_dir,
        **env_kwargs
    )
    # Call the eval method to perform the evaluation
    evaluator.eval()
