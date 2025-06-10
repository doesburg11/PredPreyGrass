# discretionary libraries
from predpreygrass.pettingzoo.envs import predpreygrass_aec_v0
from predpreygrass.pettingzoo.config.config_predpreygrass import (
    env_kwargs,
    local_output_root,
)

# external libraries
import os
import time
from statistics import mean, stdev


def evaluation_header_text():
    return (
        "Evaluation results:\n"
        + "-----------------------------------------------------------------------------\n"
        + "Date and Time: "
        + time_stamp_string
        + "\n"
        + "Environment: "
        + "predpreygrass_aec_v0"
        + "\n"
        + "Grid transformation: "
        + "bounded"
        + "\n"
        + "Learning algorithm: random"
        + "\n"
        + "-----------------------------------------------------------------------------\n"
    )


def evaluation_results_output(i):
    return (
        f"Eps {i} "
        + f"Lngth = {n_cycles} "
        + f"Strv Prd/cycl = {round(n_starved_predator_per_cycle[i],3)} "
        + f"Strv Pry/cycl = {round(n_starved_prey_per_cycle[i],3)} "
        + f"Eatn Pry/cycl = {round(n_eaten_prey_per_cycle[i],3)} "
        + f"Eatn Gra/cycl = {round(n_eaten_grass_per_cycle[i],3)} "
        + f"Brn Prd/cycl = {round(n_born_predator_per_cycle[i],3)} "
        + f"Brn Pry/cycle = {round(n_born_prey_per_cycle[i],3)} "
        + f"Mn age Prd = {round(mean_age_predator[i],1)} "
        + f"Mn age Pry = {round(mean_age_prey[i],1)}\n"
    )


def evaluation_results_summary():
    return (
        "-----------------------------------------------------------------------------\n"
        + f"Number of episodes = {num_episodes}"
        + "\n"
        + f"Mean episode length = {mean_episode_length}"
        + "\n"
        + f"Standard deviation episode length = {std_episode_length}"
        + "\n"
        + f"Per episode mean of per agent mean cumulative reward = {episode_mean_of_mean_cumulative_rewards}"
        + "\n"
        + f"Per episode mean of per Predator mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_predators}"
        + "\n"
        + f"Per episode mean of per Prey mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_prey}"
        + "\n"
        + f"% Predator extinct at termination = {round(predator_extinct_at_termination_count/num_episodes*100,1)}"
        + "\n"
        + f"Per episode mean of starved Predator/cycle = {round(episode_mean_of_n_starved_predator_per_cycle,3)}"
        + "\n"
        + f"Per episode mean of starved_Prey/cycle = {round(episode_mean_of_n_starved_prey_per_cycle,3)}"
        + "\n"
        + f"Per episode mean of eaten Prey/cycle = {round(episode_mean_of_n_eaten_prey_per_cycle,3)}"
        + "\n"
        + f"Per episode mean of eaten Grass/cycle = {round(episode_mean_of_n_eaten_grass_per_cycle,3)}"
        + "\n"
        + f"Per episode mean of born Predator/cycle = {round(episode_mean_of_n_born_predator_per_cycle,3)}"
        + "\n"
        + f"Per episode_mean_of_born Prey/cycle = {round(episode_mean_of_n_born_prey_per_cycle,3)}"
        + "\n"
        + f"Per episode mean of mean age Predator = {round(episode_mean_of_mean_age_predator,1)}"
        + "\n"
        + f"Per episode_mean_of_mean age Prey = {round(episode_mean_of_mean_age_prey,1)}"
        + "\n"
        + "-----------------------------------------------------------------------------\n"
    )


num_episodes = env_kwargs["num_episodes"]
render_mode = None  # "human"
time_stamp_string = str(time.strftime("%Y-%m-%d_%H:%M:%S"))
destination_output_dir = local_output_root + "/" + time_stamp_string
eval_header_text = evaluation_header_text()
os.makedirs(destination_output_dir, exist_ok=True)
saved_directory_and_evaluation_file_name = os.path.join(destination_output_dir, "evaluation.txt")
evaluation_file = open(saved_directory_and_evaluation_file_name, "w")
evaluation_file.write(eval_header_text)  # write to file
print(eval_header_text)  # write to screen

# initialization evaluation metrics
episode_mean_of_mean_cumulative_rewards = 0
episode_mean_of_mean_cumulative_rewards_predators = 0
episode_mean_of_mean_cumulative_rewards_prey = 0
predator_extinct_at_termination_count = 0
mean_episode_length = 0
std_episode_length = 0
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
total_predator_age_list = []
total_prey_age_list = []

episode_predator_age_list = []
episode_prey_age_list = []

env = predpreygrass_aec_v0.env(render_mode=render_mode, **env_kwargs)
env_base = env.predpreygrass
for i in range(num_episodes):
    env.reset(seed=1)
    possible_predator_name_list = env_base.possible_agent_name_list_type[env_base.predator_type_nr]
    possible_prey_name_list = env_base.possible_agent_name_list_type[env_base.prey_type_nr]
    possible_agent_name_list = env_base.possible_learning_agent_name_list
    cumulative_rewards = {agent: 0 for agent in possible_agent_name_list}
    cumulative_rewards_predator = {agent: 0 for agent in possible_predator_name_list}
    cumulative_rewards_prey = {agent: 0 for agent in possible_prey_name_list}
    n_cycles = 0

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        cumulative_rewards[agent] += reward
        if agent in possible_predator_name_list:
            cumulative_rewards_predator[agent] += reward
        elif agent in possible_prey_name_list:
            cumulative_rewards_prey[agent] += reward
        if env_base.is_no_predator or env_base.is_no_prey or env.truncations[agent]:
            action = None
            if env_base.is_no_predator:
                predator_extinct_at_termination[i] = 1
            break
        else:
            action = env.action_space(agent).sample()  # random policy

        env.step(action)
    n_cycles = env_base.n_cycles
    episode_length[i] = n_cycles
    n_starved_predator_per_cycle[i] = env_base.n_starved_predator / n_cycles
    n_starved_prey_per_cycle[i] = env_base.n_starved_prey / n_cycles
    n_eaten_prey_per_cycle[i] = env_base.n_eaten_prey / n_cycles
    n_eaten_grass_per_cycle[i] = env_base.n_eaten_grass / n_cycles
    n_born_predator_per_cycle[i] = env_base.n_born_predator / n_cycles
    n_born_prey_per_cycle[i] = env_base.n_born_prey / n_cycles
    episode_predator_age_list = env_base.agent_age_of_death_list_type[env_base.predator_type_nr]
    episode_prey_age_list = env_base.agent_age_of_death_list_type[env_base.prey_type_nr]
    mean_age_predator[i] = mean(episode_predator_age_list) if episode_predator_age_list else 0
    mean_age_prey[i] = mean(episode_prey_age_list) if episode_prey_age_list else 0
    total_predator_age_list += episode_predator_age_list
    total_prey_age_list += episode_prey_age_list
    mean_cumulative_rewards[i] = mean(cumulative_rewards.values())
    mean_cumulative_rewards_predator[i] = mean(cumulative_rewards_predator.values())
    mean_cumulative_rewards_prey[i] = mean(cumulative_rewards_prey.values())
    std_cumulative_rewards[i] = stdev(cumulative_rewards.values())
    std_cumulative_rewards_predator[i] = stdev(cumulative_rewards_predator.values())
    std_cumulative_rewards_prey[i] = stdev(cumulative_rewards_prey.values())

    eval_results_output = evaluation_results_output(i)
    print(eval_results_output)
    evaluation_file.write(eval_results_output)
env.close

predator_extinct_at_termination_count = sum(predator_extinct_at_termination)
episode_mean_of_mean_cumulative_rewards = round(mean(mean_cumulative_rewards), 1)
episode_mean_of_mean_cumulative_rewards_predators = round(mean(mean_cumulative_rewards_predator), 1)
episode_mean_of_mean_cumulative_rewards_prey = round(mean(mean_cumulative_rewards_prey), 1)
mean_episode_length = round(mean(episode_length), 1)
std_episode_length = round(stdev(episode_length), 1) if num_episodes > 1 else None
episode_mean_of_n_starved_predator_per_cycle = round(mean(n_starved_predator_per_cycle), 3)
episode_mean_of_n_starved_prey_per_cycle = round(mean(n_starved_prey_per_cycle), 3)
episode_mean_of_n_eaten_prey_per_cycle = round(mean(n_eaten_prey_per_cycle), 3)
episode_mean_of_n_eaten_grass_per_cycle = round(mean(n_eaten_grass_per_cycle), 3)
episode_mean_of_n_born_predator_per_cycle = round(mean(n_born_predator_per_cycle), 3)
episode_mean_of_n_born_prey_per_cycle = round(mean(n_born_prey_per_cycle), 3)
episode_mean_of_mean_age_predator = round(sum(total_predator_age_list) / len(total_predator_age_list), 1)
episode_mean_of_mean_age_prey = round(sum(total_prey_age_list) / len(total_prey_age_list), 1)

# save evaluation results to evaluation_file
evaluation_results_summary = evaluation_results_summary()
evaluation_file.write(evaluation_results_summary)
# additionally save the config file
evaluation_file.write("-----------------------------------------------------------------------------\n")
evaluation_file.write("Evaluation parameters:\n")
for item in env_kwargs:
    evaluation_file.write(str(item) + " = " + str(env_kwargs[item]) + "\n")
evaluation_file.write("-----------------------------------------------------------------------------\n")
evaluation_file.close()

# print to console
print(evaluation_results_summary)
