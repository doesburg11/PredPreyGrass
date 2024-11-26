# discretionary libraries
from utils.population_plotter import PopulationPlotter

# external libraries
import os
import csv
import time
from statistics import mean, stdev
from stable_baselines3 import PPO


class Evaluator:
    def __init__(
        self,
        env_fn,
        destination_output_dir,
        loaded_policy,
        destination_root_dir,
        render_mode,
        destination_source_code_dir,
        **env_kwargs,
    ):
        self.env_fn = env_fn
        self.destination_output_dir = destination_output_dir
        self.loaded_policy = loaded_policy
        self.destination_root_dir = destination_root_dir
        self.render_mode = render_mode
        self.destination_source_code_dir = destination_source_code_dir

        self.env_kwargs = env_kwargs
        self.training_steps_string = env_kwargs["training_steps_string"]
        self.num_episodes = env_kwargs["num_episodes"]
        self.environment_name = str(env_fn.parallel_env.metadata["name"]) if env_kwargs["is_parallel"] else str(env_fn.raw_env.metadata["name"])
        self.torus = env_kwargs["torus"]
        self.grid_transformation = "torus transformation" if self.torus else "bounded grid"
        self.evaluation_time_stamp = str(time.strftime("%Y-%m-%d_%H:%M:%S"))

    def initialize_evaluation_metrics(self):

            # initialization evaluation metrics
            self.self.episode_mean_of_mean_cumulative_rewards = 0
            self.self.episode_mean_of_mean_cumulative_rewards_predators = 0
            self.self.episode_mean_of_mean_cumulative_rewards_prey = 0
            self.self.predator_extinct_at_termination_count = 0
            self.self.mean_episode_length = 0
            self.self.std_episode_length = 0
            self.episode_length = [0 for _ in range(self.num_episodes)]
            self.predator_extinct_at_termination = [0 for _ in range(self.num_episodes)]
            self.n_starved_predator_per_cycle = [0 for _ in range(self.num_episodes)]
            self.n_starved_prey_per_cycle = [0 for _ in range(self.num_episodes)]
            self.n_eaten_prey_per_cycle = [0 for _ in range(self.num_episodes)]
            self.n_eaten_grass_per_cycle = [0 for _ in range(self.num_episodes)]
            self.n_born_predator_per_cycle = [0 for _ in range(self.num_episodes)]
            self.n_born_prey_per_cycle = [0 for _ in range(self.num_episodes)]
            self.mean_cumulative_rewards = [0 for _ in range(self.num_episodes)]
            self.mean_cumulative_rewards_predator = [0 for _ in range(self.num_episodes)]
            self.mean_cumulative_rewards_prey = [0 for _ in range(self.num_episodes)]
            self.std_cumulative_rewards = [0 for _ in range(self.num_episodes)]
            self.std_cumulative_rewards_predator = [0 for _ in range(self.num_episodes)]
            self.std_cumulative_rewards_prey = [0 for _ in range(self.num_episodes)]
            self.mean_age_predator = [0 for _ in range(self.num_episodes)]
            self.mean_age_prey = [0 for _ in range(self.num_episodes)]
            self.total_predator_age_list = []
            self.total_prey_age_list = []

            self.episode_predator_age_list = []
            self.episode_prey_age_list = []

    def evaluation_header_text(self):
        return (
             "Evaluation results:\n"
            + "--------------------------\n"
            + "environment: "
            + self.environment_name
            + "\n"
            + "policy algorithm: PPO"
            + "loaded_policy: "
            + self.loaded_policy
            + "\n"
            + "evaluation directory: "
            + self.destination_source_code_dir
            + "\n"
            + "training steps: "
            + self.training_steps_string
            + "\n"
            + "Date and Time: "
            + self.evaluation_time_stamp
            + "\n"
            + "--------------------------\n"
        )

    def evaluation_results_output(self,i):
        return (
                f"Eps {i} "
                + f"Lngth = {self.n_cycles} "
                + f"Strv Prd/cycl = {round(self.n_starved_predator_per_cycle[i],3)} "
                + f"Strv Pry/cycl = {round(self.n_starved_prey_per_cycle[i],3)} "
                + f"Eatn Pry/cycl = {round(self.n_eaten_prey_per_cycle[i],3)} "
                + f"Eatn Gra/cycl = {round(self.n_eaten_grass_per_cycle[i],3)} "
                + f"Brn Prd/cycl = {round(self.n_born_predator_per_cycle[i],3)} "
                + f"Brn Pry/cycle = {round(self.n_born_prey_per_cycle[i],3)} "
                + f"Mn age Prd = {round(self.mean_age_predator[i],1)} "
                + f"Mn age Pry = {round(self.mean_age_prey[i],1)}\n"
            )

    def evaluation_results_summary(self):
        return (
            "---------------------------------------------------\n" 
            + f"Number of episodes = {self.num_episodes}" + "\n"
            + f"Mean episode length = {self.mean_episode_length}" + "\n"
            + f"Standard deviation episode length = {self.std_episode_length}" + "\n"
            + f"Per episode mean of per agent mean cumulative reward = {self.episode_mean_of_mean_cumulative_rewards}"
            + "\n"
            + f"Per episode mean of per Predator mean cumulative reward = {self.episode_mean_of_mean_cumulative_rewards_predators}"
            + "\n"
            + f"Per episode mean of per Prey mean cumulative reward = {self.episode_mean_of_mean_cumulative_rewards_prey}"
            + "\n"
            + f"% Predator extinct at termination = {round(self.predator_extinct_at_termination_count/self.num_episodes*100,1)}"
            + "\n"
            + f"Per episode mean of starved Predator/cycle = {round(self.episode_mean_of_n_starved_predator_per_cycle,3)}"
            + "\n"
            + f"Per episode mean of starved_Prey/cycle = {round(self.episode_mean_of_n_starved_prey_per_cycle,3)}"
            + "\n"
            + f"Per episode mean of eaten Prey/cycle = {round(self.episode_mean_of_n_eaten_prey_per_cycle,3)}"
            + "\n"
            + f"Per episode mean of eaten Grass/cycle = {round(self.episode_mean_of_n_eaten_grass_per_cycle,3)}"
            + "\n"
            + f"Per episode mean of born Predator/cycle = {round(self.episode_mean_of_n_born_predator_per_cycle,3)}"
            + "\n"
            + f"Per episode_mean_of_born Prey/cycle = {round(self.episode_mean_of_n_born_prey_per_cycle,3)}"
            + "\n"
            + f"Per episode mean of mean age Predator = {round(self.episode_mean_of_mean_age_predator,1)}"
            + "\n"
            + f"Per episode_mean_of_mean age Prey = {round(self.episode_mean_of_mean_age_prey,1)}"
            + "\n"
            + "---------------------------------------------------\n"
        )

    def save_combined_population_data(
        self, predator_population, prey_population, episode_index
    ):
        """
        Save the combined predator and prey population data to a CSV file for each episode.

        Parameters:
        - predator_population: list of predator population counts per cycle.
        - prey_population: list of prey population counts per cycle.
        - episode_index: int, index of the current episode.
        """
        file_path = os.path.join(
            self.destination_output_dir, "population_data", f"episode_{episode_index}_population.csv"
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists

        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write the header row
            writer.writerow(["Cycle", "Predator Population", "Prey Population"])
            # Write each cycle's data
            for cycle, (pred_count, prey_count) in enumerate(zip(predator_population, prey_population)):
                writer.writerow([cycle + 1, pred_count, prey_count])

    def aec_evaluation(self):
        env = self.env_fn.env(render_mode=self.render_mode, **self.env_kwargs)
        model = PPO.load(self.loaded_policy)
        cumulative_rewards = {agent: 0 for agent in env.possible_agents}
        plotter = PopulationPlotter(self.destination_output_dir)

        # inserted
        saved_directory_and_evaluation_file_name = os.path.join(
            self.destination_output_dir, "evaluation.txt"
        )
        print("Start evaluation on: " + self.destination_root_dir)
        eval_header_text = self.evaluation_header_text()
        evaluation_file = open(saved_directory_and_evaluation_file_name, "w")
        evaluation_file.write(eval_header_text)  # write to file
        print(eval_header_text)  # write to screen

        self.initialize_evaluation_metrics()

        env_base = env.predpreygrass
        for i in range(self.num_episodes):
            env.reset(seed=i)
            possible_predator_name_list = env_base.possible_agent_name_list_type[
                env_base.predator_type_nr
            ]
            possible_prey_name_list = env_base.possible_agent_name_list_type[
                env_base.prey_type_nr
            ]
            possible_agent_name_list = env_base.possible_learning_agent_name_list
            cumulative_rewards = {agent: 0 for agent in possible_agent_name_list}
            cumulative_rewards_predator = {
                agent: 0 for agent in possible_predator_name_list
            }
            cumulative_rewards_prey = {agent: 0 for agent in possible_prey_name_list}
            self.n_cycles = 0
            for agent in env.agent_iter():
                observation, reward = env.last()[:2]
                cumulative_rewards[agent] += reward
                if agent in possible_predator_name_list:
                    cumulative_rewards_predator[agent] += reward
                elif agent in possible_prey_name_list:
                    cumulative_rewards_prey[agent] += reward

                if env_base.is_no_predator or env_base.is_no_prey or env.truncations[agent]:
                    action = None
                    if env_base.is_no_predator:
                        self.predator_extinct_at_termination[i] = 1
                    break
                else:
                    action = model.predict(observation, deterministic=True)[0]
                env.step(action)
            self.n_cycles = env_base.self.n_cycles
            plotter.plot_population(
                env_base.n_active_agent_list_type[env_base.predator_type_nr],
                env_base.n_active_agent_list_type[env_base.prey_type_nr],
                self.n_cycles,
                i,
                title="Predator and Prey Population Over Time",
            )

            self.episode_length[i] = self.n_cycles
            self.n_starved_predator_per_cycle[i] = env_base.n_starved_predator / self.n_cycles
            self.n_starved_prey_per_cycle[i] = env_base.n_starved_prey / self.n_cycles
            self.n_eaten_prey_per_cycle[i] = env_base.n_eaten_prey / self.n_cycles
            self.n_eaten_grass_per_cycle[i] = env_base.n_eaten_grass / self.n_cycles
            self.n_born_predator_per_cycle[i] = env_base.n_born_predator / self.n_cycles
            self.n_born_prey_per_cycle[i] = env_base.n_born_prey / self.n_cycles
            self.episode_predator_age_list = env_base.agent_age_of_death_list_type[
                env_base.predator_type_nr
            ]
            self.episode_prey_age_list = env_base.agent_age_of_death_list_type[
                env_base.prey_type_nr
            ]

            self.mean_age_predator[i] = (
                mean(self.episode_predator_age_list) if self.episode_predator_age_list else 0
            )
            self.mean_age_prey[i] = (
                mean(self.episode_prey_age_list) if self.episode_prey_age_list else 0
            )
            self.total_predator_age_list += self.episode_predator_age_list
            self.total_prey_age_list += self.episode_prey_age_list
            self.mean_cumulative_rewards[i] = mean(cumulative_rewards.values())
            self.mean_cumulative_rewards_predator[i] = mean(
                cumulative_rewards_predator.values()
            )
            self.mean_cumulative_rewards_prey[i] = mean(cumulative_rewards_prey.values())
            self.std_cumulative_rewards[i] = stdev(cumulative_rewards.values())
            self.std_cumulative_rewards_predator[i] = stdev(
                cumulative_rewards_predator.values()
            )
            self.std_cumulative_rewards_prey[i] = stdev(cumulative_rewards_prey.values())
            eval_results_output = self.evaluation_results_output(i)
            print(eval_results_output)
            evaluation_file.write(eval_results_output)

            # Collect predator and prey population data
            predator_population_data = env_base.n_active_agent_list_type[
                env_base.predator_type_nr
            ]
            prey_population_data = env_base.n_active_agent_list_type[
                env_base.prey_type_nr
            ]

            # Save predator and prey population data to a single file per episode
            self.save_combined_population_data(
                predator_population_data, prey_population_data, i
            )

        print("Finish evaluation.")

        env.close()
        self.predator_extinct_at_termination_count = sum(self.predator_extinct_at_termination)
        self.episode_mean_of_mean_cumulative_rewards = round(
            mean(self.mean_cumulative_rewards), 1
        )
        self.episode_mean_of_mean_cumulative_rewards_predators = round(
            mean(self.mean_cumulative_rewards_predator), 1
        )
        self.episode_mean_of_mean_cumulative_rewards_prey = round(
            mean(self.mean_cumulative_rewards_prey), 1
        )
        self.mean_episode_length = round(mean(self.episode_length), 1)
        self.std_episode_length = (
            round(stdev(self.episode_length), 1) if self.num_episodes > 1 else None
        )
        self.episode_mean_of_n_starved_predator_per_cycle = round(
            mean(self.n_starved_predator_per_cycle), 3
        )
        self.episode_mean_of_n_starved_prey_per_cycle = round(
            mean(self.n_starved_prey_per_cycle), 3
        )
        self.episode_mean_of_n_eaten_prey_per_cycle = round(mean(self.n_eaten_prey_per_cycle), 3)
        self.episode_mean_of_n_eaten_grass_per_cycle = round(
            mean(self.n_eaten_grass_per_cycle), 3
        )
        self.episode_mean_of_n_born_predator_per_cycle = round(
            mean(self.n_born_predator_per_cycle), 3
        )
        self.episode_mean_of_n_born_prey_per_cycle = round(mean(self.n_born_prey_per_cycle), 3)
        self.episode_mean_of_mean_age_predator = round(
            sum(self.total_predator_age_list) / len(self.total_predator_age_list), 1
        )
        self.episode_mean_of_mean_age_prey = round(
            sum(self.total_prey_age_list) / len(self.total_prey_age_list), 1
        )

        # save evaluation results to evaluation_file
        evaluation_results_summary= self.evaluation_results_summary()   
        evaluation_file.write(evaluation_results_summary)
        # additionally save the config file
        evaluation_file.write("---------------------------------------------------\n")
        evaluation_file.write("Evaluation parameters:\n")
        for item in self.env_kwargs:
            evaluation_file.write(str(item) + " = " + str(self.env_kwargs[item]) + "\n")
        evaluation_file.write("---------------------------------------------------\n")
        evaluation_file.close()

        # print to console
        print(evaluation_results_summary)
        

    def parallel_evaluation(self):
        # env = self.env_fn.env(render_mode=self.render_mode, **self.env_kwargs)
        model = PPO.load(self.loaded_policy)
        parallel_env = self.env_fn.parallel_env(
            **self.env_kwargs, render_mode=self.render_mode
        )
        env_base = parallel_env.predpreygrass
        cumulative_rewards = {agent: 0 for agent in parallel_env.possible_agents}
        plotter = PopulationPlotter(self.destination_output_dir)

        # inserted
        saved_directory_and_evaluation_file_name = os.path.join(
            self.destination_output_dir, "evaluation.txt"
        )
        print("Start evaluation on: " + self.destination_root_dir)
        eval_header_text = self.evaluation_header_text()
        evaluation_file = open(saved_directory_and_evaluation_file_name, "w")
        evaluation_file.write(eval_header_text)  # write to file
        print(eval_header_text)  # write to screen
        self.initialize_evaluation_metrics()

        env_base = parallel_env.predpreygrass
        for i in range(self.num_episodes):
            # env.reset(seed=i)
            possible_predator_name_list = env_base.possible_agent_name_list_type[
                env_base.predator_type_nr
            ]
            possible_prey_name_list = env_base.possible_agent_name_list_type[
                env_base.prey_type_nr
            ]
            possible_agent_name_list = env_base.possible_learning_agent_name_list
            cumulative_rewards = {agent: 0 for agent in possible_agent_name_list}
            cumulative_rewards_predator = {
                agent: 0 for agent in possible_predator_name_list
            }
            cumulative_rewards_prey = {agent: 0 for agent in possible_prey_name_list}
            self.n_cycles = 0
            observations = parallel_env.reset()[0]

            done = False
            while not done:
                actions = {
                    agent: model.predict(observations[agent], deterministic=True)[0]
                    for agent in parallel_env.agents
                }
                observations, rewards = parallel_env.step(actions)[:2]
                done = env_base.is_no_prey or env_base.is_no_predator or self.n_cycles >= env_base.max_cycles

            self.n_cycles = env_base.self.n_cycles
            plotter.plot_population(
                env_base.n_active_agent_list_type[env_base.predator_type_nr],
                env_base.n_active_agent_list_type[env_base.prey_type_nr],
                self.n_cycles,
                i,
                title="Predator and Prey Population Over Time",
            )

            self.episode_length[i] = self.n_cycles
            self.n_starved_predator_per_cycle[i] = env_base.n_starved_predator / self.n_cycles
            self.n_starved_prey_per_cycle[i] = env_base.n_starved_prey / self.n_cycles
            self.n_eaten_prey_per_cycle[i] = env_base.n_eaten_prey / self.n_cycles
            self.n_eaten_grass_per_cycle[i] = env_base.n_eaten_grass / self.n_cycles
            self.n_born_predator_per_cycle[i] = env_base.n_born_predator / self.n_cycles
            self.self.n_born_prey_per_cycle[i] = env_base.n_born_prey / self.n_cycles
            self.episode_predator_age_list = env_base.agent_age_of_death_list_type[
                env_base.predator_type_nr
            ]
            self.episode_prey_age_list = env_base.agent_age_of_death_list_type[
                env_base.prey_type_nr
            ]
            self.mean_age_predator[i] = (
                mean(self.episode_predator_age_list) if self.episode_predator_age_list else 0
            )
            self.mean_age_prey[i] = (
                mean(self.episode_prey_age_list) if self.episode_prey_age_list else 0
            )
            self.total_predator_age_list += self.episode_predator_age_list
            self.total_prey_age_list += self.episode_prey_age_list
            self.mean_cumulative_rewards[i] = mean(cumulative_rewards.values())
            self.mean_cumulative_rewards_predator[i] = mean(
                cumulative_rewards_predator.values()
            )
            self.mean_cumulative_rewards_prey[i] = mean(cumulative_rewards_prey.values())
            self.std_cumulative_rewards[i] = stdev(cumulative_rewards.values())
            self.std_cumulative_rewards_predator[i] = stdev(
                cumulative_rewards_predator.values()
            )
            self.std_cumulative_rewards_prey[i] = stdev(cumulative_rewards_prey.values())

            eval_results_output = self.evaluation_results_output(i)
            print(eval_results_output)
            evaluation_file.write(eval_results_output)  

            # Collect predator and prey population data
            predator_population_data = env_base.n_active_agent_list_type[
                env_base.predator_type_nr
            ]
            prey_population_data = env_base.n_active_agent_list_type[
                env_base.prey_type_nr
            ]

            # Save predator and prey population data to a single file per episode
            self.save_combined_population_data(
                predator_population_data, prey_population_data, i
            )

        print("Finish evaluation.")

        env_base.close()
        self.predator_extinct_at_termination_count = sum(self.predator_extinct_at_termination)
        self.episode_mean_of_mean_cumulative_rewards = round(
            mean(self.mean_cumulative_rewards), 1
        )
        self.episode_mean_of_mean_cumulative_rewards_predators = round(
            mean(self.mean_cumulative_rewards_predator), 1
        )
        self.episode_mean_of_mean_cumulative_rewards_prey = round(
            mean(self.mean_cumulative_rewards_prey), 1
        )
        self.mean_episode_length = round(mean(self.episode_length), 1)
        self.std_episode_length = (
            round(stdev(self.episode_length), 1) if self.num_episodes > 1 else None
        )
        self.episode_mean_of_n_starved_predator_per_cycle = round(
            mean(self.n_starved_predator_per_cycle), 3
        )
        self.episode_mean_of_n_starved_prey_per_cycle = round(
            mean(self.n_starved_prey_per_cycle), 3
        )
        self.episode_mean_of_n_eaten_prey_per_cycle = round(mean(self.n_eaten_prey_per_cycle), 3)
        self.episode_mean_of_n_eaten_grass_per_cycle = round(
            mean(self.n_eaten_grass_per_cycle), 3
        )
        self.episode_mean_of_n_born_predator_per_cycle = round(
            mean(self.n_born_predator_per_cycle), 3
        )
        self.episode_mean_of_n_born_prey_per_cycle = round(mean(self.n_born_prey_per_cycle), 3)
        self.episode_mean_of_mean_age_predator = round(
            sum(self.total_predator_age_list) / len(self.total_predator_age_list), 1
        )
        self.episode_mean_of_mean_age_prey = round(
            sum(self.total_prey_age_list) / len(self.total_prey_age_list), 1
        )

        # save evaluation results to evaluation_file
        evaluation_results_summary= self.evaluation_results_summary()   
        evaluation_file.write(evaluation_results_summary)
        evaluation_file.write("---------------------------------------------------\n")
        evaluation_file.write("Evaluation parameters:\n")
        for item in self.env_kwargs:
            evaluation_file.write(str(item) + " = " + str(self.env_kwargs[item]) + "\n")
        evaluation_file.write("---------------------------------------------------\n")
        evaluation_file.close()

        # print to console
        print(evaluation_results_summary)
