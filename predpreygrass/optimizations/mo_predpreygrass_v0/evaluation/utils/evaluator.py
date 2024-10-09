# dicretionary libraries
from predpreygrass.optimizations.mo_predpreygrass_v0.evaluation.utils.population_plotter import PopulationPlotter
from predpreygrass.optimizations.mo_predpreygrass_v0.training.utils.linearization_weights_constructor import construct_linearalized_weights

# external libraries
from momaland.utils.aec_wrappers import LinearizeReward
import os
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
        training_steps_string,
        destination_source_code_dir,
        **env_kwargs,
    ):
        self.env_fn = env_fn
        self.destination_output_dir = destination_output_dir
        self.loaded_policy = loaded_policy
        self.destination_root_dir = destination_root_dir
        self.render_mode = render_mode
        self.training_steps_string = training_steps_string
        self.destination_source_code_dir = destination_source_code_dir
        self.env_kwargs = env_kwargs
        self.num_episodes = env_kwargs["num_episodes"]

    def eval(self):
        # Define the number of predators and prey
        num_predators = self.env_kwargs["n_possible_predator"]
        num_prey = self.env_kwargs["n_possible_prey"]
        # Construct the weights for linearization of the multi objective rewards
        weights = construct_linearalized_weights(num_predators, num_prey)
        
        env = self.env_fn.env(render_mode=self.render_mode, **self.env_kwargs)
        env = LinearizeReward(env, weights)
        model = PPO.load(self.loaded_policy)
        cumulative_rewards = {agent: 0 for agent in env.possible_agents}
        plotter = PopulationPlotter(self.destination_output_dir)

        saved_directory_and_evaluation_file_name = os.path.join(
            self.destination_output_dir, "evaluation.txt"
        )
        print("Start evaluation on: " + self.destination_root_dir)
        eval_header_text = (
            "Evaluation results:\n" +
            "--------------------------\n" +
            "loaded_policy: " + self.loaded_policy + "\n" +
            "environment: " + str(env.metadata['name']) + "\n" +
            "evaluation: " + self.destination_source_code_dir + "\n" +  
            "training steps: " + self.training_steps_string + "\n" +
            "--------------------------\n"  
        )
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
        episode_length = [0 for _ in range(self.num_episodes)]
        predator_extinct_at_termination = [0 for _ in range(self.num_episodes)]
        n_starved_predator_per_cycle = [0 for _ in range(self.num_episodes)]
        n_starved_prey_per_cycle = [0 for _ in range(self.num_episodes)]
        n_eaten_prey_per_cycle = [0 for _ in range(self.num_episodes)]
        n_eaten_grass_per_cycle = [0 for _ in range(self.num_episodes)]
        n_born_predator_per_cycle = [0 for _ in range(self.num_episodes)]
        n_born_prey_per_cycle = [0 for _ in range(self.num_episodes)]
        mean_cumulative_rewards = [0 for _ in range(self.num_episodes)]
        mean_cumulative_rewards_predator = [0 for _ in range(self.num_episodes)]
        mean_cumulative_rewards_prey = [0 for _ in range(self.num_episodes)]
        std_cumulative_rewards = [0 for _ in range(self.num_episodes)]
        std_cumulative_rewards_predator = [0 for _ in range(self.num_episodes)]
        std_cumulative_rewards_prey = [0 for _ in range(self.num_episodes)]
        mean_age_predator = [0 for _ in range(self.num_episodes)]
        mean_age_prey = [0 for _ in range(self.num_episodes)]

        total_predator_age_list = []
        total_prey_age_list = []
        for i in range(self.num_episodes):
            env.reset(seed=i)
            possible_predator_name_list = env.predpreygrass.possible_predator_name_list
            possible_prey_name_list = env.predpreygrass.possible_prey_name_list
            possible_agent_name_list = env.predpreygrass.possible_agent_name_list
            cumulative_rewards = {agent: 0 for agent in possible_agent_name_list}
            cumulative_rewards_predator = {
                agent: 0 for agent in possible_predator_name_list
            }
            cumulative_rewards_prey = {agent: 0 for agent in possible_prey_name_list}
            n_aec_cycles = 0
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                cumulative_rewards[agent] += reward
                if agent in possible_predator_name_list:
                    cumulative_rewards_predator[agent] += reward
                elif agent in possible_prey_name_list:
                    cumulative_rewards_prey[agent] += reward

                if termination or truncation:
                    action = None
                    if env.predpreygrass.is_no_predator:
                        predator_extinct_at_termination[i] = 1
                        break
                else:
                    action = model.predict(observation, deterministic=True)[0]
                env.step(action)
            n_aec_cycles = env.predpreygrass.n_aec_cycles
            plotter.plot_population(
                env.predpreygrass.n_active_predator_list,
                env.predpreygrass.n_active_prey_list,
                n_aec_cycles,
                i,
                title="Predator and Prey Population Over Time",
            )

            episode_length[i] = n_aec_cycles
            n_starved_predator_per_cycle[i] = (
                env.predpreygrass.n_starved_predator / n_aec_cycles
            )
            n_starved_prey_per_cycle[i] = (
                env.predpreygrass.n_starved_prey / n_aec_cycles
            )
            n_eaten_prey_per_cycle[i] = env.predpreygrass.n_eaten_prey / n_aec_cycles
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
            mean_age_prey[i] = (
                mean(episode_prey_age_list) if episode_prey_age_list else 0
            )
            total_predator_age_list += episode_predator_age_list
            total_prey_age_list += episode_prey_age_list
            mean_cumulative_rewards[i] = mean(cumulative_rewards.values())
            mean_cumulative_rewards_predator[i] = mean(
                cumulative_rewards_predator.values()
            )
            mean_cumulative_rewards_prey[i] = mean(cumulative_rewards_prey.values())
            std_cumulative_rewards[i] = stdev(cumulative_rewards.values())
            std_cumulative_rewards_predator[i] = stdev(
                cumulative_rewards_predator.values()
            )
            std_cumulative_rewards_prey[i] = stdev(cumulative_rewards_prey.values())

            eval_results_text =(
                f"Eps {i} " +
                f"Lngth = {n_aec_cycles} " +
                f"Strv Prd/cycl = {round(n_starved_predator_per_cycle[i],3)} " +
                f"Strv Pry/cycl = {round(n_starved_prey_per_cycle[i],3)} " +
                f"Eatn Pry/cycl = {round(n_eaten_prey_per_cycle[i],3)} " +
                f"Eatn Gra/cycl = {round(n_eaten_grass_per_cycle[i],3)} " +
                f"Brn Prd/cycl = {round(n_born_predator_per_cycle[i],3)} " +
                f"Brn Pry/cycle = {round(n_born_prey_per_cycle[i],3)} " +
                f"Mn age Prd = {round(mean_age_predator[i],1)} " +
                f"Mn age Pry = {round(mean_age_prey[i],1)}\n"

            )
            print(eval_results_text)
            evaluation_file.write(eval_results_text)
            """
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
            evaluation_file.write(f"Eps {i} ")
            evaluation_file.write(f"Lngth = {n_aec_cycles} ")
            evaluation_file.write(
                f"Strv Prd/cycl = {round(n_starved_predator_per_cycle[i],3)} "
            )
            evaluation_file.write(
                f"Strv Pry/cycl = {round(n_starved_prey_per_cycle[i],3)} "
            )
            evaluation_file.write(
                f"Eeatn Pry/cycl = {round(n_eaten_prey_per_cycle[i],3)} "
            )
            evaluation_file.write(
                f"Eeatn Gra/cycl = {round(n_eaten_grass_per_cycle[i],3)} "
            )
            evaluation_file.write(
                f"Brn Prd/cycl = {round(n_born_predator_per_cycle[i],3)} "
            )
            evaluation_file.write(
                f"Brn Pry/cycl = {round(n_born_prey_per_cycle[i],3)} "
            )
            evaluation_file.write(f"Mn age Prd = {round(mean_age_predator[i],1)} ")
            evaluation_file.write(f"Mn age Pry = {round(mean_age_prey[i],1)}\n")
            """
        print("Finish evaluation.")

        env.close()
        predator_extinct_at_termination_count = sum(predator_extinct_at_termination)
        episode_mean_of_mean_cumulative_rewards = round(
            mean(mean_cumulative_rewards), 1
        )
        episode_mean_of_mean_cumulative_rewards_predators = round(
            mean(mean_cumulative_rewards_predator), 1
        )
        episode_mean_of_mean_cumulative_rewards_prey = round(
            mean(mean_cumulative_rewards_prey), 1
        )
        mean_episode_length = round(mean(episode_length), 1)
        std_episode_length = round(stdev(episode_length), 1) if self.num_episodes > 1 else None
        episode_mean_of_n_starved_predator_per_cycle = round(
            mean(n_starved_predator_per_cycle), 3
        )
        episode_mean_of_n_starved_prey_per_cycle = round(
            mean(n_starved_prey_per_cycle), 3
        )
        episode_mean_of_n_eaten_prey_per_cycle = round(mean(n_eaten_prey_per_cycle), 3)
        episode_mean_of_n_eaten_grass_per_cycle = round(
            mean(n_eaten_grass_per_cycle), 3
        )
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

        # save evaluation results to evaluation_file
        evaluation_file.write("--------------------------\n")
        evaluation_file.write(f"Number of episodes = {self.num_episodes}" + "\n")
        evaluation_file.write(f"Mean episode length = {mean_episode_length}" + "\n")
        evaluation_file.write(
            f"Standard deviation episode length = {std_episode_length}" + "\n"
        )
        evaluation_file.write(
            f"Per episode mean of per agent mean cumulative reward = {episode_mean_of_mean_cumulative_rewards}"
            + "\n"
        )
        evaluation_file.write(
            f"Per episode mean of per Predator mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_predators}"
            + "\n"
        )
        evaluation_file.write(
            f"Per episode mean of per Prey mean cumulative reward = {episode_mean_of_mean_cumulative_rewards_prey}"
            + "\n"
        )
        evaluation_file.write(
            f"% Predator extinct at termination = {round(predator_extinct_at_termination_count/self.num_episodes*100,1)}"
            + "\n"
        )
        evaluation_file.write(
            f"Per episode mean of starved Predator/cycle = {round(episode_mean_of_n_starved_predator_per_cycle,3)}"
            + "\n"
        )
        evaluation_file.write(
            f"Per episode mean of starved_Prey/cycle = {round(episode_mean_of_n_starved_prey_per_cycle,3)}"
            + "\n"
        )
        evaluation_file.write(
            f"Per episode mean of eaten Prey/cycle = {round(episode_mean_of_n_eaten_prey_per_cycle,3)}"
            + "\n"
        )
        evaluation_file.write(
            f"Per episode mean of eaten Grass/cycle = {round(episode_mean_of_n_eaten_grass_per_cycle,3)}"
            + "\n"
        )
        evaluation_file.write(
            f"Per episode mean of born Predator/cycle = {round(episode_mean_of_n_born_predator_per_cycle,3)}"
            + "\n"
        )
        evaluation_file.write(
            f"Per episode_mean_of_born Prey/cycle = {round(episode_mean_of_n_born_prey_per_cycle,3)}"
            + "\n"
        )
        evaluation_file.write(
            f"Per episode mean of mean age Predator = {round(episode_mean_of_mean_age_predator,1)}"
            + "\n"
        )
        evaluation_file.write(
            f"Per episode_mean_of_mean age Prey = {round(episode_mean_of_mean_age_prey,1)}"
            + "\n"
        )
        evaluation_file.write("--------------------------\n")
        evaluation_file.write("Evaluation parameters:\n")
        for item in self.env_kwargs:
            evaluation_file.write(str(item) + " = " + str(self.env_kwargs[item]) + "\n")
        evaluation_file.write("--------------------------\n")
        evaluation_file.close()

        # print to console
        print("--------------------------")
        print(f"Number of episodes = {self.num_episodes}")
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
            f"% Predator extinct at termination = {round(predator_extinct_at_termination_count/self.num_episodes*100,1)}"
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
