from ray.rllib.callbacks.callbacks import RLlibCallback
import time


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0
        self._pending_episode_metrics = []

    def on_episode_end(self, *, episode, env, env_index, **kwargs):
        """
        Called at the end of each episode.
        Logs the total and average rewards separately for predators and prey.
        """
        self.num_episodes += 1
        self.overall_sum_of_rewards += episode.get_return()

        # Initialize reward tracking for predators and prey
        predator_total_reward = 0.0
        prey_total_reward = 0.0
        predator_count = 0
        prey_count = 0

        # Retrieve rewards
        #rewards = episode.get_rewards()  # Dictionary of {agent_id: list_of_rewards}

        for agent_id, reward_list in episode.get_rewards().items():
            total_reward = sum(reward_list)  # Sum all rewards for the episode

            if "predator" in agent_id:
                predator_total_reward += total_reward
                predator_count += 1
            elif "prey" in agent_id:
                prey_total_reward += total_reward
                prey_count += 1

        # Compute average rewards (avoid division by zero)
        predator_avg_reward = predator_total_reward / predator_count if predator_count > 0 else 0
        prey_avg_reward = prey_total_reward / prey_count if prey_count > 0 else 0

        # Print episode logs
        print(f"Episode {self.num_episodes}: R={episode.get_return()} Global SUM={self.overall_sum_of_rewards}")
        print(f"  - Predators: Total Reward = {predator_total_reward:.2f}, Avg Reward = {predator_avg_reward:.2f}")
        print(f"  - Prey: Total Reward = {prey_total_reward:.2f}, Avg Reward = {prey_avg_reward:.2f}")

        # Initialize reward tracking for all subgroups
        group_rewards = {
            "speed_1_predator": [],
            "speed_2_predator": [],
            "speed_1_prey": [],
            "speed_2_prey": [],
        }

        for agent_id, rewards in episode.get_rewards().items():
            total = sum(rewards)
            if "speed_1_predator" in agent_id:
                group_rewards["speed_1_predator"].append(total)
            elif "speed_2_predator" in agent_id:
                group_rewards["speed_2_predator"].append(total)
            elif "speed_1_prey" in agent_id:
                group_rewards["speed_1_prey"].append(total)
            elif "speed_2_prey" in agent_id:
                group_rewards["speed_2_prey"].append(total)

        avg_rewards = {
            k: (sum(v) / len(v) if v else 0.0)
            for k, v in group_rewards.items()
        }
        total_rewards = {
            k: (sum(v) if v else 0.0)
            for k, v in group_rewards.items()
        }

        # Queue it for logging in `on_train_result`
        self._pending_episode_metrics.append(avg_rewards)

        # Console logging
        print(f"\n Episode summary:")
        print(f"  Global return: {episode.get_return():.2f}")
        for group, avg in avg_rewards.items():
            print(f"  {group}: Avg reward = {avg:.2f} over {len(group_rewards[group])} agents")
        for group, total in total_rewards.items():
            print(f"  {group}: Total reward = {total:.2f} over {len(group_rewards[group])} agents")

      
    def on_train_result(self, *, result, **kwargs):
        # Aggregate and log pending episode metrics
        if self._pending_episode_metrics:
            group_sums = {}
            group_counts = {}

            for episode_metrics in self._pending_episode_metrics:
                for group, value in episode_metrics.items():
                    group_sums[group] = group_sums.get(group, 0.0) + value
                    group_counts[group] = group_counts.get(group, 0) + 1

            for group in group_sums:
                result[f"custom/{group}_avg_reward"] = group_sums[group] / group_counts[group]
            for group in group_sums:
                result[f"custom/{group}_total_reward"] = group_sums[group] 

            # Clear buffer
            self._pending_episode_metrics.clear()

        # Lazy initialization: ensures timing vars exist in all worker contexts
        if not hasattr(self, "start_time"):
            self.start_time = time.time()
            self.last_iteration_time = self.start_time

        now = time.time()
        total_elapsed = now - self.start_time
        iter_num = result["training_iteration"]
        avg_time_per_iter = total_elapsed / iter_num if iter_num > 0 else 0

        iter_time = now - self.last_iteration_time
        self.last_iteration_time = now

        # Log to TensorBoard
        result["timing/iter_minutes"] = iter_time / 60.0 
        result["timing/avg_minutes_per_iter"] = avg_time_per_iter / 60.0
        result["timing/total_hours_elapsed"] = total_elapsed / 3600.0
