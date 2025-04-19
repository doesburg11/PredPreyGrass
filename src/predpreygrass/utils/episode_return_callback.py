from ray.rllib.callbacks.callbacks import RLlibCallback
import time


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0

    def on_episode_end(self, *, episode, **kwargs):
        """
        Called at the end of each episode.
        Logs the total and average rewards separately for predators and prey.
        """
        self.num_episodes += 1
        self.overall_sum_of_rewards += episode.get_return()

        # Initialize reward tracking
        predator_total_reward = 0.0
        prey_total_reward = 0.0
        predator_count = 0
        prey_count = 0

        # Retrieve rewards
        rewards = episode.get_rewards()  # Dictionary of {agent_id: list_of_rewards}

        for agent_id, reward_list in rewards.items():
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

    def on_train_result(self, *, result, **kwargs):
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
