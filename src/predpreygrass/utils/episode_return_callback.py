from ray.rllib.callbacks.callbacks import RLlibCallback
import time
from collections import defaultdict


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0
        self._pending_episode_metrics = []
        self.start_time = time.time()
        self.last_iteration_time = self.start_time


    def on_episode_end(self, *, episode, **kwargs):
        """
        Called at the end of each episode.
        Logs the total and average rewards separately for predators and prey.
        """
        self.num_episodes += 1
        episode_return = episode.get_return()
        self.overall_sum_of_rewards += episode_return

        # Accumulate rewards by group
        group_rewards = defaultdict(list)
        predator_total = prey_total = 0.0
        predator_count = prey_count = 0

        for agent_id, rewards in episode.get_rewards().items():
            total = sum(rewards)
            if "predator" in agent_id:
                predator_total += total
                predator_count += 1
            elif "prey" in agent_id:
                prey_total += total
                prey_count += 1

            # Match subgroup
            for group in ["speed_1_predator", "speed_2_predator", "speed_1_prey", "speed_2_prey"]:
                if group in agent_id:
                    group_rewards[group].append(total)
                    break

        # Compute average rewards (avoid division by zero)
        predator_avg = predator_total / predator_count if predator_count else 0.0
        prey_avg = prey_total / prey_count if prey_count else 0.0

        # Episode summary log
        print(f"Episode {self.num_episodes}: R={episode_return:.2f} | Global SUM={self.overall_sum_of_rewards:.2f}")
        print(f"  - Predators: Total = {predator_total:.2f}, Avg = {predator_avg:.2f}")
        print(f"  - Prey:      Total = {prey_total:.2f}, Avg = {prey_avg:.2f}")

        avg_metrics = {}
        for group, totals in group_rewards.items():
            avg = sum(totals) / len(totals) if totals else 0.0
            avg_metrics[group] = avg
            print(f"  - {group}: Total = {sum(totals):.2f}, Avg = {avg:.2f},  Agents = {len(totals)}")

        self._pending_episode_metrics.append(avg_metrics)

      
    def on_train_result(self, *, result, **kwargs):
        if self._pending_episode_metrics:
            group_sums = defaultdict(float)
            group_counts = defaultdict(int)

            for episode_metrics in self._pending_episode_metrics:
                for group, value in episode_metrics.items():
                    group_sums[group] += value
                    group_counts[group] += 1

            for group in group_sums:
                result[f"custom/{group}_avg_reward"] = group_sums[group] / group_counts[group]
                result[f"custom/{group}_total_reward"] = group_sums[group]

            self._pending_episode_metrics.clear()

        now = time.time()
        total_elapsed = now - self.start_time
        iter_num = result.get("training_iteration", 1)
        iter_time = now - self.last_iteration_time
        self.last_iteration_time = now

        result["timing/iter_minutes"] = iter_time / 60.0
        result["timing/avg_minutes_per_iter"] = total_elapsed / 60.0 / iter_num
        result["timing/total_hours_elapsed"] = total_elapsed / 3600.0