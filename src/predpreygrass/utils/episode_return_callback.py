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
        self.episode_lengths = {}  # manual episode length tracking

    def on_episode_step(self, *, episode, **kwargs):
        eid = episode.id_
        self.episode_lengths[eid] = self.episode_lengths.get(eid, 0) + 1

    def on_episode_end(self, *, episode, **kwargs):
        """
        Called at the end of each episode.
        Logs the total and average rewards separately for predators and prey.
        """
        self.num_episodes += 1
        episode_return = episode.get_return()
        episode_id = episode.id_
        episode_length = self.episode_lengths.pop(episode_id, 0)
        self.overall_sum_of_rewards += episode_return

        print(f"Episode {self.num_episodes} ended with return: {episode_return:.2f} | Length: {episode_length}")
        #print(f"[DEBUG] episode dir: {dir(episode)}")

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

        # Episode summary log
        print(f"Episode {self.num_episodes} ended with return: {episode_return:.2f} | Length: {episode_length}")
        print(f"Episode {self.num_episodes}: R={episode_return:.2f} | Global SUM={self.overall_sum_of_rewards:.2f}")
        print(f"  - Predators: Total = {predator_total:.2f}")
        print(f"  - Prey:      Total = {prey_total:.2f}")

        for group, totals in group_rewards.items():
            print(f"  - {group}: Total = {sum(totals):.2f}")

        # Store both avg rewards and episode length
        self._pending_episode_metrics.append({
            "length": episode_length,
        })

      
    def on_train_result(self, *, result, **kwargs):
        if self._pending_episode_metrics:
            group_sums = defaultdict(float)
            group_counts = defaultdict(int)
            episode_lengths = []

            for episode_metrics in self._pending_episode_metrics:
                for group, value in episode_metrics.items():
                    group_sums[group] += value
                    group_counts[group] += 1
                episode_lengths.append(episode_metrics["length"])

            for group in group_sums:
                result[f"custom/{group}_avg_reward"] = group_sums[group] / group_counts[group]
                result[f"custom/{group}_total_reward"] = group_sums[group]

            if episode_lengths:
                avg_length = sum(episode_lengths) / len(episode_lengths)
                result["custom/episode_length_avg"] = avg_length

            self._pending_episode_metrics.clear()

        now = time.time()
        total_elapsed = now - self.start_time
        iter_num = result.get("training_iteration", 1)
        iter_time = now - self.last_iteration_time
        self.last_iteration_time = now

        result["timing/iter_minutes"] = iter_time / 60.0
        result["timing/avg_minutes_per_iter"] = total_elapsed / 60.0 / iter_num
        result["timing/total_hours_elapsed"] = total_elapsed / 3600.0