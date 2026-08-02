from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
import time
from collections import defaultdict
import numpy as np


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0
        self._pending_episode_metrics = []
        self.start_time = time.time()
        self.last_iteration_time = self.start_time
        # rely on built-in RLlib episode length metrics instead of manual counting

    def on_episode_end(self, *, episode, metrics_logger: MetricsLogger, **kwargs):
        """
        Called at the end of each episode.
        Logs the total and average rewards separately for predators and prey.
        """
        self.num_episodes += 1
        episode_return = episode.get_return()
        episode_length = getattr(episode, "length", 0)
        self.overall_sum_of_rewards += episode_return

        # Accumulate rewards by group
        group_rewards = defaultdict(list)
        predator_total = prey_total = 0.0
        predator_totals = []
        prey_totals = []

        for agent_id, rewards in episode.get_rewards().items():
            total = sum(rewards)
            if "predator" in agent_id:
                predator_total += total
                predator_totals.append(total)
            elif "prey" in agent_id:
                prey_total += total
                prey_totals.append(total)

            # Match subgroup
            for group in ["type_1_predator", "type_2_predator", "type_1_prey", "type_2_prey"]:
                if group in agent_id:
                    group_rewards[group].append(total)
                    break

        # Episode summary log
        print(
            f"Episode {self.num_episodes}: Length: {episode_length} | R={episode_return:.2f} | Global SUM={self.overall_sum_of_rewards:.2f}"
        )
        print(f"  - Predators: Total = {predator_total:.2f}")
        print(f"  - Prey:      Total = {prey_total:.2f}")

        for group, totals in group_rewards.items():
            print(f"  - {group}: Total = {sum(totals):.2f}")

        # Percentile scalars for TensorBoard (appears under Scalars tab)
        if predator_totals:
            p25, p50, p75 = np.percentile(predator_totals, [25, 50, 75])
            metrics_logger.log_value("predator_episode_return_p25", float(p25))
            metrics_logger.log_value("predator_episode_return_p50", float(p50))
            metrics_logger.log_value("predator_episode_return_p75", float(p75))

        if prey_totals:
            p25, p50, p75 = np.percentile(prey_totals, [25, 50, 75])
            metrics_logger.log_value("prey_episode_return_p25", float(p25))
            metrics_logger.log_value("prey_episode_return_p50", float(p50))
            metrics_logger.log_value("prey_episode_return_p75", float(p75))

        # Optional: normalize returns by lifetime (if the env exposes lifetime_steps in infos)
        infos = self._episode_last_infos(episode)
        predator_lifetimes, predator_return_per_life = [], []
        prey_lifetimes, prey_return_per_life = [], []
        for aid, info in infos.items():
            if not isinstance(info, dict):
                continue
            life = info.get("lifetime_steps")
            final_ret = info.get("final_cumulative_reward")
            if life is None or final_ret is None or life <= 0:
                continue
            ret_per_life = float(final_ret) / float(life)
            if "predator" in aid:
                predator_lifetimes.append(life)
                predator_return_per_life.append(ret_per_life)
            elif "prey" in aid:
                prey_lifetimes.append(life)
                prey_return_per_life.append(ret_per_life)

        if predator_lifetimes:
            metrics_logger.log_value("predator_lifetime_steps_median", float(np.median(predator_lifetimes)))
            metrics_logger.log_value("predator_return_per_lifetime_mean", float(np.mean(predator_return_per_life)))
        if prey_lifetimes:
            metrics_logger.log_value("prey_lifetime_steps_median", float(np.median(prey_lifetimes)))
            metrics_logger.log_value("prey_return_per_lifetime_mean", float(np.mean(prey_return_per_life)))

        # RLlib already emits episode_len_* metrics for TensorBoard; no extra episode-length metrics_logger entry needed here

    def on_train_result(self, *, result, **kwargs):
        # Add training time metrics
        now = time.time()
        total_elapsed = now - self.start_time
        iter_num = result.get("training_iteration", 1)
        iter_time = now - self.last_iteration_time
        self.last_iteration_time = now

        result["timing/iter_minutes"] = iter_time / 60.0
        result["timing/avg_minutes_per_iter"] = total_elapsed / 60.0 / iter_num
        result["timing/total_hours_elapsed"] = total_elapsed / 3600.0
        # Optional: surface custom metric if available in learner results aggregation
        # (Ray will automatically aggregate logged metrics like los_rejected_moves across episodes)

    # ---- Compatibility helpers ----
    def _episode_last_infos(self, episode) -> dict:
        """
        Return a mapping of agent_id -> last info dict for this episode, with
        compatibility across RLlib API changes.
        """
        for name in ("get_last_infos", "get_infos"):
            if hasattr(episode, name):
                try:
                    infos = getattr(episode, name)()
                    if isinstance(infos, dict):
                        return infos
                except Exception:
                    pass
        for name in ("last_infos", "infos"):
            if hasattr(episode, name):
                try:
                    infos = getattr(episode, name)
                    if isinstance(infos, dict):
                        return infos
                except Exception:
                    pass
        if hasattr(episode, "_agent_to_last_info"):
            try:
                mapping = getattr(episode, "_agent_to_last_info")
                if isinstance(mapping, dict):
                    return mapping
            except Exception:
                pass
        return {}
