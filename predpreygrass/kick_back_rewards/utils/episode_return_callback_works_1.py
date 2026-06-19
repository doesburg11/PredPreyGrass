from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
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

    def _episode_agent_ids(self, episode) -> list:
        """
        Return a list of agent IDs for the current episode in a way that's
        compatible with RLlib's evolving Episode APIs.
        """
        # Try common stable attributes/methods first
        for attr in ("agent_ids", "get_agent_ids", "get_agents"):
            if hasattr(episode, attr):
                obj = getattr(episode, attr)
                try:
                    return list(obj() if callable(obj) else obj)
                except Exception:
                    pass
        # Last resort: peek into known internal mapping used by last_info_for
        if hasattr(episode, "last_info_for") and hasattr(episode, "_agent_to_last_info"):
            try:
                return list(getattr(episode, "_agent_to_last_info").keys())
            except Exception:
                pass
        return []

    def on_episode_step(self, *, episode, **kwargs):
        eid = episode.id_
        self.episode_lengths[eid] = self.episode_lengths.get(eid, 0) + 1

    def on_episode_end(self, *, episode, metrics_logger: MetricsLogger, **kwargs):
        """
        Called at the end of each episode.
        Logs the total and average rewards separately for predators and prey.
        """
        self.num_episodes += 1
        episode_return = episode.get_return()
        episode_id = episode.id_
        episode_length = self.episode_lengths.pop(episode_id, 0)
        self.overall_sum_of_rewards += episode_return

        # Accumulate rewards by group
        group_rewards = defaultdict(list)
        predator_total = prey_total = 0.0

        for agent_id, rewards in episode.get_rewards().items():
            total = sum(rewards)
            if "predator" in agent_id:
                predator_total += total
            elif "prey" in agent_id:
                prey_total += total

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

        # Log episode length and LOS-rejected count using MetricsLogger
        metrics_logger.log_value("episode_length", episode_length, reduce="mean")

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
        # Preferred: public getters
        for name in ("get_last_infos", "get_infos"):
            if hasattr(episode, name):
                try:
                    infos = getattr(episode, name)()
                    if isinstance(infos, dict):
                        return infos
                except Exception:
                    pass
        # Common attrs in some versions
        for name in ("last_infos", "infos"):
            if hasattr(episode, name):
                try:
                    infos = getattr(episode, name)
                    if isinstance(infos, dict):
                        return infos
                except Exception:
                    pass
        # Fallback: internal mapping used by older last_info_for implementations
        if hasattr(episode, "_agent_to_last_info"):
            try:
                mapping = getattr(episode, "_agent_to_last_info")
                if isinstance(mapping, dict):
                    return mapping
            except Exception:
                pass
        return {}
