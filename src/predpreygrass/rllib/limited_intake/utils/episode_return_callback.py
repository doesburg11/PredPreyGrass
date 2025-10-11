from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
import time
from collections import defaultdict

import os
import json



class EpisodeReturn(RLlibCallback):
    def __init__(self, log_trajectories=False):
        super().__init__()
        self.log_trajectories = log_trajectories
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0
        self._pending_episode_metrics = []
        self.start_time = time.time()
        self.last_iteration_time = self.start_time
        self.episode_lengths = {}  # manual episode length tracking
        self._episode_los_rejected = {}
        # Sharing analytics: resource_consumers[resource_id] = {consumers: {agent: amount}, first: step, last: step}
        self._episode_resource_consumers = {}

    def _append_agent_trajectories(self, episode):
        # Accumulate per-agent, per-step info for this episode
        infos_map = self._episode_last_infos(episode)
        agent_ids = self._episode_agent_ids(episode)
        episode_steps = episode.length if hasattr(episode, 'length') else self.episode_lengths.get(episode.id_, 0)
        # Build: {unique_id, agent_id, episode_id, step, ...fields from infos...}
        per_agent_trajectories = []
        for step in range(episode_steps):
            # RLlib does not expose all per-step infos, so we use only last info for each agent
            for agent_id in agent_ids:
                info = infos_map.get(agent_id, {})
                if not info:
                    continue
                traj = {
                    "unique_id": info.get("unique_id", agent_id),
                    "agent_id": agent_id,
                    "episode_id": episode.id_,
                    "step": step,
                }
                # Always-logged fields
                for k in ["energy", "energy_decay", "energy_movement", "energy_eating", "energy_reproduction", "age", "offspring_count", "offspring_ids"]:
                    if k in info:
                        traj[k] = info[k]
                # Event fields (movement, death, reproduction, reward, consumption_log, etc.)
                for k in ["movement", "death", "reproduction", "reward", "consumption_log", "move_blocked_reason", "los_rejected"]:
                    if k in info:
                        traj[k] = info[k]
                per_agent_trajectories.append(traj)
        # Write each episode's trajectories to a separate file to avoid concurrent write corruption
        out_dir = os.path.join(os.path.dirname(__file__), '../trajectories_output')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.abspath(os.path.join(out_dir, f'agent_trajectories_{episode.id_}.json'))
        try:
            with open(out_path, 'w') as f:
                json.dump(per_agent_trajectories, f, indent=2)
        except Exception as e:
            print(f"[TrajectoryLogger] Failed to write trajectories for episode {episode.id_}: {e}")

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
        # Aggregate per-agent infos for LOS rejections into an episode counter
        los_count = 0
        infos_map = self._episode_last_infos(episode)
        if isinstance(infos_map, dict):
            for info in infos_map.values():
                if info and isinstance(info, dict):
                    los_count += int(info.get("los_rejected", 0))
                    # --- Sharing analytics: look for 'consumption_log' in info ---
                    consumption_log = info.get("consumption_log")
                    if consumption_log and isinstance(consumption_log, list):
                        for event in consumption_log:
                            # event: (resource_id, agent_id, bite_size, step)
                            resource_id, agent_id, bite_size, step = event
                            rec = self._episode_resource_consumers.setdefault(resource_id, {"consumers": {}, "first": None, "last": None})
                            rec["consumers"][agent_id] = rec["consumers"].get(agent_id, 0.0) + bite_size
                            if rec["first"] is None:
                                rec["first"] = step
                            rec["last"] = step
        self._episode_los_rejected[eid] = self._episode_los_rejected.get(eid, 0) + los_count

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

        # --- Trajectory logging ---
        if self.log_trajectories:
            self._append_agent_trajectories(episode)

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
        los_rejected = self._episode_los_rejected.pop(episode_id, 0)
        metrics_logger.log_value("los_rejected_moves", los_rejected, reduce="mean")

        # --- Sharing analytics: compute and log sharing summary ---
        consumers_per_resource = []
        sharing_delays = []
        gini_per_resource = []
        n_shared = 0
        for res_id, rec in self._episode_resource_consumers.items():
            n_cons = len(rec["consumers"])
            consumers_per_resource.append(n_cons)
            if n_cons > 1:
                n_shared += 1
                sharing_delays.append(rec["last"] - rec["first"])
            # Gini index for bite shares
            bites = list(rec["consumers"].values())
            if bites:
                sorted_bites = sorted(bites)
                n = len(bites)
                s = sum(sorted_bites)
                gini = (sum(abs(x - y) for x in sorted_bites for y in sorted_bites)) / (2 * n * s) if s > 0 else 0.0
                gini_per_resource.append(gini)
        total = len(self._episode_resource_consumers)
        shared_fraction = n_shared / total if total > 0 else 0.0
        avg_consumers = float(sum(consumers_per_resource) / len(consumers_per_resource)) if consumers_per_resource else 0.0
        avg_sharing_delay = float(sum(sharing_delays) / len(sharing_delays)) if sharing_delays else 0.0
        avg_gini = float(sum(gini_per_resource) / len(gini_per_resource)) if gini_per_resource else 0.0
        metrics_logger.log_value("sharing/shared_fraction", shared_fraction, reduce="mean")
        metrics_logger.log_value("sharing/avg_consumers_per_resource", avg_consumers, reduce="mean")
        metrics_logger.log_value("sharing/avg_sharing_delay", avg_sharing_delay, reduce="mean")
        metrics_logger.log_value("sharing/avg_share_gini", avg_gini, reduce="mean")
        metrics_logger.log_value("sharing/n_resources", total, reduce="mean")
        # Reset for next episode
        self._episode_resource_consumers = {}

        # Note: Sharing metrics are fully accumulated in this callback from infos['consumption_log'].
        # No direct env access is needed (or used) in the new RLlib stack.

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
