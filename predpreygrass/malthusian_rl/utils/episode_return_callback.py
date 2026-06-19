from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
import time
from collections import defaultdict
from collections.abc import Iterable


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0
        self._pending_episode_metrics = []
        self.start_time = time.time()
        self.last_iteration_time = self.start_time
        self.episode_lengths = {}  # manual episode length tracking
        self._episode_los_rejected = {}

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
                    values = obj() if callable(obj) else obj
                    if isinstance(values, Iterable):
                        return list(values)
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
        self._episode_los_rejected[eid] = self._episode_los_rejected.get(eid, 0) + los_count

    def on_episode_end(self, *, episode, metrics_logger: MetricsLogger | None = None, env=None, env_index=None, **kwargs):
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

        for agent_id, agent_episode in episode.agent_episodes.items():
            agent_name = str(agent_id)
            total = agent_episode.get_return()
            if "predator" in agent_name:
                predator_total += total
            elif "prey" in agent_name:
                prey_total += total

            # Match subgroup
            for group in ["type_1_predator", "type_2_predator", "type_1_prey", "type_2_prey"]:
                if group in agent_name:
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
        if metrics_logger is None:
            return

        metrics_logger.log_value("episode_length", episode_length, reduce="mean")
        los_rejected = self._episode_los_rejected.pop(episode_id, 0)
        metrics_logger.log_value("los_rejected_moves", los_rejected, reduce="mean")

        # Log Malthusian diagnostics from env-provided episode summary when available.
        global_info = self._episode_global_info(episode)
        if not global_info:
            global_info = self._episode_global_info_from_env(env=env, env_index=env_index)
        if global_info:
            self._log_malthusian_metrics(global_info, metrics_logger)

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

    def _episode_global_info(self, episode) -> dict:
        """
        Return episode-level info dictionary from env (infos['__all__']) if present.
        """
        infos = self._episode_last_infos(episode)
        if not isinstance(infos, dict):
            return {}
        global_info = infos.get("__all__", {})
        if isinstance(global_info, dict) and global_info:
            return global_info

        # RLlib new API stack may not preserve infos["__all__"] in episodes.
        # Fallback: find a per-agent info payload carrying the episode summary.
        for info in infos.values():
            if not isinstance(info, dict):
                continue
            if (
                "mu_by_species" in info
                or "phi_by_species" in info
                or "counts_by_species" in info
            ):
                return info
        return {}

    def _episode_global_info_from_env(self, env=None, env_index=None) -> dict:
        """
        Best-effort extraction of episode summary directly from env wrappers.
        Needed because some RLlib stacks don't preserve infos['__all__'].
        """
        if env is None:
            return {}

        candidates = [env]
        if hasattr(env, "envs"):
            try:
                candidates.extend(list(getattr(env, "envs")))
            except Exception:
                pass
        if hasattr(env, "get_sub_environments"):
            try:
                candidates.extend(list(env.get_sub_environments()))
            except Exception:
                pass

        if env_index is not None and isinstance(env_index, int):
            try:
                if hasattr(env, "envs") and 0 <= env_index < len(env.envs):
                    candidates.insert(0, env.envs[env_index])
            except Exception:
                pass

        for cand in candidates:
            if cand is None:
                continue
            for obj in (cand, getattr(cand, "unwrapped", None)):
                if obj is None:
                    continue
                summary = getattr(obj, "last_episode_summary", None)
                if isinstance(summary, dict) and summary:
                    return summary
        return {}

    def _log_malthusian_metrics(self, global_info: dict, metrics_logger: MetricsLogger):
        """
        Flatten and log mu/phi/count summaries into RLlib metrics for TensorBoard/Tune.
        """
        mu = global_info.get("mu_by_species", {})
        phi = global_info.get("phi_by_species", {})
        components = global_info.get("phi_components_by_species", {})
        counts = global_info.get("counts_by_species", {})
        switching_costs = global_info.get("switching_cost_by_island", {})
        solitary_returns = global_info.get("solitary_return_by_species", {})
        solitary_counts = global_info.get("solitary_count_by_species", {})

        if isinstance(mu, dict):
            for species, island_map in mu.items():
                if not isinstance(island_map, dict):
                    continue
                for island_id, value in island_map.items():
                    try:
                        v = float(value)
                    except Exception:
                        continue
                    metrics_logger.log_value(
                        f"malthusian/mu/{species}/island_{island_id}",
                        v,
                        reduce="mean",
                    )

        if isinstance(phi, dict):
            for species, island_map in phi.items():
                if not isinstance(island_map, dict):
                    continue
                for island_id, value in island_map.items():
                    try:
                        v = float(value)
                    except Exception:
                        continue
                    metrics_logger.log_value(
                        f"malthusian/phi/{species}/island_{island_id}",
                        v,
                        reduce="mean",
                    )

        if isinstance(components, dict):
            for species, island_map in components.items():
                if not isinstance(island_map, dict):
                    continue
                for island_id, comp_map in island_map.items():
                    if not isinstance(comp_map, dict):
                        continue
                    for comp_name, value in comp_map.items():
                        try:
                            v = float(value)
                        except Exception:
                            continue
                        metrics_logger.log_value(
                            f"malthusian/phi_component/{comp_name}/{species}/island_{island_id}",
                            v,
                            reduce="mean",
                        )

        if isinstance(counts, dict):
            predator_total = 0.0
            prey_total = 0.0
            for species, island_map in counts.items():
                if not isinstance(island_map, dict):
                    continue
                species_total = 0.0
                for island_id, value in island_map.items():
                    try:
                        v = float(value)
                    except Exception:
                        continue
                    species_total += v
                    metrics_logger.log_value(
                        f"malthusian/count/{species}/island_{island_id}",
                        v,
                        reduce="mean",
                    )
                metrics_logger.log_value(
                    f"malthusian/count_total/{species}",
                    species_total,
                    reduce="mean",
                )
                if "predator" in species:
                    predator_total += species_total
                elif "prey" in species:
                    prey_total += species_total

            metrics_logger.log_value(
                "malthusian/count_total/predators",
                predator_total,
                reduce="mean",
            )
            metrics_logger.log_value(
                "malthusian/count_total/prey",
                prey_total,
                reduce="mean",
            )

        if isinstance(switching_costs, dict):
            for island_id, value in switching_costs.items():
                try:
                    v = float(value)
                except Exception:
                    continue
                metrics_logger.log_value(
                    f"malthusian/switching_cost/island_{island_id}",
                        v,
                        reduce="mean",
                    )

        if isinstance(solitary_returns, dict):
            for species, value in solitary_returns.items():
                try:
                    v = float(value)
                except Exception:
                    continue
                metrics_logger.log_value(
                    f"malthusian/solitary_return/{species}",
                    v,
                    reduce="mean",
                )

        if isinstance(solitary_counts, dict):
            for species, value in solitary_counts.items():
                try:
                    v = float(value)
                except Exception:
                    continue
                metrics_logger.log_value(
                    f"malthusian/solitary_count/{species}",
                    v,
                    reduce="mean",
                )
