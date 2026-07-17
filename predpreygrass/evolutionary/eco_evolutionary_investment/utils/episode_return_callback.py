from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
import time
from collections import defaultdict
from typing import Any, Optional
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

    def on_episode_end(
        self,
        *,
        episode,
        metrics_logger: Optional[MetricsLogger] = None,
        env=None,
        env_index: int = 0,
        **kwargs,
    ):
        """
        Called at the end of each episode.
        Logs the total and average rewards separately for predators and prey.
        """
        self.num_episodes += 1
        episode_return = episode.get_return()
        episode_length = episode.env_steps() if callable(getattr(episode, "env_steps", None)) else 0
        self.overall_sum_of_rewards += episode_return

        def log_value(name: str, value: float):
            if metrics_logger is not None:
                metrics_logger.log_value(name, float(value))

        # Accumulate rewards by group
        group_rewards = defaultdict(list)
        predator_total = prey_total = 0.0
        predator_totals = []
        prey_totals = []

        # episode.get_rewards() indexes a global-step-aligned lookback buffer that
        # can go out of range for agents whose local step count diverges from the
        # episode's (e.g. short-lived offspring under a low fixed investment
        # fraction) -- causes recurring env-runner crashes (see RESULTS.md).
        # agent_episodes + per-agent get_return() avoids that global indexing.
        for agent_id, agent_ep in episode.agent_episodes.items():
            total = agent_ep.get_return()
            if "predator" in agent_id:
                predator_total += total
                predator_totals.append(total)
                group_rewards["predator"].append(total)
            elif "prey" in agent_id:
                prey_total += total
                prey_totals.append(total)
                group_rewards["prey"].append(total)

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
            log_value("predator_episode_return_p25", float(p25))
            log_value("predator_episode_return_p50", float(p50))
            log_value("predator_episode_return_p75", float(p75))

        if prey_totals:
            p25, p50, p75 = np.percentile(prey_totals, [25, 50, 75])
            log_value("prey_episode_return_p25", float(p25))
            log_value("prey_episode_return_p50", float(p50))
            log_value("prey_episode_return_p75", float(p75))

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
            log_value("predator_lifetime_steps_median", float(np.median(predator_lifetimes)))
            log_value("predator_return_per_lifetime_mean", float(np.mean(predator_return_per_life)))
        if prey_lifetimes:
            log_value("prey_lifetime_steps_median", float(np.median(prey_lifetimes)))
            log_value("prey_return_per_lifetime_mean", float(np.mean(prey_return_per_life)))

        training_metrics = self._extract_training_metrics(infos, env=env, env_index=env_index, **kwargs)
        for metric_name, metric_value in training_metrics.items():
            if isinstance(metric_value, (int, float, np.integer, np.floating)):
                log_value(f"eco_evolution/{metric_name}", float(metric_value))

        # RLlib already emits episode_len_* metrics for TensorBoard; no extra episode-length metrics_logger entry needed here

    def on_episode_step(
        self,
        *,
        episode,
        metrics_logger: Optional[MetricsLogger] = None,
        env=None,
        env_index: int = 0,
        **kwargs,
    ):
        if metrics_logger is None:
            return
        resolved_env = self._resolve_env(env=env, env_index=env_index, **kwargs)
        live_metrics = getattr(resolved_env, "_last_live_investment_metrics", None)
        if not isinstance(live_metrics, dict):
            return
        for metric_name, metric_value in live_metrics.items():
            if isinstance(metric_value, (int, float, np.integer, np.floating)):
                metrics_logger.log_value(f"live_investment/{metric_name}", float(metric_value))

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
        # Ray will automatically aggregate logged metrics across episodes.

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

    @staticmethod
    def _resolve_env(env=None, env_index: int = 0, **kwargs) -> Any:
        """Return the underlying PredPreyGrass env from RLlib wrapper shapes."""

        def looks_like_metrics_env(obj) -> bool:
            return obj is not None and hasattr(obj, "_build_episode_training_metrics")

        def safe_index(value) -> int:
            try:
                return int(value)
            except Exception:
                return 0

        def unwrap(candidate, index: int):
            current = candidate
            seen = set()
            for _ in range(10):
                if current is None:
                    return None
                if id(current) in seen:
                    return None
                seen.add(id(current))

                if looks_like_metrics_env(current):
                    return current

                if isinstance(current, (list, tuple)):
                    if not current:
                        return None
                    current = current[index] if 0 <= index < len(current) else current[0]
                    continue

                unwrapped = getattr(current, "unwrapped", None)
                if unwrapped is not None and unwrapped is not current:
                    current = unwrapped
                    continue

                if hasattr(current, "get_sub_environments"):
                    try:
                        sub_envs = current.get_sub_environments() or []
                    except Exception:
                        sub_envs = []
                    if sub_envs:
                        current = sub_envs[index] if 0 <= index < len(sub_envs) else sub_envs[0]
                        continue

                for attr in ("envs", "_envs"):
                    sub_envs = getattr(current, attr, None)
                    if isinstance(sub_envs, (list, tuple)) and sub_envs:
                        current = sub_envs[index] if 0 <= index < len(sub_envs) else sub_envs[0]
                        break
                else:
                    sub_envs = None
                if sub_envs is not None:
                    continue

                for attr in ("env", "_env", "vector_env", "_vector_env", "base_env"):
                    inner = getattr(current, attr, None)
                    if inner is not None and inner is not current:
                        current = inner
                        break
                else:
                    return None

            return None

        index = safe_index(env_index)
        for candidate in (
            env,
            kwargs.get("env_runner"),
            getattr(kwargs.get("env_runner"), "env", None),
            kwargs.get("worker"),
            getattr(kwargs.get("worker"), "env", None),
            kwargs.get("base_env"),
        ):
            resolved = unwrap(candidate, index)
            if resolved is not None:
                return resolved
        return None

    def _extract_training_metrics(self, infos: dict, env=None, env_index: int = 0, **kwargs) -> dict:
        if not isinstance(infos, dict):
            infos = {}
        else:
            all_info = infos.get("__all__")
            if isinstance(all_info, dict):
                metrics = all_info.get("training_metrics")
                if isinstance(metrics, dict):
                    return metrics
            metrics = infos.get("training_metrics")
            if isinstance(metrics, dict):
                return metrics

        resolved_env = self._resolve_env(env=env, env_index=env_index, **kwargs)
        build_metrics = getattr(resolved_env, "_build_episode_training_metrics", None)
        if callable(build_metrics):
            metrics = build_metrics()
            if isinstance(metrics, dict):
                return metrics
        return {}
