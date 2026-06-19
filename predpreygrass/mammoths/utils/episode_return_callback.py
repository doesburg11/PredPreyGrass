from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
import time
from collections import defaultdict
import numbers


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0
        self._pending_episode_metrics = []
        self.start_time = time.time()
        self.last_iteration_time = self.start_time
        self._last_episode_len_mean = None
        self._last_episode_return_mean = None

    @staticmethod
    def _get_last_infos(episode):
        infos = {}
        if hasattr(episode, "get_last_infos"):
            infos = episode.get_last_infos() or {}
        elif hasattr(episode, "last_infos"):
            infos = episode.last_infos or {}
        if not isinstance(infos, dict):
            infos = {}
        return infos

    @staticmethod
    def _resolve_env(episode, **kwargs):
        def _looks_like_target_env(obj) -> bool:
            if obj is None:
                return False
            return any(
                hasattr(obj, attr)
                for attr in (
                    "team_capture_successes",
                    "team_capture_failures",
                    "team_capture_coop_successes",
                    "team_capture_coop_failures",
                )
            )

        def _safe_int(val, default=0):
            try:
                return int(val)
            except Exception:
                return default

        def _unwrap(obj, env_index=None, max_depth=10):
            seen = set()
            current = obj
            depth = 0
            idx = _safe_int(env_index, 0)

            while current is not None and depth < max_depth:
                depth += 1
                obj_id = id(current)
                if obj_id in seen:
                    break
                seen.add(obj_id)

                if _looks_like_target_env(current):
                    return current

                if isinstance(current, (list, tuple)):
                    if current:
                        current = current[idx] if 0 <= idx < len(current) else current[0]
                        continue
                    return None

                unwrapped = getattr(current, "unwrapped", None)
                if unwrapped is not None and unwrapped is not current:
                    current = unwrapped
                    continue

                if hasattr(current, "get_sub_environments"):
                    try:
                        subs = current.get_sub_environments() or []
                    except Exception:
                        subs = []
                    if subs:
                        current = subs[idx] if 0 <= idx < len(subs) else subs[0]
                        continue

                for attr in ("envs", "_envs"):
                    subs = getattr(current, attr, None)
                    if isinstance(subs, (list, tuple)) and subs:
                        current = subs[idx] if 0 <= idx < len(subs) else subs[0]
                        break
                else:
                    subs = None
                if subs is not None:
                    continue

                for attr in ("env", "_env", "vector_env", "_vector_env"):
                    inner = getattr(current, attr, None)
                    if inner is not None and inner is not current:
                        current = inner
                        break
                else:
                    break

            return current if _looks_like_target_env(current) else None

        env_index = kwargs.get("env_index")
        if env_index is None:
            env_index = getattr(episode, "env_id", None)
        if env_index is None:
            env_index = getattr(episode, "env_index", None)

        env = kwargs.get("env")
        resolved = _unwrap(env, env_index)
        if resolved is not None:
            return resolved

        env_runner = kwargs.get("env_runner") or kwargs.get("runner")
        if env_runner is not None:
            for attr in ("env", "_env", "envs", "_envs", "vector_env", "_vector_env", "base_env"):
                candidate = getattr(env_runner, attr, None)
                resolved = _unwrap(candidate, env_index)
                if resolved is not None:
                    return resolved

        base_env = kwargs.get("base_env")
        if base_env is None:
            worker = kwargs.get("worker")
            if worker is not None:
                base_env = getattr(worker, "base_env", None)
        resolved = _unwrap(base_env, env_index)
        if resolved is not None:
            return resolved

        return None

    def on_episode_end(self, *, episode, metrics_logger: MetricsLogger, **kwargs):
        """
        Called at the end of each episode.
        Logs the total and average rewards separately for predators and prey.
        """
        self.num_episodes += 1
        episode_return = episode.get_return()
        self.overall_sum_of_rewards += episode_return
        infos = self._get_last_infos(episode)
        info_all = infos.get("__all__", {}) if isinstance(infos, dict) else {}
        if not info_all and infos:
            info_all = next(iter(infos.values()), {})

        coop_success = info_all.get("team_capture_coop_successes")
        coop_fail = info_all.get("team_capture_coop_failures")
        total_success = info_all.get("team_capture_successes")
        total_fail = info_all.get("team_capture_failures")

        env_for_capture = self._resolve_env(episode, **kwargs)
        if env_for_capture is not None:
            env_coop_success = getattr(env_for_capture, "team_capture_coop_successes", None)
            env_coop_fail = getattr(env_for_capture, "team_capture_coop_failures", None)
            env_total_success = getattr(env_for_capture, "team_capture_successes", None)
            env_total_fail = getattr(env_for_capture, "team_capture_failures", None)

            def _pick_value(info_val, env_val):
                if env_val is None:
                    return info_val
                if info_val is None or not isinstance(info_val, numbers.Real):
                    return env_val
                try:
                    return env_val if env_val > info_val else info_val
                except Exception:
                    return env_val

            coop_success = _pick_value(coop_success, env_coop_success)
            coop_fail = _pick_value(coop_fail, env_coop_fail)
            total_success = _pick_value(total_success, env_total_success)
            total_fail = _pick_value(total_fail, env_total_fail)

        coop_success = coop_success or 0
        coop_fail = coop_fail or 0
        total_success = total_success or 0
        total_fail = total_fail or 0

        coop_denom = coop_success + coop_fail
        total_denom = total_success + total_fail
        coop_success_rate = (coop_success / coop_denom * 100.0) if coop_denom else 0.0
        total_success_rate = (total_success / total_denom * 100.0) if total_denom else 0.0
        if metrics_logger is not None:
            metrics_logger.log_value("custom_metrics/team_capture_successes", coop_success)
            metrics_logger.log_value("custom_metrics/team_capture_failures", coop_fail)
            metrics_logger.log_value("custom_metrics/team_capture_total_successes", total_success)
            metrics_logger.log_value("custom_metrics/team_capture_total_failures", total_fail)
            metrics_logger.log_value("custom_metrics/team_capture_successes_rate", coop_success_rate)
            metrics_logger.log_value(
                "custom_metrics/team_capture_total_successes_rate", total_success_rate
            )

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
        print(f"Episode {self.num_episodes}: R={episode_return:.2f} | Global SUM={self.overall_sum_of_rewards:.2f}")
        print(f"  - Coop: successes={coop_success} failures={coop_fail}")
        print(f"  - Total capture: successes={total_success} failures={total_fail}")
        print(f"  - Predators: Total = {predator_total:.2f}")
        print(f"  - Prey:      Total = {prey_total:.2f}")

        for group, totals in group_rewards.items():
            print(f"  - {group}: Total = {sum(totals):.2f}")

    def on_train_result(self, *, result, **kwargs):
        # Add training time metrics
        now = time.time()
        total_elapsed = now - self.start_time
        iter_num = result.get("training_iteration", 1)
        iter_time = now - self.last_iteration_time
        self.last_iteration_time = now

        timing_iter_minutes = iter_time / 60.0
        timing_avg_minutes_per_iter = total_elapsed / 60.0 / iter_num
        timing_total_hours_elapsed = total_elapsed / 3600.0

        custom_metrics = result.get("custom_metrics")
        if not isinstance(custom_metrics, dict):
            custom_metrics = {}
            result["custom_metrics"] = custom_metrics

        custom_metrics["timing_iter_minutes"] = timing_iter_minutes
        custom_metrics["timing_avg_minutes_per_iter"] = timing_avg_minutes_per_iter
        custom_metrics["timing_total_hours_elapsed"] = timing_total_hours_elapsed

        # Stable display metrics: carry forward last non-zero episode stats.
        env_stats = result.get("env_runners", {})
        if not isinstance(env_stats, dict):
            env_stats = {}
        num_eps = env_stats.get("num_episodes", result.get("num_episodes", 0)) or 0
        len_mean = env_stats.get(
            "episode_len_mean", result.get("episode_len_mean")
        )
        ret_mean = env_stats.get(
            "episode_return_mean", result.get("episode_reward_mean")
        )
        if num_eps and len_mean is not None:
            self._last_episode_len_mean = len_mean
            self._last_episode_return_mean = ret_mean


