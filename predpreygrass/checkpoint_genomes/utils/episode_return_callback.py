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
    def _get_last_actions(episode):
        actions = {}
        if hasattr(episode, "get_last_actions"):
            actions = episode.get_last_actions() or {}
        elif hasattr(episode, "last_actions"):
            actions = episode.last_actions or {}
        elif hasattr(episode, "_agent_to_last_action"):
            actions = episode._agent_to_last_action or {}
        if not isinstance(actions, dict):
            actions = {}
        return actions

    @staticmethod
    def _resolve_env(episode, **kwargs):
        """Best-effort: return the *underlying* (unwrapped) sub-environment.

        In RLlib's EnvRunner/VectorEnv stacks, `env_runner.env` is often a wrapper
        that does not expose our debug/metrics attributes (e.g. `per_step_agent_data`).
        We therefore aggressively unwrap common container/wrapper shapes.
        """

        def _looks_like_target_env(obj) -> bool:
            if obj is None:
                return False
            # Attributes we rely on for fallbacks.
            return any(
                hasattr(obj, attr)
                for attr in (
                    "per_step_agent_data",
                    "agent_event_log",
                    "team_capture_successes",
                    "team_capture_coop_successes",
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

                # Common container shapes.
                if isinstance(current, (list, tuple)):
                    if current:
                        current = current[idx] if 0 <= idx < len(current) else current[0]
                        continue
                    return None

                # Gymnasium-style unwrapping.
                unwrapped = getattr(current, "unwrapped", None)
                if unwrapped is not None and unwrapped is not current:
                    current = unwrapped
                    continue

                # RLlib BaseEnv / VectorEnv access.
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

                # Single-env wrappers.
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

        # Direct env passed in.
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

    @staticmethod
    def _get_user_data(episode):
        # RLlib >=2.9 uses custom_data; older versions expose user_data.
        user_data = getattr(episode, "user_data", None)
        if isinstance(user_data, dict):
            return user_data
        user_data = getattr(episode, "custom_data", None)
        if isinstance(user_data, dict):
            return user_data
        user_data = {}
        if hasattr(episode, "custom_data"):
            try:
                setattr(episode, "_custom_data", user_data)
            except Exception:
                pass
        elif hasattr(episode, "user_data"):
            try:
                setattr(episode, "user_data", user_data)
            except Exception:
                pass
        return user_data

    def on_episode_step(self, *, episode, **kwargs):
        infos = self._get_last_infos(episode)

        user_data = self._get_user_data(episode)

        info_all = infos.get("__all__", {})
        if not isinstance(info_all, dict):
            return

        current_successes = info_all.get("team_capture_successes")
        if current_successes is None:
            return

        last_successes = user_data.get("_last_team_capture_successes")
        user_data["_last_team_capture_successes"] = current_successes
        if last_successes is None:
            return

        delta_successes = int(current_successes) - int(last_successes)
        if delta_successes <= 0:
            return

        if delta_successes > 1:
            # Skip ambiguous multi-capture steps (only last capture details are available).
            user_data["multi_capture_steps"] = user_data.get("multi_capture_steps", 0) + 1
            user_data["multi_capture_successes_skipped"] = user_data.get(
                "multi_capture_successes_skipped", 0
            ) + delta_successes
            return

        helpers = int(info_all.get("team_capture_last_helpers", 0))
        free_riders = int(info_all.get("team_capture_last_free_riders", 0))
        if helpers > 0:
            user_data["captures_successful"] = user_data.get("captures_successful", 0) + 1
            if helpers == 1:
                user_data["solo_captures"] = user_data.get("solo_captures", 0) + 1
            else:
                user_data["coop_captures"] = user_data.get("coop_captures", 0) + 1
            user_data["joiners_total"] = user_data.get("joiners_total", 0) + helpers
            user_data["free_riders_total"] = user_data.get("free_riders_total", 0) + free_riders

    def on_episode_end(self, *, episode, metrics_logger: MetricsLogger, **kwargs):
        """
        Called at the end of each episode.
        Logs the total and average rewards separately for predators and prey.
        """
        self.num_episodes += 1
        episode_return = episode.get_return()
        self.overall_sum_of_rewards += episode_return
        # Safely read __all__ info across RLlib API variations
        infos = self._get_last_infos(episode)

        # Prefer __all__ when available; otherwise scan agent infos for the counters.
        info_all = infos.get("__all__", {})
        if not info_all:
            info_all = {}
            for candidate in infos.values():
                if not isinstance(candidate, dict):
                    continue
                if "team_capture_successes" in candidate or "team_capture_coop_successes" in candidate:
                    info_all = candidate
                    break

        coop_success = info_all.get("team_capture_coop_successes", 0)
        coop_fail = info_all.get("team_capture_coop_failures", 0)
        total_success = info_all.get("team_capture_successes", 0)
        total_fail = info_all.get("team_capture_failures", 0)
        mammoth_success = info_all.get("team_capture_mammoth_successes", 0)
        mammoth_fail = info_all.get("team_capture_mammoth_failures", 0)
        rabbit_success = info_all.get("team_capture_rabbit_successes", 0)
        rabbit_fail = info_all.get("team_capture_rabbit_failures", 0)

        env = self._resolve_env(episode, **kwargs)
        if env is not None and (not info_all or (total_success == 0 and total_fail == 0)):
            coop_success = getattr(env, "team_capture_coop_successes", coop_success)
            coop_fail = getattr(env, "team_capture_coop_failures", coop_fail)
            total_success = getattr(env, "team_capture_successes", total_success)
            total_fail = getattr(env, "team_capture_failures", total_fail)
        if env is not None and (mammoth_success + mammoth_fail) == 0:
            mammoth_success = getattr(env, "team_capture_mammoth_successes", mammoth_success)
            mammoth_fail = getattr(env, "team_capture_mammoth_failures", mammoth_fail)
        if env is not None and (rabbit_success + rabbit_fail) == 0:
            rabbit_success = getattr(env, "team_capture_rabbit_successes", rabbit_success)
            rabbit_fail = getattr(env, "team_capture_rabbit_failures", rabbit_fail)
        mammoth_attempts = mammoth_success + mammoth_fail
        mammoth_success_rate = mammoth_success / mammoth_attempts if mammoth_attempts else 0.0
        mammoth_success_rate_pct = mammoth_success_rate * 100.0
        rabbit_attempts = rabbit_success + rabbit_fail
        rabbit_success_rate = rabbit_success / rabbit_attempts if rabbit_attempts else 0.0
        rabbit_success_rate_pct = rabbit_success_rate * 100.0
        if metrics_logger is not None:
            metrics_logger.log_value("custom_metrics/team_capture_coop_successes", coop_success)
            metrics_logger.log_value("custom_metrics/team_capture_coop_failures", coop_fail)
            metrics_logger.log_value("custom_metrics/team_capture_total_successes", total_success)
            metrics_logger.log_value("custom_metrics/team_capture_total_failures", total_fail)
            metrics_logger.log_value("custom_metrics/mammoth_capture_successes", mammoth_success)
            metrics_logger.log_value("custom_metrics/mammoth_capture_failures", mammoth_fail)
            metrics_logger.log_value("custom_metrics/mammoth_capture_attempts", mammoth_attempts)
            metrics_logger.log_value("custom_metrics/mammoth_capture_success_rate", mammoth_success_rate_pct)
            metrics_logger.log_value("custom_metrics/rabbit_capture_successes", rabbit_success)
            metrics_logger.log_value("custom_metrics/rabbit_capture_failures", rabbit_fail)
            metrics_logger.log_value("custom_metrics/rabbit_capture_attempts", rabbit_attempts)
            metrics_logger.log_value("custom_metrics/rabbit_capture_success_rate", rabbit_success_rate_pct)
        if hasattr(episode, "custom_metrics"):
            episode.custom_metrics["team_capture_coop_successes"] = coop_success
            episode.custom_metrics["team_capture_coop_failures"] = coop_fail
            episode.custom_metrics["team_capture_total_successes"] = total_success
            episode.custom_metrics["team_capture_total_failures"] = total_fail
            episode.custom_metrics["mammoth_capture_successes"] = mammoth_success
            episode.custom_metrics["mammoth_capture_failures"] = mammoth_fail
            episode.custom_metrics["mammoth_capture_attempts"] = mammoth_attempts
            episode.custom_metrics["mammoth_capture_success_rate"] = mammoth_success_rate_pct
            episode.custom_metrics["rabbit_capture_successes"] = rabbit_success
            episode.custom_metrics["rabbit_capture_failures"] = rabbit_fail
            episode.custom_metrics["rabbit_capture_attempts"] = rabbit_attempts
            episode.custom_metrics["rabbit_capture_success_rate"] = rabbit_success_rate_pct

        # Join/defect and capture-style metrics
        user_data = self._get_user_data(episode)

        solo_captures = user_data.get("solo_captures", 0)
        coop_captures = user_data.get("coop_captures", 0)
        if env is not None and (solo_captures + coop_captures) == 0:
            env_total_success = getattr(env, "team_capture_successes", 0) or 0
            if env_total_success:
                env_coop_success = getattr(env, "team_capture_coop_successes", 0) or 0
                coop_captures = int(env_coop_success)
                solo_captures = int(max(env_total_success - coop_captures, 0))
                user_data["solo_captures"] = solo_captures
                user_data["coop_captures"] = coop_captures
        capture_successes = solo_captures + coop_captures
        solo_capture_rate = solo_captures / capture_successes if capture_successes else 0.0
        coop_capture_rate = coop_captures / capture_successes if capture_successes else 0.0
        solo_capture_rate_pct = solo_capture_rate * 100.0
        coop_capture_rate_pct = coop_capture_rate * 100.0

        multi_capture_steps = user_data.get("multi_capture_steps", 0)
        multi_capture_successes_skipped = user_data.get("multi_capture_successes_skipped", 0)

        if metrics_logger is not None:
            metrics_logger.log_value("custom_metrics/solo_captures", solo_captures)
            metrics_logger.log_value("custom_metrics/coop_captures", coop_captures)
            metrics_logger.log_value("custom_metrics/solo_capture_rate", solo_capture_rate_pct)
            metrics_logger.log_value("custom_metrics/coop_capture_rate", coop_capture_rate_pct)
            metrics_logger.log_value("custom_metrics/multi_capture_steps", multi_capture_steps)
            metrics_logger.log_value(
                "custom_metrics/multi_capture_successes_skipped", multi_capture_successes_skipped
            )

        if hasattr(episode, "custom_metrics"):
            episode.custom_metrics["solo_captures"] = solo_captures
            episode.custom_metrics["coop_captures"] = coop_captures
            episode.custom_metrics["solo_capture_rate"] = solo_capture_rate_pct
            episode.custom_metrics["coop_capture_rate"] = coop_capture_rate_pct
            episode.custom_metrics["multi_capture_steps"] = multi_capture_steps
            episode.custom_metrics["multi_capture_successes_skipped"] = multi_capture_successes_skipped

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
        print(
            "  - Solo/Coop captures: solo={} coop={} solo_capture_rate={:.2f} coop_capture_rate={:.2f}".format(
                solo_captures, coop_captures, solo_capture_rate_pct, coop_capture_rate_pct
            )
        )
        if multi_capture_steps:
            print(
                "  - Multi-capture steps skipped: steps={} successes_skipped={}".format(
                    multi_capture_steps, multi_capture_successes_skipped
                )
            )
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

        result["timing/iter_minutes"] = iter_time / 60.0
        result["timing/avg_minutes_per_iter"] = total_elapsed / 60.0 / iter_num
        result["timing/total_hours_elapsed"] = total_elapsed / 3600.0
