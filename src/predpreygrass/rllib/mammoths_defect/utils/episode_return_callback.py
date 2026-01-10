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
    def _decode_join_hunt_action(action):
        if action is None:
            return None
        if isinstance(action, dict):
            if "join_hunt" in action:
                return bool(action["join_hunt"])
            return None
        try:
            if len(action) >= 2:
                return bool(int(action[1]))
        except Exception:
            return None
        return None

    @staticmethod
    def _resolve_env(episode, **kwargs):
        def _looks_like_target_env(obj) -> bool:
            if obj is None:
                return False
            return any(
                hasattr(obj, attr)
                for attr in (
                    "per_step_agent_data",
                    "agent_event_log",
                    "team_capture_successes",
                    "predator_join_intent",
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

    @staticmethod
    def _count_join_defect_from_env(env):
        join_steps = 0
        defect_steps = 0
        step_data_list = getattr(env, "per_step_agent_data", None)
        if not isinstance(step_data_list, list):
            return join_steps, defect_steps
        for step_data in step_data_list:
            if not isinstance(step_data, dict):
                continue
            for agent_id, data in step_data.items():
                if "predator" not in agent_id or not isinstance(data, dict):
                    continue
                if "join_hunt" not in data:
                    continue
                if data["join_hunt"]:
                    join_steps += 1
                else:
                    defect_steps += 1
        return join_steps, defect_steps

    @staticmethod
    def _count_free_riders_from_env(env):
        total = 0
        event_log = getattr(env, "agent_event_log", None)
        if not isinstance(event_log, dict):
            return total
        for agent_id, record in event_log.items():
            if "predator" not in agent_id or not isinstance(record, dict):
                continue
            for evt in record.get("eating_events", []) or []:
                if not evt.get("join_hunt", True):
                    total += 1
        return total

    @staticmethod
    def _get_user_data(episode):
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
        _ = infos  # unused but kept for parity with defection callback

        user_data = self._get_user_data(episode)
        join_steps = user_data.get("join_steps", 0)
        defect_steps = user_data.get("defect_steps", 0)
        saw_join_hunt = False

        last_actions = self._get_last_actions(episode)
        for agent_id, action in last_actions.items():
            if "predator" not in agent_id:
                continue
            join_flag = self._decode_join_hunt_action(action)
            if join_flag is None:
                continue
            saw_join_hunt = True
            if join_flag:
                join_steps += 1
            else:
                defect_steps += 1

        if saw_join_hunt:
            user_data["join_steps"] = join_steps
            user_data["defect_steps"] = defect_steps
            return

        env = self._resolve_env(episode, **kwargs)
        if env is not None:
            env_join, env_defect = self._count_join_defect_from_env(env)
            user_data["join_steps"] = env_join
            user_data["defect_steps"] = env_defect

    def on_episode_end(self, *, episode, metrics_logger: MetricsLogger, **kwargs):
        """
        Called at the end of each episode.
        Logs the total and average rewards separately for predators and prey, plus join/defect stats.
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

        user_data = self._get_user_data(episode)
        join_steps = user_data.get("join_steps", 0)
        defect_steps = user_data.get("defect_steps", 0)
        if join_steps == 0 and defect_steps == 0:
            env = self._resolve_env(episode, **kwargs)
            if env is not None:
                join_steps, defect_steps = self._count_join_defect_from_env(env)
                user_data["join_steps"] = join_steps
                user_data["defect_steps"] = defect_steps
        join_rate = join_steps / max(1, join_steps + defect_steps)
        env_for_free = self._resolve_env(episode, **kwargs)
        free_riders = self._count_free_riders_from_env(env_for_free) if env_for_free is not None else 0

        if metrics_logger is not None:
            metrics_logger.log_value("custom_metrics/team_capture_successes", coop_success)
            metrics_logger.log_value("custom_metrics/team_capture_failures", coop_fail)
            metrics_logger.log_value("custom_metrics/team_capture_total_successes", total_success)
            metrics_logger.log_value("custom_metrics/team_capture_total_failures", total_fail)
            metrics_logger.log_value("custom_metrics/join_steps", join_steps)
            metrics_logger.log_value("custom_metrics/defect_steps", defect_steps)
            metrics_logger.log_value("custom_metrics/join_rate", join_rate)
            metrics_logger.log_value("custom_metrics/free_rider_events", free_riders)

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

            for group in ["type_1_predator", "type_2_predator", "type_1_prey", "type_2_prey"]:
                if group in agent_id:
                    group_rewards[group].append(total)
                    break

        predator_mean = predator_total / max(1, len(predator_totals))
        prey_mean = prey_total / max(1, len(prey_totals))

        if metrics_logger is not None:
            metrics_logger.log_value("custom_metrics/predator_episode_return_mean", predator_mean)
            metrics_logger.log_value("custom_metrics/prey_episode_return_mean", prey_mean)

        # Episode summary log
        print(f"Episode {self.num_episodes}: R={episode_return:.2f} | Global SUM={self.overall_sum_of_rewards:.2f}")
        print(f"  - Coop: successes={coop_success} failures={coop_fail}")
        print(f"  - Total capture: successes={total_success} failures={total_fail}")
        print(f"  - Predators: Total = {predator_total:.2f}")
        print(f"  - Prey:      Total = {prey_total:.2f}")
        print(f"  - Join/Defect: join_steps={join_steps} defect_steps={defect_steps} join_rate={join_rate:.2f} free_riders={free_riders}")

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
