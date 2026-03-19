from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
import time
from collections import defaultdict

from predpreygrass.rllib.direct_reciprocity.utils.reciprocity_metrics import (
    aggregate_direct_reciprocity_metrics,
    aggregate_share_decisions_from_event_log,
)


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0
        self._pending_episode_metrics = []
        self.start_time = time.time()
        self.last_iteration_time = self.start_time

    @staticmethod
    def _resolve_env(episode, **kwargs):
        def _looks_like_target_env(obj) -> bool:
            if obj is None:
                return False
            return any(
                hasattr(obj, attr)
                for attr in ("agent_event_log", "per_step_agent_data", "share_events_total", "share_opportunities_total")
            )

        def _unwrap(obj, max_depth=10):
            seen = set()
            current = obj
            depth = 0
            while current is not None and depth < max_depth:
                depth += 1
                obj_id = id(current)
                if obj_id in seen:
                    break
                seen.add(obj_id)

                if _looks_like_target_env(current):
                    return current

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
                        current = subs[0]
                        continue

                for attr in ("envs", "_envs"):
                    subs = getattr(current, attr, None)
                    if isinstance(subs, (list, tuple)) and subs:
                        current = subs[0]
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

        for candidate in (
            kwargs.get("env"),
            getattr(kwargs.get("env_runner"), "env", None),
            getattr(kwargs.get("worker"), "base_env", None),
            kwargs.get("base_env"),
        ):
            resolved = _unwrap(candidate)
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
        # Safely read __all__ info across RLlib API variations
        infos = {}
        if hasattr(episode, "get_last_infos"):
            infos = episode.get_last_infos() or {}
        elif hasattr(episode, "last_infos"):
            infos = episode.last_infos or {}
        if isinstance(infos, dict):
            info_all = infos.get("__all__", {})
            if not info_all and infos:
                info_all = next(iter(infos.values()), {})
        else:
            info_all = {}
        coop_success = info_all.get("team_capture_coop_successes", 0)
        coop_fail = info_all.get("team_capture_coop_failures", 0)
        total_success = info_all.get("team_capture_successes", 0)
        total_fail = info_all.get("team_capture_failures", 0)
        share_events_total = info_all.get("share_events_total", 0)
        share_opportunities_total = info_all.get("share_opportunities_total", 0)
        share_refusals_total = info_all.get("share_refusals_total", 0)
        env = self._resolve_env(episode, **kwargs)

        share_metrics = {}
        reciprocity_metrics = {}
        if env is not None:
            event_log = getattr(env, "agent_event_log", {})
            share_metrics = aggregate_share_decisions_from_event_log(event_log)
            reciprocity_metrics = aggregate_direct_reciprocity_metrics(event_log)
            if not share_events_total:
                share_events_total = share_metrics.get("share_events", 0)
            if not share_opportunities_total:
                share_opportunities_total = share_metrics.get("share_opportunities", 0)
            if not share_refusals_total:
                share_refusals_total = share_metrics.get("share_refusals", 0)
        share_decision_rate_pct = 100.0 * (
            share_metrics.get("share_decision_rate", 0.0)
            if share_metrics
            else (share_events_total / share_opportunities_total if share_opportunities_total else 0.0)
        )
        reciprocal_share_rate_pct = 100.0 * reciprocity_metrics.get("reciprocal_share_rate", 0.0)
        prior_helper_share_rate_pct = 100.0 * reciprocity_metrics.get("share_rate_when_prior_helper_available", 0.0)
        no_helper_share_rate_pct = 100.0 * reciprocity_metrics.get("share_rate_when_no_prior_helper_available", 0.0)
        if metrics_logger is not None:
            metrics_logger.log_value("custom_metrics/team_capture_successes", coop_success)
            metrics_logger.log_value("custom_metrics/team_capture_failures", coop_fail)
            metrics_logger.log_value("custom_metrics/team_capture_total_successes", total_success)
            metrics_logger.log_value("custom_metrics/team_capture_total_failures", total_fail)
            metrics_logger.log_value("custom_metrics/share_events_total", share_events_total)
            metrics_logger.log_value("custom_metrics/share_opportunities_total", share_opportunities_total)
            metrics_logger.log_value("custom_metrics/share_refusals_total", share_refusals_total)
            metrics_logger.log_value("custom_metrics/share_decision_rate", share_decision_rate_pct)
            metrics_logger.log_value("custom_metrics/share_rate_when_prior_helper_available", prior_helper_share_rate_pct)
            metrics_logger.log_value("custom_metrics/share_rate_when_no_prior_helper_available", no_helper_share_rate_pct)
            metrics_logger.log_value("custom_metrics/reciprocal_share_rate", reciprocal_share_rate_pct)
            metrics_logger.log_value("custom_metrics/reciprocal_dyads", reciprocity_metrics.get("reciprocal_dyads", 0))
            metrics_logger.log_value(
                "custom_metrics/share_to_prior_helper_rate",
                100.0 * reciprocity_metrics.get("share_to_prior_helper_rate", 0.0),
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
        print(
            "  - Sharing:"
            f" opportunities={share_opportunities_total}"
            f" shares={share_events_total}"
            f" refusals={share_refusals_total}"
            f" share_rate={share_decision_rate_pct:.2f}%"
        )
        print(
            "  - Reciprocity:"
            f" prior_helper_share_rate={prior_helper_share_rate_pct:.2f}%"
            f" no_helper_share_rate={no_helper_share_rate_pct:.2f}%"
            f" reciprocal_share_rate={reciprocal_share_rate_pct:.2f}%"
            f" reciprocal_dyads={reciprocity_metrics.get('reciprocal_dyads', 0)}"
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
