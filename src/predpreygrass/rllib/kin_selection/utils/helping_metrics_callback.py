from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

class HelpingMetricsCallback(RLlibCallback):
    """Logs cooperation metrics online from per-step infos.

    Metrics:
    - helping_rate: mean fraction of steps where an agent shared (donor-side)
    - received_share_mean: mean amount received per episode (recipient-side)
    - shares_per_episode: total share events per episode
    """
    def __init__(self):
        super().__init__()
        self._episode_help_counts = {}
        self._episode_receive_amounts = {}
        self._episode_steps = {}
        self._episode_share_attempts = {}
        # Per-type aggregations
        self._episode_help_by_group = {}
        self._episode_attempts_by_group = {}
        self._episode_received_by_group = {}
        # Donor -> recipient routing (same vs other type)
        self._episode_same_type_shares = {}
        self._episode_other_type_shares = {}
        # Population fractions over steps (sum of fractions to later average)
        self._episode_frac_type2_prey_sum = {}
        self._episode_frac_type2_pred_sum = {}

    def on_episode_start(self, *, episode, **kwargs):
        eid = episode.id_
        self._episode_help_counts[eid] = 0
        self._episode_receive_amounts[eid] = 0.0
        self._episode_steps[eid] = 0
        self._episode_share_attempts[eid] = 0
        self._episode_help_by_group[eid] = {}
        self._episode_attempts_by_group[eid] = {}
        self._episode_received_by_group[eid] = {}
        self._episode_same_type_shares[eid] = 0
        self._episode_other_type_shares[eid] = 0
        self._episode_frac_type2_prey_sum[eid] = 0.0
        self._episode_frac_type2_pred_sum[eid] = 0.0

    def on_episode_step(self, *, episode, **kwargs):
        eid = episode.id_
        self._episode_steps[eid] = self._episode_steps.get(eid, 0) + 1
        infos = self._last_infos(episode)
        if isinstance(infos, dict):
            # Count donor shares, attempts, and received amounts; tolerate list-shaped infos
            # Per-agent loop to access ids for group breakdown and population fractions
            for agent_id, info_val in infos.items():
                group = self._group_from_agent_id(agent_id)
                helps_inc, recv_inc, attempts_inc, route = self._accumulate_from_info(info_val, agent_id)
                if helps_inc:
                    self._episode_help_counts[eid] += helps_inc
                    if group:
                        self._episode_help_by_group[eid][group] = self._episode_help_by_group[eid].get(group, 0) + helps_inc
                if recv_inc:
                    self._episode_receive_amounts[eid] += recv_inc
                    if group:
                        self._episode_received_by_group[eid][group] = self._episode_received_by_group[eid].get(group, 0.0) + recv_inc
                if attempts_inc:
                    self._episode_share_attempts[eid] += attempts_inc
                    if group:
                        self._episode_attempts_by_group[eid][group] = self._episode_attempts_by_group[eid].get(group, 0) + attempts_inc
                if route == "same":
                    self._episode_same_type_shares[eid] += 1
                elif route == "other":
                    self._episode_other_type_shares[eid] += 1

            # Population fractions this step (based on present agent ids)
            prey1 = sum(1 for aid in infos.keys() if "type_1_prey" in aid)
            prey2 = sum(1 for aid in infos.keys() if "type_2_prey" in aid)
            pred1 = sum(1 for aid in infos.keys() if "type_1_predator" in aid)
            pred2 = sum(1 for aid in infos.keys() if "type_2_predator" in aid)
            total_prey = prey1 + prey2
            total_pred = pred1 + pred2
            if total_prey > 0:
                self._episode_frac_type2_prey_sum[eid] += prey2 / total_prey
            if total_pred > 0:
                self._episode_frac_type2_pred_sum[eid] += pred2 / total_pred

    def on_episode_end(self, *, episode, metrics_logger: MetricsLogger, **kwargs):
        eid = episode.id_
        steps = max(self._episode_steps.pop(eid, 0), 1)
        helps = self._episode_help_counts.pop(eid, 0)
        recv = self._episode_receive_amounts.pop(eid, 0.0)
        metrics_logger.log_value("custom_metrics/helping_rate", helps / steps, reduce="mean")
        metrics_logger.log_value("custom_metrics/received_share_mean", recv / steps, reduce="mean")
        metrics_logger.log_value("custom_metrics/shares_per_episode", helps, reduce="mean")
        attempts = self._episode_share_attempts.pop(eid, 0)
        metrics_logger.log_value("custom_metrics/share_attempt_rate", attempts / steps, reduce="mean")

        # Per-group metrics
        help_by_group = self._episode_help_by_group.pop(eid, {})
        attempts_by_group = self._episode_attempts_by_group.pop(eid, {})
        received_by_group = self._episode_received_by_group.pop(eid, {})
        for group, h in help_by_group.items():
            metrics_logger.log_value(f"custom_metrics/helping_rate_{group}", h / steps, reduce="mean")
        for group, a in attempts_by_group.items():
            metrics_logger.log_value(f"custom_metrics/share_attempt_rate_{group}", a / steps, reduce="mean")
        for group, rsum in received_by_group.items():
            metrics_logger.log_value(f"custom_metrics/received_share_mean_{group}", rsum / steps, reduce="mean")

        # Routing: same vs other type (as rate per step)
        same = self._episode_same_type_shares.pop(eid, 0)
        other = self._episode_other_type_shares.pop(eid, 0)
        metrics_logger.log_value("custom_metrics/shares_to_same_type_rate", same / steps, reduce="mean")
        metrics_logger.log_value("custom_metrics/shares_to_other_type_rate", other / steps, reduce="mean")

        # Population fractions (averaged over steps)
        prey_frac_sum = self._episode_frac_type2_prey_sum.pop(eid, 0.0)
        pred_frac_sum = self._episode_frac_type2_pred_sum.pop(eid, 0.0)
        metrics_logger.log_value("custom_metrics/fraction_type_2_prey", prey_frac_sum / steps, reduce="mean")
        metrics_logger.log_value("custom_metrics/fraction_type_2_predator", pred_frac_sum / steps, reduce="mean")

    # ---- helpers ----
    def _last_infos(self, episode) -> dict:
        for name in ("get_last_infos", "get_infos"):
            if hasattr(episode, name):
                try:
                    m = getattr(episode, name)()
                    if isinstance(m, dict):
                        return m
                except Exception:
                    pass
        for name in ("last_infos", "infos"):
            if hasattr(episode, name):
                try:
                    m = getattr(episode, name)
                    if isinstance(m, dict):
                        return m
                except Exception:
                    pass
        if hasattr(episode, "_agent_to_last_info"):
            try:
                m = getattr(episode, "_agent_to_last_info")
                if isinstance(m, dict):
                    return m
            except Exception:
                pass
        return {}

    def _accumulate_from_info(self, info_val, agent_id=None):
        """Return (helps_count_inc, received_sum_inc, attempts_inc, route) from a single info value.

        route is one of {None, "same", "other"} for donor->recipient type match when known.

        Handles dict, list/tuple-of-dicts, or other types gracefully.
        """
        helps = 0
        recv_sum = 0.0
        attempts = 0
        route = None
        if not info_val:
            return helps, recv_sum, attempts, route
        # Single dict case
        if isinstance(info_val, dict):
            try:
                if int(info_val.get("shared", 0)) == 1:
                    helps += 1
                # Presence of 'shared' indicates a share attempt (success or failure)
                if "shared" in info_val:
                    attempts += 1
            except Exception:
                # If casting fails, ignore this flag
                pass
            rec = info_val.get("received_share", 0.0)
            try:
                if rec:
                    recv_sum += float(rec)
            except Exception:
                pass
            # Route tagging for donor entries
            try:
                recipient_id = info_val.get("shared_to")
                if recipient_id and agent_id:
                    donor_type = "type_2" if "type_2" in agent_id else ("type_1" if "type_1" in agent_id else None)
                    recip_type = "type_2" if "type_2" in recipient_id else ("type_1" if "type_1" in recipient_id else None)
                    if donor_type and recip_type:
                        route = "same" if donor_type == recip_type else "other"
            except Exception:
                pass
            return helps, recv_sum, attempts, route
        # List/tuple case (e.g., vectorized runner returning a list of infos)
        if isinstance(info_val, (list, tuple)):
            for elem in info_val:
                h, r, a, rt = self._accumulate_from_info(elem, agent_id)
                helps += h
                recv_sum += r
                attempts += a
                # If any sub-entry provides routing, prefer the explicit tag
                if rt and not route:
                    route = rt
            return helps, recv_sum, attempts, route
        # Any other type: ignore
        return helps, recv_sum, attempts, route

    def _group_from_agent_id(self, agent_id: str):
        # Expected patterns: 'type_1_prey_#', 'type_2_predator_#'
        if not isinstance(agent_id, str):
            return None
        if "type_1_prey" in agent_id:
            return "type_1_prey"
        if "type_2_prey" in agent_id:
            return "type_2_prey"
        if "type_1_predator" in agent_id:
            return "type_1_predator"
        if "type_2_predator" in agent_id:
            return "type_2_predator"
        return None
