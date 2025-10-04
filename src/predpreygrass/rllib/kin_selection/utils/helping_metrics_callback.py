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

    def on_episode_start(self, *, episode, **kwargs):
        eid = episode.id_
        self._episode_help_counts[eid] = 0
        self._episode_receive_amounts[eid] = 0.0
        self._episode_steps[eid] = 0
        self._episode_share_attempts[eid] = 0

    def on_episode_step(self, *, episode, **kwargs):
        eid = episode.id_
        self._episode_steps[eid] = self._episode_steps.get(eid, 0) + 1
        infos = self._last_infos(episode)
        if isinstance(infos, dict):
            # Count donor shares, attempts, and received amounts; tolerate list-shaped infos
            for info_val in infos.values():
                helps_inc, recv_inc, attempts_inc = self._accumulate_from_info(info_val)
                if helps_inc:
                    self._episode_help_counts[eid] += helps_inc
                if recv_inc:
                    self._episode_receive_amounts[eid] += recv_inc
                if attempts_inc:
                    self._episode_share_attempts[eid] += attempts_inc

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

    def _accumulate_from_info(self, info_val):
        """Return (helps_count_inc, received_sum_inc, attempts_inc) from a single info value.

        Handles dict, list/tuple-of-dicts, or other types gracefully.
        """
        helps = 0
        recv_sum = 0.0
        attempts = 0
        if not info_val:
            return helps, recv_sum, attempts
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
            return helps, recv_sum, attempts
        # List/tuple case (e.g., vectorized runner returning a list of infos)
        if isinstance(info_val, (list, tuple)):
            for elem in info_val:
                h, r, a = self._accumulate_from_info(elem)
                helps += h
                recv_sum += r
                attempts += a
            return helps, recv_sum, attempts
        # Any other type: ignore
        return helps, recv_sum, attempts
