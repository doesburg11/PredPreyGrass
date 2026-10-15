"""Logs the evolving traits: at the end of every episode, each species' mean
and spread of every gene, its mean cost multiplier and its numbers."""

from ray.rllib.callbacks.callbacks import RLlibCallback


def _unwrapped(env, env_index):
    """Our environment inside RLlib's vector env."""
    envs = getattr(env, "envs", None)
    inner = envs[env_index] if envs is not None else env
    return getattr(inner, "unwrapped", inner)


class TraitLogger(RLlibCallback):
    def on_episode_end(self, *, episode, env=None, env_index=0, metrics_logger=None, **kwargs):
        if env is None or metrics_logger is None:
            return
        stats = getattr(_unwrapped(env, env_index), "episode_trait_stats", None)
        if not stats:
            return
        for species, entry in stats.items():
            for key, value in entry.items():
                metrics_logger.log_value(f"traits/{species}_{key}", float(value))
