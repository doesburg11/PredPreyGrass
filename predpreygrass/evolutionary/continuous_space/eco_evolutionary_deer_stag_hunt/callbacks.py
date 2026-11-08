"""Logs every finished episode: the environment's catch counters
(environment.new_stats), derived rates, and the hunters' gene statistics
(environment.gene_stats: means, spreads, selection differentials)."""

from ray.rllib.callbacks.callbacks import RLlibCallback

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.environment import KINDS


def _unwrapped(env, env_index):
    """Our environment inside RLlib's vector env."""
    envs = getattr(env, "envs", None)
    inner = envs[env_index] if envs is not None else env
    return getattr(inner, "unwrapped", inner)


def derived_rates(stats):
    """Share of the predators' energy from deer that came from adults, the
    adults' share of catches, and adult attempt success: adults caught over
    adults caught plus failed adult attempts (physics steps in which ready
    predators touched an adult with no second toucher in the window)."""
    rates = {}
    energy = sum(stats[f"energy_from_{kind}"] for kind in KINDS)
    if energy:
        rates["adult_energy_share"] = stats["energy_from_adult"] / energy
    if stats["deer_caught"]:
        rates["adult_catch_share"] = stats["adults_caught"] / stats["deer_caught"]
    if stats["adults_caught"] + stats["adult_failures"]:
        rates["adult_attempt_success"] = stats["adults_caught"] / (stats["adults_caught"] + stats["adult_failures"])
    return rates


class CatchLogger(RLlibCallback):
    def on_episode_end(self, *, episode, env=None, env_index=0, metrics_logger=None, **kwargs):
        if env is None or metrics_logger is None:
            return
        stats = getattr(_unwrapped(env, env_index), "episode_stats", None)
        if not stats:
            return
        for key, value in {**stats, **derived_rates(stats)}.items():
            if isinstance(value, (int, float)):
                metrics_logger.log_value(f"catches/{key}", float(value))
