"""Logs the catches: at the end of every episode, the environment's counters
(environment.new_stats) and derived rates."""

from ray.rllib.callbacks.callbacks import RLlibCallback

from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.environment import KINDS


def _unwrapped(env, env_index):
    """Our environment inside RLlib's vector env."""
    envs = getattr(env, "envs", None)
    inner = envs[env_index] if envs is not None else env
    return getattr(inner, "unwrapped", inner)


def derived_rates(stats):
    """Share of the predators' energy from deer that came from adults, the
    adults' share of catches, and the share of adult attempts (physics steps
    in which ready predators touched an adult) that caught it."""
    rates = {}
    energy = sum(stats[f"energy_from_{kind}"] for kind in KINDS)
    if energy:
        rates["adult_energy_share"] = stats["energy_from_adult"] / energy
    if stats["deer_caught"]:
        rates["adult_catch_share"] = stats["adults_caught"] / stats["deer_caught"]
    if stats["adults_caught"] + stats["adult_failures"]:
        rates["adult_attempt_success"] = stats["adults_caught"] / (stats["adults_caught"] + stats["adult_failures"])
    if stats.get("energy_from_adult"):
        rates["adult_energy_to_bystanders"] = stats.get("energy_adult_to_bystanders", 0.0) / stats["energy_from_adult"]
    return rates


class CatchLogger(RLlibCallback):
    def on_episode_end(self, *, episode, env=None, env_index=0, metrics_logger=None, **kwargs):
        if env is None or metrics_logger is None:
            return
        stats = getattr(_unwrapped(env, env_index), "episode_stats", None)
        if not stats:
            return
        for key, value in {**stats, **derived_rates(stats)}.items():
            metrics_logger.log_value(f"catches/{key}", float(value))
