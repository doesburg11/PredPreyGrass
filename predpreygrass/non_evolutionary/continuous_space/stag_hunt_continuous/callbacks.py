"""Logs the hunts: at the end of every episode, the environment's hunt,
catch and join counters (environment.new_hunt_stats) and derived rates."""

from ray.rllib.callbacks.callbacks import RLlibCallback

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import PARTY_SIZES


def _unwrapped(env, env_index):
    """Our environment inside RLlib's vector env."""
    envs = getattr(env, "envs", None)
    inner = envs[env_index] if envs is not None else env
    return getattr(inner, "unwrapped", inner)


def derived_rates(stats):
    """Join rates, joint arrivals per encounter episode and the share of
    hunts by parties of two or more."""
    rates = {}
    # Join rates at mammoth contact: all, outside a hunt cooldown, with a
    # partner near, alone.
    for key, denominator in (
        ("contact", "contact_decisions"),
        ("contact_ready", "contact_ready"),
        ("contact_with_partner", "contact_with_partner"),
        ("contact_alone", "contact_alone"),
    ):
        if stats[denominator]:
            rates[f"{key}_join_rate"] = stats[f"{key}_joins"] / stats[denominator]
    # Joint arrivals per mammoth encounter episode (DESIGN.md, 4; the chance
    # level needs the trajectories and is computed in evaluation).
    if stats["encounter_episodes"]:
        rates["joint_arrival_rate"] = stats["joint_arrivals"] / stats["encounter_episodes"]
    if stats["decisions"]:
        rates["join_rate"] = stats["joins"] / stats["decisions"]
    hunts = sum(stats[f"hunts_party_{size}"] for size in range(1, PARTY_SIZES + 1))
    if hunts:
        rates["party_2plus_share"] = 1 - stats["hunts_party_1"] / hunts
    return rates


class HuntLogger(RLlibCallback):
    def on_episode_end(self, *, episode, env=None, env_index=0, metrics_logger=None, **kwargs):
        if env is None or metrics_logger is None:
            return
        stats = getattr(_unwrapped(env, env_index), "episode_hunt_stats", None)
        if not stats:
            return
        for key, value in {**stats, **derived_rates(stats)}.items():
            metrics_logger.log_value(f"hunts/{key}", float(value))
