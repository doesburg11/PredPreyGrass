"""Configuration for the lifetime linear actor-critic Baldwin variant."""

from copy import deepcopy

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl as _baseline_config


config_actor_critic = deepcopy(_baseline_config)
config_actor_critic.update(
    {
        "actor_alpha": 0.01,
        "critic_beta": 0.05,
        "actor_critic_gamma": 0.9,
        "actor_max_update_norm": 0.1,
        "critic_max_update_norm": 0.5,
        "actor_critic_terminal_bonus": 0.0,
    }
)

config_erl = config_actor_critic
