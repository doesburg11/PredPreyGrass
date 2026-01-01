import copy

from predpreygrass.rllib.stag_hunt_vectorized.config.config_ppo_cpu_stag_hunt_vectorized import (
    config_ppo as base_config_ppo,
)

# Resume-specific PPO config. Start from the base config and override as needed.
config_ppo = copy.deepcopy(base_config_ppo)
