import copy

from predpreygrass.non_evolutionary.project_cooperation.stag_hunt.config.config_ppo_gpu_stag_hunt import (
    config_ppo as base_config_ppo,
)

# Resume-specific PPO config. Start from the base config and override as needed.
config_ppo = copy.deepcopy(base_config_ppo)
