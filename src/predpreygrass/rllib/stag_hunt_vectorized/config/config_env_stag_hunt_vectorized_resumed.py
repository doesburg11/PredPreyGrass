import copy

from predpreygrass.rllib.stag_hunt_vectorized.config.config_env_stag_hunt_vectorized import config_env as base_config_env

# Resume-specific environment config. Start from the base config and override as needed.
config_env = copy.deepcopy(base_config_env)

# Example override used for resumed runs.
config_env["team_capture_margin"] = 1.5

# Optional curriculum settings for team_capture_margin.
# Set ramp_iters > 0 and change end_margin to enable a linear ramp.
config_env["margin_curriculum"] = {
    "start_margin": config_env.get("team_capture_margin", 0.0),
    "end_margin": config_env.get("team_capture_margin", 0.0),
    "ramp_iters": 0,
    "start_iter": None,
}
