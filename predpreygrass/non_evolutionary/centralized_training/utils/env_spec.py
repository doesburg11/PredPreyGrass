from gymnasium import spaces
import numpy as np

def spaces_from_config(cfg):
    # derive per-role obs/action spaces from cfg (num_obs_channels, *_obs_range, *_action_range, etc.)
    # Example (adjust to your env):
    def obs_box(r):
        shape = (cfg["num_obs_channels"], r, r)
        return spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)

    predator_obs = obs_box(cfg["predator_obs_range"])
    prey_obs     = obs_box(cfg["prey_obs_range"])

    def act_space(range_steps):
        # whatever your env uses (Discrete, MultiDiscrete, etc.)
        return spaces.Discrete(max(1, 1 + 8 * range_steps))

    predator_act = act_space(cfg["type_1_action_range"])
    prey_act     = act_space(cfg["type_1_action_range"])

    obs_by_policy = {
        "type_1_predator": predator_obs,
        "type_1_prey": prey_obs,
        # add type_2_* if present
    }
    act_by_policy = {
        "type_1_predator": predator_act,
        "type_1_prey": prey_act,
    }
    return obs_by_policy, act_by_policy
