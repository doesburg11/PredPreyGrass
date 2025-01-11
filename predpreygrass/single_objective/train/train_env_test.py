"""
-This file trains a multi agent reinforcement model in a parallel environment. 
-After traing evaluation can be done using the AEC API.
-The source code and the trained model are saved in a separate 
directory, for reuse and analysis. 
-The algorithm used is PPO from stable_baselines3. 
"""
# discretionary libraries
from predpreygrass.single_objective.envs import predpreygrass_aec_v0
from predpreygrass.single_objective.config.config_predpreygrass import (
    env_kwargs,
    local_output_root,
    training_steps_string
)
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

env = predpreygrass_aec_v0.env(render_mode=None, **env_kwargs)
env1 = predpreygrass_aec_v0.raw_env(render_mode=None, **env_kwargs)


print(str(env),'has "agents" attribute', hasattr(env, "agents"))
print(str(env),'has "possible_agents" atribute', hasattr(env, "possible_agents"))
print(str(env1),'has "agents" attribute', hasattr(env1, "agents"))
print(str(env1),'has "possible_agents" atribute', hasattr(env1, "possible_agents"))

env2 = PettingZooEnv(env)
print(str(env2),'has "agents" attribute', hasattr(env, "agents"))
print(str(env2),'has "possible_agents" atribute', hasattr(env, "possible_agents"))


import predpreygrass.single_objective.envs.base_env.predpreygrass_aec as predpreygrass_aec
raw_env_instance = predpreygrass_aec.raw_env()
print(hasattr(raw_env_instance, "agents"))  # Should return True
