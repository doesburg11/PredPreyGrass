# discretionary libraries
from predpreygrass.parallel_predpreygrass.envs import predpreygrass_parallel_v0
from predpreygrass.parallel_predpreygrass.config.config_predpreygrass import (
    env_kwargs,
)

env = predpreygrass_parallel_v0.env(render_mode="human", **env_kwargs)
# attributes of base environment cannot be accessed directly via wrapped
# parallel to AEC environment
env_base = env.unwrapped.predpreygrass
env.reset(seed=1)
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if reward > 0.0:
        print(f"{agent}, reward: {reward}")
    if env_base.is_no_prey or env_base.is_no_predator:  
        break
    else:
        action = env.action_space(agent).sample() # random policy
    env.step(action)
env.close()