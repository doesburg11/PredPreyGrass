# discretionary libraries
from predpreygrass.envs import so_predpreygrass_v0
from predpreygrass.envs._so_predpreygrass_v0.config.so_config_predpreygrass import env_kwargs

env = so_predpreygrass_v0.env(render_mode='human', **env_kwargs)

env.reset(seed=1)
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if reward > 0.0:
        print(f"agent: {agent}, reward: {reward}")
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # random policy

    env.step(action)
env.close()