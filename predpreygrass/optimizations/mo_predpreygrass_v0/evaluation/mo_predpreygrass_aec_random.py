from predpreygrass.envs._mo_predpreygrass_v0.config.mo_config_predpreygrass import env_kwargs
# environment loop is aec
env_kwargs["is_parallel_wrapped"] = False
from predpreygrass.envs import mo_predpreygrass_v0

env = mo_predpreygrass_v0.env(render_mode='human', **env_kwargs)

env.reset()
for agent in env.agent_iter():
    observation, vec_reward, termination, truncation, info = env.last()
    if vec_reward[0] > 0.0 or vec_reward[1] > 0.0:
        print(f"agent: {agent}, reward: {vec_reward}")
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy

    env.step(action)
env.close()