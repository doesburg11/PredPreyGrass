from predpreygrass.envs._mo_predpreygrass_v0.config.mo_config_predpreygrass import env_kwargs
# environment loop is parallel
env_kwargs["is_parallel_wrapped"] = True
from predpreygrass.envs import mo_predpreygrass_v0

env = mo_predpreygrass_v0.parallel_env(render_mode='human', **env_kwargs)

observations, infos = env.reset()
episode_rewards = []


while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    #print("actions", actions)
    observations, vec_rewards, terminations, truncations, infos = env.step(actions)
    for agent in env.agents:
        if vec_rewards[agent][0] > 0.0 or vec_rewards[agent][1] > 0.0:
            print(f"agent: {agent}, reward: {vec_rewards[agent]}")
    episode_rewards.append(vec_rewards)
env.close()

# rewards are stored in a dictionary, can be accessed per agent
episode_rewards[0]