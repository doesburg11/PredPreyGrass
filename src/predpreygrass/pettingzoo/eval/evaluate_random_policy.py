# discretionary libraries
from predpreygrass.pettingzoo.envs import predpreygrass_aec_v0
from predpreygrass.pettingzoo.config.config_predpreygrass import env_kwargs

seed = 42
env = predpreygrass_aec_v0.env(render_mode="human", **env_kwargs)
env.reset(seed=seed)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if reward > 0.0:
        print(f"agent: {agent}, reward: {reward}")
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()  # random policy
        # print(f"agent: {agent}, action: {action}")

    env.step(action)
env.close()
