# discretionary libraries
from predpreygrass.envs import so_predpreygrass_v0_1
from predpreygrass.envs._so_predpreygrass_v0.config.so_config_predpreygrass_1 import env_kwargs

parallel_env = so_predpreygrass_v0_1.parallel_env(render_mode="human", **env_kwargs)
observations, infos = parallel_env.reset(seed=42)

#print(f"observations: {observations}")

while parallel_env.agents:
    # this is where you would insert your policy
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}

    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    for agent in rewards:
        if rewards[agent] > 0.0:
            print("========================================================================================================")
            print(f"agent: {agent}, reward: {rewards[agent]}")
            print("========================================================================================================")


parallel_env.close()