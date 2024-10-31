# discretionary libraries
from predpreygrass.envs import predpreygrass_v0
from predpreygrass.envs._so_predpreygrass_v0.config.config_predpreygrass import env_kwargs

parallel_env = predpreygrass_v0.parallel_env(render_mode="human")
observations, infos = parallel_env.reset(seed=42)

print(f"observations: {observations}")

while parallel_env.agents:
    # this is where you would insert your policy
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}

    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    for agent in rewards:
        if rewards[agent] > 0.0:
            print(f"agent: {agent}, reward: {rewards[agent]}")


parallel_env.close()