# discretionary libraries
from predpreygrass.predpreygrass.envs import predpreygrass_parallel_v0
from predpreygrass.predpreygrass.config.config_predpreygrass import (
    env_kwargs,
)


parallel_env = predpreygrass_parallel_v0.parallel_env(render_mode="human", **env_kwargs)
env_base = parallel_env.predpreygrass
observations, infos = parallel_env.reset(seed=1)
done = False
while not done:
    actions = {
        agent: parallel_env.action_space(agent).sample()
        for agent in parallel_env.agents
    }
    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    for agent in rewards:
        if rewards[agent] > 0.0:
            print(f"{agent}, reward: {rewards[agent]}")
    done = env_base.is_no_prey or env_base.is_no_predator
parallel_env.close()
