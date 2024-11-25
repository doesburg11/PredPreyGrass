# discretionary libraries
from predpreygrass.single_objective.envs import (
    predpreygrass_parallel_v0,
    predpreygrass_aec_v0,
)
from predpreygrass.single_objective.config.config_predpreygrass import env_kwargs

is_parallel = env_kwargs["is_parallel"]

if is_parallel:
    parallel_env = predpreygrass_parallel_v0.parallel_env(
        render_mode="human", **env_kwargs
    )
    env_base = parallel_env.predpreygrass
    observations, infos = parallel_env.reset(seed=1)
    done = False
    while not done:
        actions = {
            agent: parallel_env.action_space(agent).sample()
            for agent in parallel_env.agents
        }
        observations, rewards, terminations, truncations, infos = parallel_env.step(
            actions
        )
        for agent in rewards:
            if rewards[agent] > 0.0:
                print(f"{agent}, reward: {rewards[agent]}")
        done = env_base.is_no_prey or env_base.is_no_predator
    parallel_env.close()
else:
    env = predpreygrass_aec_v0.env(render_mode="human", **env_kwargs)
    env.reset(seed=1)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if reward > 0.0:
            print(f"agent: {agent}, reward: {reward}")
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()  # random policy

        env.step(action)
    env.close()
