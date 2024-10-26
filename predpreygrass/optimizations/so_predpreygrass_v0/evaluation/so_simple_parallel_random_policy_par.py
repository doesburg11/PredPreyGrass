# discretionary libraries
from predpreygrass.envs import so_predpreygrass_v0_par
from predpreygrass.envs._so_predpreygrass_v0.config.so_config_predpreygrass_par import (
    env_kwargs,
)
parallel_env = so_predpreygrass_v0_par.parallel_env(render_mode="human", **env_kwargs)

parallel_env = so_predpreygrass_v0_par.parallel_env(render_mode="human", **env_kwargs)
observations, infos = parallel_env.reset(seed=1)
observations, infos = parallel_env.reset(seed=1)


done = False
while not done:
    actions = {
        agent: parallel_env.action_space(agent).sample()
        for agent in parallel_env.agents
    }
    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    done = (
        parallel_env.predpreygrass.n_active_predator == 0
        or parallel_env.predpreygrass.n_active_prey== 0
    )

parallel_env.close()
