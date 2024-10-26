# discretionary libraries
from predpreygrass.envs import so_predpreygrass_v0_par
from predpreygrass.envs._so_predpreygrass_v0.config.so_config_predpreygrass_par import (
    env_kwargs,
)

# external libraries
from pettingzoo.test import parallel_api_test
from pettingzoo.test import parallel_seed_test

parallel_env = so_predpreygrass_v0_par.parallel_env(render_mode= None, **env_kwargs)

parallel_api_test(parallel_env, num_cycles=1000)
