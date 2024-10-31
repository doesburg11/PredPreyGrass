# discretionary libraries
from predpreygrass.envs import predpreygrass_parallel_v0
from predpreygrass.envs._so_predpreygrass_v0.config.config_predpreygrass_parallel import (
    env_kwargs,
)

# external libraries
from pettingzoo.test import parallel_api_test
from pettingzoo.test import parallel_seed_test

parallel_env = predpreygrass_parallel_v0.parallel_env(render_mode= None, **env_kwargs)

parallel_api_test(parallel_env, num_cycles=1000)
