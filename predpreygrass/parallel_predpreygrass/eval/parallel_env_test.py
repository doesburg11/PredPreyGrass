# discretionary libraries
from predpreygrass.parallel_predpreygrass.envs import predpreygrass_parallel_v0
from predpreygrass.parallel_predpreygrass.config.config_predpreygrass import (
    env_kwargs,
    training_steps_string,
)

# external libraries
from pettingzoo.test import parallel_api_test
from pettingzoo.test import parallel_seed_test

parallel_env = predpreygrass_parallel_v0.parallel_env(render_mode= None, **env_kwargs)

parallel_api_test(parallel_env, num_cycles=1000)
