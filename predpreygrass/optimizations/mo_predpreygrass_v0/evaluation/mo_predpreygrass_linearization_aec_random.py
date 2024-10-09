# discretionary libraries
from predpreygrass.envs import mo_predpreygrass_v0
from predpreygrass.envs._mo_predpreygrass_v0.config.mo_config_predpreygrass import env_kwargs
# environment loop is aec
env_kwargs["is_parallel_wrapped"] = False
from predpreygrass.optimizations.mo_predpreygrass_v0.training.utils.linearization_weights_constructor import (
    construct_linearalized_weights
)

# external libraries
from momaland.utils.aec_wrappers import LinearizeReward

env = mo_predpreygrass_v0.env(render_mode='human', **env_kwargs)

# Define the number of predators and prey
num_predators = env_kwargs["n_possible_predator"]
num_prey = env_kwargs["n_possible_prey"]
# Construct the weights for linearization of the multi objective rewards
weights = construct_linearalized_weights(num_predators, num_prey)

env = LinearizeReward(env, weights)

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if reward > 0.0:
        print(f"agent: {agent}, reward: {reward}")
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy

    env.step(action)
env.close()