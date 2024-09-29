from predpreygrass.envs._mo_predpreygrass_v0.config.mo_config_predpreygrass import env_kwargs
# environment loop is aec
env_kwargs["is_parallel_wrapped"] = False

from predpreygrass.envs import mo_predpreygrass_v0

from momaland.utils.aec_wrappers import LinearizeReward

env = mo_predpreygrass_v0.env(render_mode='human', **env_kwargs)

# weights for linearization rewards
weights = {}

# Define the number of predators and prey
num_predators = env_kwargs["n_possible_predator"]
num_prey = env_kwargs["n_possible_prey"]

# Populate the weights dictionary for predators
for i in range(num_predators):
    weights[f"predator_{i}"] = [0.5, 0.5]

# Populate the weights dictionary for prey
for i in range(num_prey):
    weights[f"prey_{i + num_predators}"] = [0.5, 0.5]

env = LinearizeReward(env, weights)


#print("env_kwargs", env_kwargs)

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