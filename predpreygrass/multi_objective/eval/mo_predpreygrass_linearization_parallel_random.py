# discretionary libraries
from predpreygrass.envs._mo_predpreygrass_v0.config.mo_config_predpreygrass import (
    env_kwargs,
)

# environment loop is parallel
env_kwargs["is_parallel_wrapped"] = True
from predpreygrass.envs import mo_predpreygrass_v0 as _env
from predpreygrass.optimizations.mo_predpreygrass_v0.training.utils.linearization_weights_constructor import (
    construct_linearalized_weights
)
# external libraries
from momaland.utils.parallel_wrappers import LinearizeReward

# .parallel_env() function will return a Parallel environment, as per PZ standard
parallel_env = _env.parallel_env(render_mode="human", **env_kwargs)

# Define the number of predators and prey
num_predators = env_kwargs["n_possible_predator"]
num_prey = env_kwargs["n_possible_prey"]
# Construct the weights for linearization of the multi objective rewards
weights = construct_linearalized_weights(num_predators, num_prey)

# optionally, you can scalarize the reward with weights
parallel_env = LinearizeReward(parallel_env, weights)

cycle = 0
parallel_env.reset(seed=42)
while parallel_env.agents:
    # this is where you would insert your policy
    actions = {
        agent: parallel_env.action_space(agent).sample()
        for agent in parallel_env.agents
    }
    # print("actions", actions)
    # vec_reward is a dict[str, numpy array]
    observations, linearized_rewards, terminations, truncations, infos = (
        parallel_env.step(actions)
    )
    cycle += 1
    for agent in parallel_env.agents:
        if linearized_rewards[agent] > 0:
            print(f"cycle: {cycle}")
            print(f"agent: {agent}, reward: {linearized_rewards[agent]}")

parallel_env.close()
