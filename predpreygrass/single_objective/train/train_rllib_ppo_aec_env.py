"""
-This file trains a multi agent reinforcement model in RLlib. 
"""
# discretionary libraries
from predpreygrass.single_objective.envs import predpreygrass_aec_v0
from predpreygrass.single_objective.config.config_predpreygrass import (
    env_kwargs,
    local_output_root,
    training_steps_string
)
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

env = predpreygrass_aec_v0.env(render_mode=None, **env_kwargs)

# Define and register the environment
def env_creator(config):
    return PettingZooEnv(env)

register_env("predpreygrass_env", env_creator)

# Policy mapping function
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "predator" if "predator" in agent_id else "prey"

# Create environment to fetch observation and action spaces
dummy_env = env_creator({})

config = (
    PPOConfig()
    .environment("predpreygrass_env", env_config={})
    .framework("torch")  # Use PyTorch
    .env_runners(num_env_runners=4)  # New API for parallel environments
    .multi_agent(
        policies={
            "predator": (None, dummy_env.observation_space, dummy_env.action_space, {"lr": 0.0003}),
            "prey": (None, dummy_env.observation_space, dummy_env.action_space, {"lr": 0.0001}),
        },
        policy_mapping_fn=policy_mapping_fn,  # Map agents to their respective policies
    )
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,  # Enable new RLlib API stack
    )
)

algo = PPO(config=config)

# Training loop
for _ in range(1):
    result = algo.train()
    print(f"Episode reward: {result['episode_reward_mean']}")
