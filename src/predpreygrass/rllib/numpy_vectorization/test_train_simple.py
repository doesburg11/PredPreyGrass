import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import ray

from predpreygrass.rllib.numpy_vectorization.np_vec_env import PredPreyGrassEnv
from ray.rllib.policy.policy import PolicySpec


def env_creator(config=None):
    if config is None:
        config = {}
    return PredPreyGrassEnv(**config)

# Minimal config for a quick test
config_env = {
    "grid_shape": (8, 8),
    "num_possible_predators": 2,
    "num_possible_prey": 2,
    "initial_num_predators": 1,
    "initial_num_prey": 1,
    "initial_num_grass": 5,
    "initial_energy_grass": 2.0,
    "initial_energy_predator": 5.0,
    "initial_energy_prey": 3.0,
    "seed": 123,
    "predator_creation_energy_threshold": 8.0,
    "prey_creation_energy_threshold": 6.0,
    "max_episode_steps": 20,
}

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    register_env("PredPreyGrassTest", lambda cfg: env_creator(cfg))

    # Create a sample env to get per-agent spaces
    sample_env = env_creator(config_env)
    obs_space = sample_env.observation_space
    act_space = sample_env.action_space
    def policy_mapping_fn(agent_id, episode=None, **kwargs):
        return "shared_policy"

    config = (
        PPOConfig()
        .environment(
            env="PredPreyGrassTest",
            env_config=config_env
        )
        .framework("torch")
        .env_runners(num_env_runners=0)
        .multi_agent(
            policies={
                "shared_policy": PolicySpec(
                    observation_space=obs_space,
                    action_space=act_space,
                    config={}
                )
            },
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(train_batch_size=64, minibatch_size=16, num_epochs=1)
        .rl_module(model_config={
            "conv_filters": [
                [16, 3, 1],
                [32, 3, 1]
            ],
            "conv_activation": "relu",
        })
        .resources(num_gpus=0)
    )

    algo = config.build()
    print("Training for 2 iterations...")
    for i in range(2):
        result = algo.train()
        reward_mean = result.get('episode_return_mean', result.get('episode_reward_mean', result.get('env_runners/episode_return_mean')))
        print(f"Iter {i}: reward_mean={reward_mean}")
    print("Done.")
    ray.shutdown()
