# disrcetionary libaries
from predpreygrass_12 import PredPreyGrass  # Import your custom environment

# external libraries
import gymnasium.spaces as spaces
import ray
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.tune.registry import register_env


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.episode_count = 0

    def on_episode_end(self, *, episode, **kwargs):
        self.overall_sum_of_rewards += episode.get_return()
        self.episode_count += 1
        if self.episode_count % 10 == 0:  # Print every 10 episodes
            print(f"Episode {self.episode_count} done. Reward = {episode.get_return()} Global SUM={self.overall_sum_of_rewards}")
            

def env_creator(config):
    return PredPreyGrass(config)

register_env("PredPreyGrass", env_creator)

def policy_mapping_fn(agent_id, *args, **kwargs):
    if "predator" in agent_id:
        return "predator_policy"
    elif "prey" in agent_id:
        return "prey_policy"
    else:
        raise ValueError(f"Unknown agent_id: {agent_id}")


if __name__ == "__main__":
    # Initialize Ray
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True, local_mode=True)

    # Configure PPO for RLlib
    ppo = (
        PPOConfig()
        .environment(
            env="PredPreyGrass", 
            disable_env_checking=True
        )
        .framework("torch")
        .multi_agent(
            policies={
                "predator_policy": (
                    None,
                    spaces.Dict({
                        "obs": spaces.Box(low=-1.0, high=100.0, shape=(4, 7, 7)) # TODO: inherit shape from env
                    }),
                    spaces.Discrete(5),
                    {},
                ),
                "prey_policy": (
                    None,
                    spaces.Dict({
                        "obs": spaces.Box(low=-1.0, high=100.0, shape=(4, 7, 7))
                    }),
                    spaces.Discrete(5),
                    {},
                ),
            },           
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size=2048,
            gamma=0.99,
            lr=0.0003,
        )
        .resources(num_gpus=0)
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "predator_policy": RLModuleSpec(
                        observation_space=spaces.Dict({
                            "obs": spaces.Box(low=-1.0, high=100.0, shape=(4, 7, 7))
                        }),
                        action_space=spaces.Discrete(5),
                        inference_only=False,
                        learner_only=False,
                        model_config={
                            "conv_filters": [
                                [16, [3, 3], 1],
                                [32, [3, 3], 1],
                                [64, [3, 3], 1],
                            ],
                            "fcnet_hiddens": [256, 256],
                            "fcnet_activation": "relu",
                        }
                    ),
                    "prey_policy": RLModuleSpec(
                        observation_space=spaces.Box(low=-1.0, high=100.0, shape=(4, 7, 7)),
                        action_space=spaces.Discrete(5),
                        inference_only=False,
                        learner_only=False,
                        model_config={
                            "conv_filters": [
                                [16, [3, 3], 1],
                                [32, [3, 3], 1],
                                [64, [3, 3], 1],
                            ],
                            "fcnet_hiddens": [256, 256],
                            "fcnet_activation": "relu",
                        }
                    ),
                }
            )
        )

        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .env_runners(
            num_envs_per_env_runner=1,  
            rollout_fragment_length=64,
            sample_timeout_s=300,  # Increase timeout


        )
        .callbacks(EpisodeReturn)
        .build_algo()  # new method to build the algorithm
    )



    # Training Loop
    for i in range(10):
        result = ppo.train()
        # ✅ Print available keys in `result` to check for reward-related keys



    print("Training finished.")
    # Shutdown Ray
    ray.shutdown()