import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from predpreygrass_14 import PredPreyGrass  # Import the custom environment
from predpreygrass.single_objective.utils.renderer import MatPlotLibRenderer
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune, train
from ray.rllib.callbacks.callbacks import RLlibCallback


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0

    def on_episode_end(self, *, episode, **kwargs):
        self.num_episodes += 1
        self.overall_sum_of_rewards += episode.get_return()
        print(f"Episode {self.num_episodes}: R={episode.get_return()} Global SUM={self.overall_sum_of_rewards}")

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Define environment registration
def env_creator(config):
    return PredPreyGrass(config)

register_env("PredPreyGrass", lambda config: env_creator(config))

# Policy mapping function
def policy_mapping_fn(agent_id, *args, **kwargs):
    if "predator" in agent_id:
        return "predator_policy"
    elif "prey" in agent_id:
        return "prey_policy"
    return None

# Path to checkpoint
checkpoint_path = "/home/doesburg/ray_results/PPO_2025-03-03_18-53-43/PPO_PredPreyGrass_72755_00000_0_2025-03-03_18-53-43/"

# Initialize the environment (only for getting observation/action spaces)
sample_env = env_creator({}) 

# Configure PPO
ppo_config = (
    PPOConfig()
    .environment(env="PredPreyGrass")
    .framework("torch")
    .multi_agent(
        policies={
            "predator_policy": (None, sample_env.observation_space, sample_env.action_space, {}),
            "prey_policy": (None, sample_env.observation_space, sample_env.action_space, {}),
        },
        policy_mapping_fn=policy_mapping_fn,
    )
    .training(
        train_batch_size=2048,
        gamma=0.99,
        lr=0.0003,
    )
        .rl_module(
            model_config_dict={
                "conv_filters": [  # Ensure CNN expects 4 input channels
                    [16, [3, 3], 1],  # 16 filters, 3x3 kernel, stride 1
                    [32, [3, 3], 1],  # 32 filters, 3x3 kernel, stride 1
                    [64, [3, 3], 1],  # 64 filters, 3x3 kernel, stride 1
                ],
                "fcnet_hiddens": [256, 256],  # Fully connected layers
                "fcnet_activation": "relu"
            },
        )
        . env_runners(
            num_env_runners=6,  
            num_envs_per_env_runner=4,  
            rollout_fragment_length="auto",
            sample_timeout_s=300,  
            num_cpus_per_env_runner=1  
        )
        .resources(
            num_gpus=0,  # Use GPU if available
            num_cpus_for_main_process=2  
        )       
        .callbacks(EpisodeReturn)
    )

# Resume training with Tune
tuner = tune.Tuner.restore(
    path=checkpoint_path, 
    trainable="PPO",
    resume_errored=True
    )
"""
tuner = tune.Tuner.restore(
    path=checkpoint_path,  # Provide the experiment directory containing the checkpoint
    trainable="PPO",  # RLlib expects the algorithm as a string here
    param_space=ppo_config.to_dict(),  # Convert PPOConfig to dict
    resume_errored=True,  # Resume even if errors occurred
    run_config=train.RunConfig(
        stop={"training_iteration": 1000},
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=5,  # Keep last 5 checkpoints to save space
            checkpoint_frequency=10,
            checkpoint_at_end=True,
        ),
    ),
)
"""

# Start training
tuner.fit()

# Shutdown Ray after training
ray.shutdown()
