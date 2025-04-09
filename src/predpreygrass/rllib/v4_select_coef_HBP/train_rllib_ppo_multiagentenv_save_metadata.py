"""
HBP COMPUTER
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
"""

from predpreygrass.rllib.v4_select_coef_HBP.predpreygrass_rllib_env import PredPreyGrass 
from predpreygrass.rllib.v4_select_coef_HBP.config_env import config_env
from predpreygrass.rllib.utils.save_metadata import save_run_metadata_from_trial

# External libraries
import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.tune.registry import register_env
import torch
torch.set_default_device("cuda")
import os

# Training parameters
train_batch_size = 1024

# Learners
num_gpus_per_learner = 1
num_learners = 1

# Environment runners
num_env_runners = 8
num_envs_per_env_runner = 3
rollout_fragment_length = "auto"
sample_timeout_s = 600
num_cpus_per_env_runner = 3

# Other resources
num_cpus_for_main_process = 4


class EpisodeReturn(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0
        self.metadata_saved = False

    def on_algorithm_init(self, *, algorithm, **kwargs):
        if not self.metadata_saved:
            trial_dir = getattr(algorithm, "log_dir", None)
            if trial_dir:
                save_run_metadata_from_trial(
                    config_env=config_env,
                    ppo_config={
                        "train_batch_size": train_batch_size,
                        "gamma": 0.99,
                        "lr": 0.0003,
                        "rollout_fragment_length": rollout_fragment_length,
                        "num_env_runners": num_env_runners,
                        "num_envs_per_env_runner": num_envs_per_env_runner,
                        "num_cpus_per_env_runner": num_cpus_per_env_runner,
                        "num_gpus_per_learner": num_gpus_per_learner,
                    },
                    model_arch={
                        "conv_filters": [
                            [16, [3, 3], 1],
                            [32, [3, 3], 1],
                            [64, [3, 3], 1],
                        ],
                        "fcnet_hiddens": [256, 256],
                    },
                    trial=algorithm
                )
                self.metadata_saved = True

    def on_episode_end(self, *, episode, **kwargs):
        self.num_episodes += 1
        self.overall_sum_of_rewards += episode.get_return()

        predator_total_reward = 0.0
        prey_total_reward = 0.0
        predator_count = 0
        prey_count = 0

        rewards = episode.get_rewards()
        for agent_id, reward_list in rewards.items():
            total_reward = sum(reward_list)
            if "predator" in agent_id:
                predator_total_reward += total_reward
                predator_count += 1
            elif "prey" in agent_id:
                prey_total_reward += total_reward
                prey_count += 1

        predator_avg_reward = predator_total_reward / predator_count if predator_count > 0 else 0
        prey_avg_reward = prey_total_reward / prey_count if prey_count > 0 else 0

        print(f"Episode {self.num_episodes}: R={episode.get_return()} Global SUM={self.overall_sum_of_rewards}")
        print(f"  - Predators: Total Reward = {predator_total_reward:.2f}, Avg Reward = {predator_avg_reward:.2f}")
        print(f"  - Prey: Total Reward = {prey_total_reward:.2f}, Avg Reward = {prey_avg_reward:.2f}")


def env_creator(config):
    return PredPreyGrass(config or config_env)

def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    speed = parts[1]
    role = parts[2]
    return f"speed_{speed}_{role}"

def build_module_spec(obs_space, act_space):
    return RLModuleSpec(
        module_class=DefaultPPOTorchRLModule,
        observation_space=obs_space,
        action_space=act_space,
        inference_only=False,
        model_config={
            "conv_filters": [
                [16, [3, 3], 1],
                [32, [3, 3], 1],
                [64, [3, 3], 1],
            ],
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        catalog_class=None,
    )


if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)
    checkpoint_dir = "/checkpoint_dir"  # Change this if needed
    sample_env = env_creator({})
    sample_agents = ["speed_1_predator_0", "speed_2_predator_0", "speed_1_prey_0", "speed_2_prey_0"]

    module_specs = {
        policy_mapping_fn(agent): build_module_spec(
            sample_env.observation_spaces[agent],
            sample_env.action_spaces[agent]
        ) for agent in sample_agents
    }

    multi_module_spec = MultiRLModuleSpec(rl_module_specs=module_specs)

    try:
        restored_tuner = tune.Tuner.restore(
            path=checkpoint_dir,
            resume_errored=True,
            trainable=PPOConfig().algo_class,
        )
        print("Successfully restored training from checkpoint.")
        results = restored_tuner.fit()

    except:
        print("Starting new training experiment.")
        ppo = (
            PPOConfig()
            .environment(env="PredPreyGrass")
            .framework("torch")
            .multi_agent(
                policies={pid: (None, module_specs[pid].observation_space, module_specs[pid].action_space, {}) for pid in module_specs},
                policy_mapping_fn=policy_mapping_fn,
            )
            .training(
                train_batch_size=train_batch_size,
                gamma=0.99,
                lr=0.0003,
            )
            .rl_module(rl_module_spec=multi_module_spec)
            .learners(
                num_gpus_per_learner=num_gpus_per_learner,
                num_learners=num_learners,
            )
            .env_runners(
                num_env_runners=num_env_runners,
                num_envs_per_env_runner=num_envs_per_env_runner,
                rollout_fragment_length=rollout_fragment_length,
                sample_timeout_s=sample_timeout_s,
                num_cpus_per_env_runner=num_cpus_per_env_runner,
            )
            .resources(
                num_cpus_for_main_process=num_cpus_for_main_process,
            )
            .callbacks(EpisodeReturn)
        )

        tuner = tune.Tuner(
            ppo.algo_class,
            param_space=ppo,
            run_config=train.RunConfig(
                stop={"training_iteration": 1000},
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=100,
                    checkpoint_frequency=10,
                    checkpoint_at_end=True,
                ),
            ),
        )
        results = tuner.fit()

    ray.shutdown()
