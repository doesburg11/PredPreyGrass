"""
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
The agents are specialized based on their speed (high and low) and role (predator or prey).
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
Improvements versus v5_move_energy: 
- This training usses pre-trained RLModules for speed_1 agents.
- The RLModuleSpec is built for speed_2 agents.
"""

from predpreygrass.rllib.v6_mini_grid.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v6_mini_grid.config.config_env_train import config_env
from predpreygrass.utils.episode_return_callback import EpisodeReturn

#  external libraries
import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec, RLModule
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.tune.registry import register_env
from pathlib import Path
from datetime import datetime
import os, json

# === Helper Functions ===
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
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1], [64, [3, 3], 1]],
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
    )

if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=True)
    register_env("PredPreyGrass", env_creator)
    # Load pre-trained RLModules
    checkpoint_root = "/path/to/your/checkpoint_root/checkpoint_000099/learner_group/learner/rl_module"
    pretrained_modules = {
        "speed_1_predator": RLModule.from_checkpoint(os.path.join(checkpoint_root, "speed_1_predator")),
        "speed_1_prey": RLModule.from_checkpoint(os.path.join(checkpoint_root, "speed_1_prey")),
    }

    # Sample environment for space definitions
    sample_env = env_creator(config=config_env)
    sample_agents = ["speed_2_predator_0", "speed_2_prey_0"]

    # Build new module specs for untrained policies
    module_specs = {
        pid: build_module_spec(
            sample_env.observation_spaces[agent],
            sample_env.action_spaces[agent]
        )
        for agent, pid in zip(sample_agents, ["speed_2_predator", "speed_2_prey"])
    }

    # Combine pre-trained and new module specs
    multi_module_spec = MultiRLModuleSpec(rl_module_specs={
        **pretrained_modules,
        **module_specs,
    })

    # PPO Config
    ppo = (
        PPOConfig()
        .environment(env="PredPreyGrass")
        .framework("torch")
        .multi_agent(
            policies={
                pid: (None, multi_module_spec[pid].observation_space, multi_module_spec[pid].action_space, {})
                for pid in multi_module_spec.keys()
            },
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(train_batch_size=4096, gamma=0.99, lr=5e-5)
        .rl_module(rl_module_spec=multi_module_spec)
        .learners(num_gpus_per_learner=1, num_learners=1)
        .env_runners(num_env_runners=2, num_envs_per_env_runner=4, rollout_fragment_length=200, sample_timeout_s=60, num_cpus_per_env_runner=1)
        .resources(num_cpus_for_main_process=1)
        .callbacks(EpisodeReturn)
    )

    # Experiment setup
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_specialized_speed2_{timestamp}"
    tuner = tune.Tuner(
        ppo.algo_class,
        param_space=ppo,
        run_config=train.RunConfig(
            name=experiment_name,
            storage_path=os.path.expanduser("~/ray_results/"),
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
