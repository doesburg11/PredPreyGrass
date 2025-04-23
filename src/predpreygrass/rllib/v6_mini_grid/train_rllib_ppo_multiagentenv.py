"""
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
Improvements versus v4_age: 
- externalized PPO config to config_ppo.py
- added RLlibCallback to log episode returns externally
- implemented energy move cost to the environment
"""
import os
os.environ["TORCH_GLOBAL_FOREACH"] = "0"

from predpreygrass.rllib.v6_mini_grid.predpreygrass_rllib_env import PredPreyGrass 
from predpreygrass.rllib.v6_mini_grid.config.config_env_train import config_env
from predpreygrass.utils.episode_return_callback import EpisodeReturn

#  external libraries
import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.tune.registry import register_env
import os
from datetime import datetime
from pathlib import Path
import json


def get_config_ppo():
    """
    ADjust to your systems.
    Dynamically select the appropriate PPO config based on system resources.
    Returns:
        config_ppo (dict): The selected PPO config module.
    Raises:
        RuntimeError if no suitable config is matched.
    """
    num_cpus = os.cpu_count()

    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.v6_mini_grid.config.config_ppo_gpu import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.v6_mini_grid.config.config_ppo_cpu import config_ppo
    elif num_cpus == 2:
        from predpreygrass.rllib.v6_mini_grid.config.config_ppo_colab import config_ppo
    else:
        raise RuntimeError(f"Unsupported cpu_count={num_cpus}. Please add matching config_ppo.")

    return config_ppo

def env_creator(config):
    return PredPreyGrass(config or config_env)

def policy_mapping_fn(agent_id, *args, **kwargs):  # Expected format: "speed_1_predator_0", "speed_2_prey_5"
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
    )


if __name__ == "__main__":
    ray.init(log_to_driver=True, ignore_reinit_error=True)
    register_env("PredPreyGrass", env_creator)
    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    existing_experiment_dir = "PPO_2025-04-23_21-51-36"
    experiment_path = ray_results_path / existing_experiment_dir
    if (experiment_path / "tuner.pkl").exists():
        restored_tuner = tune.Tuner.restore(
            path=str(experiment_path),  # The directory where Tune stores experiment results
            resume_errored=True,  # Resume even if the last trial errored
            trainable=PPOConfig().algo_class,  # The algorithm class used in the experiment
        )
        print("=== Successfully restored training from checkpoint ===")
        results = restored_tuner.fit()  # Continue training
    else:
        print(" === Start a new experiment === ")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = f"PPO_{timestamp}"
        trial_dir = ray_results_path / experiment_name / "meta_config_train"
        trial_dir.mkdir(parents=True, exist_ok=True)
        sample_env = env_creator(config=config_env)  # Sample env for observation/action space setup
        sample_agents = ["speed_1_predator_0", "speed_2_predator_0", "speed_1_prey_0", "speed_2_prey_0"]
        module_specs = {}
        for sample_agent in sample_agents:
            policy = policy_mapping_fn(sample_agent)
            module_specs[policy] = build_module_spec(
                sample_env.observation_spaces[sample_agent],
                sample_env.action_spaces[sample_agent]
            )
        multi_module_spec = MultiRLModuleSpec(rl_module_specs=module_specs)
        # Prepare and save config metadata before training
        config_ppo = get_config_ppo() # ppo config depending on system
        config_metadata = {
            "config_env": config_env,
            "config_ppo": config_ppo,
        }
        with open(trial_dir / "run_config.json", "w") as f:
            json.dump(config_metadata, f, indent=4)
        print(f"Saved config to: {trial_dir/'run_config.json'}")
        # Create a fresh PPO configuration if no checkpoint is found
        ppo = (
            PPOConfig()
            .environment(env="PredPreyGrass")
            .framework("torch")
            .multi_agent(
                # This ensures that each policy is trained on the right observation/action space.
                policies = {pid: (None, module_specs[pid].observation_space, module_specs[pid].action_space, {}) for pid in module_specs},
                policy_mapping_fn=policy_mapping_fn,
            )
            .training(
                train_batch_size=config_ppo["train_batch_size"], 
                gamma=config_ppo["gamma"],
                lr=config_ppo["lr"],            
            )
            .rl_module(
                rl_module_spec=multi_module_spec
            )
            .learners(
                num_gpus_per_learner=config_ppo["num_gpus_per_learner"],
                num_learners=config_ppo["num_learners"],
            )
            .env_runners(
                num_env_runners=config_ppo["num_env_runners"],  
                num_envs_per_env_runner=config_ppo["num_envs_per_env_runner"],  
                rollout_fragment_length=config_ppo["rollout_fragment_length"],
                sample_timeout_s=config_ppo["sample_timeout_s"],  
                num_cpus_per_env_runner=config_ppo["num_cpus_per_env_runner"] 
            )
            .resources(
                num_cpus_for_main_process=config_ppo["num_cpus_for_main_process"],
            )       
            .callbacks(EpisodeReturn)
        )

 
        # Start a new experiment if no checkpoint is found
        tuner = tune.Tuner(
            ppo.algo_class,
            param_space=ppo,
            run_config=train.RunConfig(
                name=experiment_name,
                storage_path=os.path.expanduser(ray_results_dir),
                stop={"training_iteration": 1000},
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=100,
                    checkpoint_frequency=10,
                    checkpoint_at_end=True,
                ),
            ),
        )
        # Run the Tuner and capture the results.
        results = tuner.fit()
    ray.shutdown()
