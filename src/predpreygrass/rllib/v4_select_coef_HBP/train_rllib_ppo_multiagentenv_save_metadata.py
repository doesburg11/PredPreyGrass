"""
HBP COMPUTER
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
The environment is implemented in the file predpreygrass/rllib/predpreygrass_rllib_env_moore_speed.py.
The environment configuration is in the file predpreygrass/rllib/config_env.py.

This implements MultiRLModuleSpec explicitly to define the policies for predators and prey.
"""
from predpreygrass.rllib.v4_select_coef_HBP.predpreygrass_rllib_env import PredPreyGrass 
from predpreygrass.rllib.v4_select_coef_HBP.config_env import config_env

#  external libraries
import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.tune.registry import register_env
import os
from datetime import datetime
from pathlib import Path
import json


# ======= hyperparameters =======
# training
train_batch_size=1024
gamma=0.99
lr=0.0003            

# learners
num_gpus_per_learner=1
num_learners=1

# env_runners
num_env_runners=8
num_envs_per_env_runner=3
rollout_fragment_length="auto"
sample_timeout_s=600
num_cpus_per_env_runner=3 

#resources
num_cpus_for_main_process=4
    

class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0

    def on_episode_end(self, *, episode, **kwargs):
        """
        Called at the end of each episode.
        Logs the total and average rewards separately for predators and prey.
        """
        self.num_episodes += 1
        self.overall_sum_of_rewards += episode.get_return()

        # Initialize reward tracking
        predator_total_reward = 0.0
        prey_total_reward = 0.0
        predator_count = 0
        prey_count = 0

        # Retrieve rewards
        rewards = episode.get_rewards()  # Dictionary of {agent_id: list_of_rewards}

        for agent_id, reward_list in rewards.items():
            total_reward = sum(reward_list)  # Sum all rewards for the episode

            if "predator" in agent_id:
                predator_total_reward += total_reward
                predator_count += 1
            elif "prey" in agent_id:
                prey_total_reward += total_reward
                prey_count += 1

        # Compute average rewards (avoid division by zero)
        predator_avg_reward = predator_total_reward / predator_count if predator_count > 0 else 0
        prey_avg_reward = prey_total_reward / prey_count if prey_count > 0 else 0

        # Print episode logs
        print(f"Episode {self.num_episodes}: R={episode.get_return()} Global SUM={self.overall_sum_of_rewards}")
        print(f"  - Predators: Total Reward = {predator_total_reward:.2f}, Avg Reward = {predator_avg_reward:.2f}")
        print(f"  - Prey: Total Reward = {prey_total_reward:.2f}, Avg Reward = {prey_avg_reward:.2f}")

def env_creator(config):
    return PredPreyGrass(config or config_env)

def policy_mapping_fn(agent_id, *args, **kwargs):
    # Expected format: "speed_1_predator_0", "speed_2_prey_5"
    parts = agent_id.split("_")
    # parts[0] = "speed"
    # parts[1] = speed value
    # parts[2] = role ("predator" or "prey")
    # parts[3] = agent index (not needed)
    speed = parts[1]
    role = parts[2]
    # Example output:
    # "speed_1_predator_0" → "speed_1_predator"
    # "speed_2_prey_5" → "speed_2_prey"
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
    ray.init(
        log_to_driver=True,
        ignore_reinit_error=True,
    )
    register_env("PredPreyGrass", env_creator)

    """================= Create you own Ray Result output directory here ==================="""
    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results"
    ray_results_path = Path(ray_results_dir).expanduser()
    existing_experiment_dir = "PPO_2025-04-10_12-15-07/"
    experiment_path = Path(ray_results_dir+existing_experiment_dir).expanduser()

    # === Checkpoint restore path (full experiment folder!) ===
    if (experiment_path / "tuner.pkl").exists():
        restored_tuner = tune.Tuner.restore(
            path=str(experiment_path),  # The directory where Tune stores experiment results
            resume_errored=True,  # Resume even if the last trial errored
            trainable=PPOConfig().algo_class,  # The algorithm class used in the experiment
        )
        print("Successfully restored training from checkpoint.")

        # Continue training
        results = restored_tuner.fit()

    else:
        print(" === No checkpoint found, start a new experiment === ")
        print(f"Starting new training experiment.")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = f"PPO_{timestamp}"

        trial_dir = ray_results_path / experiment_name / "PPO_PredPreyGrass_00000"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Sample env for observation/action space setup
        sample_env = env_creator({})  # Create a single instance
        sample_agents = ["speed_1_predator_0", "speed_2_predator_0", "speed_1_prey_0", "speed_2_prey_0"]
        module_specs = {}
        for sample_agent in sample_agents:
            policy = policy_mapping_fn(sample_agent)
            module_specs[policy] = build_module_spec(
                sample_env.observation_spaces[sample_agent],
                sample_env.action_spaces[sample_agent]
            )

        # alternative
        """
        module_specs = {
            policy_mapping_fn(agent): build_module_spec(
                sample_env.observation_spaces[agent],
                sample_env.action_spaces[agent]
            ) for agent in sample_agents
        }
        """

        multi_module_spec = MultiRLModuleSpec(rl_module_specs=module_specs)

       # Prepare and save config metadata before training
        config_metadata = {
            "config_env": config_env,
            "ppo_config": {
                "train_batch_size": train_batch_size,
                "gamma": gamma,
                "lr": lr,
                "rollout_fragment_length": rollout_fragment_length,
                "num_env_runners": num_env_runners,
                "num_envs_per_env_runner": num_envs_per_env_runner,
                "num_cpus_per_env_runner": num_cpus_per_env_runner,
                "num_gpus_per_learner": num_gpus_per_learner,
            },
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
                train_batch_size=train_batch_size, 
                gamma=gamma,
                lr=lr,            
            )
            .rl_module(
                rl_module_spec=multi_module_spec
            )
            .learners(
                num_gpus_per_learner=num_gpus_per_learner,
                num_learners=num_learners,
            )
            .env_runners(
                num_env_runners=num_env_runners,  
                num_envs_per_env_runner=num_envs_per_env_runner,  
                rollout_fragment_length=rollout_fragment_length,
                sample_timeout_s=sample_timeout_s,  
                num_cpus_per_env_runner=num_cpus_per_env_runner 
            )
            .resources(
                num_cpus_for_main_process=num_cpus_for_main_process,
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
