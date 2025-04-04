"""
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
The environment is implemented in the file predpreygrass/rllib/predpreygrass_rllib_env_moore_speed.py.
The environment configuration is in the file predpreygrass/rllib/config_env.py.

This implements MultiRLModuleSpec explicitly to define the policies for predators and prey.
"""
from predpreygrass.rllib.v4_select_coef.predpreygrass_rllib_env import PredPreyGrass 
from predpreygrass.rllib.v4_select_coef.config_env import config_env

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
        num_cpus=32,  # Reserve 2 for OS/system overhead
        num_gpus=0,
        log_to_driver=True,
        ignore_reinit_error=True,
    )
    register_env("PredPreyGrass", env_creator)
    # Set your actual checkpoint path if you want to restore training
    #checkpoint_dir = f"file://{os.path.abspath('./predpreygrass/rllib/v4_select_coef/trained_models/config_2')}"
    checkpoint_dir = "/checkpoint_dir"  # Placeholder for the checkpoint directory

    sample_env = env_creator({})  # Create a single instance
    sample_agents = ["speed_1_predator_0", "speed_2_predator_0", "speed_1_prey_0", "speed_2_prey_0"]
    module_specs = {}
    for sample_agent in sample_agents:
        policy = policy_mapping_fn(sample_agent)
        module_specs[policy] = build_module_spec(
            sample_env.observation_spaces[sample_agent],
            sample_env.action_spaces[sample_agent]
        )

    multi_module_spec = MultiRLModuleSpec(rl_module_specs=module_specs)

    # Try restoring from an existing experiment if available
    try:
        restored_tuner = tune.Tuner.restore(
            path=checkpoint_dir,  # The directory where Tune stores experiment results
            resume_errored=True,  # Resume even if the last trial errored
            trainable=PPOConfig().algo_class,  # The algorithm class used in the experiment
        )
        print("Successfully restored training from checkpoint.")

        # Continue training
        results = restored_tuner.fit()

    except:
        print(f"Starting new training experiment.")

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
                train_batch_size=1024, 
                gamma=0.99,
                lr=0.0003,            
            )
            .rl_module(
                rl_module_spec=multi_module_spec
            )
            .env_runners(
                num_env_runners=12,  
                num_envs_per_env_runner=2,  
                rollout_fragment_length="auto",
                sample_timeout_s=600,  
                num_cpus_per_env_runner=2 
            )
            .learners(
                num_learners=1,
                num_cpus_per_learner=6,
                num_gpus_per_learner=0
            )
            .resources(
                num_cpus_for_main_process=2,
                num_gpus=0  # Enables GPU usage

            )       
            .callbacks(EpisodeReturn)
        )

        # Start a new experiment if no checkpoint is found
        tuner = tune.Tuner(
            ppo.algo_class,
            param_space=ppo,
            run_config=train.RunConfig(
                stop={"training_iteration": 1000},
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=100,  # Keep only the last 5 checkpoints to save disk space
                    checkpoint_frequency=10,  # Save every 10 iterations
                    checkpoint_at_end=True,  # Ensure a checkpoint is saved at the end
                ),
            ),
        )
        # Run the Tuner and capture the results.
        results = tuner.fit()
    #print(f"Training results: {results}")
    ray.shutdown()
