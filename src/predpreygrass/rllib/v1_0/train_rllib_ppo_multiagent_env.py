"""
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
This implements MultiRLModuleSpec explicitly to define the policies for predators
and prey separately.
"""
from predpreygrass.rllib.v1_0.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v1_0.config_env import config_env

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
    if "predator" in agent_id:
        return "predator_policy"
    elif "prey" in agent_id:
        return "prey_policy"
    return None


if __name__ == "__main__":
    register_env("PredPreyGrass", env_creator)
    ray.shutdown()
    ray.init(
        log_to_driver=True,
        ignore_reinit_error=True,
    )
    # Set your actual checkpoint path if you want to restore training
    checkpoint_dir = f"file://{os.path.abspath('./src/predpreygrass/rllib/v1_0/trained_model/')}"

    sample_env = env_creator({})  # Create a single instance
    # Observation/action spaces for the sample policies
    obs_space_pred = sample_env.observation_spaces["predator_0"]
    act_space_pred = sample_env.action_spaces["predator_0"]
    obs_space_prey = sample_env.observation_spaces["prey_0"]
    act_space_prey = sample_env.action_spaces["prey_0"]

    multi_module_spec = MultiRLModuleSpec(
        rl_module_specs={
            "predator_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_pred,
                action_space=act_space_pred,
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
            ),
            "prey_policy": RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=obs_space_prey,
                action_space=act_space_prey,
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
            ),
        }
    )

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
    except Exception:
        print("No checkpoint found. Starting new training experiment.")
        print("Starting new training experiment.")

        # Create a fresh PPO configuration if no checkpoint is found
        ppo = (
            PPOConfig()
            .environment(env="PredPreyGrass")
            .framework("torch")
            .multi_agent(
                # This ensures that each policy is trained on the right observation/action space.
                policies={
                    "predator_policy": (
                        None,
                        sample_env.observation_spaces["predator_0"],
                        sample_env.action_spaces["predator_0"],
                        {},
                    ),
                    "prey_policy": (None, sample_env.observation_spaces["prey_0"], sample_env.action_spaces["prey_0"], {}),
                },
                policy_mapping_fn=policy_mapping_fn,
            )
            .training(
                train_batch_size=1024,
                gamma=0.99,
                lr=0.0003,
            )
            .rl_module(rl_module_spec=multi_module_spec)
            .env_runners(
                num_env_runners=4,
                num_envs_per_env_runner=4,
                rollout_fragment_length="auto",
                sample_timeout_s=600,
                num_cpus_per_env_runner=1,
            )
            .resources(num_cpus_for_main_process=2)
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
    # print(f"Training results: {results}")
    ray.shutdown()
