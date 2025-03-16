# external libraries
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray import tune
from ray.tune.registry import register_env
from ray.tune import RunConfig  

# discretionary libraries
from predpreygrass.rllib.predpreygrass_rllib_env import PredPreyGrass  
from predpreygrass.rllib.config_env import config_env

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

register_env("PredPreyGrass", env_creator)

def policy_mapping_fn(agent_id, *args, **kwargs):
    if "predator" in agent_id:
        return "predator_policy"
    elif "prey" in agent_id:
        return "prey_policy"
    return None

if __name__ == "__main__":
    ray.shutdown()
    ray.init(
        num_cpus=8,
        log_to_driver=True,
        ignore_reinit_error=True,
    )

    checkpoint_dir = "/home/doesburg/ray_results/PPO_2025-03-14_11-46-25"  # Set your actual checkpoint path
    #checkpoint_dir = "path_to_checkpoints_dir"  # Set your actual checkpoint path
    
    sample_env = env_creator({})  # Create a single instance

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

    except Exception as e:
        print(f"Starting new training experiment.")

        # Define the checkpoint configuration for a new training session
        checkpoint_config = tune.CheckpointConfig(
            num_to_keep=100,
            checkpoint_frequency=10,
            checkpoint_at_end=True,
        )

        # Create a fresh PPO configuration if no checkpoint is found
        ppo = (
            PPOConfig()
            .environment(env="PredPreyGrass")
            .framework("torch")
            .multi_agent(
                # This ensures that each policy is trained on the right observation/action space.
                policies={
                    "predator_policy": (None, sample_env.observation_spaces["predator_0"], sample_env.action_spaces["predator_0"], {}),
                    "prey_policy": (None, sample_env.observation_spaces["prey_0"], sample_env.action_spaces["prey_0"], {}),
                },
                policy_mapping_fn=policy_mapping_fn,
            )
            .training(
                train_batch_size=512,  # for memory overload: reduced batch size (was 1024)
                minibatch_size=128,  # reduced minibatch size (was full batch)
                num_epochs=5,  # reduce number of passes (was deafult)
                gamma=0.99,
                lr=0.0003,
                lambda_=0.95,  # GAE smoothing
                clip_param=0.2,  # stable PPO clipping
            )
            . rl_module(
                model_config_dict={
                    "conv_filters": [
                        [16, [3, 3], 1],
                        [32, [3, 3], 2],
                        [64, [3, 3], 1],
                    ],
                    "fcnet_hiddens": [128, 128],
                    "fcnet_activation": "relu",
                },
                catalog_class=None,  # ✅ Explicitly set catalog_class (prevents fallback)
                observation_space=sample_env.observation_spaces["predator_0"],  # ✅ Explicitly set observation space
                action_space=sample_env.action_spaces["predator_0"],  # ✅ Explicitly set action space
                inference_only=False,  # ✅ Ensure training happens
            )
            . env_runners(
                num_env_runners=4,  
                num_envs_per_env_runner=4,  
                rollout_fragment_length="auto",
                sample_timeout_s=600,  
                num_cpus_per_env_runner=1  
            )
            .resources(
                num_cpus_for_main_process=2  
            )       
            .callbacks(EpisodeReturn)
        )

        # Start a new experiment if no checkpoint is found
        tuner = tune.Tuner(
            ppo.algo_class,
            param_space=ppo,
            run_config=RunConfig(
                stop={"training_iteration": 1000},
                checkpoint_config=checkpoint_config,  # ✅ Use pre-defined checkpoint config
                verbose=2,
            ),
        )
        # Run the Tuner and capture the results.
        results = tuner.fit()
    #print(f"Training results: {results}")
    ray.shutdown()
