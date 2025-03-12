import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray import train, tune
from ray.rllib.callbacks.callbacks import RLlibCallback

from predpreygrass.rllib.predpreygrass_16 import PredPreyGrass  # Import your custom environment
from config_env_16 import config_env

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


# env_creator is already a function that takes a config argument and returns an instance of PredPreyGrass
# directly pass it without using lambda
register_env("PredPreyGrass", env_creator)

def policy_mapping_fn(agent_id, *args, **kwargs):
    if "predator" in agent_id:
        return "predator_policy"
    elif "prey" in agent_id:
        return "prey_policy"
    return None

if __name__ == "__main__":
    # Initialize Ray
    ray.shutdown()
    ray.init(
        num_cpus=8,
        log_to_driver=True,
        ignore_reinit_error=True,
    )

    sample_env = env_creator({})  # Create a single instance

    # Configure PPO for RLlib
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
            train_batch_size=1024,
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
            num_env_runners=4,  
            num_envs_per_env_runner=4,  
            rollout_fragment_length="auto",
            sample_timeout_s=600,  
            num_cpus_per_env_runner=1  
        )
        .resources(
            num_gpus=0,  # Use GPU if available
            num_cpus_for_main_process=2  
        )       
        .callbacks(EpisodeReturn)
    )

   # Create a Tuner instance
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
