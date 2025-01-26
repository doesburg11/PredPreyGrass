import gymnasium as gym
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from predpreygrass_10 import PredPreyGrass  # Import your custom environment
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import logging
logging.basicConfig(level=logging.ERROR)  # Only show errors
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



class CustomCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        print(f"Episode {episode.episode_id} ended with reward {episode.total_reward}.")


def env_creator(config):
    """
    Create an instance of the PredPreyGrass environment.
    """
    return PredPreyGrass(config)


# Register the custom environment
register_env("PredPreyGrass", env_creator)

def policy_mapping_fn(agent_id, episode):
    if "predator" in agent_id:
        return "predator_policy"
    elif "prey" in agent_id:
        return "prey_policy"
    return None

if __name__ == "__main__":
    # Initialize Ray
    ray.init()

    # Define the RLlib configuration
    config = (
        PPOConfig()
        .environment(
            env="PredPreyGrass", 
            env_config={"max_steps": 100}, 
            disable_env_checking=True
        )
        .framework("torch")
        .multi_agent(
            policies={
                "predator_policy": (
                    None,
                    gym.spaces.Box(low=-1.0, high=100.0, shape=(4, 7, 7)),
                    gym.spaces.Discrete(5),
                    {},
                ),
                "prey_policy": (
                    None,
                    gym.spaces.Box(low=-1.0, high=100.0, shape=(4, 7, 7)),
                    gym.spaces.Discrete(5),
                    {},
                ),
            },           
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size=4000,
            gamma=0.99,
            lr=0.0003,
        )
        .rl_module(
            model_config={
                "conv_filters": [
                    [16, [3, 3], 1],  # 16 filters, 3x3 kernel, stride 1
                    [32, [3, 3], 1],  # 32 filters, 3x3 kernel, stride 1
                    [64, [3, 3], 1],  # 64 filters, 3x3 kernel, stride 1
                ]
            }
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .rollouts(
            num_env_runners=2,
            rollout_fragment_length=50, 
            sample_timeout_s=120)

    )
    

    # Build the PPO algorithm with the configuration
    algo = config.build()

    # Train the algorithm
    for i in range(10):  # Train for 100 iterations
        try:
            result = algo.train()
            print(f"Iteration {i}: reward = {result.get('episode_reward_mean', 'N/A')}")
        except KeyError:
            print("Error in training metrics. Ensure episodes terminate correctly.")
            break

    # Save the trained model
    checkpoint = algo.save("checkpoints")
    print(f"Checkpoint saved at {checkpoint}")

    # Shutdown Ray
    ray.shutdown()
