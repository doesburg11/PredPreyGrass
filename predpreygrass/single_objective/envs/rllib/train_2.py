import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import gymnasium
from gymnasium.envs.registration import registry
from ray.rllib.callbacks.callbacks import RLlibCallback
import warnings
from predpreygrass_10 import PredPreyGrass  # Import your custom environment

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Could not create a Catalog object for your RLModule")


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        # Keep some global state in between individual callback events.
        self.overall_sum_of_rewards = 0.0

    def on_episode_end(self, *, episode, **kwargs):
        self.overall_sum_of_rewards += episode.get_return()
        print(f"Episode done. R={episode.get_return()} Global SUM={self.overall_sum_of_rewards}")


def env_creator(config):
    return PredPreyGrass(config)

# Ensure the environment is registered only once
if "PredPreyGrass" not in registry:
    register_env("PredPreyGrass", env_creator)

def policy_mapping_fn(agent_id, episode):
    if "predator" in agent_id:
        return "predator_policy"
    elif "prey" in agent_id:
        return "prey_policy"
    return None

if __name__ == "__main__":
    # Initialize Ray
    ray.shutdown()
    ray.init(log_to_driver=True, logging_level="DEBUG", ignore_reinit_error=True, local_mode=True)

    # Configure PPO for RLlib
    ppo = (
        PPOConfig()
        .environment(
            env="PredPreyGrass", 
            disable_env_checking=True
        )
        .framework("torch")
        .multi_agent(
            policies={
                "predator_policy": (
                    None,
                    gymnasium.spaces.Box(low=-1.0, high=100.0, shape=(4, 7, 7)),
                    gymnasium.spaces.Discrete(5),
                    {},
                ),
                "prey_policy": (
                    None,
                    gymnasium.spaces.Box(low=-1.0, high=100.0, shape=(4, 7, 7)),
                    gymnasium.spaces.Discrete(5),
                    {},
                ),
            },           
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size=128,
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
        .env_runners(
            num_envs_per_env_runner=1,  
            rollout_fragment_length=64,
            sample_timeout_s=300,  # Increase timeout


        )
        .callbacks(EpisodeReturn)
        .build_algo()
    )

    # Visualization setup
    env = PredPreyGrass()
    grid_size = (env.x_grid_size, env.y_grid_size)
    all_agents = env.possible_agents + env.grass_agents
    #visualizer = GridVisualizer(grid_size, all_agents, trace_length=1)

    results = ppo.train()
    print(f"Training results: {results.keys()}")
 
    import time
    time.sleep(2)
    ppo.stop()
    ray.shutdown()
