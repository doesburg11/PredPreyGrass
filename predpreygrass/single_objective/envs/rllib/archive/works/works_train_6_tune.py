import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray import train, tune
from ray.rllib.callbacks.callbacks import RLlibCallback

from works_predpreygrass_6 import PredPreyGrass  # Import your custom environment

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


register_env("PredPreyGrass", lambda config: env_creator(config))


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
        #logging_level="DEBUG", 
        ignore_reinit_error=True, 
        #local_mode=True
    )

    # Configure PPO for RLlib
    ppo = (
        PPOConfig()
        .environment(env="PredPreyGrass")
        .framework("torch")
        .multi_agent(
            policies = {
                "predator_policy",
                "prey_policy",
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
                "conv_filters": [  # Ensure CNN expects 4 input channels
                    [16, [3, 3], 1],  # 16 filters, 3x3 kernel, stride 1
                    [32, [3, 3], 1],  # 32 filters, 3x3 kernel, stride 1
                    [64, [3, 3], 1],  # 64 filters, 3x3 kernel, stride 1
                ],
                "fcnet_hiddens": [256, 256],  # Fully connected layers
                "fcnet_activation": "relu"
            }
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .env_runners(
            num_envs_per_env_runner=1,  
            rollout_fragment_length=64,
            sample_timeout_s=300,  # Increase timeout to 5 minutes
        )
        .callbacks(EpisodeReturn)
    )

    # Visualization setup
    env = PredPreyGrass()
    grid_size = (env.grid_size, env.grid_size)
    all_agents = env.possible_agents + env.grass_agents
    # Create a Tuner instance to manage the trials.
    tuner = tune.Tuner(
        ppo.algo_class,
        param_space=ppo,
        run_config=train.RunConfig(
            stop={"training_iteration": 20},  # ✅ Corrected stopping criterion
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=5,  # Keep only the last 5 checkpoints to save disk space
                checkpoint_frequency=5,  # Save every 10 iterations
                checkpoint_at_end=True,  # Ensure a checkpoint is saved at the end
            ),
        ),
    )

    # Run the Tuner and capture the results.
    results = tuner.fit()
    ray.shutdown()
