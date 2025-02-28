import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray import train, tune
from ray.rllib.callbacks.callbacks import RLlibCallback

from predpreygrass5_ import PredPreyGrass  # Import your custom environment


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        # Keep some global state in between individual callback events.
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0

    def on_episode_end(self, *, episode, **kwargs):
        self.num_episodes += 1
        self.overall_sum_of_rewards += episode.get_return()
        print(f"Episode {self.num_episodes}: R={episode.get_return()} Global SUM={self.overall_sum_of_rewards}")

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

    sample_env = env_creator({})  # Create a single instance

    # Configure PPO for RLlib
    ppo = (
        PPOConfig()
        .environment(env="PredPreyGrass")
        .framework("torch")
        # chatGPT
        .multi_agent(
            policies={
                "predator_policy": (None, sample_env.observation_space, env_creator({}).action_space, {}),
                "prey_policy": (None, sample_env.observation_space, env_creator({}).action_space, {}),
            },
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            # charGPT
            train_batch_size=2048,  # 🔹 Search over batch sizes
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
            num_env_runners=6,  # Equivalent to num_rollout_workers
            num_envs_per_env_runner=1,  
             num_cpus_per_env_runner=1,
            rollout_fragment_length="auto",  
            sample_timeout_s=300,  # Increase timeout to 5 minutes
        )
        .resources(
            num_gpus=0,  # Set to 1 if using a GPU
            num_cpus_for_main_process=2 ,
        )
       .callbacks(EpisodeReturn)
    )

    # Visualization setup
    #env = PredPreyGrass()
    grid_size = (sample_env.grid_size, sample_env.grid_size)
    all_agents = sample_env.possible_agents + sample_env.grass_agents
    # Create a Tuner instance to manage the trials.
    tuner = tune.Tuner(
        ppo.algo_class,
        param_space=ppo,
        run_config=train.RunConfig(
            stop={"training_iteration": 5000},  # ✅ Corrected stopping criterion
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=100,  # Keep only the last 5 checkpoints to save disk space
                checkpoint_frequency=50,  # Save every 10 iterations
                checkpoint_at_end=True,  # Ensure a checkpoint is saved at the end
            ),
        ),
    )

    # Run the Tuner and capture the results.
    results = tuner.fit()
    #print(f"Training results: {results}")
    ray.shutdown()


from predpreygrass.single_objective.utils.renderer import MatPlotLibRenderer
from predpreygrass5_ import PredPreyGrass  # Import your custom environment
from time import sleep

verbose = False

if __name__ == "__main__":
    # Grid size
    env = PredPreyGrass()
    # Reset the environment and get initial observations
    observations, _ = env.reset(seed=41)

    print("\nRESET:")
    env._print_grid_state()
    env._print_grid_from_state()
    
    # Get the grid size
    grid_size = (env.grid_size, env.grid_size)
    
    # Combine predator, prey, and grass agents
    all_agents = env.agents + env.grass_agents
    # Initialize the visualizer
    visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=5)

    step = 0
    while env.agents:  # Stop when no agents are left
        print(f"STEP {step}:")
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        
        observations, rewards, terminations, truncations, info = env.step(action_dict)

        env._print_grid_state()
        env._print_grid_from_state()

        # Stop the loop if all agents are terminated
        if all(terminations.values()):
            print("All agents are terminated. Ending simulation.")
            break

        # Merge agent and grass positions for visualization
        merged_positions = {**env.agent_positions, **env.grass_positions}
        visualizer.update(merged_positions, step)
        
        sleep(0.1)  # Slow down visualization
        step += 1

    visualizer.close()

