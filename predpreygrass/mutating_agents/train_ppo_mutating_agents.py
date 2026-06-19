"""
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
Predators and prey both either posses speed_1 or speed_2.
speed 1: action_space(9); Moore neighborhood movement (including "stay")
speed_2: action_space(25); Extended Moore neighborhood movement (including "stay")
"""
from predpreygrass.mutating_agents.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.mutating_agents.config.config_env_train import config_env
from predpreygrass.mutating_agents.utils.episode_return_callback import EpisodeReturn

# External libraries
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.tune.registry import register_env
import os
from datetime import datetime
from pathlib import Path
import json
import math
import time


def get_config_ppo():
    """
    Personalize to your system(s).
    Dynamically select the appropriate PPO config based on system resources.
    Returns:
        config_ppo (dict): The selected PPO config module.
    Raises:
        RuntimeError if no suitable config is matched.
    """
    num_cpus = os.cpu_count()
    # GPU configuration
    if num_cpus == 32:
        from predpreygrass.mutating_agents.config.config_ppo_gpu import config_ppo
    # CPU configuration
    elif num_cpus == 8:
        from predpreygrass.mutating_agents.config.config_ppo_cpu import config_ppo
    # Colab configuration
    elif num_cpus == 2:
        from predpreygrass.mutating_agents.config.config_ppo_colab import config_ppo
    else:
        raise RuntimeError(f"Unsupported cpu_count={num_cpus}. Please add matching config_ppo.")
    return config_ppo


def env_creator(config):
    return PredPreyGrass(config or config_env)


def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    Maps agent IDs to policies based on their speed and role.
    This function is used to determine which policy to apply for each agent.
    Args:
        agent_id (str): The ID of the agent, expected to be in the format "speed_X_role_Y".
    Returns:
        str: The policy name for the agent, formatted as "speed_X_role_Y".
    """
    # Expected format: "speed_1_predator_0", "speed_2_prey_5"
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


def get_result_metric(result, flat_key, *nested_keys, default=float("nan")):
    if flat_key in result:
        return result[flat_key]

    value = result
    for key in nested_keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def format_metric_with_last(value, last_value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        value = float(value)
        return f"{value:.2f}", value
    if last_value is not None:
        return f"no completed episodes (last={last_value:.2f})", last_value
    return "no completed episodes yet", last_value


# --- Main training loop ---

if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)
    # adjust the path to your personal results directory
    ray_results_dir = "~/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Save config metadata
    config_ppo = get_config_ppo()
    config_metadata = {
        "config_env": config_env,
        "config_ppo": config_ppo,
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)
    print(f"Saved config to: {experiment_path/'run_config.json'}")

    # Build MultiRLModuleSpec
    sample_env = env_creator(config=config_env)
    observation_spaces = sample_env.observation_spaces
    action_spaces = sample_env.action_spaces
    if observation_spaces is None or action_spaces is None:
        raise RuntimeError("PredPreyGrass must define observation_spaces and action_spaces before training.")

    sample_agents = ["speed_1_predator_0", "speed_2_predator_0", "speed_1_prey_0", "speed_2_prey_0"]
    module_specs = {}
    policy_specs = {}
    for sample_agent in sample_agents:
        policy = policy_mapping_fn(sample_agent)
        obs_space = observation_spaces[sample_agent]
        act_space = action_spaces[sample_agent]
        module_specs[policy] = build_module_spec(obs_space, act_space)
        policy_specs[policy] = (None, obs_space, act_space, {})
    multi_module_spec = MultiRLModuleSpec(rl_module_specs=module_specs)

    # Build PPO algorithm
    ppo_algo = (
        PPOConfig()
        .environment(env="PredPreyGrass")
        .framework("torch")
        .multi_agent(
            policies=policy_specs,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size=config_ppo["train_batch_size"],
            gamma=config_ppo["gamma"],
            lr=config_ppo["lr"],
        )
        .rl_module(rl_module_spec=multi_module_spec)
        .learners(
            num_gpus_per_learner=config_ppo["num_gpus_per_learner"],
            num_learners=config_ppo["num_learners"],
        )
        .env_runners(
            num_env_runners=config_ppo["num_env_runners"],
            num_envs_per_env_runner=config_ppo["num_envs_per_env_runner"],
            rollout_fragment_length=config_ppo["rollout_fragment_length"],
            sample_timeout_s=config_ppo["sample_timeout_s"],
            num_cpus_per_env_runner=config_ppo["num_cpus_per_env_runner"],
        )
        .resources(
            num_cpus_for_main_process=config_ppo["num_cpus_for_main_process"],
        )
        .callbacks(EpisodeReturn)
        .build_algo()
    )

    # Manual training loop
    max_iters = int(os.environ.get("PPO_MAX_ITERS", "1000"))
    checkpoint_every = int(os.environ.get("PPO_CHECKPOINT_EVERY", "10"))
    last_mean_return = None
    last_mean_len = None

    for iter in range(max_iters):
        print(f"\n=== Training iteration {iter + 1}/{max_iters} ===")
        result = ppo_algo.train()

        mean_return = get_result_metric(result, "env_runners/episode_return_mean", "env_runners", "episode_return_mean")
        mean_len = get_result_metric(result, "env_runners/episode_len_mean", "env_runners", "episode_len_mean")
        mean_return_text, last_mean_return = format_metric_with_last(mean_return, last_mean_return)
        mean_len_text, last_mean_len = format_metric_with_last(mean_len, last_mean_len)

        print(
            f"Iteration {iter + 1}: "
            f"Env steps sampled={result['num_env_steps_sampled_lifetime']}, "
            f"Mean episode return={mean_return_text}, "
            f"Mean episode length={mean_len_text}"
        )
        # Save checkpoint manually every N iterations
        if (iter + 1) % checkpoint_every == 0 or (iter + 1) == max_iters:
            checkpoint_path = experiment_path / f"checkpoint_iter_{iter + 1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # Save Algorithm checkpoint
            ppo_algo.save_to_path(str(checkpoint_path))
            print(f"Saved Algorithm checkpoint to {checkpoint_path}")

    # Delay shutdown to give Ray time to clean up, to avoid crashing
    ppo_algo.stop()
    time.sleep(2)
    ray.shutdown()
