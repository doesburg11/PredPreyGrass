"""
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
Predators and prey both either can be of type_1 or type_2.
"""
from predpreygrass.rllib.ppg_visibility.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.ppg_visibility.config.config_env_train_v1_0 import config_env
from predpreygrass.rllib.ppg_visibility.utils.episode_return_callback import EpisodeReturn
from predpreygrass.rllib.ppg_visibility.utils.networks import build_module_spec

# External libraries
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.registry import register_env
import os
from datetime import datetime
from pathlib import Path
import json
import time


# Custom logger_creator to redirect RLlib logs to our experiment_path
def custom_logger_creator(config):
    def logger_creator_func(config_):
        from ray.tune.logger import UnifiedLogger

        logdir = str(experiment_path)
        print(f"Redirecting RLlib logging to {logdir}")
        return UnifiedLogger(config_, logdir, loggers=None)

    return logger_creator_func


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
        from predpreygrass.rllib.ppg_visibility.config.config_ppo_gpu_default import config_ppo
    # CPU configuration
    elif num_cpus == 8:
        from predpreygrass.rllib.ppg_visibility.config.config_ppo_cpu import config_ppo
    else:
        raise RuntimeError(f"Unsupported cpu_count={num_cpus}. Please add matching config_ppo.")
    return config_ppo


def env_creator(config):
    return PredPreyGrass(config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    Maps agent IDs to policies based on their type and role.
    This function is used to determine which policy to apply for each agent.
    Args:
        agent_id (str): The ID of the agent, expected to be in the format "type_X_role_Y".
    Returns:
        str: The policy name for the agent, formatted as "type_X_role_Y".
    """
    # Expected format: "type_1_predator_0", "type_2_prey_5"
    parts = agent_id.split("_")
    type = parts[1]
    role = parts[2]
    return f"type_{type}_{role}"


# --- Main training loop ---

if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)
    # adjust the path to your personal results directory
    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
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
    # print(f"Saved config to: {experiment_path/'run_config.json'}")

    # Build MultiRLModuleSpec
    sample_env = env_creator(config=config_env)
    # Dynamically detect available sample agents
    module_specs = {}
    for agent_id in sample_env.observation_spaces:
        policy = policy_mapping_fn(agent_id)
        if policy not in module_specs:
            module_specs[policy] = build_module_spec(
                sample_env.observation_spaces[agent_id],
                sample_env.action_spaces[agent_id],
            )

    multi_module_spec = MultiRLModuleSpec(rl_module_specs=module_specs)

    # Build PPO algorithm
    ppo_config = (
        PPOConfig()
        .environment(env="PredPreyGrass", env_config=config_env)
        .framework("torch")
        .multi_agent(
            policies={pid: (None, module_specs[pid].observation_space, module_specs[pid].action_space, {}) for pid in module_specs},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size_per_learner=config_ppo["train_batch_size_per_learner"],
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
    )

    ppo_algo = ppo_config.build_algo(logger_creator=custom_logger_creator({}))

    # Manual training loop
    max_iters = config_ppo["max_iters"]
    checkpoint_every = 10

    for iter in range(max_iters):
        print(f"\n=== Training iteration {iter + 1}/{max_iters} ===")
        t0 = time.perf_counter()
        result = ppo_algo.train()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        steps = result.get("num_env_steps_sampled_lifetime", 0)

        mean_return = result.get("env_runners/episode_return_mean", float("nan"))
        mean_len = result.get("env_runners/episode_len_mean", float("nan"))

        print(f"Iteration {iter + 1}: Time={elapsed:.2f}s, " f"Steps={steps}, Return={mean_return:.2f}, Length={mean_len:.2f}")

        # Save checkpoint
        if (iter + 1) % checkpoint_every == 0 or (iter + 1) == max_iters:
            checkpoint_path = experiment_path / f"checkpoint_iter_{iter + 1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ppo_algo.save_to_path(str(checkpoint_path))
            print(f"Saved Algorithm checkpoint to {checkpoint_path}")

    # Delay shutdown to give Ray time to clean up, to avoid crashing
    time.sleep(2)
    del ppo_algo
    ray.shutdown()
