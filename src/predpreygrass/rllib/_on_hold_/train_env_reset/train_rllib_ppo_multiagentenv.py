"""
THIS IS A TEST SCRIPT FOR CHANGING REWARDS DURING TRAINING
It runs 1 iteration, saves checkpoints, then runs another iteration with modified rewards.
"""
from predpreygrass.rllib._on_hold_train_env_reset.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib._on_hold_train_env_reset.config.config_env_train import config_env
from predpreygrass.rllib._on_hold_train_env_reset.utils.episode_return_callback import EpisodeReturn

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
        from predpreygrass.rllib._on_hold_train_env_reset.config.config_ppo_gpu import config_ppo
    # CPU configuration
    elif num_cpus == 8:
        from predpreygrass.rllib._on_hold_train_env_reset.config.config_ppo_cpu import config_ppo
    # Colab configuration
    elif num_cpus == 2:
        from predpreygrass.rllib._on_hold_train_env_reset.config.config_ppo_colab import config_ppo
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
    print(f"Saved config to: {experiment_path/'run_config.json'}")

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
    ppo_algo = (
        PPOConfig()
        .environment(env="PredPreyGrass")
        .framework("torch")
        .multi_agent(
            policies={pid: (None, module_specs[pid].observation_space, module_specs[pid].action_space, {}) for pid in module_specs},
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
        .build_algo(logger_creator=custom_logger_creator({}))
    )

    # Manual training loop
    max_iters = 1
    checkpoint_every = 1

    for iter in range(max_iters):
        print(f"\n=== Training iteration {iter + 1}/{max_iters} ===")
        result = ppo_algo.train()

        mean_return = result.get("env_runners/episode_return_mean", float("nan"))
        mean_len = result.get("env_runners/episode_len_mean", float("nan"))

        print(f"Iteration {iter + 1}: " f"Env steps sampled={result['num_env_steps_sampled_lifetime']}, ")
        # Save checkpoint manually every N iterations
        if (iter + 1) % checkpoint_every == 0 or (iter + 1) == max_iters:
            checkpoint_path = experiment_path / f"checkpoint_iter_{iter + 1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # Save Algorithm checkpoint
            ppo_algo.save_to_path(str(checkpoint_path))
            print(f"Saved Algorithm checkpoint to {checkpoint_path}")

    # Delay shutdown to give Ray time to clean up, to avoid crashing
    time.sleep(2)
    del ppo_algo

    checkpoint_str = str(experiment_path / "checkpoint_iter_1")

    # === STEP 2: Resume from checkpoint with modified grass-eating rewards ===

    print("\n=== Step 2: Resuming from checkpoint with updated reward config ===")

    # Modify reward for both prey types
    config_env["reward_prey_eat_grass"] = 1.0

    # Restore algorithm from checkpoint
    from ray.rllib.algorithms.algorithm import Algorithm

    resumed_algo = Algorithm.from_checkpoint(checkpoint_str)

    # Update the environment config with new reward values
    resumed_algo.config["env_config"] = config_env

    print(f"Resumed Algorithm from checkpoint: {checkpoint_str}")
    print(f"Updated environment config: {resumed_algo.config['env_config']}")

    # Resume training for 1 iteration only
    print("🚀 Resuming training for 1 additional iteration with new config...")
    result = resumed_algo.train()

    # Save another checkpoint after resumed training
    resumed_ckpt_path = experiment_path / f"checkpoint_iter_{2}"
    resumed_ckpt_path.mkdir(parents=True, exist_ok=True)
    resumed_algo.save_to_path(str(resumed_ckpt_path))
    print(f"✅ Saved resumed checkpoint to: {resumed_ckpt_path}")

    # Shutdown Ray
    time.sleep(2)
    del resumed_algo
    ray.shutdown()
