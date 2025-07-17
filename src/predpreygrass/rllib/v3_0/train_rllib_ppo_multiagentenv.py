"""
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
Predators and prey both either posses speed_1 or speed_2.
speed 1: action_space(9); Moore neighborhood movement (including "stay")
speed_2: action_space(25); Extended Moore neighborhood movement (including "stay")
"""
from predpreygrass.rllib.v3_0.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v3_0.config.config_env_train import config_env
from predpreygrass.rllib.v3_0.utils.episode_return_callback import EpisodeReturn
from predpreygrass.rllib.v3_0.utils.fitness_tracker import mutate_reward_config_from_stats, get_offspring_totals

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
        from predpreygrass.rllib.v2_3.config.config_ppo_gpu import config_ppo
    # CPU configuration
    elif num_cpus == 8:
        from predpreygrass.rllib.v2_3.config.config_ppo_cpu import config_ppo
    # Colab configuration
    elif num_cpus == 2:
        from predpreygrass.rllib.v2_3.config.config_ppo_colab import config_ppo
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

    # adjust the path to your personal results directory
    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_{timestamp}"
    trial_dir = ray_results_path / experiment_name / "PPO_PredPreyGrass_00000"
    trial_dir.mkdir(parents=True, exist_ok=True)
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    # Inject into config_env dynamically
    config_env["log_dir"] = str(trial_dir)  # <== ✅ FIXED: add log_dir to config_env
    # Now register and launch the environment
    register_env("PredPreyGrass", lambda config: PredPreyGrass(config))

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

    config_env["log_dir"] = str(experiment_path / "reward_evolution")
    config_env["iteration"] = 0

    # Build PPO algorithm
    ppo_config = PPOConfig()
    ppo_config = (
        ppo_config.environment(env="PredPreyGrass", env_config=config_env)
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
    )

    ppo_algo = ppo_config.build_algo(logger_creator=custom_logger_creator({}))

    # Manual training loop
    max_iters = 1000
    checkpoint_every = 10

    # Reward mutation output dir
    log_dir = str(experiment_path / "reward_evolution")
    os.makedirs(log_dir, exist_ok=True)
    current_reward_config = config_env["reward_config"].copy()

    for iter in range(max_iters):
        print(f"\n=== Training iteration {iter + 1}/{max_iters} ===")
        # Update the env_config dict with current iteration
        ppo_algo.config["env_config"]["iteration"] = iter
        ppo_algo.config["env_config"]["log_dir"] = log_dir
        print(f"[DEBUG] Current iteration: {iter}, log_dir: {log_dir}")
        result = ppo_algo.train()

        # Define where the mutated reward config will be stored
        save_path = os.path.join(log_dir, f"mutated_reward_config_{iter}.json")

        # Look for file saved by environment at reset
        stats_path = os.path.join(log_dir, f"offspring_stats_iter_{iter}.json")
        if not os.path.isfile(stats_path):
            print(f"[META-SELECTION] Skipped mutation: stats file missing at {stats_path}")
            continue

        with open(stats_path, "r") as f:
            stats_data = json.load(f)

        offspring_totals = get_offspring_totals(stats_data)

        # Mutate and save new reward config based on offspring stats
        mutate_reward_config_from_stats(
            current_reward_config, offspring_totals, target_key="reward_prey_eat_grass", save_path=save_path
        )

        # Load and apply updated reward config to all workers
        with open(save_path, "r") as f:
            updated_reward_config = json.load(f)

        ppo_algo.workers.foreach_worker(lambda w: w.env.set_reward_config(updated_reward_config))

        sorted_types = sorted(
            [(k, v) for k, v in offspring_totals.items() if "prey" in k],
            key=lambda x: x[1],
            reverse=True,
        )

        if len(sorted_types) >= 2:
            winner, loser = sorted_types[0][0], sorted_types[1][0]
            print(
                f"[ITERATION LOG] Winner: {winner} ({sorted_types[0][1]} offspring), Loser: {loser} ({sorted_types[1][1]} offspring)"
            )
        else:
            print("[ITERATION LOG] Not enough prey types to determine winner.")

        print("[ITERATION LOG] Completed iteration", iter)
        print("[ITERATION LOG] Saved reward config to:", save_path)
        print("[ITERATION LOG] Current reward_prey_eat_grass values:")
        for key in sorted(current_reward_config.keys()):
            if "prey" in key:
                reward_val = current_reward_config[key].get("reward_prey_eat_grass", 0)
                print(f"    {key}: {reward_val:.3f}")

        # Checkpoint
        if (iter + 1) % checkpoint_every == 0 or (iter + 1) == max_iters:
            checkpoint_path = experiment_path / f"checkpoint_iter_{iter + 1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ppo_algo.save_to_path(str(checkpoint_path))
            print(f"Saved Algorithm checkpoint to {checkpoint_path}")

    # Delay shutdown to give Ray time to clean up, to avoid crashing
    time.sleep(2)
    del ppo_algo
    ray.shutdown()
