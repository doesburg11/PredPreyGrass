"""
This script trains a multi-agent environment with APPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
Predators and prey both either can be of type_1 or type_2.
"""
from predpreygrass.rllib.shared_prey.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.shared_prey.config.config_env_shared_prey import config_env
from predpreygrass.rllib.shared_prey.utils.episode_return_callback import EpisodeReturn
from predpreygrass.rllib.shared_prey.utils.networks import build_multi_module_spec

import ray
from ray.rllib.algorithms.appo import APPOConfig
from ray.tune.registry import register_env
from ray.tune import Tuner, RunConfig, CheckpointConfig

import os
from datetime import datetime
from pathlib import Path
import json
import shutil


def get_config_appo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.shared_prey.config.config_appo_gpu_shared_prey import config_appo
    elif num_cpus == 8:
        from predpreygrass.rllib.shared_prey.config.config_appo_cpu_shared_prey import config_appo
    else:
        # Default to CPU config for other CPU counts to keep training usable across machines.
        from predpreygrass.rllib.shared_prey.config.config_appo_cpu_shared_prey import config_appo
    return config_appo


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
    parts = agent_id.split("_")
    type = parts[1]
    role = parts[2]
    return f"type_{type}_{role}"


# --- Main training setup ---

if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)

    # Override static seed at runtime to avoid deterministic placements; keep config file unchanged.
    # Enable strict RLlib outputs so only live agent IDs are emitted each step.
    env_config = {**config_env, "seed": None, "strict_rllib_output": True}


    ray_results_dir = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/shared_prey/ray_results/pred_decay_0_20/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    version = "GRID_30_PRED_OBS_RANGE_9_INITS_15_INIT_PREY_ENERGY_2_5"
    experiment_name = f"{version}_{timestamp}"
    experiment_path = ray_results_path / experiment_name 

    experiment_path.mkdir(parents=True, exist_ok=True)
    # --- Save environment source file for provenance ---
    source_dir = experiment_path / "SOURCE_CODE"
    source_dir.mkdir(exist_ok=True)
    env_file = Path(__file__).parent / "predpreygrass_rllib_env.py"
    shutil.copy2(env_file, source_dir / f"predpreygrass_rllib_env_{version}.py")

    config_appo = get_config_appo()
    config_metadata = {
        "config_env": config_env,
        "config_appo": config_appo,
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)
    # print(f"Saved config to: {experiment_path/'run_config.json'}")

    sample_env = env_creator(config=env_config)
    # Ensure spaces are populated before extracting
    sample_env.reset(seed=None)

    # Group spaces per policy id (first agent of each policy defines the space)
    obs_by_policy, act_by_policy = {}, {}
    for agent_id, obs_space in sample_env.observation_spaces.items():
        pid = policy_mapping_fn(agent_id)
        if pid not in obs_by_policy:
            obs_by_policy[pid] = obs_space
            act_by_policy[pid] = sample_env.action_spaces[agent_id]

    # Explicitly include action_space_struct so connectors see every agent ID
    # (avoids KeyErrors when new agents appear mid-episode).
    sample_env.action_space_struct = sample_env.action_spaces

    # Build one MultiRLModuleSpec in one go
    multi_module_spec = build_multi_module_spec(obs_by_policy, act_by_policy)

    # Policies dict for RLlib
    policies = {
        pid: (None, obs_by_policy[pid], act_by_policy[pid], {})
        for pid in obs_by_policy
    }

    # Build config dictionary for Tune
    appo_config = (
        APPOConfig()
        .environment(env="PredPreyGrass", env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size_per_learner=config_appo["train_batch_size_per_learner"],
            gamma=config_appo["gamma"],
            lr=config_appo["lr"],
            entropy_coeff=config_appo["entropy_coeff"],
            vf_loss_coeff=config_appo["vf_loss_coeff"],
            clip_param=config_appo["clip_param"],
            kl_coeff=config_appo["kl_coeff"],
            kl_target=config_appo["kl_target"],
            use_kl_loss=config_appo["use_kl_loss"],
            vtrace=config_appo["vtrace"],
            vf_clip_param=config_appo["vf_clip_param"],
            grad_clip=config_appo["grad_clip"],
        )
        .rl_module(rl_module_spec=multi_module_spec)
        .learners(
            num_gpus_per_learner=config_appo["num_gpus_per_learner"],
            num_learners=config_appo["num_learners"],
        )
        .env_runners(
            num_env_runners=config_appo["num_env_runners"],
            num_envs_per_env_runner=config_appo["num_envs_per_env_runner"],
            rollout_fragment_length=config_appo["rollout_fragment_length"],
            sample_timeout_s=config_appo["sample_timeout_s"],
            num_cpus_per_env_runner=config_appo["num_cpus_per_env_runner"],
        )
        
        .resources(
            num_cpus_for_main_process=config_appo["num_cpus_for_main_process"],
        )
        .callbacks(EpisodeReturn)
    )

    max_iters = config_appo["max_iters"]
    checkpoint_every = 10
    del sample_env  # to avoid any stray references

    tuner = Tuner(
        appo_config.algo_class,
        param_space=appo_config,
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_path),
            stop={"training_iteration": max_iters},
            checkpoint_config=CheckpointConfig(
                num_to_keep=100,
                checkpoint_frequency=checkpoint_every,
                checkpoint_at_end=True,
            ),
        ),
    )

    result = tuner.fit()
    ray.shutdown()
