"""
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
"""
from predpreygrass.rllib.numpy_vectorization.np_vec_env import PredPreyGrassEnv as PredPreyGrass
from predpreygrass.rllib.numpy_vectorization.utils.networks import build_multi_module_spec
from predpreygrass.rllib.numpy_vectorization.config.config_env_vectorization import config_env
from predpreygrass.rllib.numpy_vectorization.config.config_ppo_cpu_vectorization import config_ppo

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune import Tuner, RunConfig, CheckpointConfig

from datetime import datetime
from pathlib import Path
import json
import os


def get_config_ppo():
    # Directly return imported config_ppo
    return config_ppo


def env_creator(config):
    return PredPreyGrass(**config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    # For this environment, just return the agent_id as the policy name (single policy)
    return "shared_policy"


# --- Main training setup ---

if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)

    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_VECTORIZATION_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    config_ppo = get_config_ppo()
    # Optional runtime overrides via environment variables for quick smoke tests
    max_iters_override = os.environ.get("MAX_ITERS")
    if max_iters_override is not None:
        try:
            config_ppo["max_iters"] = int(max_iters_override)
        except ValueError:
            pass
    num_env_runners_override = os.environ.get("NUM_ENV_RUNNERS")
    if num_env_runners_override is not None:
        try:
            config_ppo["num_env_runners"] = int(num_env_runners_override)
        except ValueError:
            pass
    num_envs_per_runner_override = os.environ.get("NUM_ENVS_PER_ENV_RUNNER")
    if num_envs_per_runner_override is not None:
        try:
            config_ppo["num_envs_per_env_runner"] = int(num_envs_per_runner_override)
        except ValueError:
            pass
    config_metadata = {
        "config_env": config_env,
        "config_ppo": config_ppo,
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)
    # print(f"Saved config to: {experiment_path/'run_config.json'}")

    sample_env = env_creator(config=config_env)

    # Use a single shared policy for all agents
    obs_space = list(sample_env.observation_spaces.values())[0]
    act_space = list(sample_env.action_spaces.values())[0]
    multi_module_spec = build_multi_module_spec({"shared_policy": obs_space}, {"shared_policy": act_space})
    policies = {"shared_policy": (None, obs_space, act_space, {})}

    # Build config dictionary for Tune
    ppo_config = (
        PPOConfig()
        .environment(env="PredPreyGrass", env_config=config_env)
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size_per_learner=config_ppo["train_batch_size_per_learner"],
            minibatch_size=config_ppo["minibatch_size"],
            num_epochs=config_ppo["num_epochs"],
            gamma=config_ppo["gamma"],
            lr=config_ppo["lr"],
            lambda_=config_ppo["lambda_"],
            entropy_coeff=config_ppo["entropy_coeff"],
            vf_loss_coeff=config_ppo["vf_loss_coeff"],
            clip_param=config_ppo["clip_param"],
            kl_coeff=config_ppo["kl_coeff"],
            kl_target=config_ppo["kl_target"],
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
    # No custom callbacks
    )

    max_iters = config_ppo["max_iters"]
    checkpoint_every = 10
    del sample_env  # to avoid any stray references

    tuner = Tuner(
        ppo_config.algo_class,
        param_space=ppo_config,
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
