from predpreygrass.rllib.v2_7.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v2_7.config.config_env_train_v1_0 import config_env
from predpreygrass.rllib.v2_7.utils.episode_return_callback import EpisodeReturn

import random
from ray.tune.schedulers import PopulationBasedTraining
from ray import tune

import ray
from ray import train
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.tune.registry import register_env
from ray.tune import Tuner

import os
from datetime import datetime
from pathlib import Path
import json
import pprint


def custom_logger_creator(config):
    def logger_creator_func(config_):
        from ray.tune.logger import UnifiedLogger

        logdir = str(experiment_path)
        print(f"Redirecting RLlib logging to {logdir}")
        return UnifiedLogger(config_, logdir, loggers=None)

    return logger_creator_func


def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.v2_7.config.config_ppo_gpu_pbt import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.v2_7.config.config_ppo_cpu_pbt import config_ppo
    else:
        raise RuntimeError(f"Unsupported cpu_count={num_cpus}. Please add matching config_ppo.")
    return config_ppo


def env_creator(config):
    return PredPreyGrass(config or config_env)


def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    type = parts[1]
    role = parts[2]
    return f"type_{type}_{role}"


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


# --- Main training setup ---

if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)

    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    config_ppo = get_config_ppo()
    config_metadata = {
        "config_env": config_env,
        "config_ppo": config_ppo,
    }

    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)
    # print(f"Saved config to: {experiment_path/'run_config.json'}")

    sample_env = env_creator(config=config_env)
    module_specs = {}
    for agent_id in sample_env.observation_spaces:
        policy = policy_mapping_fn(agent_id)
        if policy not in module_specs:
            module_specs[policy] = build_module_spec(
                sample_env.observation_spaces[agent_id],
                sample_env.action_spaces[agent_id],
            )

    multi_module_spec = MultiRLModuleSpec(rl_module_specs=module_specs)

    # Build config dictionary for Tune
    ppo_config = (
        PPOConfig()
        .environment(env="PredPreyGrass")
        .framework("torch")
        .multi_agent(
            policies={pid: (None, module_specs[pid].observation_space, module_specs[pid].action_space, {}) for pid in module_specs},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size_per_learner=config_ppo["train_batch_size_per_learner"],
            gamma=config_ppo["gamma"],
            lr=config_ppo["lr"],
            minibatch_size=config_ppo["minibatch_size"],
            num_epochs=config_ppo["num_epochs"],  # Required for PBT to mutate
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

    max_iters = config_ppo["max_iters"]
    checkpoint_every = 10

    def explore(config):
        if config["train_batch_size_per_learner"] < config["minibatch_size"] * 2:
            config["train_batch_size_per_learner"] = config["minibatch_size"] * 2
        if config.get("num_epochs", 1) < 1:
            config["num_epochs"] = 1
        return config

    hyperparam_mutations = {
        "lr": [1e-3, 5e-4, 1e-4, 5e-5],
        "minibatch_size": lambda: random.randint(128, 1024),
        "train_batch_size_per_learner": lambda: random.randint(2000, 16000),
        "num_epochs": lambda: random.randint(1, 10),
    }

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
    )

    tuner = tune.Tuner(
        trainable=ppo_config.algo_class,
        param_space=ppo_config,
        run_config=train.RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_path),
            stop={"training_iteration": max_iters},
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=100,
                checkpoint_frequency=checkpoint_every,
                checkpoint_at_end=True,
            ),
        ),
        tune_config=tune.TuneConfig(
            scheduler=pbt,
            metric="env_runners/episode_return_mean",
            mode="max",
            num_samples=4,  # ← You can scale this up
        ),
    )

    result = tuner.fit()


    best_result = result.get_best_result()

    print("Best performing trial's final set of hyperparameters:\n")
    pprint.pprint(
        {k: v for k, v in best_result.config.items() if k in hyperparam_mutations}
    )

    print("\nBest performing trial's final reported metrics:\n")

    metrics_to_print = [
        "episode_reward_mean",
        "episode_reward_max",
        "episode_reward_min",
        "episode_len_mean",
    ]
    pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})

    ray.shutdown()
