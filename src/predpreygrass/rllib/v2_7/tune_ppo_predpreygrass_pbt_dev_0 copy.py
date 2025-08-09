from predpreygrass.rllib.v2_7.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v2_7.config.config_env_train_v1_0 import config_env

import os

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

from ray import tune
from ray import train
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.tune.registry import register_env
from ray.tune import Tuner
from ray.tune.schedulers import PopulationBasedTraining

import pprint
import json
import random
from datetime import datetime
from pathlib import Path


def env_creator(config):
    return PredPreyGrass(config or config_env)


def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        # Workaround to avoid PyTorch CUDA memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        from predpreygrass.rllib.v2_7.config.config_ppo_gpu_pbt import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.v2_7.config.config_ppo_cpu_pbt import config_ppo
    else:
        raise RuntimeError(f"Unsupported cpu_count={num_cpus}. Please add matching config_ppo.")
    return config_ppo


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


# Postprocess the perturbed PBT config to ensure it's still valid
def explore(config):
    config["train_batch_size_per_learner"] = min(config["train_batch_size_per_learner"], 2048)
    if config["train_batch_size_per_learner"] < config["minibatch_size"] * 2:
        config["train_batch_size_per_learner"] = config["minibatch_size"] * 2
    config["num_epochs"] = min(config["num_epochs"], 30)
    if config["num_epochs"] < 1:
        config["num_epochs"] = 1
    return config


if __name__ == "__main__":
    register_env("PredPreyGrass", env_creator)

    # experiment output directory
    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_PBT_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Get the PPO config based on the number of CPUs and save into experiment
    config_ppo = get_config_ppo()
    config_metadata = {
        "config_env": config_env,
        "config_ppo": config_ppo,
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)

    # Create a sample environment to build (multi) module specs
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

    ppo_config = (
        PPOConfig()
        .environment(env="PredPreyGrass")
        .framework("torch")
        .multi_agent(
            policies={pid: (None, module_specs[pid].observation_space, module_specs[pid].action_space, {}) for pid in module_specs},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            # These params are tuned from a fixed starting value.
            clip_param=config_ppo["clip_param"],
            lr=config_ppo["lr"],
            entropy_coeff=config_ppo["entropy_coeff"],
            # These params start off randomly drawn from a set.
            num_epochs=tune.choice([10, 30]),
            minibatch_size=tune.choice([128, 256, 512]),
            train_batch_size_per_learner=tune.choice([1024, 2048]),
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
    )

    # PBT setup
    hyperparam_mutations = {
        "lr": [1e-3, 5e-4, 1e-4],
        "clip_param": lambda: random.uniform(0.1, 0.3),
        "entropy_coeff": [0.0, 0.001, 0.005],
        "num_epochs": lambda: random.randint(10, 30),
        "minibatch_size": lambda: random.choice([128, 256, 512]),
        "train_batch_size_per_learner": lambda: random.choice([1024, 2048]),
    }
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=3,  # ← Cloning can occur after 3 iterations
        resample_probability=0.25,
        hyperparam_mutations=hyperparam_mutations,  # Specifies the mutations of hyperparams
        custom_explore_fn=explore,
        log_config=True,
        metric="env_runners/episode_return_mean",  # ← moved here
        mode="max",                                 # ← moved here

    )

    # Stopping criteria
    stopping_criteria = {"training_iteration": config_ppo["max_iters"]}

    checkpoint_every = 1

    tuner = Tuner(
        ppo_config.algo_class,
        param_space=ppo_config,
        tune_config=tune.TuneConfig(
            # metric="env_runners/episode_return_mean",
            # mode="max",
            scheduler=pbt,
            num_samples=5,
            reuse_actors=False,
        ),
        run_config=tune.RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_path),
            stop=stopping_criteria,
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=100,
                checkpoint_frequency=checkpoint_every,
                checkpoint_at_end=True,
            ),
      ),
    )

    result = tuner.fit()

    best_result = result.get_best_result(
        metric="env_runners/episode_return_mean",
        mode="max",
    )
    for result in best_result:
        print(result.metrics.get("env_runners/episode_return_mean"))

    print("Best performing trial's final set of hyperparameters:\n")
    pprint.pprint({k: v for k, v in best_result.config.items() if k in hyperparam_mutations})

    env_metrics = best_result.metrics.get("env_runners", {})

    metrics_to_print = {
        "episode_return_mean": env_metrics.get("episode_return_mean"),
        "episode_return_max": env_metrics.get("episode_return_max"),
        "episode_return_min": env_metrics.get("episode_return_min"),
        "episode_len_mean": env_metrics.get("episode_len_mean"),
    }
    print("\nBest performing trial's final reported metrics:\n")
    pprint.pprint(metrics_to_print)
