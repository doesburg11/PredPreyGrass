# https://docs.ray.io/en/latest/tune/examples/pbt_ppo_example.html

from predpreygrass.rllib.v2_7.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v2_7.config.config_env_train_v1_0 import config_env
from predpreygrass.rllib.v2_7.tune_ppo_multiagent_v2_7 import Path, datetime

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # <- NEW
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# keep each worker from spawning too many threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import random

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.registry import register_env
import pprint

from datetime import datetime
from pathlib import Path
import json


def env_creator(config):
    return PredPreyGrass(config or config_env)


def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.v2_7.config.config_ppo_gpu_pbt import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.v2_7.config.config_ppo_cpu import config_ppo
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


# Postprocess the perturbed config to ensure it's still valid
def explore(config):
    config["train_batch_size_per_learner"] = min(config["train_batch_size_per_learner"], 40000)
    config["minibatch_size"] = min(config["minibatch_size"], 512)
    config["num_epochs"] = min(config["num_epochs"], 10)
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

    # Get the PPO config based on the number of CPUs and sae it to a file
    config_ppo = get_config_ppo()
    config_metadata = {
        "config_env": config_env,
        "config_ppo": config_ppo,
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)

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

    hyperparam_mutations = {
        "lr": [1e-3, 5e-4, 1e-4],
        "clip_param": lambda: random.uniform(0.1, 0.4),
        "entropy_coeff": [0.0, 0.001, 0.005],
        "num_epochs": lambda: random.randint(1, 30),
        "minibatch_size": lambda: random.choice([128, 256, 512]),
        "train_batch_size_per_learner": lambda: random.choice([512, 1024, 2048]),
    }

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
    )

    # Stop when we've either reached 100 training iterations or reward=300
    stopping_criteria = {"training_iteration": 10, "episode_reward_mean": 300}

    config = (
        PPOConfig()
        .environment(env="PredPreyGrass")
        .framework("torch")
        .multi_agent(
            policies={pid: (None, module_specs[pid].observation_space, module_specs[pid].action_space, {}) for pid in module_specs},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            # These params are tuned from a fixed starting value.
            kl_coeff=config_ppo["kl_coeff"],
            lambda_=config_ppo["lambda_"],
            clip_param=config_ppo["clip_param"],
            lr=config_ppo["lr"],
            vf_loss_coeff=config_ppo["vf_loss_coeff"],
            # These params start off randomly drawn from a set.
            num_epochs=tune.choice([10, 20]),
            minibatch_size=tune.choice([128, 1024]),
            train_batch_size_per_learner=tune.choice([512, 2048]),
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

    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_return_mean",
            mode="max",
            scheduler=pbt,
            num_samples=2,
        ),
        param_space=config,
        run_config=tune.RunConfig(name=experiment_name, storage_path=str(ray_results_path), stop=stopping_criteria),
    )
    results = tuner.fit()

    best_result = results.get_best_result()

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
