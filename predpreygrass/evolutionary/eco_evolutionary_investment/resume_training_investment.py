"""
Resume PPO training for eco_evolutionary_investment from the latest checkpoint.

Automatically finds the most recent PPO_ECO_EVOLUTION_INVESTMENT_* experiment
directory under ~/ray_results/ and calls Tuner.restore() to continue training.
"""

from predpreygrass.evolutionary.eco_evolutionary_investment.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.evolutionary.eco_evolutionary_investment.config.config_env_eco_evolutionary import config_env
from predpreygrass.evolutionary.eco_evolutionary_investment.utils.episode_return_callback import EpisodeReturn
from predpreygrass.evolutionary.eco_evolutionary_investment.utils.networks import build_multi_module_spec

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune import Tuner

import os
from pathlib import Path
from typing import Any


def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.evolutionary.eco_evolutionary_investment.config.config_ppo_gpu_eco_evolutionary import config_ppo
    else:
        from predpreygrass.evolutionary.eco_evolutionary_investment.config.config_ppo_cpu_eco_evolutionary import config_ppo
    return config_ppo


def env_creator(config):
    return PredPreyGrass(config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    if "predator" in agent_id:
        return "predator"
    if "prey" in agent_id:
        return "prey"
    raise ValueError(f"Unrecognized agent_id format: {agent_id}")


def find_latest_experiment(ray_results_path: Path, prefix: str) -> Path:
    candidates = sorted(
        [d for d in ray_results_path.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No experiment directories found with prefix '{prefix}' in {ray_results_path}")
    latest = candidates[-1]
    print(f"Resuming from: {latest}")
    return latest


# Higher entropy after resume encourages exploration during the cold-start phase before the
# ecosystem stabilizes. Drop back to config_ppo["entropy_coeff"] (0.01) once training
# recovers (typically after ~10-20 iterations).
RESUME_ENTROPY_COEFF = 0.05


if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)

    ray_results_path = Path("~/ray_results/").expanduser()
    experiment_path = find_latest_experiment(ray_results_path, "PPO_ECO_EVOLUTION_INVESTMENT_")

    config_ppo = get_config_ppo()

    sample_env = env_creator(config=config_env)
    assert sample_env.observation_spaces is not None and sample_env.action_spaces is not None
    obs_by_policy: dict[str, Any] = {}
    act_by_policy: dict[str, Any] = {}
    for agent_id, obs_space in sample_env.observation_spaces.items():
        pid = policy_mapping_fn(agent_id)
        if pid not in obs_by_policy:
            obs_by_policy[pid] = obs_space
            act_by_policy[pid] = sample_env.action_spaces[agent_id]
    del sample_env

    multi_module_spec = build_multi_module_spec(obs_by_policy, act_by_policy)
    policies = {
        pid: (None, obs_by_policy[pid], act_by_policy[pid], {})
        for pid in obs_by_policy
    }

    ppo_config = (
        PPOConfig()
        .environment(env="PredPreyGrass", env_config=config_env, disable_env_checking=True)
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
            entropy_coeff=RESUME_ENTROPY_COEFF,
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
        .callbacks(EpisodeReturn)
    )

    tuner = Tuner.restore(
        path=str(experiment_path),
        trainable=ppo_config.algo_class,  # type: ignore[arg-type]
        resume_unfinished=True,
        resume_errored=False,
        restart_errored=False,
        param_space=ppo_config.to_dict(),
    )

    result = tuner.fit()
    ray.shutdown()
