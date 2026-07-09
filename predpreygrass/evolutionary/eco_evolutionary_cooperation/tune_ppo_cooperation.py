"""
Trains the eco_evolutionary_cooperation environment with PPO using the Ray RLlib
new API stack.

The environment is a predator-prey-grass grid world in which agents carry a
heritable cooperation_rate genome trait. Each step, an agent that successfully
hunts or grazes donates cooperation_rate * (that step's energy gain) to same-
species neighbors within cooperation_range (meal-sharing, not a continuous tax
on stock energy). Because offspring spawn adjacent to their parent, spatial
neighbors are more likely to be kin than a random draw from the population
(population viscosity) — so donations are kin-biased without any explicit
kin-recognition mechanism, the standard substrate for kin selection. Within-
lifetime foraging/hunting/dispersal behavior is learned by shared PPO policies
and is not inherited (Baldwinian layer); the RL-learned dispersal pattern in
turn determines local relatedness, closing the two-way Darwin/Baldwin loop.

Checkpoints and a copy of the environment source are saved under ~/ray_results/
for provenance. Cooperation-rate genome statistics are logged to TensorBoard
via the EpisodeReturn callback.
"""

from predpreygrass.evolutionary.eco_evolutionary_cooperation.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.evolutionary.eco_evolutionary_cooperation.config.config_env_eco_evolutionary import config_env
from predpreygrass.evolutionary.eco_evolutionary_cooperation.utils.episode_return_callback import EpisodeReturn
from predpreygrass.evolutionary.eco_evolutionary_cooperation.utils.networks import build_multi_module_spec

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune import Tuner, RunConfig, CheckpointConfig

from datetime import datetime
from pathlib import Path
import json
import shutil
from typing import Any


def get_config_ppo():
    import torch
    if torch.cuda.is_available():
        from predpreygrass.evolutionary.eco_evolutionary_cooperation.config.config_ppo_gpu_eco_evolutionary import config_ppo
    else:
        from predpreygrass.evolutionary.eco_evolutionary_cooperation.config.config_ppo_cpu_eco_evolutionary import config_ppo
    return config_ppo


def env_creator(config):
    return PredPreyGrass(config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    if "predator" in agent_id:
        return "predator"
    if "prey" in agent_id:
        return "prey"
    raise ValueError(f"Unrecognized agent_id format: {agent_id}")


# --- Main training setup ---

if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)

    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
    # ray_results_dir = "~/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    version = "ECO_EVOLUTION_COOPERATION"
    experiment_name = f"PPO_{version}_{timestamp}"
    experiment_path = ray_results_path / experiment_name

    experiment_path.mkdir(parents=True, exist_ok=True)
    # --- Save environment source file for provenance ---
    source_dir = experiment_path / "SOURCE_CODE"
    source_dir.mkdir(exist_ok=True)
    env_file = Path(__file__).parent / "predpreygrass_rllib_env.py"
    shutil.copy2(env_file, source_dir / f"predpreygrass_rllib_env_{version}.py")

    config_ppo = get_config_ppo()
    config_metadata = {
        "config_env": config_env,
        "config_ppo": config_ppo,
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)
    # print(f"Saved config to: {experiment_path/'run_config.json'}")

    sample_env = env_creator(config=config_env)
    if sample_env.observation_spaces is None or sample_env.action_spaces is None:
        raise RuntimeError("PredPreyGrass must define observation_spaces and action_spaces for all policies.")

    # Group spaces per policy id (first agent of each policy defines the space)
    obs_by_policy: dict[str, Any] = {}
    act_by_policy: dict[str, Any] = {}
    for agent_id, obs_space in sample_env.observation_spaces.items():
        pid = policy_mapping_fn(agent_id)
        if pid not in obs_by_policy:
            obs_by_policy[pid] = obs_space
            act_by_policy[pid] = sample_env.action_spaces[agent_id]

    # Build one MultiRLModuleSpec in one go
    multi_module_spec = build_multi_module_spec(obs_by_policy, act_by_policy)

    # Policies dict for RLlib
    policies = {
        pid: (None, obs_by_policy[pid], act_by_policy[pid], {})
        for pid in obs_by_policy
    }

    # Build config dictionary for Tune
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
        .callbacks(EpisodeReturn)
    )

    max_iters = config_ppo["max_iters"]
    checkpoint_every = 10
    del sample_env  # to avoid any stray references

    tuner = Tuner(
        ppo_config.algo_class,
        param_space=ppo_config.to_dict(),
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
