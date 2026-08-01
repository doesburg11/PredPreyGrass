"""
Trains the eco_evolutionary_nuptial_gift environment with PPO using the Ray RLlib
new API stack.

The environment splits predators into two sexes with separate policies:
predator_male hunts prey; predator_female grazes grass only, capped low enough
that grazing alone cannot reach the reproduction threshold. Only predator_female
reproduces. Each step a predator_male that successfully hunts donates
male_donation_rate * (that step's hunting gain) -- a heritable, mechanically-
executed genome trait, not a learned action -- to predator_female neighbors
within cooperation_range (a nuptial gift). Because offspring spawn adjacent to
their mother, spatial neighbors are more likely to be kin than a random draw
from the population (population viscosity), so gifts are kin-biased without
any explicit kin-recognition mechanism. Within-lifetime foraging/hunting/
dispersal behavior is learned by shared PPO policies (one per sex, plus prey)
and is not inherited (Baldwinian layer).

Pass --fixed-donation-rate to run the fixed-genome fitness-sweep mode instead
of the real evolutionary mode: genome_enabled is forced False and every
predator_male uses the same fixed, non-inherited donation rate (see
run_fixed_genome_sweep.sh and README.md's staged rollout plan).

Checkpoints and a copy of the environment source are saved under ~/ray_results/
for provenance. male_donation_rate genome statistics and the female
reproduction-gift-share metric are logged to TensorBoard via the EpisodeReturn
callback. See README.md for the full argument.
"""

from predpreygrass.evolutionary.eco_evolutionary_nuptial_gift.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.evolutionary.eco_evolutionary_nuptial_gift.config.config_env_eco_evolutionary import config_env as _base_config_env
from predpreygrass.evolutionary.eco_evolutionary_nuptial_gift.utils.episode_return_callback import EpisodeReturn
from predpreygrass.evolutionary.eco_evolutionary_nuptial_gift.utils.networks import build_multi_module_spec

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune import Tuner, RunConfig, CheckpointConfig

import argparse
import copy
from datetime import datetime
from pathlib import Path
import json
import shutil
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override config_env['seed'] for this run.",
    )
    parser.add_argument(
        "--max-iters", type=int, default=None,
        help="Override config_ppo['max_iters'] for this run.",
    )
    parser.add_argument(
        "--fixed-donation-rate", type=float, default=None,
        help="Run in fixed-genome fitness-sweep mode: forces genome_enabled=False "
             "and fixes every predator_male's donation rate at this value (no "
             "inheritance, no mutation, no per-agent variation). Omit for the "
             "real evolutionary mode.",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force the CPU PPO config regardless of GPU availability. Use this "
             "when a GPU exists on the machine but is already reserved by another "
             "training run (e.g. via a shared Ray cluster) -- requesting a GPU "
             "PPOConfig in that situation queues forever instead of erroring.",
    )
    return parser.parse_args()


def get_config_ppo(force_cpu: bool = False):
    import torch
    if not force_cpu and torch.cuda.is_available():
        from predpreygrass.evolutionary.eco_evolutionary_nuptial_gift.config.config_ppo_gpu_eco_evolutionary import config_ppo
    else:
        from predpreygrass.evolutionary.eco_evolutionary_nuptial_gift.config.config_ppo_cpu_eco_evolutionary import config_ppo
    return config_ppo


def env_creator(config):
    return PredPreyGrass(config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    if "predator_male" in agent_id:
        return "predator_male"
    if "predator_female" in agent_id:
        return "predator_female"
    if "prey" in agent_id:
        return "prey"
    raise ValueError(f"Unrecognized agent_id format: {agent_id}")


# --- Main training setup ---

if __name__ == "__main__":
    args = parse_args()

    config_env = copy.deepcopy(_base_config_env)
    if args.seed is not None:
        config_env["seed"] = args.seed
    fixed_tag = ""
    if args.fixed_donation_rate is not None:
        config_env["genome_enabled"] = False
        config_env["founder_genome"]["predator_male"]["male_donation_rate_mean"] = args.fixed_donation_rate
        config_env["founder_genome"]["predator_female"]["male_donation_rate_mean"] = args.fixed_donation_rate
        fixed_tag = f"_FIXED{args.fixed_donation_rate}"

    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)

    ray_results_dir = "~/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    version = f"ECO_EVOLUTION_NUPTIAL_GIFT{fixed_tag}"
    experiment_name = f"PPO_{version}_{timestamp}"
    experiment_path = ray_results_path / experiment_name

    experiment_path.mkdir(parents=True, exist_ok=True)
    # --- Save environment source file for provenance ---
    source_dir = experiment_path / "SOURCE_CODE"
    source_dir.mkdir(exist_ok=True)
    env_file = Path(__file__).parent / "predpreygrass_rllib_env.py"
    shutil.copy2(env_file, source_dir / f"predpreygrass_rllib_env_{version}.py")

    config_ppo = get_config_ppo(force_cpu=args.cpu)
    if args.max_iters is not None:
        config_ppo = dict(config_ppo, max_iters=args.max_iters)
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
