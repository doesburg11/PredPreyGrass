# File: tune_ppo_multiagentenv_search.py
# Full replacement script (concurrency-focused, no env tuning, predator-only metric)

from predpreygrass.rllib.v3_1.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v3_1.config.config_env_train_v1_0 import config_env
from predpreygrass.rllib.v3_1.utils.networks import build_multi_module_spec

import os
from datetime import datetime
from pathlib import Path
import json

import ray
from ray import tune
from ray.tune import Tuner, RunConfig, CheckpointConfig
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.stopper import CombinedStopper, TrialPlateauStopper, FunctionStopper

# -----------------------------------------------------------------------------
# Safety knobs
# -----------------------------------------------------------------------------
# Keep the CUDA allocator resilient (useful even if GPU is OFF here).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------------------------------------------------------
# Predator-only metric
# -----------------------------------------------------------------------------
class PredatorScore(DefaultCallbacks):
    """Adds 'score_pred' = pred/100. Target >= 1.0 means predator returns >= 100."""
    def on_train_result(self, *, algorithm, result, **kwargs):
        m = result.get("env_runners", {}).get("agent_episode_returns_mean", {})
        pred = float(m.get("type_1_predator_0", 0.0))
        result["score_pred"] = pred / 100.0


def get_config_ppo():
    # Choose your base PPO defaults by CPU count (keeping your original logic)
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.v3_1.config.config_ppo_hbp_search import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.v3_1.config.config_ppo_cpu import config_ppo
    else:
        raise RuntimeError(f"Unsupported cpu_count={num_cpus}. Please add matching config_ppo.")
    return config_ppo


def env_creator(config):
    return PredPreyGrass(config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    type_ = parts[1]
    role = parts[2]
    return f"type_{type_}_{role}"


if __name__ == "__main__":
    # Fresh Ray
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    # Env registration
    register_env("PredPreyGrass", env_creator)

    # Paths & experiment meta
    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    config_ppo = get_config_ppo()
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump({"config_env": config_env, "config_ppo": config_ppo}, f, indent=4)

    # Probe spaces per policy
    sample_env = env_creator(config=config_env)
    obs_by_policy, act_by_policy = {}, {}
    for agent_id, obs_space in sample_env.observation_spaces.items():
        pid = policy_mapping_fn(agent_id)
        if pid not in obs_by_policy:
            obs_by_policy[pid] = obs_space
            act_by_policy[pid] = sample_env.action_spaces[agent_id]
    del sample_env

    multi_module_spec = build_multi_module_spec(obs_by_policy, act_by_policy)
    policies = {pid: (None, obs_by_policy[pid], act_by_policy[pid], {}) for pid in obs_by_policy}

    # ------------------------------
    # Base PPO config (fixed env, small per-trial resources)
    # ------------------------------
    ppo_config = (
        PPOConfig()
        .environment(env="PredPreyGrass", env_config=config_env)
        .framework("torch")
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
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
            num_learners=config_ppo["num_learners"],
            num_gpus_per_learner=config_ppo["num_gpus_per_learner"],
            num_cpus_per_learner=config_ppo["num_cpus_per_learner"],
        )
        .env_runners(
            num_env_runners=config_ppo["num_env_runners"],
            num_envs_per_env_runner=config_ppo["num_envs_per_env_runner"],
            rollout_fragment_length=config_ppo["rollout_fragment_length"],
            sample_timeout_s=config_ppo["sample_timeout_s"],
            num_cpus_per_env_runner=config_ppo["num_cpus_per_env_runner"],
        )
        .resources(
            num_cpus_for_main_process=1,  # driver thread budget (low on purpose)
        )
        .callbacks(PredatorScore)
    )

    max_iters = config_ppo["max_iters"]
    checkpoint_every = 10

    # ------------------------------
    # Search space: ONLY PPO hypers + light runtime knobs (no env tuning)
    # Use update_from_dict for Tune objects to bypass config validation at build time.
    # ------------------------------
    search_space = ppo_config.copy(copy_frozen=False)

    # PPO hyperparameters (only)
    search_space.training(
        num_epochs=tune.qrandint(20, 40, q=2),
        #train_batch_size_per_learner=tune.qrandint(1024, 3072, q=256),
    )

    # ------------------------------
    # Early stopping & scheduler
    # ------------------------------
    def target_hit(_trial_id, result):
        # Stop a trial as soon as predator average ≥ 100.
        return result.get("score_pred", 0.0) >= 1.0

    stopper = CombinedStopper(
        FunctionStopper(target_hit),
        TrialPlateauStopper(
            metric="score_pred",
            mode="max",
            std=0.02,
            num_results=12,
            grace_period=24,
            metric_threshold=0.05,
        ),
    )

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=max_iters,
        grace_period=20,
        reduction_factor=4,
    )

    searcher = OptunaSearch()

    # ------------------------------
    # Tuner: explicit 4-way concurrency
    # ------------------------------
    tuner = Tuner(
        ppo_config.algo_class,
        param_space=search_space,
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_path),
            stop=stopper,
            checkpoint_config=CheckpointConfig(
                num_to_keep=100,
                checkpoint_frequency=checkpoint_every,
                checkpoint_at_end=True,
            ),
        ),
        tune_config=tune.TuneConfig(
            metric="score_pred",
            mode="max",
            scheduler=scheduler,
            search_alg=searcher,
            num_samples=48,
            max_concurrent_trials=4,
        ),
    )

    result = tuner.fit()
    ray.shutdown()
