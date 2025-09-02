# File: tune_ppo_multiagentenv_search.py
# Full replacement script

from predpreygrass.rllib.v3_1.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v3_1.config.config_env_train_v1_0 import config_env
from predpreygrass.rllib.v3_1.utils.networks import build_multi_module_spec

import os
from datetime import datetime
from pathlib import Path
import json
import random

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
# Runtime knobs
# -----------------------------------------------------------------------------
# Keep this False to ensure concurrency on your single-GPU box (all trials CPU-only).
# If you set it True, we’ll sample GPU trials rarely (still allows concurrency
# with CPU-only trials, but a GPU-using trial will hold the single GPU).
USE_GPU_TRIALS = False

# Reduce per-trial CPU footprint so several trials can run in parallel on 32 CPUs.
PER_TRIAL_NUM_ENVS = 1
PER_TRIAL_NUM_ENV_RUNNERS_CHOICES = [2, 3]   # small and steady
PER_TRIAL_CPUS_PER_ENV_RUNNER = 2
CPUS_FOR_DRIVER = 2

# Safer CUDA allocator (helps fragmentation when you enable GPU trials later).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------------------------------------------------------
# Composite metric: push both predator and prey past targets
# -----------------------------------------------------------------------------
class CompositeScore(DefaultCallbacks):
    """Adds 'score_both' = min(pred/100, prey/400) to training results."""
    def on_train_result(self, *, algorithm, result, **kwargs):
        m = result.get("env_runners", {}).get("agent_episode_returns_mean", {})
        pred = float(m.get("type_1_predator_0", 0.0))
        prey = float(m.get("type_1_prey_0", 0.0))
        result["score_both"] = min(pred / 100.0, prey / 400.0)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.v3_1.config.config_ppo_gpu_default import config_ppo
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

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)

    # Paths & experiment metadata
    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    config_ppo = get_config_ppo()
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(
            {"config_env": config_env, "config_ppo": config_ppo},
            f,
            indent=4,
        )

    # Probe spaces per policy from a sample env instance
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

    # Base PPO config (new API stack)
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
        .resources(num_cpus_for_main_process=config_ppo["num_cpus_for_main_process"])
        .callbacks(CompositeScore)
    )

    max_iters = config_ppo["max_iters"]
    checkpoint_every = 10

    # ------------------------------
    # Search space (clone + tune)
    # ------------------------------
    search_space = ppo_config.copy(copy_frozen=False)

    # Resource knobs as top-level keys (avoid early validation on tune objects)
    # CPU-only by default to guarantee concurrency on a 1-GPU machine.
    if USE_GPU_TRIALS:
        # Rarely pick GPU (3/4 = 0, 1/4 = 1)
        gpu_choice = tune.choice([0, 0, 0, 1])
    else:
        gpu_choice = tune.choice([0])

    search_space.update_from_dict({
        "num_gpus_per_learner": gpu_choice,
        "num_cpus_for_main_process": CPUS_FOR_DRIVER,
        "num_env_runners": tune.choice(PER_TRIAL_NUM_ENV_RUNNERS_CHOICES),
        "num_cpus_per_env_runner": PER_TRIAL_CPUS_PER_ENV_RUNNER,
        "num_envs_per_env_runner": PER_TRIAL_NUM_ENVS,
    })

    # GPU-aware batch sizing without triggering early validation
    def _pick_train_batch(spec):
        use_gpu = spec.config.get("num_gpus_per_learner", 0) == 1
        return random.choice(
            [2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096] if use_gpu
            else [1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072]
        )

    def _pick_minibatch(spec):
        use_gpu = spec.config.get("num_gpus_per_learner", 0) == 1
        return random.choice([96, 128, 160] if use_gpu else [64, 96, 128])

    def _pick_num_epochs(spec):
        use_gpu = spec.config.get("num_gpus_per_learner", 0) == 1
        return random.randint(15, 30) if use_gpu else random.randint(8, 20)

    search_space.update_from_dict({
        "train_batch_size_per_learner": tune.sample_from(_pick_train_batch),
        "minibatch_size": tune.sample_from(_pick_minibatch),
        "num_epochs": tune.sample_from(_pick_num_epochs),
    })

    # Core PPO hypers (don’t re-set train_batch/minibatch/epochs here)
    search_space.training(
        lr=tune.loguniform(1e-5, 5e-4),
        gamma=tune.uniform(0.92, 0.995),
        lambda_=tune.uniform(0.9, 1.0),
        entropy_coeff=tune.uniform(0.0, 0.02),
        clip_param=tune.uniform(0.15, 0.35),
        vf_loss_coeff=tune.uniform(0.6, 1.4),
    )

    # Gentle env learnability knobs
    env_cfg = config_env.copy()
    env_cfg.update({
        "move_energy_cost_factor": tune.loguniform(5e-4, 8e-3),
        "energy_gain_per_step_grass": tune.uniform(0.035, 0.06),
        "max_energy_grass": env_cfg["max_energy_grass"],
    })
    search_space.environment(env_config=env_cfg)

    # ------------------------------
    # Stoppers & Scheduler
    # ------------------------------
    def target_hit(_trial_id, result):
        return result.get("score_both", 0.0) >= 1.0

    stopper = CombinedStopper(
        FunctionStopper(target_hit),
        TrialPlateauStopper(
            metric="score_both",
            mode="max",                # <-- fixes your ValueError
            std=0.02,
            num_results=12,
            grace_period=24,
            metric_threshold=0.05
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
    # Tuner: enable real concurrency
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
            metric="score_both",
            mode="max",
            scheduler=scheduler,
            search_alg=searcher,
            num_samples=48,
            max_concurrent_trials=4,   # <-- with the small per-trial CPU, this *will* run concurrently
        ),
    )

    result = tuner.fit()
    ray.shutdown()
