# File: tune_ppo_multiagentenv_search.py
# Full replacement script (32-CPU optimized, 4 concurrent trials, no env tuning, predator-only metric)

from predpreygrass.rllib.v3_1.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v3_1.config.config_env_train_v1_0 import config_env
from predpreygrass.rllib.v3_1.utils.networks import build_multi_module_spec

import os
import csv
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
from ray.tune.stopper import CombinedStopper, TrialPlateauStopper, MaximumIterationStopper, Stopper

# --------------------------------------------------------------------------------
# Make CUDA allocator robust, even though we run CPU-only here.
# --------------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# --------------------------------------------------------------------------------
# Callback: report predator-only metric and write a CSV row once a trial hits >=100
# --------------------------------------------------------------------------------
class PredatorScore(DefaultCallbacks):
    """
    Adds 'score_pred' = predator_return/100.0.
    When a trial first reaches score_pred >= 1.0 (i.e., >=100 predator reward),
    append one row to <experiment_dir>/predator_100_hits.csv with:
    [trial_name, iteration, score_pred, lr, num_epochs].
    """
    def on_train_result(self, *, algorithm, result, **kwargs):
        # RLlib (new stack) nests returns under "env_runners"
        m = result.get("env_runners", {}).get("agent_episode_returns_mean", {})
        pred = float(m.get("type_1_predator_0", 0.0))
        score_pred = pred / 100.0

        # Expose metric to Tune
        result["score_pred"] = score_pred

        # One-shot CSV append when first crossing >= 1.0
        if score_pred >= 1.0 and not getattr(algorithm, "_saved_pred100_once", False):
            try:
                # trial local dir -> experiment dir
                trial_dir = getattr(algorithm, "local_path", None) or algorithm.logdir
                experiment_dir = os.path.dirname(trial_dir)
                csv_path = os.path.join(experiment_dir, "predator_100_hits.csv")

                cfg_from_result = result.get("config", {}) or {}
                row = {
                    "trial_name": os.path.basename(trial_dir),
                    "iteration": int(result.get("training_iteration", -1)),
                    "score_pred": float(score_pred),
                    "lr": float(cfg_from_result.get("lr", algorithm.config.get("lr"))),
                    "num_epochs": int(cfg_from_result.get("num_epochs", algorithm.config.get("num_epochs"))),
                }

                write_header = not os.path.exists(csv_path)
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["trial_name", "iteration", "score_pred", "lr", "num_epochs"]
                    )
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

                setattr(algorithm, "_saved_pred100_once", True)
            except Exception as e:
                print(f"[PredatorScore] Failed to append predator_100_hits.csv: {e}")


# --------------------------------------------------------------------------------
# Stopper: kill a trial if recent metric drops by more than threshold
# --------------------------------------------------------------------------------
class DropStopper(Stopper):
    """
    Stop a trial if the metric drops by more than `drop_threshold`
    over the last `window` iterations. Never stops the whole experiment.
    """
    def __init__(self, window: int = 5, drop_threshold: float = 0.2):
        self.window = int(window)
        self.drop_threshold = float(drop_threshold)
        self.history = {}  # trial_id -> list of recent scores

    def __call__(self, trial_id: str, result: dict) -> bool:
        score = result.get("score_pred")
        if score is None:
            return False

        hist = self.history.setdefault(trial_id, [])
        hist.append(float(score))
        if len(hist) > self.window:
            hist.pop(0)

        # Stop if last vs first in window drops more than threshold
        if len(hist) == self.window and (hist[-1] < hist[0] - self.drop_threshold):
            return True
        return False

    def stop_all(self) -> bool:
        return False  # never halt the whole study


# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------
def get_config_ppo():
    # Keep your original branch: pick the 32-CPU default bundle
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
    # e.g. "agent_type_1_predator_0" -> "type_1_predator"
    parts = agent_id.split("_")
    type_ = parts[1]
    role = parts[2]
    return f"type_{type_}_{role}"


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Make sure Ray reserves all 32 logical CPUs that we want to drive
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True, num_cpus=32)

    register_env("PredPreyGrass", env_creator)

    # Paths & experiment meta
    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Save the static configs used
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

    # --------------------------------------------------------------------------------
    # Fixed per-trial resources to hit 32 CPUs with 4 concurrent trials
    #   driver:                1 CPU
    #   learner:               1 CPU
    #   env-runners:           3 runners × 2 CPUs each = 6 CPUs
    #   TOTAL per trial:       8 CPUs  ->  4 trials × 8 = 32 CPUs
    # --------------------------------------------------------------------------------
    PER_TRIAL_NUM_ENV_RUNNERS = 3
    PER_TRIAL_CPUS_PER_ENV_RUNNER = 2
    PER_TRIAL_CPUS_LEARNER = 1
    CPUS_FOR_DRIVER = 1
    MAX_CONCURRENT_TRIALS = 4  # 4 × 8 CPUs = 32

    # --------------------------------------------------------------------------------
    # Base PPO config (no env tuning)
    # --------------------------------------------------------------------------------
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
            num_learners=1,
            num_gpus_per_learner=0,                    # CPU-only trials
            num_cpus_per_learner=PER_TRIAL_CPUS_LEARNER,
        )
        .env_runners(
            num_env_runners=PER_TRIAL_NUM_ENV_RUNNERS,
            num_envs_per_env_runner=1,
            rollout_fragment_length=config_ppo["rollout_fragment_length"],
            sample_timeout_s=config_ppo["sample_timeout_s"],
            num_cpus_per_env_runner=PER_TRIAL_CPUS_PER_ENV_RUNNER,
        )
        .resources(
            num_cpus_for_main_process=CPUS_FOR_DRIVER,
            placement_strategy="PACK",  # single node; PACK is fine
        )
        .callbacks(PredatorScore)
    )

    max_iters = config_ppo["max_iters"]
    checkpoint_every = 10

    # --------------------------------------------------------------------------------
    # Search space: ONLY PPO hypers (no env tuning)
    # --------------------------------------------------------------------------------
    search_space = ppo_config.copy(copy_frozen=False)
    search_space.training(
        # moderate ranges; you can widen once stable
        num_epochs=tune.qrandint(20, 40, q=2),
        lr=tune.loguniform(1e-4, 1e-3),  # around 3e-4
        # If you want to explore batch size too, uncomment:
        # train_batch_size_per_learner=tune.qrandint(1024, 3072, q=256),
    )

    # --------------------------------------------------------------------------------
    # Stoppers & Scheduler
    # --------------------------------------------------------------------------------
    # Plateau: if metric wiggles less than std over the last num_results iters (after grace)
    plateau = TrialPlateauStopper(
        metric="score_pred",
        mode="max",
        std=0.02,
        num_results=12,
        grace_period=24,
    )
    drop = DropStopper(window=5, drop_threshold=0.2)
    cap = MaximumIterationStopper(max_iters)

    stopper = CombinedStopper(plateau, drop, cap)

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=max_iters,
        grace_period=20,
        reduction_factor=4,
    )

    searcher = OptunaSearch()

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
            num_samples=4,                 
            max_concurrent_trials=MAX_CONCURRENT_TRIALS,  # =4
        ),
    )

    result = tuner.fit()
    ray.shutdown()
