# File: tune_ppo_multiagentenv_search.py
# Full replacement script (concurrency-focused, no env tuning, predator-only metric)

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
from ray.tune.stopper import CombinedStopper, TrialPlateauStopper, MaximumIterationStopper
from ray.tune.stopper import Stopper

# Keep the CUDA allocator resilient (useful even if GPU is OFF here).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class PredatorScore(DefaultCallbacks):
    """
    Adds 'score_pred' = predator_return/100.0.
    When a trial first reaches score_pred >= 1.0 (i.e., >=100 predator reward),
    append one row to <experiment_dir>/predator_100_hits.csv with:
    [trial_name, iteration, score_pred, lr, num_epochs].
    """
    def on_train_result(self, *, algorithm, result, **kwargs):
        # Pull mean episode returns per agent from RLlib's "env_runners" block.
        m = result.get("env_runners", {}).get("agent_episode_returns_mean", {})
        pred = float(m.get("type_1_predator_0", 0.0))
        score_pred = pred / 100.0
        result["score_pred"] = score_pred  # so Tune can use it as the metric

        # Only record ONCE per trial when it first crosses the threshold.
        if score_pred >= 1.0 and not getattr(algorithm, "_saved_pred100_once", False):
            try:
                # trial logdir: .../<experiment_name>/<trial_name>
                trial_dir = algorithm.logdir
                experiment_dir = os.path.dirname(trial_dir)
                csv_path = os.path.join(experiment_dir, "predator_100_hits.csv")

                cfg_from_result = result.get("config", {})
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

                # Mark so we don't write multiple times for this same trial.
                setattr(algorithm, "_saved_pred100_once", True)
            except Exception as e:
                # Don't crash training if logging fails; you’ll still see the exception in logs.
                print(f"[PredatorScore] Failed to append predator_100_hits.csv: {e}")


class DropStopper(Stopper):
    """
    Stop a trial if the metric drops by more than `drop_threshold`
    over the last `window` iterations. Never stops the whole experiment.
    """
    def __init__(self, window: int = 5, drop_threshold: float = 0.1):
        self.window = int(window)
        self.drop_threshold = float(drop_threshold)
        self.history = {}  # trial_id -> list of recent scores

    def __call__(self, trial_id: str, result: dict) -> bool:
        score = result.get("score_pred")
        if score is None:
            return False

        hist = self.history.setdefault(trial_id, [])
        hist.append(float(score))
        # Keep only the recent `window` points
        if len(hist) > self.window:
            hist.pop(0)

        # If we have a full window, stop on significant drop vs. window start
        if len(hist) == self.window and (hist[-1] < hist[0] - self.drop_threshold):
            return True

        return False

    def stop_all(self) -> bool:
        # Never stop the entire experiment; only individual trials.
        return False


# -----------------------------------------------------------------------------
# Safety knobs
# -----------------------------------------------------------------------------


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

    search_space = ppo_config.copy(copy_frozen=False)

    # PPO hyperparameters (only)
    search_space.training(
        num_epochs=tune.qrandint(20, 40, q=2),
        lr=tune.loguniform(1e-4, 1e-3),  # centered around 0.0003 (3e-4)
        #train_batch_size_per_learner=tune.qrandint(1024, 3072, q=256),
    )

    # ------------------------------
    # Stoppers & Scheduler  (no hard 100+ cutoff)
    # ------------------------------
    """
        std:    
            Standard deviation threshold. Plateau tolerance (smaller = more sensitive)
            If the metrics values over the last num_results iterations vary less than this, it’s considered a plateau.
            Example: if score_pred is bouncing between 0.72 and 0.75 for 12 iterations and std < 0.02,
            the stopper says “this trial is stuck.
        num_results:       
            How many most recent results to consider when checking for plateau.
            In this case: the last 12 reported values.
        grace_period:       
            Minimum number of iterations before the stopper starts checking for plateau.
            Here: don’t even consider stopping until the trial has run at least 24 iterations.
    """
    stopper = CombinedStopper(
        TrialPlateauStopper(
            metric="score_pred",
            mode="max",
            std=0.02,           
            num_results=12,
            grace_period=24     
            # NOTE: no metric_threshold here -> no "stop at >= 1.0" behavior
        ),
        DropStopper(window=5, drop_threshold=0.2),
        MaximumIterationStopper(max_iters)  # optional safety cap
    )

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
            stop=stopper,  # <-- only plateau-based early stopping
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
            num_samples=20,
            max_concurrent_trials=4,
        ),
    )

    result = tuner.fit()
    ray.shutdown()
