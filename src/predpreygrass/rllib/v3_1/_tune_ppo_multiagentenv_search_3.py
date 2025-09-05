# File: tune_ppo_multiagentenv_search.py
# Two-phase Ray Tune run:
#  - Phase A: CPU-only, 4 concurrent trials (auto-fills ~32 CPUs)
#  - Phase B: GPU single-trial, local search around Phase A's best
# Predator-only metric, robust CSV logging, no env tuning.

from predpreygrass.rllib.v3_1.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v3_1.config.config_env_train_v1_0 import config_env
from predpreygrass.rllib.v3_1.utils.networks import build_multi_module_spec

import os
import csv
from datetime import datetime
from pathlib import Path
import json
import math

import ray
from ray import tune
from ray.tune import Tuner, RunConfig, CheckpointConfig, FailureConfig
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.registry import register_env
from ray.tune.callback import Callback as TuneCallback
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.stopper import (
    CombinedStopper,
    TrialPlateauStopper,
    MaximumIterationStopper,
    Stopper,
)

# -----------------------------------------------------------------------------
# Safety & allocator
# -----------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -----------------------------------------------------------------------------
# RLlib Callback: metric + one-shot ≥100 predator hit logging
# -----------------------------------------------------------------------------
class PredatorScore(DefaultCallbacks):
    """
    Adds 'score_pred' = predator_return/100.0.
    When a trial first reaches score_pred >= 1.0 (i.e., >=100 predator reward),
    append one row to <experiment_dir>/predator_100_hits.csv with:
    [trial_name, iteration, score_pred, lr, num_epochs].
    """
    def on_train_result(self, *, algorithm, result, **kwargs):
        m = result.get("env_runners", {}).get("agent_episode_returns_mean", {})
        pred = float(m.get("type_1_predator_0", 0.0))
        score_pred = pred / 100.0
        result["score_pred"] = score_pred

        # stash latest on algo (for debugging/fallbacks if needed)
        algorithm._last_score_pred = score_pred
        algorithm._last_iter = int(result.get("training_iteration", -1))
        cfg_from_result = result.get("config", {}) or {}
        algorithm._last_lr = float(cfg_from_result.get("lr", algorithm.config.get("lr")))
        algorithm._last_num_epochs = int(cfg_from_result.get("num_epochs", algorithm.config.get("num_epochs")))

        # One-shot CSV append when first crossing >= 1.0
        if score_pred >= 1.0 and not getattr(algorithm, "_saved_pred100_once", False):
            try:
                # This is safe to use for RLlib Algorithm; Tune deprec warning is for Trial.
                trial_dir = getattr(algorithm, "logdir", None)
                if trial_dir is None:
                    return
                experiment_dir = os.path.dirname(trial_dir)
                csv_path = os.path.join(experiment_dir, "predator_100_hits.csv")
                row = {
                    "trial_name": os.path.basename(trial_dir),
                    "iteration": algorithm._last_iter,
                    "score_pred": score_pred,
                    "lr": algorithm._last_lr,
                    "num_epochs": algorithm._last_num_epochs,
                }
                write_header = not os.path.exists(csv_path)
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["trial_name", "iteration", "score_pred", "lr", "num_epochs"],
                    )
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
                setattr(algorithm, "_saved_pred100_once", True)
            except Exception as e:
                print(f"[PredatorScore] Failed to append predator_100_hits.csv: {e}")

# -----------------------------------------------------------------------------
# Per-trial Drop stopper: stops a trial if recent window drops too much
# -----------------------------------------------------------------------------
class DropStopper(Stopper):
    """
    Stop a trial if 'score_pred' drops by more than `drop_threshold` across the
    last `window` iterations (compared start-of-window to latest). Never stops
    the whole experiment.
    """
    def __init__(self, window: int = 5, drop_threshold: float = 0.2, name: str = "drop"):
        self.window = int(window)
        self.drop_threshold = float(drop_threshold)
        self.history = {}  # trial_id -> recent scores
        self.name = name

    def __call__(self, trial_id: str, result: dict) -> bool:
        score = result.get("score_pred")
        if score is None:
            return False

        hist = self.history.setdefault(trial_id, [])
        hist.append(float(score))
        if len(hist) > self.window:
            hist.pop(0)

        if len(hist) == self.window and (hist[-1] < hist[0] - self.drop_threshold):
            return True
        return False

    def stop_all(self) -> bool:
        return False  # never stop the whole study

# -----------------------------------------------------------------------------
# Stopper wrapper that remembers WHICH stopper fired
# -----------------------------------------------------------------------------
class ReasonedStopper(Stopper):
    """
    Wraps multiple stoppers and records which one triggered per trial_id.
    """
    def __init__(self, *stoppers):
        self.stoppers = list(stoppers)
        for s in self.stoppers:
            if not hasattr(s, "name"):
                s.name = s.__class__.__name__.replace("Stopper", "").lower()
        self._reasons = {}  # trial_id -> reason string

    def __call__(self, trial_id: str, result: dict) -> bool:
        for s in self.stoppers:
            if s(trial_id, result):
                reason = getattr(s, "name", s.__class__.__name__)
                self._reasons[trial_id] = reason
                result["__stop_reason"] = reason
                return True
        return False

    def stop_all(self) -> bool:
        return any(getattr(s, "stop_all", lambda: False)() for s in self.stoppers)

    def reason_for(self, trial_id: str):
        return self._reasons.get(trial_id)

# -----------------------------------------------------------------------------
# Tune Callback to write final line for every finished/errored trial
# -----------------------------------------------------------------------------
class FinalMetricsLogger(TuneCallback):
    """
    Writes <experiment_dir>/predator_final.csv when a trial finishes or errors,
    including the stop reason captured by ReasonedStopper.
    """
    def __init__(self, reasoned_stop: ReasonedStopper):
        super().__init__()
        self._reasoned_stop = reasoned_stop

    # NOTE: Tune (v2+) does NOT pass `result` here reliably. Use trial.last_result or info.get("result")
    def on_trial_complete(self, iteration, trials, trial, **info):
        try:
            experiment_dir = os.path.dirname(trial.local_path)
            csv_path = os.path.join(experiment_dir, "predator_final.csv")

            # Try best effort to get the final result dict
            result = info.get("result") or getattr(trial, "last_result", {}) or {}
            reason = self._reasoned_stop.reason_for(trial.trial_id) or result.get("__stop_reason") or "completed"

            score_pred = float(result.get("score_pred", float("nan")))
            final_iter = int(result.get("training_iteration", -1))
            cfg = result.get("config", {}) or {}
            lr = float(cfg.get("lr", float("nan")))
            num_epochs = int(cfg.get("num_epochs", -1))

            row = {
                "trial_name": os.path.basename(trial.local_path),
                "iteration": final_iter,
                "score_pred": score_pred,
                "lr": lr,
                "num_epochs": num_epochs,
                "stop_reason": reason,
            }

            write_header = not os.path.exists(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["trial_name", "iteration", "score_pred", "lr", "num_epochs", "stop_reason"],
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"[FinalMetricsLogger] Failed to append predator_final.csv: {e}")

    def on_trial_error(self, iteration, trials, trial, **info):
        try:
            experiment_dir = os.path.dirname(trial.local_path)
            csv_path = os.path.join(experiment_dir, "predator_final.csv")
            row = {
                "trial_name": os.path.basename(trial.local_path),
                "iteration": -1,
                "score_pred": float("nan"),
                "lr": float("nan"),
                "num_epochs": -1,
                "stop_reason": "error",
            }
            write_header = not os.path.exists(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["trial_name", "iteration", "score_pred", "lr", "num_epochs", "stop_reason"],
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"[FinalMetricsLogger] Failed to append predator_final.csv on error: {e}")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_config_ppo():
    # Follow your existing branching
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.v3_1.config.config_ppo_hbp_search import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.v3_1.config.config_ppo_cpu import config_ppo
    else:
        # Fall back to HBP config if present; else raise
        try:
            from predpreygrass.rllib.v3_1.config.config_ppo_hbp_search import config_ppo
        except Exception as _:
            raise RuntimeError(f"Unsupported cpu_count={num_cpus}. Please add matching config_ppo.")
    return config_ppo

def env_creator(config):
    return PredPreyGrass(config)

def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    type_ = parts[1]
    role = parts[2]
    return f"type_{type_}_{role}"

def build_base_ppo(obs_by_policy, act_by_policy, rl_module_spec, cfg_env, cfg_ppo) -> PPOConfig:
    policies = {pid: (None, obs_by_policy[pid], act_by_policy[pid], {}) for pid in obs_by_policy}
    ppo = (
        PPOConfig()
        .environment(env="PredPreyGrass", env_config=cfg_env)
        .framework("torch")
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .training(
            train_batch_size_per_learner=cfg_ppo["train_batch_size_per_learner"],
            minibatch_size=cfg_ppo["minibatch_size"],
            num_epochs=cfg_ppo["num_epochs"],
            gamma=cfg_ppo["gamma"],
            lr=cfg_ppo["lr"],
            lambda_=cfg_ppo["lambda_"],
            entropy_coeff=cfg_ppo["entropy_coeff"],
            vf_loss_coeff=cfg_ppo["vf_loss_coeff"],
            clip_param=cfg_ppo["clip_param"],
            kl_coeff=cfg_ppo["kl_coeff"],
            kl_target=cfg_ppo["kl_target"],
        )
        .rl_module(rl_module_spec=rl_module_spec)
        .callbacks(PredatorScore)
    )
    return ppo

def compute_phase_a_resources(max_concurrent_trials: int, num_cpus_total: int):
    """
    Spread ~all CPUs across N concurrent trials.
    Each trial uses:
      - 1 CPU driver (num_cpus_for_main_process=1)
      - 1 CPU learner (num_cpus_per_learner=1)
      - X env runners each with 1 CPU (num_cpus_per_env_runner=1)
    Solve X so that N * (2 + X) ≈ num_cpus_total
    """
    overhead_per_trial = 2  # driver + learner
    # conservative: leave a tiny margin of 0-? CPUs
    per_trial_budget = max(2, (num_cpus_total // max_concurrent_trials) - overhead_per_trial)
    num_env_runners = max(2, per_trial_budget)
    return {
        "num_env_runners": num_env_runners,
        "num_envs_per_env_runner": 1,
        "num_cpus_per_env_runner": 3,
        "num_cpus_per_learner": 1,
        "num_cpus_for_main_process": 1,
        "num_gpus_per_learner": 0,
        "max_concurrent_trials": max_concurrent_trials,
    }

def compute_phase_b_resources():
    # GPU single-trial; keep sampler modest
    return {
        "num_env_runners": 3,
        "num_envs_per_env_runner": 1,
        "num_cpus_per_env_runner": 1,
        "num_cpus_per_learner": 1,
        "num_cpus_for_main_process": 1,
        "num_gpus_per_learner": 1,
        "max_concurrent_trials": 1,
    }

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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
    experiment_name_a = f"PPO_{timestamp}_A_cpu"
    experiment_name_b = f"PPO_{timestamp}_B_gpu"
    (ray_results_path / experiment_name_a).mkdir(parents=True, exist_ok=True)
    (ray_results_path / experiment_name_b).mkdir(parents=True, exist_ok=True)

    config_ppo = get_config_ppo()

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

    # --------------------------------------------------------------------------------
    # PHASE A: CPU-only, saturate ~32 CPUs with 4 concurrent trials (auto resources)
    # --------------------------------------------------------------------------------
    total_cpus = os.cpu_count() or 32
    phase_a_res = compute_phase_a_resources(max_concurrent_trials=4, num_cpus_total=min(32, total_cpus))
    print(
        f"\n[PHASE A] CPU-only parallel search\n"
        f"  total_cpus_used≈{phase_a_res['max_concurrent_trials']} * "
        f"(driver 1 + learner 1 + env_runners {phase_a_res['num_env_runners']}*1) "
        f"= ~{phase_a_res['max_concurrent_trials']*(2+phase_a_res['num_env_runners'])} CPUs\n"
    )

    ppo_a = build_base_ppo(obs_by_policy, act_by_policy, multi_module_spec, config_env, config_ppo)
    ppo_a = (
        ppo_a
        .learners(
            num_learners=1,
            num_gpus_per_learner=0,
            num_cpus_per_learner=phase_a_res["num_cpus_per_learner"],
        )
        .env_runners(
            num_env_runners=phase_a_res["num_env_runners"],
            num_envs_per_env_runner=phase_a_res["num_envs_per_env_runner"],
            rollout_fragment_length="auto",          # avoids batch/fragment mismatch
            sample_timeout_s=config_ppo["sample_timeout_s"],
            num_cpus_per_env_runner=phase_a_res["num_cpus_per_env_runner"],
        )
        .resources(
            num_cpus_for_main_process=phase_a_res["num_cpus_for_main_process"],
        )
    )

    # Search space (ONLY PPO hypers)
    search_space_a = ppo_a.copy(copy_frozen=False).training(
        num_epochs=tune.qrandint(20, 40, q=2),
        lr=tune.loguniform(1e-4, 1e-3),  # centered near 3e-4
    )

    # Stoppers & scheduler
    plateau_a = TrialPlateauStopper(
        metric="score_pred", mode="max",
        std=0.02, num_results=12, grace_period=24
    )
    plateau_a.name = "plateau"
    drop_a = DropStopper(window=5, drop_threshold=0.2, name="drop")
    maxcap_a = MaximumIterationStopper(config_ppo["max_iters"])
    maxcap_a.name = "max_iters"
    stop_a = ReasonedStopper(plateau_a, drop_a, maxcap_a)

    scheduler_a = ASHAScheduler(
        time_attr="training_iteration",
        max_t=config_ppo["max_iters"],
        grace_period=20,
        reduction_factor=4,
    )
    searcher_a = OptunaSearch()

    # Loggers + configs
    run_config_a = RunConfig(
        name=experiment_name_a,
        storage_path=str(ray_results_path),
        stop=stop_a,
        checkpoint_config=CheckpointConfig(
            num_to_keep=50,
            checkpoint_frequency=10,
            checkpoint_at_end=True,
        ),
        callbacks=[FinalMetricsLogger(stop_a)],
        failure_config=FailureConfig(fail_fast=False),
    )

    tuner_a = Tuner(
        ppo_a.algo_class,
        param_space=search_space_a,
        run_config=run_config_a,
        tune_config=tune.TuneConfig(
            metric="score_pred",
            mode="max",
            scheduler=scheduler_a,
            search_alg=searcher_a,
            num_samples=24,                       # decent breadth
            max_concurrent_trials=phase_a_res["max_concurrent_trials"],
        ),
    )

    results_a = tuner_a.fit()  # ---- run Phase A

    # Best result from A (if none finished, we still proceed cautiously)
    try:
        best_a = results_a.get_best_result(metric="score_pred", mode="max")
        base_lr = float(best_a.config.get("lr", config_ppo["lr"]))
        base_epochs = int(best_a.config.get("num_epochs", config_ppo["num_epochs"]))
        print(f"\n[PHASE A] Best config → lr={base_lr:.6g}, num_epochs={base_epochs}\n")
    except Exception as e:
        print(f"[PHASE A] Could not obtain best result ({e}). Falling back to base config.")
        best_a = None
        base_lr = float(config_ppo["lr"])
        base_epochs = int(config_ppo["num_epochs"])

    # --------------------------------------------------------------------------------
    # PHASE B: GPU single-trial local search around Phase A's best (if GPU available)
    # --------------------------------------------------------------------------------
    # Detect GPU
    try:
        import torch
        have_gpu = torch.cuda.is_available()
    except Exception:
        have_gpu = False

    if not have_gpu:
        print("[PHASE B] No GPU detected; skipping Phase B.")
        ray.shutdown()
        raise SystemExit(0)

    res_b = compute_phase_b_resources()
    print(
        f"\n[PHASE B] GPU single-trial local search\n"
        f"  num_env_runners={res_b['num_env_runners']}, "
        f"num_envs_per_env_runner={res_b['num_envs_per_env_runner']}, "
        f"rollout_fragment_length=auto, "
        f"num_gpus_per_learner={res_b['num_gpus_per_learner']}\n"
    )

    ppo_b = build_base_ppo(obs_by_policy, act_by_policy, multi_module_spec, config_env, config_ppo)
    ppo_b = (
        ppo_b
        .learners(
            num_learners=1,
            num_gpus_per_learner=res_b["num_gpus_per_learner"],
            num_cpus_per_learner=res_b["num_cpus_per_learner"],
        )
        .env_runners(
            num_env_runners=res_b["num_env_runners"],
            num_envs_per_env_runner=res_b["num_envs_per_env_runner"],
            rollout_fragment_length="auto",
            sample_timeout_s=config_ppo["sample_timeout_s"],
            num_cpus_per_env_runner=res_b["num_cpus_per_env_runner"],
        )
        .resources(
            num_cpus_for_main_process=res_b["num_cpus_for_main_process"],
        )
    )

    # Local search around best A:
    #  - lr in [base_lr/2, base_lr*2], clipped to [1e-5, 5e-3]
    #  - num_epochs in [base_epochs-6, base_epochs+6], clipped to [12, 48]
    lr_low = max(1e-5, base_lr / 2.0)
    lr_high = min(5e-3, base_lr * 2.0)
    ep_low = max(12, base_epochs - 6)
    ep_high = min(48, base_epochs + 6)
    if ep_low > ep_high:
        ep_low, ep_high = ep_high, ep_low

    search_space_b = ppo_b.copy(copy_frozen=False).training(
        lr=tune.loguniform(lr_low, lr_high),
        num_epochs=tune.qrandint(ep_low, ep_high, q=2),
    )

    plateau_b = TrialPlateauStopper(
        metric="score_pred", mode="max",
        std=0.02, num_results=12, grace_period=24
    )
    plateau_b.name = "plateau"
    drop_b = DropStopper(window=5, drop_threshold=0.2, name="drop")
    maxcap_b = MaximumIterationStopper(config_ppo["max_iters"])
    maxcap_b.name = "max_iters"
    stop_b = ReasonedStopper(plateau_b, drop_b, maxcap_b)

    scheduler_b = ASHAScheduler(
        time_attr="training_iteration",
        max_t=config_ppo["max_iters"],
        grace_period=20,
        reduction_factor=4,
    )
    searcher_b = OptunaSearch()

    run_config_b = RunConfig(
        name=experiment_name_b,
        storage_path=str(ray_results_path),
        stop=stop_b,
        checkpoint_config=CheckpointConfig(
            num_to_keep=50,
            checkpoint_frequency=10,
            checkpoint_at_end=True,
        ),
        callbacks=[FinalMetricsLogger(stop_b)],
        failure_config=FailureConfig(fail_fast=False),
    )

    tuner_b = Tuner(
        ppo_b.algo_class,
        param_space=search_space_b,
        run_config=run_config_b,
        tune_config=tune.TuneConfig(
            metric="score_pred",
            mode="max",
            scheduler=scheduler_b,
            search_alg=searcher_b,
            num_samples=8,                   # small local search on GPU
            max_concurrent_trials=res_b["max_concurrent_trials"],  # 1
        ),
    )

    results_b = tuner_b.fit()  # ---- run Phase B
    ray.shutdown()
