# File: tune_ppo_multiagentenv_search.py
# Two-phase search:
#   Phase A: CPU-only, 4 concurrent trials (~24 CPUs total), explore lr/num_epochs.
#   Phase B: Single GPU trial, restore best CPU checkpoint if available, exploit.
#
# Predator-only metric + CSV logs for >=100 hits and final rows. No env tuning.

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
from ray.tune.callback import Callback as TuneCallback
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.stopper import (
    CombinedStopper,
    TrialPlateauStopper,
    MaximumIterationStopper,
    Stopper,
)

# -------------------------
# Runtime/top-level knobs
# -------------------------
# Phase A: exploration budget (iterations and samples)
PHASE_A_MAX_ITERS = 30            # early-budget per CPU trial (stopped earlier if plateau/drop)
PHASE_A_NUM_SAMPLES = 16          # total trials to try on CPU
PHASE_A_MAX_CONCURRENT = 4        # run 4 in parallel
# per-trial CPU sizing (→ ~6 CPUs per trial; 4 trials ≈ 24 CPUs)
PHASE_A_NUM_ENV_RUNNERS = 2
PHASE_A_CPUS_PER_ENV_RUNNER = 2
PHASE_A_CPUS_PER_LEARNER = 1
PHASE_A_CPUS_DRIVER = 1

# Phase B: exploit on GPU (single trial)
PHASE_B_MAX_ITERS = 200           # or leave high; plateau/drop will stop when done
PHASE_B_NUM_ENV_RUNNERS = 3       # modest sampler on GPU
PHASE_B_CPUS_PER_ENV_RUNNER = 1
PHASE_B_CPUS_PER_LEARNER = 1

# Safer CUDA allocator (reduces fragmentation)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# -------------------------
# RLlib Callback: metric + hit logging
# -------------------------
class PredatorScore(DefaultCallbacks):
    """
    Adds 'score_pred' = predator_return/100.0.
    When a trial first reaches score_pred >= 1.0 (i.e., >=100 predator reward),
    append a row to <experiment_dir>/predator_100_hits.csv with:
    [trial_name, iteration, score_pred, lr, num_epochs].
    """
    def on_train_result(self, *, algorithm, result, **kwargs):
        m = result.get("env_runners", {}).get("agent_episode_returns_mean", {})
        pred = float(m.get("type_1_predator_0", 0.0))
        score_pred = pred / 100.0
        result["score_pred"] = score_pred

        # Cache latest on algo (used by completion logger as fallback)
        algorithm._last_score_pred = score_pred
        algorithm._last_iter = int(result.get("training_iteration", -1))
        cfg_from_result = result.get("config", {}) or {}
        algorithm._last_lr = float(cfg_from_result.get("lr", algorithm.config.get("lr")))
        algorithm._last_num_epochs = int(cfg_from_result.get("num_epochs", algorithm.config.get("num_epochs")))

        # One-shot CSV append on first >=1.0
        if score_pred >= 1.0 and not getattr(algorithm, "_saved_pred100_once", False):
            try:
                trial_dir = algorithm.logdir
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
                with open(csv_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["trial_name","iteration","score_pred","lr","num_epochs"])
                    if write_header:
                        w.writeheader()
                    w.writerow(row)
                setattr(algorithm, "_saved_pred100_once", True)
            except Exception as e:
                print(f"[PredatorScore] Failed to append predator_100_hits.csv: {e}")


# -------------------------
# Per-trial Drop stopper
# -------------------------
class DropStopper(Stopper):
    """
    Stop a trial if 'score_pred' drops by more than `drop_threshold` across the
    last `window` iterations (compare oldest in window to newest).
    """
    def __init__(self, window=5, drop_threshold=0.2, name="drop"):
        self.window = int(window)
        self.drop_threshold = float(drop_threshold)
        self.name = name
        self._hist = {}  # trial_id -> list[recent scores]

    def __call__(self, trial_id, result):
        s = result.get("score_pred")
        if s is None:
            return False
        h = self._hist.setdefault(trial_id, [])
        h.append(float(s))
        if len(h) > self.window:
            h.pop(0)
        if len(h) == self.window and (h[-1] < h[0] - self.drop_threshold):
            return True
        return False

    def stop_all(self):
        return False


# -------------------------
# Reasoned stopper wrapper
# -------------------------
class ReasonedStopper(Stopper):
    """Wraps multiple stoppers and records which one triggered per trial."""
    def __init__(self, *stoppers):
        self.stoppers = list(stoppers)
        for s in self.stoppers:
            if not hasattr(s, "name"):
                s.name = s.__class__.__name__.replace("Stopper", "").lower()
        self._reason = {}  # trial_id -> str

    def __call__(self, trial_id, result):
        for s in self.stoppers:
            if s(trial_id, result):
                self._reason[trial_id] = getattr(s, "name", s.__class__.__name__)
                result["__stop_reason"] = self._reason[trial_id]
                return True
        return False

    def stop_all(self):
        return any(getattr(s, "stop_all", lambda: False)() for s in self.stoppers)

    def reason_for(self, trial_id):
        return self._reason.get(trial_id)


# -------------------------
# Final line CSV logger
# -------------------------
class FinalMetricsLogger(TuneCallback):
    """
    Appends <experiment_dir>/predator_final.csv on trial completion or error:
    [trial_name, iteration, score_pred, lr, num_epochs, stop_reason]
    """
    def __init__(self, reasoned_stop: ReasonedStopper):
        super().__init__()
        self._rs = reasoned_stop

    def _append(self, trial, row):
        try:
            experiment_dir = os.path.dirname(trial.logdir)
            csv_path = os.path.join(experiment_dir, "predator_final.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["trial_name","iteration","score_pred","lr","num_epochs","stop_reason"]
                )
                if write_header:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            print(f"[FinalMetricsLogger] Failed to append predator_final.csv: {e}")

    def on_trial_complete(self, iteration, trials, trial, result, **info):
        cfg = result.get("config", {}) or {}
        reason = self._rs.reason_for(trial.trial_id) or result.get("__stop_reason") or "completed"
        row = {
            "trial_name": os.path.basename(trial.logdir),
            "iteration": int(result.get("training_iteration", -1)),
            "score_pred": float(result.get("score_pred", float("nan"))),
            "lr": float(cfg.get("lr", float("nan"))),
            "num_epochs": int(cfg.get("num_epochs", -1)),
            "stop_reason": reason,
        }
        self._append(trial, row)

    def on_trial_error(self, iteration, trials, trial, **info):
        row = {
            "trial_name": os.path.basename(trial.logdir),
            "iteration": -1,
            "score_pred": float("nan"),
            "lr": float("nan"),
            "num_epochs": -1,
            "stop_reason": "error",
        }
        self._append(trial, row)


# -------------------------
# Helpers
# -------------------------
def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.v3_1.config.config_ppo_hbp_search import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.v3_1.config.config_ppo_cpu import config_ppo
    else:
        # default to HBP layout if you're usually on 32 CPUs; adjust as needed
        from predpreygrass.rllib.v3_1.config.config_ppo_hbp_search import config_ppo
    return config_ppo


def env_creator(cfg):
    return PredPreyGrass(cfg)


def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    type_ = parts[1]
    role = parts[2]
    return f"type_{type_}_{role}"


def detect_gpu():
    if os.environ.get("RLLIB_FORCE_CPU", "0") == "1":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Fresh Ray + env
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)
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

    # Probe spaces to build MultiRLModuleSpec
    sample_env = env_creator(config_env)
    obs_by_policy, act_by_policy = {}, {}
    for agent_id, obs_space in sample_env.observation_spaces.items():
        pid = policy_mapping_fn(agent_id)
        if pid not in obs_by_policy:
            obs_by_policy[pid] = obs_space
            act_by_policy[pid] = sample_env.action_spaces[agent_id]
    del sample_env

    multi_module_spec = build_multi_module_spec(obs_by_policy, act_by_policy)
    policies = {pid: (None, obs_by_policy[pid], act_by_policy[pid], {}) for pid in obs_by_policy}

    # -------------------------------------------------
    # Phase A: CPU-only, 4 concurrent trials (~24 CPUs)
    # -------------------------------------------------
    print("\n[PHASE A] CPU-only exploration (4 concurrent trials, ~24 CPUs total)\n")

    # Base CPU config (no env tuning; rollout_fragment_length='auto' for safety)
    ppo_a = (
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
            num_gpus_per_learner=0,
            num_cpus_per_learner=PHASE_A_CPUS_PER_LEARNER,
        )
        .env_runners(
            num_env_runners=PHASE_A_NUM_ENV_RUNNERS,
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto",
            sample_timeout_s=config_ppo["sample_timeout_s"],
            num_cpus_per_env_runner=PHASE_A_CPUS_PER_ENV_RUNNER,
        )
        .resources(num_cpus_for_main_process=PHASE_A_CPUS_DRIVER)
        .callbacks(PredatorScore)
    )

    # Phase A search space: only PPO hypers
    space_a = ppo_a.copy(copy_frozen=False).training(
        num_epochs=tune.qrandint(18, 40, q=2),
        lr=tune.loguniform(1e-4, 1e-3),   # centered around ~3e-4
    )

    plateau_a = TrialPlateauStopper(metric="score_pred", mode="max", std=0.02, num_results=10, grace_period=16)
    plateau_a.name = "plateau"
    drop_a = DropStopper(window=5, drop_threshold=0.2, name="drop")
    max_a = MaximumIterationStopper(PHASE_A_MAX_ITERS); max_a.name = "max_iters"
    reasoned_a = ReasonedStopper(plateau_a, drop_a, max_a)

    sched_a = ASHAScheduler(time_attr="training_iteration", max_t=PHASE_A_MAX_ITERS, grace_period=8, reduction_factor=3)
    searcher = OptunaSearch()

    tuner_a = Tuner(
        ppo_a.algo_class,
        param_space=space_a,
        run_config=RunConfig(
            name=experiment_name + "_A_cpu",
            storage_path=str(ray_results_path),
            stop=reasoned_a,
            checkpoint_config=CheckpointConfig(
                num_to_keep=50,
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
            callbacks=[FinalMetricsLogger(reasoned_a)],
        ),
        tune_config=tune.TuneConfig(
            metric="score_pred",
            mode="max",
            scheduler=sched_a,
            search_alg=searcher,
            num_samples=PHASE_A_NUM_SAMPLES,
            max_concurrent_trials=PHASE_A_MAX_CONCURRENT,
        ),
    )

    results_a = tuner_a.fit()

    # Extract best config & checkpoint (if any)
    best_a = results_a.get_best_result(metric="score_pred", mode="max")
    best_cfg = best_a.config or {}
    best_lr = float(best_cfg.get("lr", config_ppo["lr"]))
    best_epochs = int(best_cfg.get("num_epochs", config_ppo["num_epochs"]))
    restore_path = None
    try:
        if best_a.checkpoint:
            restore_path = best_a.checkpoint.path
    except Exception:
        restore_path = None

    print("\n[PHASE A] Best CPU trial:",
          f"\n  score_pred={best_a.metrics.get('score_pred')}",
          f"\n  lr={best_lr}, num_epochs={best_epochs}",
          f"\n  restore_path={'<none>' if not restore_path else restore_path}\n")

    # -------------------------------------------------
    # Phase B: Single-trial GPU exploit (or CPU if no GPU)
    # -------------------------------------------------
    USE_GPU = detect_gpu()
    print(f"[PHASE B] Exploit on {'GPU' if USE_GPU else 'CPU'} (single trial)\n")

    ppo_b = (
        PPOConfig()
        .environment(env="PredPreyGrass", env_config=config_env)
        .framework("torch")
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .training(
            # lock to best from Phase A
            train_batch_size_per_learner=config_ppo["train_batch_size_per_learner"],
            minibatch_size=config_ppo["minibatch_size"],
            num_epochs=best_epochs,
            gamma=config_ppo["gamma"],
            lr=best_lr,
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
            num_gpus_per_learner=1 if USE_GPU else 0,
            num_cpus_per_learner=PHASE_B_CPUS_PER_LEARNER,
        )
        .env_runners(
            num_env_runners=PHASE_B_NUM_ENV_RUNNERS,
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto",  # OOM- & validation-safe
            sample_timeout_s=config_ppo["sample_timeout_s"],
            num_cpus_per_env_runner=PHASE_B_CPUS_PER_ENV_RUNNER,
        )
        .resources(num_cpus_for_main_process=1)
        .callbacks(PredatorScore)
    )

    plateau_b = TrialPlateauStopper(metric="score_pred", mode="max", std=0.02, num_results=12, grace_period=24)
    plateau_b.name = "plateau"
    drop_b = DropStopper(window=6, drop_threshold=0.25, name="drop")
    max_b = MaximumIterationStopper(PHASE_B_MAX_ITERS); max_b.name = "max_iters"
    reasoned_b = ReasonedStopper(plateau_b, drop_b, max_b)

    sched_b = ASHAScheduler(time_attr="training_iteration", max_t=PHASE_B_MAX_ITERS, grace_period=12, reduction_factor=3)

    run_cfg_b = RunConfig(
        name=experiment_name + "_B_exploit",
        storage_path=str(ray_results_path),
        stop=reasoned_b,
        checkpoint_config=CheckpointConfig(
            num_to_keep=20,
            checkpoint_frequency=10,
            checkpoint_at_end=True,
        ),
        callbacks=[FinalMetricsLogger(reasoned_b)],
    )
    # If we have a checkpoint from Phase A, try to restore it
    if restore_path:
        run_cfg_b = run_cfg_b.update(restore_path=restore_path)

    tuner_b = Tuner(
        ppo_b.algo_class,
        param_space=ppo_b,  # fixed (no search) exploit run
        run_config=run_cfg_b,
        tune_config=tune.TuneConfig(
            metric="score_pred",
            mode="max",
            scheduler=sched_b,
            num_samples=1,
            max_concurrent_trials=1,
        ),
    )

    results_b = tuner_b.fit()

    # Print brief summary for Phase B best
    best_b = results_b.get_best_result(metric="score_pred", mode="max")
    print("\n[PHASE B] Final single-trial result:",
          f"\n  score_pred={best_b.metrics.get('score_pred')}",
          f"\n  iter={best_b.metrics.get('training_iteration')}\n")

    ray.shutdown()
