"""
Curriculum trainer for mammoths_defect with shared MultiDiscrete action space:
- Phase 1: defection_enabled=True, join flag active, zero join cost and zero scavenger fraction.
- Phase 2: defection_enabled=True, join flag active with chosen costs.
- Phase 1 stops when a moving-average metric hits a target or max iters; Phase 2 runs for a fixed number of iters.
"""

import os
from datetime import datetime
from pathlib import Path
import json
from collections import deque
import math
import numbers

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger

from predpreygrass.rllib.mammoths_defect.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.mammoths_defect.config.config_env_mammoths_defect import config_env
from predpreygrass.rllib.mammoths_defect.utils.episode_return_callback import EpisodeReturn
from predpreygrass.rllib.mammoths_defect.utils.networks import build_multi_module_spec


# Curriculum knobs
PHASE1_MAX_ITERS = 500
FINETUNE_ITERS = 1000
# Phase 2 settings
DEF_JOIN_COST = 0.0
DEF_SCAVENGER_FRAC = 0.0
DEF_FAILED_PENALTY = 0.0 # 0.0
PHASE2_INITIAL_TYPE_1_PREDATORS = 30
PHASE2_ENERGY_LOSS_PER_STEP_PREDATOR = 0.03

PHASE1_METRIC_KEY = "env_runners/agent_episode_returns_mean/type_1_predator_0"
PHASE1_METRIC_FALLBACKS = (
    "env_runners/custom_metrics/predator_episode_return_mean",
    PHASE1_METRIC_KEY,
    "env_runners/module_episode_returns_mean/type_1_predator",
    "env_runners/episode_return_mean",
    "episode_reward_mean",
)
PHASE1_METRIC_TARGET = 20.0
PHASE1_METRIC_WINDOW = 10
# Optional: resume Phase 2 from an existing checkpoint (skips Phase 1 if found).
PHASE2_RESTORE_CHECKPOINT = ""


def _get_metric(result: dict, path: str):
    """Fetch a metric from RLlib results, handling nested `env_runners/...` keys."""
    if path in result:
        return result[path]
    current = result
    for part in path.split("/"):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _is_finite_number(val) -> bool:
    if not isinstance(val, numbers.Real):
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def _first_finite_metric(result: dict, *paths: str):
    for path in paths:
        val = _get_metric(result, path)
        if _is_finite_number(val):
            return float(val), path
    return None, None


def _mean_predator_return(result: dict):
    custom_pred = _get_metric(result, "env_runners/custom_metrics/predator_episode_return_mean")
    if _is_finite_number(custom_pred):
        return float(custom_pred)
    agent_means = _get_metric(result, "env_runners/agent_episode_returns_mean")
    if isinstance(agent_means, dict):
        vals = [float(v) for k, v in agent_means.items() if "predator" in k and _is_finite_number(v)]
        if vals:
            return sum(vals) / len(vals)
    module_means = _get_metric(result, "env_runners/module_episode_returns_mean")
    if isinstance(module_means, dict):
        val = module_means.get("type_1_predator")
        if _is_finite_number(val):
            return float(val)
    return None


torch, _ = try_import_torch()


class PPOTorchLearnerNoForeach(PPOTorchLearner):
    def configure_optimizers_for_module(self, module_id, config=None) -> None:
        if config is None:
            config = self.config
        module = self._module[module_id]
        params = self.get_parameters(module)
        optimizer = torch.optim.Adam(params, foreach=False)
        self.register_optimizer(
            module_id=module_id,
            optimizer=optimizer,
            params=params,
            lr_or_lr_schedule=config.lr,
        )


def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.mammoths_defect.config.config_ppo_gpu_mammoths_defect import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.mammoths_defect.config.config_ppo_cpu_mammoths_defect import config_ppo
    else:
        from predpreygrass.rllib.mammoths_defect.config.config_ppo_cpu_mammoths_defect import config_ppo
    return config_ppo


def env_creator(config):
    return PredPreyGrass(config)


def policy_mapping_fn(agent_id, *_, **__):
    parts = agent_id.split("_")
    return f"type_{parts[1]}_{parts[2]}"


def build_spaces(env_config):
    sample_env = env_creator(env_config)
    sample_env.reset(seed=None)
    obs_by_policy, act_by_policy = {}, {}
    for agent_id, obs_space in sample_env.observation_spaces.items():
        pid = policy_mapping_fn(agent_id)
        if pid not in obs_by_policy:
            obs_by_policy[pid] = obs_space
            act_by_policy[pid] = sample_env.action_spaces[agent_id]
    sample_env.action_space_struct = sample_env.action_spaces
    return obs_by_policy, act_by_policy


def build_ppo_config_obj(env_config, config_ppo, multi_module_spec):
    return (
        PPOConfig()
        .environment(env="PredPreyGrass", env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies={pid: (None, None, None, {}) for pid in multi_module_spec.rl_module_specs.keys()},
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
            learner_class=PPOTorchLearnerNoForeach,
        )
        .env_runners(
            num_env_runners=config_ppo["num_env_runners"],
            num_envs_per_env_runner=config_ppo["num_envs_per_env_runner"],
            rollout_fragment_length=config_ppo["rollout_fragment_length"],
            sample_timeout_s=config_ppo["sample_timeout_s"],
            num_cpus_per_env_runner=config_ppo["num_cpus_per_env_runner"],
        )
        .resources(num_cpus_for_main_process=config_ppo["num_cpus_for_main_process"])
        .callbacks(EpisodeReturn)
    )


def make_logger(logdir: Path):
    """Return a logger_creator for RLlib that writes TB/JSON/CSV to logdir."""

    def logger_creator(config):
        return UnifiedLogger(config, str(logdir), loggers=None)

    return logger_creator


if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)
    register_env("PredPreyGrass", env_creator)

    base_env_cfg = {**config_env, "seed": None, "strict_rllib_output": True}
    # Phase 1: defection on, zero-cost cooperation, no forced join
    phase1_env = {
        **base_env_cfg,
        "defection_enabled": True,
        "force_all_join": False,
        "team_capture_join_cost": 0.0,
        "team_capture_scavenger_fraction": 0.0,
        "energy_percentage_loss_per_failed_attacked_prey": 0.0,
    }
    # Phase 2: defection on, join flag active
    phase2_env = {
        **base_env_cfg,
        "defection_enabled": True,
        "force_all_join": False,
        "team_capture_join_cost": DEF_JOIN_COST,
        "team_capture_scavenger_fraction": DEF_SCAVENGER_FRAC,
        "energy_percentage_loss_per_failed_attacked_prey": DEF_FAILED_PENALTY,
        "n_initial_active_type_1_predator": PHASE2_INITIAL_TYPE_1_PREDATORS,
        "energy_loss_per_step_predator": PHASE2_ENERGY_LOSS_PER_STEP_PREDATOR,
    }

    config_ppo = get_config_ppo()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_version = "MAMMOTHS_DEFECT_CURRICULUM_FAIL_0_01_ENERGY_LOSS_PRED_0_03"
    ray_results_dir = Path(__file__).parent / "ray_results"
    ray_results_dir.mkdir(parents=True, exist_ok=True)
    experiment_root = ray_results_dir / f"{base_version}_{timestamp}"
    experiment_root.mkdir(parents=True, exist_ok=True)

    with open(experiment_root / "run_config_phase1.json", "w") as f:
        json.dump({"config_env": phase1_env, "config_ppo": config_ppo}, f, indent=4)

    best_ckpt = None
    restore_path = Path(PHASE2_RESTORE_CHECKPOINT) if PHASE2_RESTORE_CHECKPOINT else None
    if restore_path and restore_path.exists():
        best_ckpt = str(restore_path)
        print(f"[Phase 1] Skipping Phase 1; using checkpoint {best_ckpt}", flush=True)
    else:
        if restore_path:
            print(f"[Phase 1] Checkpoint not found at {restore_path}; running Phase 1.", flush=True)
        # Phase 1 setup
        phase1_name = f"{base_version}_PHASE1_DEF_ON_FREE_{timestamp}"
        phase1_dir = ray_results_dir / phase1_name
        phase1_dir.mkdir(parents=True, exist_ok=True)
        obs_by_policy, act_by_policy = build_spaces(phase1_env)
        multi_module_spec = build_multi_module_spec(obs_by_policy, act_by_policy)
        phase1_cfg = build_ppo_config_obj(phase1_env, {**config_ppo, "max_iters": PHASE1_MAX_ITERS}, multi_module_spec)
        algo1 = phase1_cfg.build(logger_creator=make_logger(phase1_dir))

        print(
            f"[Phase 1] defection_enabled={phase1_env.get('defection_enabled')} "
            f"target={PHASE1_METRIC_TARGET} window={PHASE1_METRIC_WINDOW} "
            f"max_iters={PHASE1_MAX_ITERS}",
            flush=True,
        )
        metric_window = deque(maxlen=PHASE1_METRIC_WINDOW)
        for i in range(1, PHASE1_MAX_ITERS + 1):
            res = algo1.train()
            val, used_path = _first_finite_metric(res, *PHASE1_METRIC_FALLBACKS)
            if val is None:
                val = _mean_predator_return(res)
                used_path = "computed_mean_predator_return" if val is not None else None
            iter_num = res.get("training_iteration", i)
            if val is None:
                print(f"[Phase1] iter={iter_num} {PHASE1_METRIC_KEY} missing; no usable fallback metric", flush=True)
            else:
                metric_window.append(val)
                suffix = f"(via {used_path})" if used_path and used_path != PHASE1_METRIC_KEY else ""
                print(f"[Phase1] iter={iter_num} {PHASE1_METRIC_KEY} last={val:.3f} {suffix}".rstrip(), flush=True)
                if len(metric_window) == PHASE1_METRIC_WINDOW:
                    avg = sum(metric_window) / len(metric_window)
                    if not math.isfinite(avg):
                        metric_window.clear()
                        print(
                            f"[Phase1] iter={iter_num} {PHASE1_METRIC_KEY} window_avg non-finite; resetting window",
                            flush=True,
                        )
                    else:
                        print(
                            f"[Phase1] iter={iter_num} {PHASE1_METRIC_KEY} window_avg({PHASE1_METRIC_WINDOW})={avg:.3f}",
                            flush=True,
                        )
                        if avg >= PHASE1_METRIC_TARGET:
                            print(f"[Phase1] Target reached at iter {iter_num}", flush=True)
                            break
            ckpt_path = algo1.save(str(phase1_dir / f"checkpoint_{iter_num:06d}"))
            best_ckpt = ckpt_path
        algo1.stop()

    # Phase 2 setup
    with open(experiment_root / "run_config_phase2.json", "w") as f:
        json.dump({"config_env": phase2_env, "config_ppo": config_ppo}, f, indent=4)

    phase2_name = f"{base_version}_PHASE2_DEF_ON_{timestamp}"
    phase2_dir = ray_results_dir / phase2_name
    phase2_dir.mkdir(parents=True, exist_ok=True)
    obs_by_policy2, act_by_policy2 = build_spaces(phase2_env)
    multi_module_spec2 = build_multi_module_spec(obs_by_policy2, act_by_policy2)
    phase2_cfg = build_ppo_config_obj(phase2_env, {**config_ppo, "max_iters": FINETUNE_ITERS}, multi_module_spec2)
    algo2 = phase2_cfg.build(logger_creator=make_logger(phase2_dir))
    if best_ckpt:
        try:
            algo2.restore(best_ckpt)
            print(f"[Phase 2] Restored from {best_ckpt}", flush=True)
        except Exception as exc:
            print(f"[Phase 2] Restore failed ({exc}); training from scratch.", flush=True)
    print(
        f"[Phase 2] defection_enabled={phase2_env.get('defection_enabled')} "
        f"join_cost={DEF_JOIN_COST} scavenger_frac={DEF_SCAVENGER_FRAC} "
        f"failed_penalty={DEF_FAILED_PENALTY} max_iters={FINETUNE_ITERS}",
        flush=True,
    )
    last_ckpt = None
    for i in range(1, FINETUNE_ITERS + 1):
        res = algo2.train()
        iter_num = res.get("training_iteration", i)
        val, used_path = _first_finite_metric(res, *PHASE1_METRIC_FALLBACKS)
        if val is None:
            val = _mean_predator_return(res)
            used_path = "computed_mean_predator_return" if val is not None else None
        if val is None:
            print(
                f"[Phase 2] iter={iter_num} {PHASE1_METRIC_KEY} missing; no usable fallback metric",
                flush=True,
            )
        else:
            suffix = f"(via {used_path})" if used_path and used_path != PHASE1_METRIC_KEY else ""
            print(f"[Phase 2] iter={iter_num} {PHASE1_METRIC_KEY} last={val:.3f} {suffix}".rstrip(), flush=True)
        last_ckpt = algo2.save(str(phase2_dir / f"checkpoint_{iter_num:06d}"))
    if last_ckpt:
        print(f"[Phase 2] Last checkpoint saved at: {last_ckpt}", flush=True)
    algo2.stop()

    ray.shutdown()
