"""
Curriculum trainer for mammoths_defect with shared MultiDiscrete action space.
Uses Tune + callbacks to switch env settings from Phase 1 to Phase 2.
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
from ray.tune import Tuner, RunConfig, CheckpointConfig
from ray.tune.stopper import Stopper

from predpreygrass.rllib.mammoths_defect.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.mammoths_defect.config.config_env_mammoths_defect import config_env
from predpreygrass.rllib.mammoths_defect.utils.episode_return_callback import EpisodeReturn
from predpreygrass.rllib.mammoths_defect.utils.networks import build_multi_module_spec


# Curriculum knobs
PHASE1_MAX_ITERS = 500
FINETUNE_ITERS = 1000
DEF_JOIN_COST = 0.1
DEF_SCAVENGER_FRAC = 0.05
DEF_FAILED_PENALTY = 0.0
# Metric-based stop for Phase 1
PHASE1_METRIC_KEY = "env_runners/agent_episode_returns_mean/type_1_predator_0"
PHASE1_METRIC_FALLBACKS = (
    "env_runners/custom_metrics/predator_episode_return_mean",
    PHASE1_METRIC_KEY,
    "env_runners/module_episode_returns_mean/type_1_predator",
    "env_runners/episode_return_mean",
    "episode_reward_mean",
)
PHASE1_METRIC_TARGET = 10.0
PHASE1_METRIC_WINDOW = 20

# Base and phase environment configs (used by callback on workers).
BASE_ENV_CFG = {**config_env, "seed": None, "strict_rllib_output": True}
PHASE1_ENV = {
    **BASE_ENV_CFG,
    "defection_enabled": True,
    "force_all_join": False,
    "team_capture_join_cost": 0.0,
    "team_capture_scavenger_fraction": 0.0,
    "failed_attack_reward_penalty": 0.0,
}
PHASE2_ENV = {
    **BASE_ENV_CFG,
    "defection_enabled": True,
    "force_all_join": False,
    "team_capture_join_cost": DEF_JOIN_COST,
    "team_capture_scavenger_fraction": DEF_SCAVENGER_FRAC,
    "failed_attack_reward_penalty": DEF_FAILED_PENALTY,
}


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


def build_ppo_config_obj(env_config, config_ppo, multi_module_spec, callbacks_class=EpisodeReturn):
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
        .callbacks(callbacks_class)
    )


class CurriculumCallbacks(EpisodeReturn):
    def __init__(self):
        super().__init__()
        self.phase = "phase1"
        self.metric_window = deque(maxlen=PHASE1_METRIC_WINDOW)
        self.phase2_start_iter = None
        self.phase2_end_iter = None

    @staticmethod
    def _apply_env_settings(algorithm, settings):
        if hasattr(algorithm, "config") and hasattr(algorithm.config, "env_config"):
            try:
                algorithm.config.env_config.update(settings)
            except Exception:
                pass

        def _apply(env):
            if "defection_enabled" in settings:
                env.defection_enabled = bool(settings["defection_enabled"])
            if "force_all_join" in settings:
                env.force_all_join = bool(settings["force_all_join"])
            if "team_capture_join_cost" in settings:
                env.team_capture_join_cost = max(0.0, float(settings["team_capture_join_cost"]))
            if "team_capture_scavenger_fraction" in settings:
                frac = float(settings["team_capture_scavenger_fraction"])
                env.team_capture_scavenger_fraction = max(0.0, min(1.0, frac))
            if "failed_attack_reward_penalty" in settings:
                env.failed_attack_reward_penalty = float(settings["failed_attack_reward_penalty"])
            if isinstance(getattr(env, "config", None), dict):
                env.config.update(settings)

        env_runner_group = getattr(algorithm, "env_runner_group", None)
        if env_runner_group is None:
            return

        if hasattr(env_runner_group, "foreach_worker"):
            env_runner_group.foreach_worker(lambda worker: worker.foreach_env(_apply))
            return
        if hasattr(env_runner_group, "foreach_env"):
            env_runner_group.foreach_env(_apply)
            return
        if hasattr(env_runner_group, "foreach_runner"):
            def _apply_runner(runner):
                if hasattr(runner, "foreach_env"):
                    runner.foreach_env(_apply)
                    return
                envs = getattr(runner, "envs", None) or getattr(runner, "_envs", None)
                if isinstance(envs, (list, tuple)):
                    for env in envs:
                        _apply(env)
                    return
                env = getattr(runner, "env", None) or getattr(runner, "_env", None)
                if env is not None:
                    _apply(env)

            env_runner_group.foreach_runner(_apply_runner)
            return

        envs = getattr(env_runner_group, "envs", None) or getattr(env_runner_group, "_envs", None)
        if isinstance(envs, (list, tuple)):
            for env in envs:
                _apply(env)

    def _switch_to_phase2(self, algorithm, iter_num, reason):
        if self.phase != "phase1":
            return
        self.phase = "phase2"
        self.phase2_start_iter = iter_num + 1
        self.phase2_end_iter = self.phase2_start_iter + FINETUNE_ITERS - 1
        self._apply_env_settings(algorithm, PHASE2_ENV)
        print(
            f"[Curriculum] Switching to Phase 2 at iter {iter_num} ({reason}); "
            f"phase2_end_iter={self.phase2_end_iter}",
            flush=True,
        )

    def on_train_result(self, *, algorithm, result, **kwargs):
        super().on_train_result(result=result, **kwargs)

        iter_num = result.get("training_iteration", 1)
        val, used_path = _first_finite_metric(result, *PHASE1_METRIC_FALLBACKS)
        if val is None:
            val = _mean_predator_return(result)
            used_path = "computed_mean_predator_return" if val is not None else None

        if val is not None:
            result["curriculum_metric_last"] = val
        if used_path:
            result["curriculum_metric_source"] = used_path

        phase_label = "Phase 1" if self.phase == "phase1" else "Phase 2"
        if val is None:
            print(
                f"[{phase_label}] iter={iter_num} {PHASE1_METRIC_KEY} missing; no usable fallback metric",
                flush=True,
            )
        else:
            suffix = f"(via {used_path})" if used_path and used_path != PHASE1_METRIC_KEY else ""
            print(
                f"[{phase_label}] iter={iter_num} {PHASE1_METRIC_KEY} last={val:.3f} {suffix}".rstrip(),
                flush=True,
            )

        if self.phase == "phase1":
            if val is not None:
                self.metric_window.append(val)
                if len(self.metric_window) == PHASE1_METRIC_WINDOW:
                    avg = sum(self.metric_window) / len(self.metric_window)
                    if math.isfinite(avg):
                        result["curriculum_phase1_window_avg"] = avg
                        if avg >= PHASE1_METRIC_TARGET:
                            self._switch_to_phase2(algorithm, iter_num, "metric_target")
                    else:
                        self.metric_window.clear()

            if self.phase == "phase1" and iter_num >= PHASE1_MAX_ITERS:
                self._switch_to_phase2(algorithm, iter_num, "max_iters")

        result["curriculum_phase"] = self.phase
        if self.phase2_start_iter is not None:
            result["curriculum_phase2_start_iter"] = self.phase2_start_iter
        if self.phase2_end_iter is not None:
            result["curriculum_phase2_end_iter"] = self.phase2_end_iter


class CurriculumStopper(Stopper):
    def __init__(self):
        self._phase2_end_iters = {}

    def __call__(self, trial_id, result):
        end_iter = self._phase2_end_iters.get(trial_id)
        if end_iter is None:
            end_value = result.get("curriculum_phase2_end_iter")
            if end_value is not None:
                end_iter = int(end_value)
                self._phase2_end_iters[trial_id] = end_iter

        if end_iter is not None:
            return result.get("training_iteration", 0) >= end_iter

        return result.get("training_iteration", 0) >= PHASE1_MAX_ITERS + FINETUNE_ITERS

    def stop_all(self):
        return False


if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)
    register_env("PredPreyGrass", env_creator)

    phase1_env = dict(PHASE1_ENV)
    phase2_env = dict(PHASE2_ENV)

    config_ppo = get_config_ppo()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_version = "MAMMOTHS_DEFECT_CURRICULUM"
    ray_results_dir = Path(__file__).parent / "ray_results"
    ray_results_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = f"{base_version}_{timestamp}"
    experiment_root = ray_results_dir / experiment_name
    experiment_root.mkdir(parents=True, exist_ok=True)

    # Shared model spec (Phase 1 action space drives curriculum)
    obs_by_policy, act_by_policy = build_spaces(phase1_env)
    multi_module_spec = build_multi_module_spec(obs_by_policy, act_by_policy)
    ppo_config = build_ppo_config_obj(phase1_env, config_ppo, multi_module_spec, callbacks_class=CurriculumCallbacks)

    with open(experiment_root / "run_config_phase1.json", "w") as f:
        json.dump({"config_env": phase1_env, "config_ppo": config_ppo}, f, indent=4)

    with open(experiment_root / "run_config_phase2.json", "w") as f:
        json.dump({"config_env": phase2_env, "config_ppo": config_ppo}, f, indent=4)

    stopper = CurriculumStopper()
    checkpoint_every = 10
    tuner = Tuner(
        ppo_config.algo_class,
        param_space=ppo_config,
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_dir),
            stop=stopper,
            checkpoint_config=CheckpointConfig(
                num_to_keep=100,
                checkpoint_frequency=checkpoint_every,
                checkpoint_at_end=True,
            ),
        ),
    )
    tuner.fit()

    ray.shutdown()
