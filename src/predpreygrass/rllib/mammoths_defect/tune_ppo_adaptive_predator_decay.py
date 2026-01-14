"""
Tune-based PPO training with adaptive predator energy decay.
Increases predator energy_loss_per_step_predator when episode_len_mean EMA plateaus or declines.
"""
from predpreygrass.rllib.mammoths_defect.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.mammoths_defect.config.config_env_mammoths_defect import config_env
from predpreygrass.rllib.mammoths_defect.utils.episode_return_callback import EpisodeReturn
from predpreygrass.rllib.mammoths_defect.utils.networks import build_multi_module_spec

import ray
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.module_to_env import (
    GetActions,
    ListifyDataForVectorEnv,
    ModuleToAgentUnmapping,
    NormalizeAndClipActions,
    RemoveSingleTsTimeRankFromBatch,
    TensorToNumpy,
    UnBatchToIndividualItems,
)
from ray.rllib.core.columns import Columns
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env
from ray.tune import Tuner, RunConfig, CheckpointConfig

import os
from datetime import datetime
from pathlib import Path
import json
import shutil
import math
import numbers
import numpy as np

LEN_METRIC_KEY = "env_runners/episode_len_mean"
EMA_WEIGHT = 0.99
LEN_IMPROVEMENT_MIN = 1.0 # from previous best EMA, to count as improvement, not plateau
LEN_DECLINE_DELTA = 10.0  # from peak EMA
LEN_PLATEAU_PATIENCE = 10
PRED_DECAY_STEP = 0.005
PRED_DECAY_MAX = 0.2
BASE_PREDATOR_DECAY = float(config_env.get("energy_loss_per_step_predator", 0.0))
FORCE_JOIN_LOGIT_LOW = -1e9
FORCE_JOIN_LOGIT_HIGH = 1e9

torch, _ = try_import_torch()


def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.mammoths_defect.config.config_ppo_gpu_mammoths_defect import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.mammoths_defect.config.config_ppo_cpu_mammoths_defect import config_ppo
    else:
        # Default to CPU config for other CPU counts to keep training usable across machines.
        from predpreygrass.rllib.mammoths_defect.config.config_ppo_cpu_mammoths_defect import config_ppo
    return config_ppo


def env_creator(config):
    return PredPreyGrass(config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    Maps agent IDs to policies based on their type and role.
    This function is used to determine which policy to apply for each agent.
    Args:
        agent_id (str): The ID of the agent, expected to be in the format "type_X_role_Y".
    Returns:
        str: The policy name for the agent, formatted as "type_X_role_Y".
    """
    parts = agent_id.split("_")
    type = parts[1]
    role = parts[2]
    return f"type_{type}_{role}"


class ForceJoinHuntLogits(ConnectorV2):
    def __init__(self, force_all_join: bool, force_join_prob: float):
        super().__init__()
        self.force_all_join = bool(force_all_join)
        self.force_join_prob = float(force_join_prob)
        if self.force_join_prob < 0.0:
            self.force_join_prob = 0.0
        elif self.force_join_prob > 1.0:
            self.force_join_prob = 1.0
        self._rng = np.random.default_rng()

    def _should_force(self) -> bool:
        if self.force_all_join:
            return True
        if self.force_join_prob >= 1.0:
            return True
        if self.force_join_prob <= 0.0:
            return False
        return self._rng.random() < self.force_join_prob

    @staticmethod
    def _get_join_slice(module):
        action_space = getattr(module, "action_space", None)
        nvec = getattr(action_space, "nvec", None)
        if nvec is None or len(nvec) < 2:
            return None
        start = int(sum(nvec[:-1]))
        size = int(nvec[-1])
        if size <= 1:
            return None
        return start, size

    @staticmethod
    def _force_logits(logits, start: int, size: int):
        end = start + size
        if torch is not None and isinstance(logits, torch.Tensor):
            updated = logits.clone()
            updated[..., start:end] = FORCE_JOIN_LOGIT_LOW
            updated[..., end - 1] = FORCE_JOIN_LOGIT_HIGH
            return updated
        arr = np.array(logits, copy=True)
        arr[..., start:end] = FORCE_JOIN_LOGIT_LOW
        arr[..., end - 1] = FORCE_JOIN_LOGIT_HIGH
        return arr

    def __call__(self, *, rl_module, batch, episodes, **kwargs):
        if Columns.ACTION_DIST_INPUTS not in batch:
            return batch
        if not (self.force_all_join or self.force_join_prob > 0.0):
            return batch

        def _maybe_force(logits, _eps_id, _agent_id, module_id):
            if module_id is None or "predator" not in str(module_id):
                return logits
            if not self._should_force():
                return logits
            module = rl_module[module_id] if hasattr(rl_module, "__getitem__") else rl_module
            join_slice = self._get_join_slice(module)
            if join_slice is None:
                return logits
            start, size = join_slice
            return self._force_logits(logits, start, size)

        ConnectorV2.foreach_batch_item_change_in_place(
            batch=batch,
            column=Columns.ACTION_DIST_INPUTS,
            func=_maybe_force,
        )
        return batch


def _is_finite_number(val) -> bool:
    if not isinstance(val, numbers.Real):
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def _get_metric(result: dict, path: str):
    if path in result:
        return result[path]
    current = result
    for part in path.split("/"):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _apply_env_settings(algorithm, settings):
    if hasattr(algorithm, "config") and hasattr(algorithm.config, "env_config"):
        try:
            algorithm.config.env_config.update(settings)
        except Exception:
            pass

    def _apply(env):
        if "energy_loss_per_step_predator" in settings:
            env.energy_loss_per_step_predator = float(settings["energy_loss_per_step_predator"])
        if isinstance(getattr(env, "config", None), dict):
            env.config.update(settings)

    env_runner_group = getattr(algorithm, "env_runner_group", None)
    if env_runner_group is None:
        return

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

    if hasattr(env_runner_group, "foreach_worker"):
        env_runner_group.foreach_worker(lambda worker: worker.foreach_env(_apply))
        return
    if hasattr(env_runner_group, "foreach_env_runner"):
        env_runner_group.foreach_env_runner(_apply_runner)
        return
    if hasattr(env_runner_group, "foreach_runner"):
        env_runner_group.foreach_runner(_apply_runner)
        return
    if hasattr(env_runner_group, "foreach_env"):
        env_runner_group.foreach_env(_apply)
        return

    envs = getattr(env_runner_group, "envs", None) or getattr(env_runner_group, "_envs", None)
    if isinstance(envs, (list, tuple)):
        for env in envs:
            _apply(env)


def _build_module_to_env_connectors(force_all_join, force_join_prob, normalize_actions, clip_actions):
    return [
        ForceJoinHuntLogits(force_all_join=force_all_join, force_join_prob=force_join_prob),
        GetActions(),
        TensorToNumpy(),
        UnBatchToIndividualItems(),
        ModuleToAgentUnmapping(),
        RemoveSingleTsTimeRankFromBatch(),
        NormalizeAndClipActions(
            normalize_actions=normalize_actions,
            clip_actions=clip_actions,
        ),
        ListifyDataForVectorEnv(),
    ]


class AdaptivePredatorDecayCallbacks(EpisodeReturn):
    def __init__(self):
        super().__init__()
        self.ema = None
        self.best_ema = None
        self.no_improve = 0
        self.current_decay = None
        self._printed_start = False

    def _ensure_current_decay(self, algorithm):
        if self.current_decay is not None:
            return
        cfg_val = None
        if hasattr(algorithm, "config") and hasattr(algorithm.config, "env_config"):
            cfg_val = algorithm.config.env_config.get("energy_loss_per_step_predator")
        if cfg_val is None:
            cfg_val = BASE_PREDATOR_DECAY
        self.current_decay = float(cfg_val)

    def on_train_result(self, *, algorithm, result, **kwargs):
        super().on_train_result(result=result, **kwargs)

        if not self._printed_start:
            print("[AdaptiveDecay] Callback active", flush=True)
            self._printed_start = True
        iter_num = result.get("training_iteration", 1)
        raw_len = _get_metric(result, LEN_METRIC_KEY)
        if not _is_finite_number(raw_len):
            print(
                f"[AdaptiveDecay] iter={iter_num} {LEN_METRIC_KEY} missing or non-finite",
                flush=True,
            )
            return
        raw_len = float(raw_len)
        if self.ema is None:
            self.ema = raw_len
        else:
            self.ema = EMA_WEIGHT * self.ema + (1.0 - EMA_WEIGHT) * raw_len

        result["adaptive/episode_len_mean_ema"] = self.ema
        self._ensure_current_decay(algorithm)
        result["adaptive/energy_loss_per_step_predator"] = self.current_decay

        improved = self.best_ema is None or self.ema > self.best_ema + LEN_IMPROVEMENT_MIN
        if improved:
            self.best_ema = self.ema
            self.no_improve = 0
        else:
            self.no_improve += 1
        result["adaptive/episode_len_mean_raw"] = raw_len
        result["adaptive/best_episode_len_mean_ema"] = self.best_ema
        result["adaptive/no_improve_iters"] = self.no_improve
        best_str = f"{self.best_ema:.1f}" if self.best_ema is not None else "None"
        print(
            f"[AdaptiveDecay] iter={iter_num} len_mean={raw_len:.1f} "
            f"ema={self.ema:.1f} best_ema={best_str} "
            f"no_improve={self.no_improve} decay={self.current_decay:.3f}",
            flush=True,
        )

        declined = self.best_ema is not None and self.ema < self.best_ema - LEN_DECLINE_DELTA
        plateaued = self.no_improve >= LEN_PLATEAU_PATIENCE
        if declined or plateaued:
            new_decay = min(self.current_decay + PRED_DECAY_STEP, PRED_DECAY_MAX)
            if new_decay > self.current_decay:
                _apply_env_settings(algorithm, {"energy_loss_per_step_predator": new_decay})
                self.current_decay = new_decay
                self.best_ema = self.ema
                self.no_improve = 0
                result["adaptive/energy_loss_per_step_predator"] = new_decay
                reason = "decline" if declined else "plateau"
                print(
                    f"[AdaptiveDecay] {reason} at iter {iter_num} "
                    f"EMA={self.ema:.1f} -> energy_loss_per_step_predator={new_decay:.3f}",
                    flush=True,
                )


# --- Main training setup ---

if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)
    register_env("PredPreyGrass", env_creator)

    # Override static seed at runtime to avoid deterministic placements; keep config file unchanged.
    # Enable strict RLlib outputs so only live agent IDs are emitted each step.
    env_config = {**config_env, "seed": None, "strict_rllib_output": True}


    ray_results_dir = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/mammoths_defect/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    version = "MAMMOTHS_DEFECT_BASE_ADAPTIVE_PRED_DECAY"
    experiment_name = f"{version}_{timestamp}"
    experiment_path = ray_results_path / experiment_name 

    experiment_path.mkdir(parents=True, exist_ok=True)
    # --- Save environment source file for provenance ---
    source_dir = experiment_path / "SOURCE_CODE_ENV"
    source_dir.mkdir(exist_ok=True)
    env_file = Path(__file__).parent / "predpreygrass_rllib_env.py"
    shutil.copy2(env_file, source_dir / f"predpreygrass_rllib_env_{version}.py")

    config_ppo = get_config_ppo()
    config_metadata = {
        "config_env": config_env,
        "config_ppo": config_ppo,
        "adaptive_predator_decay": {
            "len_metric_key": LEN_METRIC_KEY,
            "ema_weight": EMA_WEIGHT,
            "len_improvement_min": LEN_IMPROVEMENT_MIN,
            "len_decline_delta": LEN_DECLINE_DELTA,
            "len_plateau_patience": LEN_PLATEAU_PATIENCE,
            "pred_decay_step": PRED_DECAY_STEP,
            "pred_decay_max": PRED_DECAY_MAX,
            "pred_decay_base": BASE_PREDATOR_DECAY,
        },
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)
    # print(f"Saved config to: {experiment_path/'run_config.json'}")

    sample_env = env_creator(config=env_config)
    # Ensure spaces are populated before extracting
    sample_env.reset(seed=None)

    # Group spaces per policy id (first agent of each policy defines the space)
    obs_by_policy, act_by_policy = {}, {}
    for agent_id, obs_space in sample_env.observation_spaces.items():
        pid = policy_mapping_fn(agent_id)
        if pid not in obs_by_policy:
            obs_by_policy[pid] = obs_space
            act_by_policy[pid] = sample_env.action_spaces[agent_id]

    # Explicitly include action_space_struct so connectors see every agent ID
    # (avoids KeyErrors when new agents appear mid-episode).
    sample_env.action_space_struct = sample_env.action_spaces

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
        .environment(env="PredPreyGrass", env_config=env_config)
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
    )
    def force_join_connector(*_):
        return _build_module_to_env_connectors(
            env_config.get("force_all_join", False),
            env_config.get("force_join_prob", 0.0),
            ppo_config.normalize_actions,
            ppo_config.clip_actions,
        )
    ppo_config = (
        ppo_config.env_runners(
            num_env_runners=config_ppo["num_env_runners"],
            num_envs_per_env_runner=config_ppo["num_envs_per_env_runner"],
            rollout_fragment_length=config_ppo["rollout_fragment_length"],
            sample_timeout_s=config_ppo["sample_timeout_s"],
            num_cpus_per_env_runner=config_ppo["num_cpus_per_env_runner"],
            module_to_env_connector=force_join_connector,
            add_default_connectors_to_module_to_env_pipeline=False,
        )
        .resources(
            num_cpus_for_main_process=config_ppo["num_cpus_for_main_process"],
        )
        .callbacks(AdaptivePredatorDecayCallbacks)
    )

    max_iters = config_ppo["max_iters"]
    checkpoint_every = 10
    del sample_env  # to avoid any stray references

    tuner = Tuner(
        ppo_config.algo_class,
        param_space=ppo_config,
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
