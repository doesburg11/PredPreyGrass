"""
Resume PPO training for Stag Hunt from a fixed checkpoint using resume configs.
"""
from predpreygrass.rllib.stag_hunt_vectorized.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.stag_hunt_vectorized.config.config_env_stag_hunt_vectorized_resumed import (
    config_env as resume_env_config,
)
from predpreygrass.rllib.stag_hunt_vectorized.utils.episode_return_callback import EpisodeReturn
from predpreygrass.rllib.stag_hunt_vectorized.utils.networks import build_multi_module_spec

import ray
from ray.tune import Trainable, Tuner, RunConfig, CheckpointConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

import torch

import copy
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.stag_hunt_vectorized.config.config_ppo_gpu_stag_hunt_vectorized_resumed import (
            config_ppo,
        )
    elif num_cpus == 8:
        from predpreygrass.rllib.stag_hunt_vectorized.config.config_ppo_cpu_stag_hunt_vectorized_resumed import (
            config_ppo,
        )
    else:
        # Default to CPU config for other CPU counts to keep training usable across machines.
        from predpreygrass.rllib.stag_hunt_vectorized.config.config_ppo_cpu_stag_hunt_vectorized_resumed import (
            config_ppo,
        )
    return config_ppo

def env_creator(config):
    return PredPreyGrass(config)

def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    type_id = parts[1]
    role = parts[2]
    return f"type_{type_id}_{role}"

def build_spaces(env_config):
    sample_env = env_creator(config=env_config)
    sample_env.reset(seed=None)

    obs_by_policy, act_by_policy = {}, {}
    for agent_id, obs_space in sample_env.observation_spaces.items():
        pid = policy_mapping_fn(agent_id)
        if pid not in obs_by_policy:
            obs_by_policy[pid] = obs_space
            act_by_policy[pid] = sample_env.action_spaces[agent_id]

    sample_env.action_space_struct = sample_env.action_spaces
    del sample_env

    return obs_by_policy, act_by_policy


def build_ppo_config(env_cfg, config_ppo, obs_by_policy, act_by_policy, multi_module_spec):
    policies = {
        pid: (None, obs_by_policy[pid], act_by_policy[pid], {})
        for pid in obs_by_policy
    }

    return (
        PPOConfig()
        .environment(env="PredPreyGrass", env_config=env_cfg)
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
            learner_class=PPOTorchLearnerNoForeach,
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
        .callbacks(ResumeCallbacks)
    )


class ResumeCallbacks(EpisodeReturn):
    def __init__(self):
        super().__init__()
        self._curriculum_start_iter = None

    def on_train_result(self, *, algorithm, metrics_logger=None, result: dict, **kwargs):
        super().on_train_result(result=result, **kwargs)
        if not CURRICULUM_ENABLED:
            return
        margin = self._get_margin(result)
        self._apply_margin(algorithm, margin)
        if metrics_logger is not None:
            metrics_logger.log_value("custom_metrics/team_capture_margin", margin)
        result["custom_metrics/team_capture_margin"] = margin

    def _get_margin(self, result):
        training_iter = int(result.get("training_iteration", 0))
        if self._curriculum_start_iter is None:
            if CURRICULUM_START_ITER is not None:
                self._curriculum_start_iter = int(CURRICULUM_START_ITER)
            else:
                self._curriculum_start_iter = training_iter

        if CURRICULUM_RAMP_ITERS <= 0:
            return float(CURRICULUM_END_MARGIN)

        progress = (training_iter - self._curriculum_start_iter) / float(CURRICULUM_RAMP_ITERS)
        progress = max(0.0, min(1.0, progress))
        return float(
            CURRICULUM_START_MARGIN
            + (CURRICULUM_END_MARGIN - CURRICULUM_START_MARGIN) * progress
        )

    def _apply_margin(self, algorithm, margin):
        if algorithm is None:
            return
        try:
            algorithm.config.env_config["team_capture_margin"] = margin
        except Exception:
            pass

        eval_cfg = getattr(algorithm.config, "evaluation_config", None)
        if isinstance(eval_cfg, dict):
            eval_env_cfg = eval_cfg.get("env_config")
            if isinstance(eval_env_cfg, dict):
                eval_env_cfg["team_capture_margin"] = margin

        def _set_margin(env_runner):
            env = getattr(env_runner, "env", None)
            if env is None:
                return
            sub_envs = getattr(env, "envs", None)
            if sub_envs:
                for sub_env in sub_envs:
                    if hasattr(sub_env, "team_capture_margin"):
                        sub_env.team_capture_margin = margin
            elif hasattr(env, "team_capture_margin"):
                env.team_capture_margin = margin

        if getattr(algorithm, "env_runner_group", None):
            algorithm.env_runner_group.foreach_env_runner(
                _set_margin,
                local_env_runner=True,
            )
        if getattr(algorithm, "eval_env_runner_group", None):
            algorithm.eval_env_runner_group.foreach_env_runner(
                _set_margin,
                local_env_runner=True,
            )


class PPOTorchLearnerNoForeach(PPOTorchLearner):
    def configure_optimizers_for_module(self, module_id, config=None):
        module = self._module[module_id]
        params = self.get_parameters(module)
        optimizer = torch.optim.Adam(params, foreach=False)
        self._sanitize_optimizer_params(optimizer)
        self.register_optimizer(
            module_id=module_id,
            optimizer=optimizer,
            params=params,
            lr_or_lr_schedule=config.lr,
        )

    def _set_optimizer_state(self, state):
        super()._set_optimizer_state(state)
        for optimizer in self._named_optimizers.values():
            self._sanitize_optimizer_params(optimizer)

    def _sanitize_optimizer_params(self, optimizer):
        # Ray converts optimizer state to tensors on restore; torch Adam+foreach can't handle tensor betas.
        if not hasattr(optimizer, "param_groups"):
            return
        for group in optimizer.param_groups:
            if "foreach" in group:
                group["foreach"] = False
            if "capturable" in group:
                group["capturable"] = False
            for key, value in list(group.items()):
                if key == "params":
                    continue
                if isinstance(value, (tuple, list)):
                    group[key] = tuple(
                        v.item() if torch.is_tensor(v) and v.numel() == 1 else v
                        for v in value
                    )
                elif torch.is_tensor(value) and value.numel() == 1:
                    group[key] = value.item()

class PPOCheckpointTrainable(Trainable):
    def setup(self, config):
        self._cfg = dict(config)
        algo_config = self._cfg["algo_config"].copy(copy_frozen=False)
        self.algo = algo_config.build_algo()

        ckpt = self._cfg.get("restore_from_path")
        if ckpt:
            self.algo.restore_from_path(ckpt)

    def step(self):
        return self.algo.train()

    def save_checkpoint(self, checkpoint_dir):
        return self.algo.save_to_path(checkpoint_dir)

    def load_checkpoint(self, path):
        self.algo.restore_from_path(path)

    def cleanup(self):
        self.algo.stop()

    @classmethod
    def default_resource_request(cls, config):
        algo_config = config["algo_config"]
        return algo_config.algo_class.default_resource_request(algo_config)


def prepare_experiment_dir(ray_results_path, experiment_name, env_cfg, config_ppo, version_tag):
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    source_dir = experiment_path / "SOURCE_CODE_ENV"
    source_dir.mkdir(exist_ok=True)
    env_file = Path(__file__).parent / "predpreygrass_rllib_env.py"
    shutil.copy2(env_file, source_dir / f"predpreygrass_rllib_env_{version_tag}.py")

    config_metadata = {
        "config_env": env_cfg,
        "config_ppo": config_ppo,
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)

    return experiment_path


RESTORE_FROM = Path(
    "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/stag_hunt_vectorized/ray_results/"
    "STAG_HUNT_EPOCH_20_SEPARATE_PREY_CHANNELS_2025-12-31_11-56-34/"
    "PPO_PredPreyGrass_5f8ac_00000_0_2025-12-31_11-56-34/checkpoint_000007"
)
RESUME_ITERS = None
EVAL_INTERVAL = 1
EVAL_EPISODES = 5
EVAL_SEED = 12345
EVAL_NUM_ENV_RUNNERS = 1
MARGIN_CURRICULUM = resume_env_config.get("margin_curriculum")
if not isinstance(MARGIN_CURRICULUM, dict):
    MARGIN_CURRICULUM = None
CURRICULUM_ENABLED = MARGIN_CURRICULUM is not None
if CURRICULUM_ENABLED:
    CURRICULUM_START_MARGIN = float(
        MARGIN_CURRICULUM.get(
            "start_margin",
            resume_env_config.get("team_capture_margin", 0.0),
        )
    )
    CURRICULUM_END_MARGIN = float(
        MARGIN_CURRICULUM.get("end_margin", CURRICULUM_START_MARGIN)
    )
    CURRICULUM_RAMP_ITERS = int(MARGIN_CURRICULUM.get("ramp_iters", 0))
    CURRICULUM_START_ITER = MARGIN_CURRICULUM.get("start_iter")
else:
    CURRICULUM_START_MARGIN = None
    CURRICULUM_END_MARGIN = None
    CURRICULUM_RAMP_ITERS = 0
    CURRICULUM_START_ITER = None


def main():

    ray.shutdown()
    os.environ.pop("RAY_ADDRESS", None)
    ray.init(address="local", log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)
    config_ppo = get_config_ppo()

    resume_iters = RESUME_ITERS or config_ppo["max_iters"]

    resume_env_cfg = copy.deepcopy(resume_env_config)
    resume_env_cfg["seed"] = None
    if CURRICULUM_ENABLED:
        if CURRICULUM_RAMP_ITERS <= 0:
            resume_env_cfg["team_capture_margin"] = CURRICULUM_END_MARGIN
        else:
            resume_env_cfg["team_capture_margin"] = CURRICULUM_START_MARGIN

    obs_by_policy, act_by_policy = build_spaces(resume_env_cfg)
    multi_module_spec = build_multi_module_spec(obs_by_policy, act_by_policy)

    ray_results_dir = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/stag_hunt_vectorized/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if RESTORE_FROM is None:
        raise ValueError("RESTORE_FROM must be set to resume from a checkpoint.")
    checkpoint_path = RESTORE_FROM.expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"restore-from checkpoint not found: {checkpoint_path}")

    resume_name = f"STAG_HUNT_EPOCH_20_SEPARATE_PREY_CHANNELS_RESUME_{timestamp}"
    prepare_experiment_dir(
        ray_results_path,
        resume_name,
        resume_env_cfg,
        config_ppo,
        "RESUME",
    )
    resume_config = build_ppo_config(
        resume_env_cfg, config_ppo, obs_by_policy, act_by_policy, multi_module_spec
    )
    eval_env_cfg = copy.deepcopy(resume_env_cfg)
    eval_env_cfg["seed"] = EVAL_SEED
    resume_config.evaluation(
        evaluation_interval=EVAL_INTERVAL,
        evaluation_duration=EVAL_EPISODES,
        evaluation_duration_unit="episodes",
        evaluation_num_env_runners=EVAL_NUM_ENV_RUNNERS,
        evaluation_config={
            "explore": False,
            "env_config": eval_env_cfg,
        },
    )
    tuner = Tuner(
        PPOCheckpointTrainable,
        param_space={
            "algo_config": resume_config,
            "restore_from_path": str(checkpoint_path),
        },
        run_config=RunConfig(
            name=resume_name,
            storage_path=str(ray_results_path),
            stop={"training_iteration": resume_iters},
            checkpoint_config=CheckpointConfig(
                num_to_keep=10,
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
        ),
    )
    tuner.fit()
    ray.shutdown()


if __name__ == "__main__":
    main()
