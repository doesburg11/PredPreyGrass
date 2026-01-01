"""
Two-phase PPO training for Stag Hunt:
1) Train with team_capture_margin = 0 for N iterations.
2) Continue from the last checkpoint with team_capture_margin = 1.5.
"""
from predpreygrass.rllib.stag_hunt.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.stag_hunt.config.config_env_stag_hunt import config_env
from predpreygrass.rllib.stag_hunt.utils.episode_return_callback import EpisodeReturn
from predpreygrass.rllib.stag_hunt.utils.networks import build_multi_module_spec

import ray
from ray.tune import Trainable, Tuner, RunConfig, CheckpointConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

import torch

import argparse
import copy
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.stag_hunt.config.config_ppo_gpu_stag_hunt import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.stag_hunt.config.config_ppo_cpu_stag_hunt import config_ppo
    else:
        from predpreygrass.rllib.stag_hunt.config.config_ppo_cpu_stag_hunt import config_ppo
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
        .callbacks(EpisodeReturn)
    )


class PPOTorchLearnerNoForeach(PPOTorchLearner):
    def configure_optimizers_for_module(self, module_id, config=None):
        module = self._module[module_id]
        params = self.get_parameters(module)
        optimizer = torch.optim.Adam(params, foreach=False)
        self.register_optimizer(
            module_id=module_id,
            optimizer=optimizer,
            params=params,
            lr_or_lr_schedule=config.lr,
        )


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


def resolve_checkpoint(result_grid):
    best_result = result_grid.get_best_result(metric="training_iteration", mode="max")
    checkpoint = best_result.get_best_checkpoint(metric="training_iteration", mode="max")
    if checkpoint is not None:
        return Path(checkpoint.to_directory())

    trial_path = Path(best_result.path)
    candidates = sorted(
        trial_path.glob("checkpoint_*"),
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
    )
    if not candidates:
        raise RuntimeError(f"No checkpoints found in {trial_path}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description="Two-phase PPO training with a team_capture_margin curriculum.")
    parser.add_argument("--phase1-iters", type=int, default=100)
    parser.add_argument("--phase2-iters", type=int, default=None)
    parser.add_argument("--phase1-margin", type=float, default=0.0)
    parser.add_argument("--phase2-margin", type=float, default=1.5)
    parser.add_argument("--restore-from", type=str, default=None)
    args = parser.parse_args()

    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    register_env("PredPreyGrass", env_creator)
    config_ppo = get_config_ppo()

    phase2_iters = args.phase2_iters
    if phase2_iters is None:
        phase2_iters = max(config_ppo["max_iters"] - args.phase1_iters, 1)

    base_env_cfg = copy.deepcopy(config_env)
    base_env_cfg["seed"] = None

    phase1_env_cfg = copy.deepcopy(base_env_cfg)
    phase1_env_cfg["team_capture_margin"] = args.phase1_margin

    phase2_env_cfg = copy.deepcopy(base_env_cfg)
    phase2_env_cfg["team_capture_margin"] = args.phase2_margin

    obs_by_policy, act_by_policy = build_spaces(phase1_env_cfg)
    multi_module_spec = build_multi_module_spec(obs_by_policy, act_by_policy)

    ray_results_dir = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/stag_hunt/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    checkpoint_path = None
    if args.restore_from:
        checkpoint_path = Path(args.restore_from).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"restore-from checkpoint not found: {checkpoint_path}")
    else:
        phase1_name = f"STAG_HUNT_MARGIN_CURRICULUM_PHASE1_M{args.phase1_margin}_{timestamp}"
        prepare_experiment_dir(
            ray_results_path,
            phase1_name,
            phase1_env_cfg,
            config_ppo,
            "PHASE1",
        )
        phase1_config = build_ppo_config(
            phase1_env_cfg, config_ppo, obs_by_policy, act_by_policy, multi_module_spec
        )
        tuner = Tuner(
            PPOCheckpointTrainable,
            param_space={"algo_config": phase1_config},
            run_config=RunConfig(
                name=phase1_name,
                storage_path=str(ray_results_path),
                stop={"training_iteration": args.phase1_iters},
                checkpoint_config=CheckpointConfig(
                    num_to_keep=10,
                    checkpoint_frequency=10,
                    checkpoint_at_end=True,
                ),
            ),
        )
        result = tuner.fit()
        checkpoint_path = resolve_checkpoint(result)
        print(f"[phase1] checkpoint: {checkpoint_path}")

    phase2_name = f"STAG_HUNT_MARGIN_CURRICULUM_PHASE2_M{args.phase2_margin}_{timestamp}"
    prepare_experiment_dir(
        ray_results_path,
        phase2_name,
        phase2_env_cfg,
        config_ppo,
        "PHASE2",
    )
    phase2_config = build_ppo_config(
        phase2_env_cfg, config_ppo, obs_by_policy, act_by_policy, multi_module_spec
    )
    tuner = Tuner(
        PPOCheckpointTrainable,
        param_space={
            "algo_config": phase2_config,
            "restore_from_path": str(checkpoint_path),
        },
        run_config=RunConfig(
            name=phase2_name,
            storage_path=str(ray_results_path),
            stop={"training_iteration": phase2_iters},
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
