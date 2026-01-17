"""
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
Predators and prey both either can be of type_1 or type_2.
"""
from predpreygrass.rllib.mammoths_defect.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.mammoths_defect.config.config_env_mammoths_defect import config_env
from predpreygrass.rllib.mammoths_defect.utils.episode_return_callback import EpisodeReturn
from predpreygrass.rllib.mammoths_defect.utils.networks import build_multi_module_spec

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune import Tuner, RunConfig, CheckpointConfig

import os
from datetime import datetime
from pathlib import Path
import json
import shutil


# Join-probability schedule (start -> 0.0 over 1000 iterations, step=0.1 every 100 iters)
JOIN_PROB_START = float(config_env.get("force_join_prob", 1.0))
JOIN_PROB_END = 0.0
JOIN_PROB_STEP = 0.1
JOIN_PROB_INTERVAL = 100
JOIN_PROB_WARMUP_ITERS = 400


def _clamp_prob(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _apply_env_settings(algorithm, settings: dict) -> None:
    if hasattr(algorithm, "config") and hasattr(algorithm.config, "env_config"):
        try:
            algorithm.config.env_config.update(settings)
        except Exception:
            pass

    def _apply_one(target):
        if target is None:
            return
        to_update = [target]
        unwrapped = getattr(target, "unwrapped", None)
        if unwrapped is not None and unwrapped is not target:
            to_update.append(unwrapped)
        for obj in to_update:
            if "force_all_join" in settings:
                obj.force_all_join = bool(settings["force_all_join"])
            if "force_join_prob" in settings:
                obj.force_join_prob = _clamp_prob(settings["force_join_prob"])
            if isinstance(getattr(obj, "config", None), dict):
                obj.config.update(settings)

    def _apply(env):
        if env is None:
            return
        for attr in ("envs", "_envs"):
            sub_envs = getattr(env, attr, None)
            if isinstance(sub_envs, (list, tuple)):
                for sub in sub_envs:
                    _apply_one(sub)
                _apply_one(env)
                return
        _apply_one(env)

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


class EpisodeReturnJoinSchedule(EpisodeReturn):
    def __init__(self):
        super().__init__()
        self._last_join_prob = None

    def on_train_result(self, *, result, **kwargs):
        iter_num = result.get("training_iteration", 0)
        steps = max(0, (iter_num - JOIN_PROB_WARMUP_ITERS) // JOIN_PROB_INTERVAL)
        join_prob = JOIN_PROB_START - JOIN_PROB_STEP * steps
        join_prob = max(JOIN_PROB_END, _clamp_prob(join_prob))

        if self._last_join_prob != join_prob:
            algorithm = kwargs.get("algorithm")
            if algorithm is not None:
                _apply_env_settings(algorithm, {"force_join_prob": join_prob})
            self._last_join_prob = join_prob

        result["custom_metrics/force_join_prob_schedule"] = join_prob
        super().on_train_result(result=result, **kwargs)


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


# --- Main training setup ---

if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)
    register_env("PredPreyGrass", env_creator)

    # Override static seed at runtime to avoid deterministic placements; keep config file unchanged.
    env_config = {
        **config_env,
        "seed": None,
    }


    ray_results_dir = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/mammoths_defect/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    version = "MAMMOTHS_DEFECT_JOIN_PROB_SCHEDULE_1_0_TO_0_0"
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
        "join_prob_schedule": {
            "start": JOIN_PROB_START,
            "end": JOIN_PROB_END,
            "step": JOIN_PROB_STEP,
            "interval": JOIN_PROB_INTERVAL,
            "warmup_iters": JOIN_PROB_WARMUP_ITERS,
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
        .callbacks(EpisodeReturnJoinSchedule)
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
