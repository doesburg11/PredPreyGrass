"""
This script trains a multi-agent environment with PPO using Ray RLlib new API stack.
It uses a custom environment that simulates a predator-prey-grass ecosystem.
The environment is a grid world where predators and prey move around.
Predators try to catch prey, and prey try to eat grass.
Predators and prey both either can be of type_1 or type_2.
"""
import os
import sys
from datetime import datetime
from pathlib import Path
import json
import shutil
import subprocess


def _prepend_snapshot_source() -> None:
    script_path = Path(__file__).resolve()
    try:
        if script_path.parents[2].name == "predpreygrass" and script_path.parents[1].name == "rllib":
            source_root = script_path.parents[3]
            if source_root.name in {"REPRODUCE_CODE", "SOURCE_CODE"}:
                source_root_str = str(source_root)
                if source_root_str not in sys.path:
                    sys.path.insert(0, source_root_str)
    except IndexError:
        return


_prepend_snapshot_source()


from predpreygrass.rllib.stag_hunt_forward_view_nature_nurture.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.stag_hunt_forward_view_nature_nurture.config.config_env_stag_hunt_forward_view import config_env
from predpreygrass.rllib.stag_hunt_forward_view_nature_nurture.utils.episode_return_callback import EpisodeReturn
from predpreygrass.rllib.stag_hunt_forward_view_nature_nurture.utils.networks import build_multi_module_spec

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune import Tuner, RunConfig, CheckpointConfig


_SNAPSHOT_EXCLUDE_DIRS = {
    "ray_results",
    "ray_results_failed",
    "trained_examples",
    "__pycache__",
}


def copy_module_snapshot(reproduce_dir: Path) -> None:
    """Copy only the local module tree into REPRODUCE_CODE for reproducibility."""
    module_dir = Path(__file__).resolve().parent
    module_name = module_dir.name

    pkg_root = reproduce_dir / "predpreygrass"
    rllib_root = pkg_root / "rllib"
    rllib_root.mkdir(parents=True, exist_ok=True)

    pkg_init = pkg_root / "__init__.py"
    if not pkg_init.exists():
        pkg_init.write_text("")
    rllib_init = rllib_root / "__init__.py"
    if not rllib_init.exists():
        rllib_init.write_text("")

    dest_dir = rllib_root / module_name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    def _ignore(path: str, entries):
        ignored = []
        for entry in entries:
            if entry in _SNAPSHOT_EXCLUDE_DIRS:
                ignored.append(entry)
                continue
            if entry.endswith(".pyc"):
                ignored.append(entry)
        return ignored

    shutil.copytree(module_dir, dest_dir, ignore=_ignore)

    assets_src = None
    for parent in module_dir.parents:
        if parent.name == "REPRODUCE_CODE":
            candidate = parent / "assets" / "images" / "icons"
            if candidate.is_dir():
                assets_src = candidate
            break
    if assets_src is None:
        candidate = module_dir.parents[4] / "assets" / "images" / "icons"
        if candidate.is_dir():
            assets_src = candidate
    if assets_src:
        assets_dest = reproduce_dir / "assets" / "images" / "icons"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        assets_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(assets_src, assets_dest)


def write_pip_freeze(output_path: Path) -> None:
    """Write the current environment's pip freeze to output_path."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        output_path.write_text(result.stdout)
        if result.stderr:
            (output_path.parent / "pip_freeze_train_stderr.txt").write_text(result.stderr)
    except Exception as exc:
        output_path.write_text(f"pip freeze failed: {exc}")


def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        from predpreygrass.rllib.stag_hunt_forward_view_nature_nurture.config.config_ppo_gpu_stag_hunt_forward_view import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.stag_hunt_forward_view_nature_nurture.config.config_ppo_cpu_stag_hunt_forward_view import config_ppo
    else:
        # Default to CPU config for other CPU counts to keep training usable across machines.
        from predpreygrass.rllib.stag_hunt_forward_view_nature_nurture.config.config_ppo_cpu_stag_hunt_forward_view import config_ppo
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
    env_config = {**config_env, "seed": None}
    trained_example_dir = os.getenv("TRAINED_EXAMPLE_DIR")
    if trained_example_dir:
        ray_results_path = Path(trained_example_dir).expanduser().resolve() / "ray_results"
    else:
        ray_results_dir = "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/stag_hunt_forward_view_nature_nurture/ray_results/"
        ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    version = "STAG_HUNT_FORWARD_NATURE_NURTURE"
    experiment_name = f"{version}_{timestamp}"
    experiment_path = ray_results_path / experiment_name 

    experiment_path.mkdir(parents=True, exist_ok=True)
    reproduce_dir = experiment_path / "REPRODUCE_CODE"
    reproduce_dir.mkdir(exist_ok=True)
    copy_module_snapshot(reproduce_dir)

    config_ppo = get_config_ppo()
    config_metadata = {
        "config_env": config_env,
        "config_ppo": config_ppo,
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)
    # print(f"Saved config to: {experiment_path/'run_config.json'}")

    config_dir = reproduce_dir / "CONFIG"
    config_dir.mkdir(parents=True, exist_ok=True)
    write_pip_freeze(reproduce_dir / "pip_freeze_train.txt")
    with open(config_dir / "config_env.json", "w") as f:
        json.dump(config_env, f, indent=4)
    with open(config_dir / "config_ppo.json", "w") as f:
        json.dump(config_ppo, f, indent=4)
    shutil.copy2(experiment_path / "run_config.json", config_dir / "run_config.json")

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
        .callbacks(EpisodeReturn)
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
