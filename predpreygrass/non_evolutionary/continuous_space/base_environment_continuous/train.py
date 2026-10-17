"""Train predators and prey using the environment and PPO configuration files.

Run from the repository root:

    python predpreygrass/non_evolutionary/continuous_space/base_environment_continuous/train.py

Each run saves what it was started with in <run>/source_code/ (see
save_source_code), so it can be reproduced later.
"""

import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from predpreygrass.non_evolutionary.continuous_space.base_environment_continuous.config.config_env import config_env
from predpreygrass.non_evolutionary.continuous_space.base_environment_continuous.config.config_ppo import config_ppo

# Set native thread limits before loading NumPy/PyTorch in CLI runs.
if __name__ == "__main__":
    for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_variable] = "1"

import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from predpreygrass.training import LocalPPO
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from tensorboardX import SummaryWriter

from predpreygrass.non_evolutionary.continuous_space.base_environment_continuous.environment import ENV_NAME, register


def _species_of(agent_id: str) -> str:
    if agent_id.startswith("predator_"):
        return "predator"
    if agent_id.startswith("prey_"):
        return "prey"
    raise ValueError(f"Unrecognized agent id {agent_id!r}: expected 'predator_<i>' or " "'prey_<j>'.")


def _flatten_scalars(prefix: str, value, out: dict) -> None:
    """Collect every finite number in a nested RLlib result dict as 'a/b/c'."""
    if isinstance(value, dict):
        for key, sub in value.items():
            _flatten_scalars(f"{prefix}/{key}" if prefix else str(key), sub, out)
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and math.isfinite(value):
        out[prefix] = float(value)


def _per_agent_mapping(agent_id, *args, **kwargs):
    return agent_id


def _per_species_mapping(agent_id, *args, **kwargs):
    return f"{_species_of(agent_id)}_policy"


def build_policies(mode: str, predator_count: int, prey_count: int):
    if mode == "il":
        policies = {f"predator_{i}" for i in range(predator_count)} | {f"prey_{j}" for j in range(prey_count)}
        return policies, _per_agent_mapping
    if mode == "ps":
        return {"predator_policy", "prey_policy"}, _per_species_mapping
    raise ValueError(f"Unknown mode: {mode!r} (expected 'il' or 'ps')")


def module_checkpoint(checkpoint, policy):
    """Where a training checkpoint keeps one policy's RLModule state."""
    return Path(checkpoint) / "learner_group" / "learner" / "rl_module" / policy


def load_settings():
    """Copy and validate editable settings without modifying the source dictionaries."""
    env = dict(config_env)
    ppo = dict(config_ppo)
    for settings, names in (
        (
            env,
            (
                "predator_count",
                "prey_count",
                "max_time_steps",
                "action_repeat",
            ),
        ),
        (ppo, ("iterations", "train_batch_size", "minibatch_size", "num_epochs")),
    ):
        for name in names:
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
    for name in ("num_env_runners", "num_learners", "checkpoint_every"):
        if isinstance(ppo[name], bool) or not isinstance(ppo[name], int) or ppo[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for settings, name, positive in (
        (ppo, "lr", True),
        (ppo, "vf_clip_param", True),
        (ppo, "entropy_coeff", False),
        (ppo, "num_gpus_per_learner", False),
        (ppo, "grad_clip", True),
    ):
        value = settings[name]
        if name == "grad_clip" and value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (positive and value == 0)
        ):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    for name in ("gamma", "gae_lambda"):
        value = ppo[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    for name in ("prey_fov", "predator_fov"):
        if name in env and (isinstance(env[name], bool) or not isinstance(env[name], int) or not 0 < env[name] <= 360):
            raise ValueError(f"{name} must be an integer in (0, 360]")
    if not isinstance(ppo["vf_share_layers"], bool):
        raise ValueError("vf_share_layers must be a boolean")
    if ppo["mode"] not in ("il", "ps"):
        raise ValueError("mode must be il or ps")
    if ppo["mode"] == "il":
        # A policy per individual needs every agent ID ahead of time, and
        # newborns get IDs no policy was built for.
        raise ValueError("mode il cannot be used with reproduction; use ps")
    if ppo["checkpoint_every"] and not ppo["checkpoint_dir"]:
        raise ValueError("checkpoint_every requires checkpoint_dir")
    ppo.setdefault("init_checkpoint", None)
    ppo["frozen_policies"] = list(ppo.get("frozen_policies") or [])
    if ppo["init_checkpoint"]:
        ppo["init_checkpoint"] = os.path.abspath(os.path.expanduser(os.fspath(ppo["init_checkpoint"])))
        policies, _ = build_policies(ppo["mode"], env["predator_count"], env["prey_count"])
        for policy in sorted(policies):
            state = module_checkpoint(ppo["init_checkpoint"], policy)
            if not (state / "module_state.pkl").is_file():
                raise ValueError(f"init_checkpoint has no {policy}: {state}")
    if ppo["frozen_policies"]:
        if not ppo["init_checkpoint"]:
            raise ValueError("frozen_policies requires init_checkpoint")
        if ppo["mode"] != "ps":
            raise ValueError("frozen_policies requires mode ps")
        unknown = set(ppo["frozen_policies"]) - {"predator_policy", "prey_policy"}
        if unknown:
            raise ValueError(f"Unknown frozen_policies: {sorted(unknown)}")
        if len(set(ppo["frozen_policies"])) == 2:
            raise ValueError("frozen_policies would leave nothing to train")
    module_name = Path(__file__).resolve().parent.name
    timestamp = datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_name = f"{module_name}_{timestamp}"
    for name in ("checkpoint_dir", "tensorboard_dir"):
        if ppo[name]:
            ppo[name] = os.path.abspath(os.path.expanduser(os.fspath(ppo[name]).replace("{run_name}", run_name)))
    return env, SimpleNamespace(**ppo)


SOURCE_FILES = ("*.py", "config/*.py", "README.md", "SPEC.md", "requirements.txt")


def run_directory(ppo):
    """The run's own folder: the parent of its checkpoint directory (or of
    its TensorBoard directory without checkpoints), runs/<run>/ by default;
    None when both outputs are switched off."""
    for output in (ppo.checkpoint_dir, ppo.tensorboard_dir):
        if output:
            return Path(output).parent
    return None


def _command(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True, timeout=60).stdout


def save_source_code(run_dir, env_config, ppo):
    """Copy what this run was started with to <run_dir>/source_code/:

    - the module's code, configuration, README and requirements (not tests,
      runs/ or caches);
    - settings.json: the settings actually used, including changes made in
      memory by a launcher script;
    - the launcher script itself, if training was started from outside the
      module;
    - git.txt (commit, branch, uncommitted files) and uncommitted.patch;
    - versions.txt: Python version and `pip freeze`.

    Only the code copy is required; failures of the other parts are printed
    as warnings and do not stop training."""
    module_dir = Path(__file__).resolve().parent
    target = Path(run_dir) / "source_code"
    target.mkdir(parents=True, exist_ok=True)
    for pattern in SOURCE_FILES:
        for path in sorted(module_dir.glob(pattern)):
            if path.name.startswith("test_") or not path.is_file():
                continue
            destination = target / path.relative_to(module_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    settings = {"config_env": env_config, "config_ppo": vars(ppo)}
    (target / "settings.json").write_text(json.dumps(settings, indent=2, sort_keys=True, default=str) + "\n")
    launcher = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    if launcher is not None and launcher.suffix == ".py" and launcher.is_file() and launcher.parent != module_dir:
        shutil.copy2(launcher, target / f"launcher_{launcher.name}")
    try:
        commit = _command(["git", "rev-parse", "HEAD"], module_dir).strip()
        branch = _command(["git", "rev-parse", "--abbrev-ref", "HEAD"], module_dir)
        status = _command(["git", "status", "--porcelain"], module_dir)
        lines = [f"commit: {commit}", f"branch: {branch.strip()}"]
        lines.append("uncommitted changes:\n" + status if status else "uncommitted changes: none")
        (target / "git.txt").write_text("\n".join(lines) + "\n")
        diff = _command(["git", "diff", "HEAD"], module_dir)
        if diff:
            (target / "uncommitted.patch").write_text(diff)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"warning: git state not saved: {error}", flush=True)
    try:
        freeze = _command([sys.executable, "-m", "pip", "freeze"], module_dir)
        (target / "versions.txt").write_text(f"python {platform.python_version()} ({sys.executable})\n\n{freeze}")
    except (OSError, subprocess.SubprocessError) as error:
        print(f"warning: package versions not saved: {error}", flush=True)
    return target


def main():
    env_config, ppo = load_settings()
    run_dir = run_directory(ppo)
    if run_dir is not None:
        saved = save_source_code(run_dir, env_config, ppo)
        print(f"source code and settings saved to {saved}", flush=True)
    # Keep each sampler and the local learner from starting competing BLAS
    # thread pools. Ray workers inherit these variables when Ray starts.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    register()

    policies, mapping_fn = build_policies(ppo.mode, env_config["predator_count"], env_config["prey_count"])

    config = (
        PPOConfig(algo_class=LocalPPO)
        .environment(
            ENV_NAME,
            env_config=env_config,
        )
        .resources(num_cpus_for_main_process=1)
        .env_runners(
            num_env_runners=ppo.num_env_runners,
            num_cpus_per_env_runner=1,
        )
        .training(
            train_batch_size_per_learner=ppo.train_batch_size,
            minibatch_size=ppo.minibatch_size,
            num_epochs=ppo.num_epochs,
            lr=ppo.lr,
            lambda_=ppo.gae_lambda,
            gamma=ppo.gamma,
            vf_clip_param=ppo.vf_clip_param,
            entropy_coeff=ppo.entropy_coeff,
            grad_clip=ppo.grad_clip,
            grad_clip_by="global_norm",
        )
        .learners(
            num_learners=ppo.num_learners,
            num_gpus_per_learner=ppo.num_gpus_per_learner,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=mapping_fn,
            # Frozen policies still act, with their loaded weights, but the
            # learner never updates them.
            policies_to_train=[p for p in policies if p not in ppo.frozen_policies],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    p: RLModuleSpec(
                        load_state_path=(str(module_checkpoint(ppo.init_checkpoint, p)) if ppo.init_checkpoint else None)
                    )
                    for p in policies
                },
            ),
            model_config=DefaultModelConfig(vf_share_layers=ppo.vf_share_layers),
        )
    )

    algo = config.build_algo()
    writer = SummaryWriter(ppo.tensorboard_dir) if ppo.tensorboard_dir else None
    try:
        for i in range(ppo.iterations):
            t0 = time.time()
            result = algo.train()
            env_runner_stats = result.get("env_runners", {})
            reward_mean = env_runner_stats.get("episode_return_mean")
            num_episodes = env_runner_stats.get("num_episodes")
            module_means = env_runner_stats.get("module_episode_returns_mean", {})
            module_returns = {k: round(v, 1) for k, v in module_means.items()}
            if writer is not None:
                scalars = {}
                _flatten_scalars("", result, scalars)
                for tag, scalar in scalars.items():
                    writer.add_scalar(tag, scalar, i + 1)
                writer.flush()
            print(
                f"iter {i + 1}/{ppo.iterations}  "
                f"episode_return_mean={reward_mean}  "
                f"num_episodes={num_episodes}  "
                f"module_returns={module_returns}  "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
            done = i + 1
            if ppo.checkpoint_every and done % ppo.checkpoint_every == 0 and done < ppo.iterations:
                path = os.path.join(ppo.checkpoint_dir, f"iter_{done:06d}")
                algo.save(path)
                print(f"checkpoint saved to {path}", flush=True)
        if ppo.checkpoint_dir:
            algo.save(ppo.checkpoint_dir)
            print(f"checkpoint saved to {ppo.checkpoint_dir}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        algo.stop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Training takes no command-line arguments. " "Edit config/config_env.py and config/config_ppo.py.")
    main()
