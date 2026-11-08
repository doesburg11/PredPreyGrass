"""Train the hunters (arms L and B, DESIGN.md) against frozen deer, using
the environment and PPO configuration files.

The predators' policy is rl_module.GeneBiasPPOTorchRLModule (beta 0 in arm
L, genes.BETA0 in arm B); the deer's is a deer_stag_hunt checkpoint's deer
policy, copied into the run (<run>/deer_policy/) and never trained. With
every checkpoint, gene_pools.json (each environment's gene pool) and
budget.json (each environment's lifetime physics steps, episodes, births
and generations) are written next to it.

Run from the repository root:

    python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/train.py

Each run saves what it was started with in <run>/source_code/ (see
save_source_code), so it can be reproduced later.
"""

import dataclasses
import json
import math
import pickle
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

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_env import config_env
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import config_ppo

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

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import genes
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.callbacks import CatchLogger
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.environment import ENV_NAME, SPECIES, EvolvingDeerStagHunt, register, species_of
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.rl_module import GeneBiasPPOTorchRLModule

POLICIES = {f"{species}_policy" for species in SPECIES}
ARMS = {"L": 0.0, "B": genes.BETA0}  # arm -> beta
GENE_POOLS = "gene_pools.json"
BUDGET = "budget.json"
DEER_POLICY = "deer_policy"


def module_checkpoint(checkpoint, policy):
    """Where a training checkpoint keeps one policy's RLModule state."""
    return Path(checkpoint) / "learner_group" / "learner" / "rl_module" / policy


def predator_model_config(env_config, arm, vf_share_layers):
    """The predators' model_config: RLlib's defaults plus the gene bias."""
    env = EvolvingDeerStagHunt(env_config)
    return dict(dataclasses.asdict(DefaultModelConfig(vf_share_layers=vf_share_layers)), gene_bias={"beta": ARMS[arm], "layout": genes.layout(env)})


def copy_deer_policy(deer_checkpoint, run_dir):
    """Copy the deer policy of a deer_stag_hunt checkpoint into the run;
    returns the copy's directory."""
    source = module_checkpoint(deer_checkpoint, "deer_policy")
    target = Path(run_dir) / DEER_POLICY
    if target.exists():
        raise ValueError(f"{target} exists")
    shutil.copytree(source, target)
    (target / "source.txt").write_text(f"{Path(deer_checkpoint).resolve()}\n")
    return target


def deer_state(path):
    with open(Path(path) / "module_state.pkl", "rb") as f:
        return pickle.load(f)


def save_evolution(algo, checkpoint) -> None:
    """Write every simulation environment's gene pool and lifetime counters
    next to a checkpoint (each environment evolves its own population)."""
    per_runner = algo.env_runner_group.foreach_env_runner(
        lambda runner: [(env.unwrapped.gene_pool_state(), dict(env.unwrapped.lifetime)) for env in runner.env.envs] if runner.env is not None else [],
        local_env_runner=False,
    )
    entries = [entry for runner in per_runner for entry in runner]
    Path(checkpoint).mkdir(parents=True, exist_ok=True)
    (Path(checkpoint) / GENE_POOLS).write_text(json.dumps([pool for pool, _ in entries]) + "\n")
    (Path(checkpoint) / BUDGET).write_text(json.dumps([budget for _, budget in entries]) + "\n")


def _flatten_scalars(prefix: str, value, out: dict) -> None:
    """Collect every finite number in a nested RLlib result dict as 'a/b/c'."""
    if isinstance(value, dict):
        for key, sub in value.items():
            _flatten_scalars(f"{prefix}/{key}" if prefix else str(key), sub, out)
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and math.isfinite(value):
        out[prefix] = float(value)


def _per_species_mapping(agent_id, *args, **kwargs):
    return f"{species_of(agent_id)}_policy"


def load_settings():
    """Copy and validate editable settings without modifying the source dictionaries."""
    env = dict(config_env)
    ppo = dict(config_ppo)
    for settings, names in (
        (
            env,
            (
                "predator_count",
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
    if not isinstance(ppo["vf_share_layers"], bool):
        raise ValueError("vf_share_layers must be a boolean")
    if ppo["checkpoint_every"] and not ppo["checkpoint_dir"]:
        raise ValueError("checkpoint_every requires checkpoint_dir")
    if ppo["arm"] not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)} (E is evolve.py)")
    ppo["beta"] = ARMS[ppo["arm"]]
    if not (module_checkpoint(ppo["deer_checkpoint"], "deer_policy") / "module_state.pkl").is_file():
        raise ValueError(f"deer_checkpoint has no deer policy: {ppo['deer_checkpoint']}")
    ppo["deer_checkpoint"] = os.path.abspath(os.path.expanduser(os.fspath(ppo["deer_checkpoint"])))
    if ppo["seed"] is not None and (isinstance(ppo["seed"], bool) or not isinstance(ppo["seed"], int)):
        raise ValueError("seed must be None or an integer")
    if not ppo["checkpoint_dir"]:
        raise ValueError("checkpoint_dir is required (the deer policy is copied next to it)")
    module_name = Path(__file__).resolve().parent.name
    timestamp = datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_name = f"{module_name}_{timestamp}"
    for name in ("checkpoint_dir", "tensorboard_dir"):
        if ppo[name]:
            ppo[name] = os.path.abspath(os.path.expanduser(os.fspath(ppo[name]).replace("{run_name}", run_name)))
    return env, SimpleNamespace(**ppo)


SOURCE_FILES = ("*.py", "config/*.py", "launchers/*.py", "README.md", "DESIGN.md")


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


ACTING = ("encoder.actor_encoder.", "pi.")  # the weights a module acts with


def _same(state, loaded, where, keys):
    for name in keys:
        if name not in state:
            raise RuntimeError(f"{where}: deer weight {name} missing")
        current = state[name]
        current = current.detach().cpu().numpy() if hasattr(current, "detach") else np.asarray(current)
        if not np.array_equal(current, np.asarray(loaded[name])):
            raise RuntimeError(f"{where}: deer weight {name} differs from the loaded deer policy")


def check_deer_frozen(algo, loaded):
    """The learner's deer weights equal the loaded deer policy, and so do the
    acting weights (actor encoder, policy head) of every remote env runner.
    Env runners receive only inference weights, so their unused critic keeps
    its initial values."""
    if algo.learner_group.is_local:
        state = algo.learner_group._learner.module["deer_policy"].get_state()
    else:
        state = algo.learner_group.foreach_learner(lambda learner: learner.module["deer_policy"].get_state())[0]
    _same(state, loaded, "learner", list(loaded))
    acting = [name for name in loaded if name.startswith(ACTING)]
    if not acting:
        raise RuntimeError("the loaded deer policy has no acting weights")
    states = algo.env_runner_group.foreach_env_runner(lambda runner: runner.module["deer_policy"].get_state(), local_env_runner=False)
    if not states:
        raise RuntimeError("no remote env runner to check the deer weights on")
    for k, state in enumerate(states):
        _same(state, loaded, f"env runner {k + 1}", acting)


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
    deer_dir = copy_deer_policy(ppo.deer_checkpoint, run_dir)
    loaded_deer = deer_state(deer_dir)

    policies, mapping_fn = POLICIES, _per_species_mapping

    config = (
        PPOConfig(algo_class=LocalPPO)
        .environment(
            ENV_NAME,
            env_config=env_config,
            # RLlib's check steps once with random actions and rejects the
            # final observation of an agent that died in that step (it is no
            # longer in env.agents, as RLlib's own convention requires), so
            # it fails whenever a deer happens to be caught in that step.
            # tests/test_environment.py covers the invariants instead.
            disable_env_checking=True,
        )
        .callbacks(CatchLogger)
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
            # The deer act with their loaded weights; the learner never
            # updates them.
            policies_to_train=["predator_policy"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "predator_policy": RLModuleSpec(
                        module_class=GeneBiasPPOTorchRLModule,
                        model_config=predator_model_config(env_config, ppo.arm, ppo.vf_share_layers),
                    ),
                    "deer_policy": RLModuleSpec(load_state_path=str(deer_dir)),
                }
            ),
            model_config=DefaultModelConfig(vf_share_layers=ppo.vf_share_layers),
        )
    )
    if ppo.seed is not None:
        config = config.debugging(seed=ppo.seed)

    algo = config.build_algo()
    check_deer_frozen(algo, loaded_deer)
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
            catches = {
                key: env_runner_stats.get(f"catches/{key}")
                for key in ("births_predator", "adults_caught", "adult_attempt_success", "gene_a_hunter_mean", "gene_a_young_mean", "gene_a_adult_mean")
            }
            catch_summary = "  ".join(f"{key}={value:.2f}" for key, value in catches.items() if isinstance(value, (int, float)))
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
                f"{catch_summary}  "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
            done = i + 1
            if ppo.checkpoint_every and done % ppo.checkpoint_every == 0 and done < ppo.iterations:
                path = os.path.join(ppo.checkpoint_dir, f"iter_{done:06d}")
                algo.save(path)
                save_evolution(algo, path)
                print(f"checkpoint saved to {path}", flush=True)
        if ppo.checkpoint_dir:
            algo.save(ppo.checkpoint_dir)
            save_evolution(algo, ppo.checkpoint_dir)
            check_deer_frozen(algo, loaded_deer)
            print(f"checkpoint saved to {ppo.checkpoint_dir}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        algo.stop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Training takes no command-line arguments. " "Edit config/config_env.py and config/config_ppo.py.")
    main()
