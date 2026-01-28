#!/usr/bin/env python3
"""Promote a single Ray run checkpoint + snapshots into rllib/trained_examples.

Edit the constants below before running. This script takes no arguments.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import shutil


# === USER-EDITABLE CONSTANTS ===
EXAMPLE_NAME = "EXAMPLE_NAME"
TRIAL_DIR_NAME = "PPO_PredPreyGrass_..."
CHECKPOINT_DIR_NAME = "checkpoint_000000"
EVAL_DIR_NAME = "eval_multiple_runs_..."  # directory name under the checkpoint dir
OVERWRITE = False


def _find_parent_named(path: Path, name: str) -> Path:
    for parent in path.parents:
        if parent.name == name:
            return parent
    raise RuntimeError(f"Could not find parent named '{name}' for {path}")


def _copytree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_tensorboard_files(trial_dir: Path, dest_dir: Path) -> None:
    tb_dir = dest_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    for path in trial_dir.rglob("events.out.tfevents*"):
        if any(part.startswith("checkpoint_") for part in path.parts):
            continue
        rel = path.relative_to(trial_dir)
        dest = tb_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

    log_dir = dest_dir / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_files = [
        "params.json",
        "params.pkl",
        "progress.csv",
        "result.json",
        "result.pkl",
        "trial_state.json",
        "experiment_state.json",
    ]
    for name in log_files:
        candidate = trial_dir / name
        if candidate.is_file():
            shutil.copy2(candidate, log_dir / name)


def _write_wrapper_scripts(example_dir: Path) -> None:
    def _write(script_name: str, target_script: str) -> None:
        content = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "EXAMPLE_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
            "export TRAINED_EXAMPLE_DIR=\"${EXAMPLE_DIR}\"\n"
            "export PYTHONPATH=\"${EXAMPLE_DIR}/SOURCE_CODE:${EXAMPLE_DIR}/eval/SOURCE_CODE:${PYTHONPATH:-}\"\n"
            f"python \"${{EXAMPLE_DIR}}/eval/SOURCE_CODE/{target_script}\"\n"
        )
        script_path = example_dir / script_name
        script_path.write_text(content)
        script_path.chmod(0o755)

    _write("run_debug.sh", "evaluate_ppo_from_checkpoint_debug.py")
    _write("run_multi.sh", "evaluate_ppo_from_checkpoint_multi_runs.py")


def main() -> None:
    script_path = Path(__file__).resolve()
    promote_dir = script_path.parent
    experiment_dir = promote_dir.parent
    rllib_dir = _find_parent_named(experiment_dir, "rllib")
    trained_examples_root = rllib_dir / "trained_examples"

    trial_dir = experiment_dir / TRIAL_DIR_NAME
    checkpoint_dir = trial_dir / CHECKPOINT_DIR_NAME
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    eval_dir = checkpoint_dir / EVAL_DIR_NAME
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

    example_dir = trained_examples_root / EXAMPLE_NAME
    example_dir.mkdir(parents=True, exist_ok=True)

    # Copy training snapshots
    training_source = experiment_dir / "SOURCE_CODE"
    training_config = experiment_dir / "CONFIG"
    if training_source.is_dir():
        _copytree(training_source, example_dir / "SOURCE_CODE", OVERWRITE)
    if training_config.is_dir():
        _copytree(training_config, example_dir / "CONFIG", OVERWRITE)

    # Copy selected checkpoint
    _copytree(checkpoint_dir, example_dir / "checkpoint", OVERWRITE)

    # Copy evaluation output (includes eval SOURCE_CODE + CONFIG + visuals)
    _copytree(eval_dir, example_dir / "eval", OVERWRITE)

    # Copy tensorboard/event files and trial metadata
    _copy_tensorboard_files(trial_dir, example_dir)

    _write_wrapper_scripts(example_dir)

    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "example_name": EXAMPLE_NAME,
        "experiment_dir": str(experiment_dir),
        "trial_dir": str(trial_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "eval_dir": str(eval_dir),
        "source_code_dir": "SOURCE_CODE",
        "config_dir": "CONFIG",
        "checkpoint_dest": "checkpoint",
        "eval_dest": "eval",
    }
    (example_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Promoted run to: {example_dir}")


if __name__ == "__main__":
    main()
