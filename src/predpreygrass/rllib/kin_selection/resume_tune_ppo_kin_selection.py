"""
Resume a previous PPO kin_selection experiment using Ray Tune (new API).

This script restores the most recent experiment directory created by
`tune_ppo_kin_selection.py` (prefix: "PPO_KIN_SELECTION_") under the
configured Ray results storage path and continues the run.

Notes
- Uses Tune's Tuner.restore to resume unfinished/errored trials.
- Keeps the original experiment's RunConfig (stopper, checkpointing, etc.).
- Environment is re-registered so the restored trainer can recreate it.
- To extend beyond the original stop criteria, start a new experiment using
  the original trainer script with a larger max_iters.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import ray
from ray.tune import Tuner
from ray.tune.registry import register_env

# Ensure the env is importable and registered before restoring
from predpreygrass.rllib.kin_selection.predpreygrass_rllib_env import PredPreyGrass


def env_creator(config):
    return PredPreyGrass(config)


def find_latest_experiment(
    storage_path: Path, prefix: str = "PPO_KIN_SELECTION_"
) -> Optional[Path]:
    if not storage_path.exists():
        return None
    # Find experiment dirs directly under storage_path matching the prefix
    cands = [
        p for p in storage_path.iterdir() if p.is_dir() and p.name.startswith(prefix)
    ]
    if not cands:
        return None
    # Sort by modification time (newest first)
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume PPO kin_selection experiment via Ray Tune"
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="~/Dropbox/02_marl_results/predpreygrass_results/ray_results/",
        help="Ray Tune storage path where experiments are saved.",
    )
    parser.add_argument(
        "--experiment-path",
        type=str,
        default=None,
        help=(
            "Path to a specific experiment directory to restore. If not provided, "
            "the latest 'PPO_KIN_SELECTION_*' under storage-path is used."
        ),
    )
    parser.add_argument(
        "--resume-unfinished",
        action="store_true",
        help="Resume unfinished trials (default: True)",
    )
    parser.add_argument(
        "--no-resume-unfinished",
        dest="resume_unfinished",
        action="store_false",
        help="Do not resume unfinished trials",
    )
    parser.set_defaults(resume_unfinished=True)
    parser.add_argument(
        "--resume-errored",
        action="store_true",
        help="Resume errored trials (default: True)",
    )
    parser.add_argument(
        "--no-resume-errored",
        dest="resume_errored",
        action="store_false",
        help="Do not resume errored trials",
    )
    parser.set_defaults(resume_errored=True)
    parser.add_argument(
        "--restart-errored",
        action="store_true",
        help="Restart errored trials from scratch instead of resuming (default: False)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    storage_path = Path(args.storage_path).expanduser().resolve()
    experiment_path = (
        Path(args.experiment_path).expanduser().resolve()
        if args.experiment_path
        else None
    )

    if experiment_path is None:
        experiment_path = find_latest_experiment(storage_path)
        if experiment_path is None:
            raise FileNotFoundError(
                f"No experiment found under: {storage_path}. "
                "Provide --experiment-path explicitly or launch a new run."
            )

    # Register env before restore so RLlib can recreate it by name
    register_env("PredPreyGrass", env_creator)

    print(f"[Resume] Storage path:   {storage_path}")
    print(f"[Resume] Experiment dir: {experiment_path}")
    print(
        f"[Resume] Options: resume_unfinished={args.resume_unfinished}, "
        f"resume_errored={args.resume_errored}, restart_errored={args.restart_errored}"
    )

    # Start Ray and restore the Tuner
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    tuner = Tuner.restore(
        str(experiment_path),
        resume_unfinished=args.resume_unfinished,
        resume_errored=args.resume_errored,
        restart_errored=args.restart_errored,
    )

    # Continue the experiment. Fit will block until completion per original stop criteria.
    result_grid = tuner.fit()

    # Print a concise summary
    try:
        best = result_grid.get_best_result()
        # Show some commonly helpful fields if present
        metrics = {
            k: best.metrics.get(k)
            for k in (
                "episode_reward_mean",
                "training_iteration",
                "episodes_total",
                "custom_metrics/helping_rate_mean",
                "custom_metrics/share_attempt_rate_mean",
                "score_pred",
            )
        }
        print(f"[Resume] Best trial: {best.path}")
        print(f"[Resume] Best checkpoint: {getattr(best, 'checkpoint', None)}")
        print(f"[Resume] Metrics: {metrics}")
    except Exception as e:
        print(f"[Resume] Completed. Could not compute best result summary: {e}")

    ray.shutdown()


if __name__ == "__main__":
    main()
