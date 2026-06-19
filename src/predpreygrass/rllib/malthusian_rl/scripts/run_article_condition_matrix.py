"""
Run all named article-task reconstruction conditions across seeds.

This is the highest-level harness for the paper-condition matrix encoded in
`config_article_protocol.py`.

Example:
    PYTHONPATH=src .conda/bin/python \
        src/predpreygrass/rllib/malthusian_rl/scripts/run_article_condition_matrix.py \
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from predpreygrass.rllib.malthusian_rl.config.config_article_protocol import (
    ARTICLE_EXPERIMENT_CONDITIONS,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _prepend_pythonpath(env: dict[str, str], repo_root: Path) -> None:
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"


def _condition_task_variant(condition: str) -> tuple[str, str]:
    config = ARTICLE_EXPERIMENT_CONDITIONS[condition]
    return str(config["task"]), str(config.get("variant", "biased"))


def run_condition_seed(
    *,
    condition: str,
    seed: int,
    results_dir: Path,
    max_iters: int | None,
    checkpoint_every: int | None,
    python_executable: str,
    dry_run: bool,
) -> None:
    repo_root = _repo_root()
    trainer = repo_root / "src/predpreygrass/rllib/malthusian_rl/tune_appo_article_exact.py"
    task, variant = _condition_task_variant(condition)
    env = os.environ.copy()
    _prepend_pythonpath(env, repo_root)
    env["ARTICLE_TASK"] = task
    env["ARTICLE_VARIANT"] = variant
    env["ARTICLE_CONDITION"] = condition
    env["ARTICLE_SEED"] = str(seed)
    env["ARTICLE_RESULTS_DIR"] = str(results_dir.expanduser())
    if max_iters is not None:
        env["ARTICLE_MAX_ITERS"] = str(max_iters)
    if checkpoint_every is not None:
        env["ARTICLE_CHECKPOINT_EVERY"] = str(checkpoint_every)

    command = [python_executable, str(trainer)]
    print(f"Running article matrix condition={condition} task={task} variant={variant} seed={seed}")
    if dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def run_matrix_evaluator(
    *,
    results_dir: Path,
    python_executable: str,
    dry_run: bool,
) -> None:
    repo_root = _repo_root()
    evaluator = repo_root / "src/predpreygrass/rllib/malthusian_rl/evaluate_exact_reproduction.py"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = results_dir / f"APPO_MALTHUSIAN_ARTICLE_MATRIX_summary_{timestamp}"
    env = os.environ.copy()
    _prepend_pythonpath(env, repo_root)
    command = [
        python_executable,
        str(evaluator),
        "--ray-results-dir",
        str(results_dir.expanduser()),
        "--experiment-glob",
        "APPO_MALTHUSIAN_ARTICLE_*_seed_*",
        "--output-dir",
        str(output_dir.expanduser()),
    ]
    print(f"Evaluating article condition matrix into {output_dir}")
    if dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def _parse_conditions(values: list[str] | None) -> list[str]:
    if not values:
        return sorted(ARTICLE_EXPERIMENT_CONDITIONS)
    unknown = sorted(set(values) - set(ARTICLE_EXPERIMENT_CONDITIONS))
    if unknown:
        known = ", ".join(sorted(ARTICLE_EXPERIMENT_CONDITIONS))
        raise ValueError(f"Unknown condition(s): {unknown}. Expected one or more of: {known}.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("~/Dropbox/02_marl_results/predpreygrass_results/ray_results/").expanduser(),
    )
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conditions = _parse_conditions(args.conditions)
    for condition in conditions:
        for seed in args.seeds:
            run_condition_seed(
                condition=condition,
                seed=seed,
                results_dir=args.results_dir,
                max_iters=args.max_iters,
                checkpoint_every=args.checkpoint_every,
                python_executable=args.python,
                dry_run=args.dry_run,
            )

    if not args.skip_eval:
        run_matrix_evaluator(
            results_dir=args.results_dir,
            python_executable=args.python,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
