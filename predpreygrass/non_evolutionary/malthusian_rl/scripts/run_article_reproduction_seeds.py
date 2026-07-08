"""
Run article-task reconstruction experiments across seeds.

Examples:
    PYTHONPATH=src .conda/bin/python \
        predpreygrass/malthusian_rl/scripts/run_article_reproduction_seeds.py \
        --task allelopathy --variant biased \
        --condition allelopathy_biased_heterogeneous_dynamic --seeds 0 1 2

    PYTHONPATH=src .conda/bin/python \
        predpreygrass/malthusian_rl/scripts/run_article_reproduction_seeds.py \
        --task clamity --condition clamity_dynamic_population --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _prepend_pythonpath(env: dict[str, str], repo_root: Path) -> None:
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"


def run_seed(
    *,
    task: str,
    variant: str,
    condition: str | None,
    seed: int,
    results_dir: Path,
    max_iters: int | None,
    checkpoint_every: int | None,
    python_executable: str,
    dry_run: bool,
) -> None:
    repo_root = _repo_root()
    trainer = repo_root / "predpreygrass/malthusian_rl/tune_appo_article_exact.py"
    env = os.environ.copy()
    _prepend_pythonpath(env, repo_root)
    env["ARTICLE_TASK"] = task
    env["ARTICLE_VARIANT"] = variant
    if condition:
        env["ARTICLE_CONDITION"] = condition
    env["ARTICLE_SEED"] = str(seed)
    env["ARTICLE_RESULTS_DIR"] = str(results_dir.expanduser())
    if max_iters is not None:
        env["ARTICLE_MAX_ITERS"] = str(max_iters)
    if checkpoint_every is not None:
        env["ARTICLE_CHECKPOINT_EVERY"] = str(checkpoint_every)

    command = [python_executable, str(trainer)]
    print(f"Running article task={task} variant={variant} condition={condition or 'default'} seed={seed}")
    if dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def run_evaluator(
    *,
    task: str,
    variant: str,
    condition: str | None,
    results_dir: Path,
    python_executable: str,
    dry_run: bool,
) -> None:
    repo_root = _repo_root()
    evaluator = repo_root / "predpreygrass/malthusian_rl/evaluate_exact_reproduction.py"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = results_dir / f"APPO_MALTHUSIAN_ARTICLE_{task}_{variant}_{condition or 'all'}_summary_{timestamp}"
    env = os.environ.copy()
    _prepend_pythonpath(env, repo_root)
    command = [
        python_executable,
        str(evaluator),
        "--ray-results-dir",
        str(results_dir.expanduser()),
        "--experiment-glob",
        f"APPO_MALTHUSIAN_ARTICLE_{task}_{variant}_{condition or '*'}_seed_*",
        "--output-dir",
        str(output_dir.expanduser()),
    ]
    print(f"Evaluating article runs into {output_dir}")
    if dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["allelopathy", "clamity"], default="allelopathy")
    parser.add_argument("--variant", choices=["biased", "unbiased"], default="biased")
    parser.add_argument("--condition", default=None)
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

    for seed in args.seeds:
        run_seed(
            task=args.task,
            variant=args.variant,
            condition=args.condition,
            seed=seed,
            results_dir=args.results_dir,
            max_iters=args.max_iters,
            checkpoint_every=args.checkpoint_every,
            python_executable=args.python,
            dry_run=args.dry_run,
        )

    if not args.skip_eval:
        run_evaluator(
            task=args.task,
            variant=args.variant,
            condition=args.condition,
            results_dir=args.results_dir,
            python_executable=args.python,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
