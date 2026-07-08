"""
Run the frozen exact Malthusian protocol across multiple seeds.

Example:
    PYTHONPATH=src .conda/bin/python \
        predpreygrass/malthusian_rl/scripts/run_exact_reproduction_seeds.py \
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from predpreygrass.non_evolutionary.malthusian_rl.config.config_paper_protocol import (
    DEFAULT_PAPER_PROTOCOL_VARIANT,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _prepend_pythonpath(env: dict[str, str], repo_root: Path) -> None:
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"


def run_seed(
    *,
    seed: int,
    variant: str,
    results_dir: Path,
    max_iters: int | None,
    checkpoint_every: int | None,
    python_executable: str,
    dry_run: bool,
) -> None:
    repo_root = _repo_root()
    trainer = repo_root / "predpreygrass/malthusian_rl/tune_appo_malthusian_exact.py"
    env = os.environ.copy()
    _prepend_pythonpath(env, repo_root)
    env["EXACT_SEED"] = str(seed)
    env["EXACT_PROTOCOL_VARIANT"] = variant
    env["EXACT_RESULTS_DIR"] = str(results_dir.expanduser())
    if max_iters is not None:
        env["EXACT_MAX_ITERS"] = str(max_iters)
    if checkpoint_every is not None:
        env["EXACT_CHECKPOINT_EVERY"] = str(checkpoint_every)

    command = [python_executable, str(trainer)]
    print(f"Running exact protocol seed={seed} variant={variant}")
    if dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def run_evaluator(
    *,
    variant: str,
    results_dir: Path,
    output_dir: Path,
    python_executable: str,
    dry_run: bool,
) -> None:
    repo_root = _repo_root()
    evaluator = repo_root / "predpreygrass/malthusian_rl/evaluate_exact_reproduction.py"
    env = os.environ.copy()
    _prepend_pythonpath(env, repo_root)
    command = [
        python_executable,
        str(evaluator),
        "--ray-results-dir",
        str(results_dir.expanduser()),
        "--experiment-glob",
        f"APPO_MALTHUSIAN_EXACT_{variant}_seed_*",
        "--output-dir",
        str(output_dir.expanduser()),
    ]
    print(f"Evaluating exact protocol runs into {output_dir}")
    if dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variant", default=DEFAULT_PAPER_PROTOCOL_VARIANT)
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
            seed=seed,
            variant=args.variant,
            results_dir=args.results_dir,
            max_iters=args.max_iters,
            checkpoint_every=args.checkpoint_every,
            python_executable=args.python,
            dry_run=args.dry_run,
        )

    if not args.skip_eval:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = args.results_dir / f"APPO_MALTHUSIAN_EXACT_{args.variant}_summary_{timestamp}"
        run_evaluator(
            variant=args.variant,
            results_dir=args.results_dir,
            output_dir=output_dir,
            python_executable=args.python,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

