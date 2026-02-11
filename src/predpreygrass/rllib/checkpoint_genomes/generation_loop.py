"""
Sketch generation loop: train -> evaluate -> select -> breed -> repeat.

This script is intentionally simple and uses subprocess calls to:
  - tune_ppo.py for training (nurture)
  - evaluate_ppo_from_checkpoint_multi_runs.py for scoring (selection)
  - make_child_checkpoint.py for breeding (nature)

Run without arguments to load defaults from genome_config.json in this folder.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


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

from predpreygrass.rllib.checkpoint_genomes.make_child_checkpoint import make_child_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Checkpoint-genome generation loop.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to genome_config.json (defaults to local genome_config.json).",
    )
    parser.add_argument("--seed-checkpoint", default=None, help="Seed checkpoint directory.")
    parser.add_argument("--out-root", default=None, help="Output root directory for generations.")
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--elite", type=int, default=None, help="Elites copied to next generation.")
    parser.add_argument("--alpha", type=float, default=None, help="Crossover mix weight.")
    parser.add_argument("--sigma", type=float, default=None, help="Mutation stddev.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for selection.")
    parser.add_argument(
        "--fitness-key",
        default=None,
        help="Dot path into defection_metrics_aggregate.json.",
    )
    parser.add_argument(
        "--train-script",
        default=None,
        help="Path to tune_ppo.py.",
    )
    parser.add_argument(
        "--eval-script",
        default=None,
        help="Path to evaluation script.",
    )
    parser.add_argument("--skip-train", action="store_true", default=None, help="Skip training step.")
    parser.add_argument("--skip-eval", action="store_true", default=None, help="Skip evaluation step.")
    return parser.parse_args()


def _resolve(path_str: str, base_dir: Optional[Path] = None) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")
        return resolved

    rel_hint = path_str.startswith(("./", "../"))
    candidates = []
    if rel_hint and base_dir is not None:
        candidates.append((base_dir / path).expanduser())
    candidates.append((Path.cwd() / path).expanduser())
    if base_dir is not None and not rel_hint:
        candidates.append((base_dir / path).expanduser())

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Path not found: {path_str}")


def _resolve_output_dir(path_str: str, base_dir: Optional[Path]) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path_str.startswith(("./", "../")) and base_dir is not None:
        return (base_dir / path).expanduser().resolve()
    return (Path.cwd() / path).expanduser().resolve()


def _find_latest_checkpoint(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [
        p for p in root.rglob("checkpoint_*") if (p / "rllib_checkpoint.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _get_by_path(data: dict, dotted: str, default: float = 0.0) -> float:
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


def _merge_config(base: dict, args: argparse.Namespace) -> dict:
    merged = dict(base)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            merged[key] = value
    return merged


def _require_config_value(cfg: dict, key: str) -> str:
    value = cfg.get(key)
    if not value:
        raise ValueError(f"Missing required config value: {key}")
    return value


def _train_from_checkpoint(
    checkpoint_dir: Path, genome_dir: Path, train_script: Path, python_exe: str
) -> Path:
    train_root = genome_dir / "train"
    train_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["RESTORE_CHECKPOINT"] = str(checkpoint_dir)
    env["TRAINED_EXAMPLE_DIR"] = str(train_root)
    subprocess.run([python_exe, str(train_script)], check=True, env=env)

    latest = _find_latest_checkpoint(train_root / "ray_results")
    if latest is None:
        raise RuntimeError(f"No checkpoint found under {train_root / 'ray_results'}")
    return latest


def _evaluate_genome(
    genome_dir: Path, eval_script: Path, python_exe: str, fitness_key: str
) -> float:
    env = os.environ.copy()
    env["TRAINED_EXAMPLE_DIR"] = str(genome_dir)
    subprocess.run([python_exe, str(eval_script)], check=True, env=env)

    runs_dir = genome_dir / "eval" / "runs"
    if not runs_dir.exists():
        return 0.0
    eval_dirs = sorted(runs_dir.glob("eval_multiple_runs_SEXUAL_REPRODUCTION_*"), key=lambda p: p.stat().st_mtime)
    if not eval_dirs:
        return 0.0

    metrics_path = eval_dirs[-1] / "summary_data" / "defection_metrics_aggregate.json"
    if not metrics_path.exists():
        return 0.0

    with metrics_path.open() as f:
        metrics = json.load(f)
    fitness = _get_by_path(metrics, fitness_key, default=0.0)

    fitness_path = genome_dir / "fitness.json"
    fitness_path.write_text(json.dumps({"fitness": fitness, "fitness_key": fitness_key}, indent=2))
    return fitness


def _sync_checkpoint(dest: Path, src: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _init_population(seed_ckpt: Path, out_root: Path, pop_size: int, sigma: float) -> list[Path]:
    gen_dir = out_root / "gen_000"
    gen_dir.mkdir(parents=True, exist_ok=True)
    genomes = []
    for i in range(pop_size):
        genome_dir = gen_dir / f"genome_{i:03d}"
        ckpt_dir = genome_dir / "checkpoint_000000"
        genome_dir.mkdir(parents=True, exist_ok=True)
        make_child_checkpoint(
            seed_ckpt,
            seed_ckpt,
            ckpt_dir,
            alpha=0.5,
            sigma=sigma,
            seed=None,
            policies=None,
            overwrite=True,
        )
        genomes.append(genome_dir)
    return genomes


def main():
    args = parse_args()
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else Path(__file__).with_name("genome_config.json")
    )
    cfg = _merge_config(_load_config(config_path), args)
    base_dir = config_path.parent

    rng = random.Random(cfg.get("seed"))

    seed_ckpt = _resolve(_require_config_value(cfg, "seed_checkpoint"), base_dir)
    out_root = _resolve_output_dir(_require_config_value(cfg, "out_root"), base_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    train_script = _resolve(cfg.get("train_script") or str(Path(__file__).parent / "tune_ppo.py"), base_dir)
    eval_script = _resolve(
        cfg.get("eval_script") or str(Path(__file__).parent / "evaluate_ppo_from_checkpoint_multi_runs.py"),
        base_dir,
    )
    python_exe = sys.executable

    population_size = int(cfg.get("population_size", 6))
    generations = int(cfg.get("generations", 3))
    elite = int(cfg.get("elite", 2))
    alpha = float(cfg.get("alpha", 0.5))
    sigma = float(cfg.get("sigma", 0.01))
    fitness_key = cfg.get("fitness_key", "capture_outcomes.coop_capture_rate")
    skip_train = bool(cfg.get("skip_train", False))
    skip_eval = bool(cfg.get("skip_eval", False))

    genomes = _init_population(seed_ckpt, out_root, population_size, sigma)

    for gen in range(generations):
        gen_dir = out_root / f"gen_{gen:03d}"
        if gen > 0:
            gen_dir.mkdir(parents=True, exist_ok=True)

        scored = []
        for genome_dir in genomes:
            ckpt_dir = genome_dir / "checkpoint_000000"
            if not skip_train:
                trained_ckpt = _train_from_checkpoint(ckpt_dir, genome_dir, train_script, python_exe)
                _sync_checkpoint(ckpt_dir, trained_ckpt)

            if skip_eval:
                fitness = 0.0
            else:
                fitness = _evaluate_genome(genome_dir, eval_script, python_exe, fitness_key)
            scored.append((genome_dir, fitness))

        scored.sort(key=lambda item: item[1], reverse=True)
        elites = scored[: max(elite, 0)]

        next_gen_dir = out_root / f"gen_{gen + 1:03d}"
        next_gen_dir.mkdir(parents=True, exist_ok=True)
        next_genomes = []

        for i, (genome_dir, _) in enumerate(elites):
            new_genome_dir = next_gen_dir / f"genome_{i:03d}"
            new_ckpt_dir = new_genome_dir / "checkpoint_000000"
            new_genome_dir.mkdir(parents=True, exist_ok=True)
            _sync_checkpoint(new_ckpt_dir, genome_dir / "checkpoint_000000")
            next_genomes.append(new_genome_dir)

        weights = [max(score, 0.0) + 1e-6 for _, score in scored]
        while len(next_genomes) < population_size:
            parent_a, parent_b = rng.choices(scored, weights=weights, k=2)
            idx = len(next_genomes)
            child_genome_dir = next_gen_dir / f"genome_{idx:03d}"
            child_ckpt_dir = child_genome_dir / "checkpoint_000000"
            child_genome_dir.mkdir(parents=True, exist_ok=True)
            make_child_checkpoint(
                parent_a[0] / "checkpoint_000000",
                parent_b[0] / "checkpoint_000000",
                child_ckpt_dir,
                alpha=alpha,
                sigma=sigma,
                seed=None,
                policies=None,
                overwrite=True,
            )
            next_genomes.append(child_genome_dir)

        genomes = next_genomes


if __name__ == "__main__":
    main()
