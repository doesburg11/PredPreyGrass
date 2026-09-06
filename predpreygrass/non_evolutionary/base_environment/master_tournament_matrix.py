"""
Master Tournament cross-checkpoint matrix for base_environment.

Cross-evaluates every predator-policy checkpoint saved during a base_environment PPO
run against every prey-policy checkpoint (mixing checkpoints from different training
iterations into one episode), and records ecology/reward outcome metrics into an
NxN matrix rendered as a heatmap. This diagnoses whether predator-prey coevolution is
making genuine directional progress (a clean diagonal gradient) versus a Red-Queen-style
non-progressing oscillation (a flat, noisy matrix) -- the same diagnostic spirit as the
project's population-level drift-vs-selection tooling, applied to policy checkpoints.

This script is READ-ONLY with respect to training: it only loads already-flushed
checkpoint directories and never touches tune_ppo_base_environment.py,
predpreygrass_rllib_env.py, or a running training process. It deliberately does NOT
call ray.init() -- RLModule.from_checkpoint() and env stepping are plain
filesystem/torch/Python operations that don't need a Ray runtime, and starting a second
Ray instance risks contending for CPU/memory with a training run that may still be
active on the same machine.

Usage:
    # 1. Sanity-check checkpoint discovery and a full-sweep size/time estimate (loads nothing):
    python -m predpreygrass.non_evolutionary.base_environment.master_tournament_matrix \\
        --run-dir <path-to-trial-dir> --stride 10 --dry-run

    # 2. Cheap pilot (short episodes, one per cell) to calibrate real per-episode timing:
    python -m predpreygrass.non_evolutionary.base_environment.master_tournament_matrix \\
        --run-dir <path-to-trial-dir> --stride 10 --episodes-per-cell 1 --max-steps 200

    # 3. Full sweep once training has produced all checkpoints, using 30 worker processes:
    python -m predpreygrass.non_evolutionary.base_environment.master_tournament_matrix \\
        --run-dir <path-to-trial-dir> --workers 30

    # 4. Re-render the heatmap from a previous sweep's results without rerunning episodes:
    python -m predpreygrass.non_evolutionary.base_environment.master_tournament_matrix \\
        --replot <path-to-trial-dir>/master_tournament --metric prey_avg_reward
"""
from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.base_environment.config_env import config_env
# Reusing the proven checkpoint-loading/inference pattern (and the numpy._core.numeric
# checkpoint compatibility shim that runs as an import-time side effect of this module)
# rather than duplicating it.
from predpreygrass.non_evolutionary.base_environment.evaluate_ppo_from_checkpoint_debug import (
    policy_pi,
    policy_mapping_fn,
)

# --- External libraries ---
import argparse
import multiprocessing as mp
import os
import pickle
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ray.rllib.core.rl_module.rl_module import RLModule

VALID_METRICS = [
    "episode_length",
    "births_predator",
    "births_prey",
    "deaths_predator",
    "deaths_prey",
    "final_num_predators",
    "final_num_prey",
    "extinct_predator",
    "extinct_prey",
    "predator_avg_reward",
    "prey_avg_reward",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Path to the Tune trial directory containing checkpoint_* subdirectories "
             "(e.g. .../PPO_BASE_ENVIRONMENT_SEED42_.../PPO_PredPreyGrass_..._0_.../). "
             "Required unless --replot is given.",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Take every Nth checkpoint (by sorted index) on both axes.",
    )
    parser.add_argument(
        "--checkpoints-limit", type=int, default=None,
        help="Cap the number of checkpoints considered after striding.",
    )
    parser.add_argument(
        "--episodes-per-cell", type=int, default=3,
        help="Episodes run per (predator_checkpoint, prey_checkpoint) pair; averaged "
             "for the matrix, kept individually in results_long.csv.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override config_env's max_steps (default 1000) for a cheap pilot. "
             "The timing summary will flag when this shortens episodes relative to "
             "a full sweep.",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Sample actions from the policy distribution instead of the default "
             "deterministic argmax. Deterministic is recommended for a tournament "
             "matrix -- it removes policy-sampling noise so cell-to-cell variance "
             "reflects env stochasticity only.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed. Episode e within any cell uses seed + e; the same set of "
             "seeds is reused across every cell so matrix patterns reflect policy "
             "pairing, not seed luck.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to write results_long.csv / matrix_<metric>.npy / "
             "checkpoint_iterations.npy / heatmap_<metric>.png. "
             "Defaults to <run-dir>/master_tournament/ (or the --replot dir).",
    )
    parser.add_argument(
        "--metric", type=str, default="final_num_prey", choices=VALID_METRICS,
        help="Which field drives the primary heatmap.",
    )
    parser.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=True,
        help="Render + save the heatmap PNG at the end (use --no-plot to skip).",
    )
    parser.add_argument(
        "--replot", type=str, default=None,
        help="Path to a previous --output-dir. Skip the sweep entirely, reload its "
             "results_long.csv, and just regenerate the heatmap.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the discovered checkpoint index/iteration grid and a full-sweep "
             "size estimate, then exit. Loads nothing (no RLModules, no env).",
    )
    parser.add_argument(
        "--workers", type=int, default=30,
        help="Number of worker processes to evaluate cells in parallel (fork start "
             "method; Linux only). Each worker pins itself to a single torch thread "
             "so total CPU demand tracks this number directly instead of every "
             "worker also fanning out its own internal thread pool. Use 1 to fall "
             "back to plain sequential execution in a single subprocess.",
    )
    args = parser.parse_args()
    if not args.replot and not args.run_dir:
        parser.error("--run-dir is required unless --replot is given.")
    return args


def discover_checkpoints(run_dir: Path) -> List[Path]:
    """All checkpoint_* directories under run_dir, sorted by their trailing index."""
    ckpts = sorted(
        (p for p in run_dir.glob("checkpoint_*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_* directories found under {run_dir}")
    return ckpts


def apply_stride_and_limit(ckpts: List[Path], stride: int, limit: Optional[int]) -> List[Path]:
    subset = ckpts[::stride]
    if limit is not None:
        subset = subset[:limit]
    return subset


def read_training_iteration(ckpt_path: Path, checkpoint_frequency_fallback: int = 10) -> int:
    """Read the actual training_iteration this checkpoint was saved at, directly
    from algorithm_state.pkl, rather than assuming the checkpoint_frequency config
    value -- so this stays correct even if a future run changes that setting."""
    state_path = ckpt_path / "algorithm_state.pkl"
    try:
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        return int(state["training_iteration"])
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: this is a best-effort label
        fallback = (int(ckpt_path.name.rsplit("_", 1)[-1]) + 1) * checkpoint_frequency_fallback
        print(
            f"[warn] Could not read training_iteration from {state_path} ({exc!r}); "
            f"falling back to the (index+1)*{checkpoint_frequency_fallback} heuristic: {fallback}"
        )
        return fallback


def load_all_rl_modules(ckpt_paths: List[Path], policy_id: str) -> Dict[int, RLModule]:
    """Preload every checkpoint's RLModule for one policy, keyed by the (strided,
    0-indexed) position used throughout the matrix -- 2*N loads total, not 2*N*N."""
    modules = {}
    for idx, ckpt_path in enumerate(ckpt_paths):
        module_path = ckpt_path / "learner_group" / "learner" / "rl_module" / policy_id
        if not module_path.is_dir():
            raise FileNotFoundError(f"Expected RLModule directory not found: {module_path}")
        modules[idx] = RLModule.from_checkpoint(module_path)
    return modules


def run_one_episode(env: PredPreyGrass, predator_module: RLModule, prey_module: RLModule, seed: int, deterministic: bool) -> dict:
    """Run one episode with the given predator/prey checkpoints mixed together,
    then return the env's own ecology metrics plus predator/prey avg reward."""
    rl_modules = {"predator_policy": predator_module, "prey_policy": prey_module}
    observations, _ = env.reset(seed=seed)
    active_agents = list(observations.keys())
    reward_history: Dict[str, list] = defaultdict(list)

    while True:
        action_dict = {
            agent_id: policy_pi(
                observations[agent_id],
                rl_modules[policy_mapping_fn(agent_id)],
                deterministic=deterministic,
            )
            for agent_id in active_agents
        }
        observations, rewards, terminations, truncations, _ = env.step(action_dict)
        for agent_id, r in rewards.items():
            reward_history[agent_id].append(r)
        # env.agents itself lags one step behind: an agent that dies this step keeps
        # its entry in env.agents until the *next* step() call prunes it (a
        # _pending_removal mechanism), even though its position/energy dict entries
        # are already deleted. Deriving next step's active roster from
        # terminations/truncations (the standard MultiAgentEnv contract) instead of
        # env.agents avoids resubmitting an action for an already-fully-deleted agent.
        active_agents = [
            agent_id for agent_id in observations
            if not terminations.get(agent_id, False) and not truncations.get(agent_id, False)
        ]
        if terminations.get("__all__", False) or truncations.get("__all__", False):
            break
        if not active_agents:
            raise RuntimeError(
                "active_agents is empty but neither terminations['__all__'] nor "
                "truncations['__all__'] is set -- env.step({}) would loop forever."
            )

    metrics = dict(env._build_episode_training_metrics())

    predator_total = prey_total = 0.0
    predator_count = prey_count = 0
    for agent_id, rs in reward_history.items():
        total = sum(rs)
        if "predator" in agent_id:
            predator_total += total
            predator_count += 1
        elif "prey" in agent_id:
            prey_total += total
            prey_count += 1
    metrics["predator_avg_reward"] = predator_total / predator_count if predator_count else 0.0
    metrics["prey_avg_reward"] = prey_total / prey_count if prey_count else 0.0
    return metrics


# --- Parallel cell evaluation -----------------------------------------------
# These globals are populated in the MAIN process, before the worker Pool is
# created. With the "fork" start method (Linux default), each worker process is
# a copy-on-write snapshot of the parent taken at fork time, so every worker
# inherits the already-preloaded RLModules and the already-constructed env for
# free -- no reloading from disk per worker, no pickling heavy objects through
# IPC. Each worker's copy of `_ENV` is independently mutated from then on
# (separate address space after fork), so there's no cross-worker interference.
_ENV: Optional[PredPreyGrass] = None
_PREDATOR_MODULES: Optional[Dict[int, RLModule]] = None
_PREY_MODULES: Optional[Dict[int, RLModule]] = None
_ITERATIONS: Optional[List[int]] = None
_EPISODES_PER_CELL: int = 1
_SEED: int = 42
_DETERMINISTIC: bool = True


def _worker_init():
    """Pool initializer: run once per worker process. Without this, every one
    of the N worker processes would independently ask torch for its own
    multi-threaded intra-op thread pool (sized off the whole machine's core
    count), causing severe oversubscription -- N processes x many threads each
    fighting over the same physical cores. Pinning each worker to a single
    thread makes total CPU demand track --workers directly."""
    torch.set_num_threads(1)


def _run_cell(task: Tuple[int, int]) -> List[dict]:
    """Run all episodes for one (predator_ckpt_index, prey_ckpt_index) cell,
    using the module-level globals inherited via fork. Returns one row dict per
    episode (kept separate, not pre-averaged, so results_long.csv stays as rich
    as the sequential version's)."""
    i, j = task
    rows = []
    for ep in range(_EPISODES_PER_CELL):
        seed = _SEED + ep
        row = run_one_episode(_ENV, _PREDATOR_MODULES[i], _PREY_MODULES[j], seed, _DETERMINISTIC)
        row.update(
            predator_ckpt_index=i,
            prey_ckpt_index=j,
            predator_iteration=_ITERATIONS[i],
            prey_iteration=_ITERATIONS[j],
            episode_index=ep,
        )
        rows.append(row)
    return rows


def build_matrix(df: pd.DataFrame, metric: str, n: int) -> np.ndarray:
    pivot = df.groupby(["predator_ckpt_index", "prey_ckpt_index"])[metric].mean().unstack()
    pivot = pivot.reindex(index=range(n), columns=range(n))
    return pivot.to_numpy()


def plot_heatmap(matrix: np.ndarray, iterations: List[int], metric_name: str, run_label: str, output_path: Path):
    n = len(iterations)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("Prey policy training iteration")
    ax.set_ylabel("Predator policy training iteration")

    tick_stride = max(1, n // 15)
    ticks = list(range(0, n, tick_stride))
    labels = [str(iterations[t]) for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    fig.colorbar(im, ax=ax, label=metric_name)
    ax.set_title(f"Master Tournament Matrix ({metric_name})\n{run_label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved heatmap to {output_path}")


def replot(args):
    replot_dir = Path(args.replot)
    csv_path = replot_dir / "results_long.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"No results_long.csv found in {replot_dir}")
    df = pd.read_csv(csv_path)

    iteration_map = (
        df.drop_duplicates("predator_ckpt_index")
        .set_index("predator_ckpt_index")["predator_iteration"]
        .to_dict()
    )
    n = len(iteration_map)
    iterations = [iteration_map[i] for i in sorted(iteration_map)]

    matrix = build_matrix(df, args.metric, n)
    output_dir = Path(args.output_dir) if args.output_dir else replot_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(matrix, iterations, args.metric, replot_dir.name, output_dir / f"heatmap_{args.metric}.png")


def main():
    args = parse_args()

    if args.replot:
        replot(args)
        return

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dir does not exist or is not a directory: {run_dir}")

    all_ckpts = discover_checkpoints(run_dir)
    strided_ckpts = apply_stride_and_limit(all_ckpts, args.stride, args.checkpoints_limit)
    n = len(strided_ckpts)
    iterations = [read_training_iteration(c) for c in strided_ckpts]

    print(f"[discover] {len(all_ckpts)} checkpoints found under {run_dir}")
    print(f"[discover] using {n} checkpoints after --stride {args.stride}"
          + (f" and --checkpoints-limit {args.checkpoints_limit}" if args.checkpoints_limit else ""))
    print("[discover] checkpoint indices -> training iterations: "
          + ", ".join(f"{i}->{it}" for i, it in enumerate(iterations)))

    episodes_per_cell = args.episodes_per_cell
    full_cells = len(all_ckpts) ** 2
    full_episodes = full_cells * episodes_per_cell
    pilot_cells = n * n
    pilot_episodes = pilot_cells * episodes_per_cell
    print(f"[discover] this run will evaluate {n}x{n}={pilot_cells} cells "
          f"x {episodes_per_cell} episodes/cell = {pilot_episodes} episodes")
    print(f"[discover] a full sweep over all {len(all_ckpts)} checkpoints would be "
          f"{len(all_ckpts)}x{len(all_ckpts)}={full_cells} cells "
          f"x {episodes_per_cell} episodes/cell = {full_episodes} episodes")

    if args.dry_run:
        print("[dry-run] stopping before loading any RLModules or running episodes.")
        return

    deterministic = not args.stochastic

    predator_modules = load_all_rl_modules(strided_ckpts, "predator_policy")
    prey_modules = load_all_rl_modules(strided_ckpts, "prey_policy")
    print(f"[load] preloaded {n} predator_policy modules and {n} prey_policy modules")

    env_config = dict(config_env)
    if args.max_steps is not None:
        env_config["max_steps"] = args.max_steps
    env = PredPreyGrass(env_config)

    # This process's own torch thread pool is irrelevant once workers fork from
    # it (each worker pins itself to 1 thread in _worker_init), but pin it here
    # too in case --workers 1 is used (no pool overhead, everything runs via
    # _run_cell directly in-process below).
    torch.set_num_threads(1)

    global _ENV, _PREDATOR_MODULES, _PREY_MODULES, _ITERATIONS, _EPISODES_PER_CELL, _SEED, _DETERMINISTIC
    _ENV = env
    _PREDATOR_MODULES = predator_modules
    _PREY_MODULES = prey_modules
    _ITERATIONS = iterations
    _EPISODES_PER_CELL = episodes_per_cell
    _SEED = args.seed
    _DETERMINISTIC = deterministic

    tasks = [(i, j) for i in range(n) for j in range(n)]
    total_episodes = n * n * episodes_per_cell
    num_workers = max(1, args.workers)
    cpu_count = os.cpu_count() or 1
    if num_workers > cpu_count:
        print(f"[warn] --workers {num_workers} exceeds the {cpu_count} logical CPUs detected "
              f"on this machine; processes will contend for cores rather than run in parallel.")
    print(f"[sweep] evaluating {len(tasks)} cells ({total_episodes} episodes total) "
          f"with {num_workers} worker process(es)")

    results: List[dict] = []
    sweep_start = time.perf_counter()
    done_cells = 0
    progress_stride = max(1, len(tasks) // 20)  # ~20 progress lines regardless of matrix size

    if num_workers == 1:
        for task in tasks:
            results.extend(_run_cell(task))
            done_cells += 1
            if done_cells % progress_stride == 0 or done_cells == len(tasks):
                elapsed = time.perf_counter() - sweep_start
                print(f"[sweep] {done_cells}/{len(tasks)} cells done ({elapsed:.1f}s elapsed)")
    else:
        # fork (not the platform default on all OSes, but this repo targets Linux):
        # workers inherit the already-preloaded modules/env from this process's
        # memory at fork time rather than re-loading anything from disk.
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=num_workers, initializer=_worker_init) as pool:
            for cell_rows in pool.imap_unordered(_run_cell, tasks):
                results.extend(cell_rows)
                done_cells += 1
                if done_cells % progress_stride == 0 or done_cells == len(tasks):
                    elapsed = time.perf_counter() - sweep_start
                    print(f"[sweep] {done_cells}/{len(tasks)} cells done ({elapsed:.1f}s elapsed)")

    total_wall = time.perf_counter() - sweep_start
    implied_per_episode = total_wall / total_episodes if total_episodes else 0.0
    effective_max_steps = env_config["max_steps"]
    print(f"[timing] {total_episodes} episodes across {len(tasks)} cells finished in "
          f"{timedelta(seconds=int(total_wall))} using {num_workers} worker(s) "
          f"(max_steps={effective_max_steps}); implied {implied_per_episode:.3f}s/episode "
          f"at this parallelism level")
    if args.max_steps is not None:
        print(f"[timing] NOTE: this run used --max-steps {args.max_steps}, shortened from "
              f"config_env's default {config_env['max_steps']}. A full sweep at the default "
              "max_steps will likely take proportionally longer per episode than the "
              "extrapolation below assumes.")
    estimated_full_seconds = full_episodes * implied_per_episode
    print(f"[timing] full-sweep extrapolation using all {len(all_ckpts)} discovered checkpoints "
          f"at this same --workers {num_workers} level "
          f"({full_cells} cells x {episodes_per_cell} episodes/cell = {full_episodes} episodes): "
          f"~{timedelta(seconds=int(estimated_full_seconds))} (HH:MM:SS)")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "master_tournament"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = output_dir / "results_long.csv"
    df.to_csv(csv_path, index=False)
    print(f"[save] wrote {len(df)} rows to {csv_path}")

    matrix = build_matrix(df, args.metric, n)
    matrix_path = output_dir / f"matrix_{args.metric}.npy"
    np.save(matrix_path, matrix)
    np.save(output_dir / "checkpoint_iterations.npy", np.array(iterations))
    print(f"[save] wrote {n}x{n} matrix for metric '{args.metric}' to {matrix_path}")

    if args.plot:
        plot_heatmap(matrix, iterations, args.metric, run_dir.name, output_dir / f"heatmap_{args.metric}.png")


if __name__ == "__main__":
    main()
