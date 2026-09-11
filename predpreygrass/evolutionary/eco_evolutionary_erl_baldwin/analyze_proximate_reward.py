"""Proximate-vs-ultimate reward analysis (Singh, Lewis, Barto & Sorg 2010).

Their claim: evolution optimizes a REWARD function for reproductive fitness, but
the reward it finds need not resemble fitness itself -- a dense, always-available
proximate substitute (e.g. "food reward") can serve evolution's purpose better
than trying to track the sparse, delayed ultimate criterion ("expected number of
descendants") directly.

This script tests that against real ERL Baldwin (Ackley & Littman 1991) data: each
agent's `genome.eval_weights` IS its evolved, lifetime-fixed reward function (the
"evaluation network"; see world.py's module docstring), and `offspring_count` is
its realized fitness. If evolution converged on a proximate substitute, the
evolved weights should load heavily on dense per-step channels (`health_norm`,
`energy_norm`) rather than on whichever channel correlates best with actual
offspring_count -- those need not be the same channel.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.analyze_proximate_reward \\
        ~/simulation_results/erl_results/ERL_BALDWIN_2026-09-11_.../lineage_fitness.csv

    # or point at a directory to combine every lineage_fitness.csv found under it:
    python -m ...analyze_proximate_reward ~/simulation_results/erl_results/erl_full_study/B_seed1

    # include a run's still-alive agents (right-censored) via its final checkpoint --
    # lineage_fitness.csv deliberately never logs survivors itself (see
    # run_erl_simulation.py: a run that gets --resume-from'd again later has no way
    # to know a given invocation's survivors are "final", so persisting them there
    # would risk duplicate/contradictory rows for the same agent across resumes):
    python -m ...analyze_proximate_reward .../lineage_fitness.csv \\
        --checkpoint .../checkpoints/checkpoint_step_1000000.pkl
"""

import argparse
import csv
from pathlib import Path

import numpy as np

# Matches world.py's OBS_DIM comment; index 7 only exists under S/ERLS (obs_dim=8).
CHANNEL_NAMES = ["visual_N", "visual_S", "visual_E", "visual_W", "in_tree", "health_norm", "energy_norm", "alarm_signal"]

Row = tuple[list[float], float, float]  # (eval_weights, offspring_count, lifespan)


def _find_csvs(paths: list[str]) -> list[Path]:
    found = []
    for p in paths:
        path = Path(p).expanduser()
        if path.is_dir():
            found.extend(sorted(path.rglob("lineage_fitness.csv")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(path)
    if not found:
        raise FileNotFoundError(f"No lineage_fitness.csv found under: {paths}")
    return found


def _check_dim(obs_dim: int | None, this_dim: int, source: Path) -> int:
    if obs_dim is not None and this_dim != obs_dim:
        raise ValueError(
            f"{source} has obs_dim={this_dim}, but earlier data had obs_dim={obs_dim} "
            "-- don't mix S/ERLS runs (obs_dim=8) with other strategies (obs_dim=7)."
        )
    return this_dim


def _rows_from_csv(path: Path) -> tuple[list[Row], int]:
    rows: list[Row] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        weight_cols = sorted(
            (k for k in reader.fieldnames if k.startswith("eval_weight_")),
            key=lambda k: int(k.split("_")[-1]),
        )
        for row in reader:
            rows.append(
                (
                    [float(row[c]) for c in weight_cols],
                    float(row["offspring_count"]),
                    float(row["lifespan"]),
                )
            )
    return rows, len(weight_cols)


def _rows_from_checkpoint(path: Path) -> tuple[list[Row], int]:
    """Currently-alive agents in a checkpoint, as right-censored lineage rows --
    see metrics.lineage_record's `censored` semantics. lifespan is a lower bound
    (the agent may go on to live/reproduce further; only real deaths in
    lineage_fitness.csv are final)."""
    from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.checkpoint import load_checkpoint

    world = load_checkpoint(path)
    rows: list[Row] = [
        (list(a.genome.eval_weights), float(a.offspring_count), float(world.current_step - a.born_step))
        for a in world.agents
        if a.alive
    ]
    return rows, world.obs_dim


def _load(
    csv_paths: list[Path], checkpoint_paths: list[Path]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Returns (eval_weights [n, obs_dim], offspring_count [n], lifespan [n], obs_dim, n_censored)."""
    all_rows: list[Row] = []
    obs_dim = None
    n_censored = 0
    for path in csv_paths:
        rows, this_dim = _rows_from_csv(path)
        obs_dim = _check_dim(obs_dim, this_dim, path)
        all_rows.extend(rows)
    for path in checkpoint_paths:
        rows, this_dim = _rows_from_checkpoint(path)
        obs_dim = _check_dim(obs_dim, this_dim, path)
        all_rows.extend(rows)
        n_censored += len(rows)
    if not all_rows:
        raise ValueError(f"No data rows found in: {csv_paths + checkpoint_paths}")
    weights = np.array([r[0] for r in all_rows])
    offspring = np.array([r[1] for r in all_rows])
    lifespan = np.array([r[2] for r in all_rows])
    return weights, offspring, lifespan, obs_dim, n_censored


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ols_standardized(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coefficients of y on z-scored columns of X (plus intercept), via least squares."""
    Xz = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    design = np.column_stack([Xz, np.ones(len(y))])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefs[:-1]  # drop intercept


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths", nargs="*", default=[],
        help="lineage_fitness.csv file(s) or directories to search recursively.",
    )
    parser.add_argument(
        "--checkpoint", nargs="*", default=[],
        help="checkpoint_step_*.pkl file(s) whose currently-alive agents are included "
             "as right-censored rows (see this script's module docstring for why "
             "lineage_fitness.csv never logs survivors itself).",
    )
    args = parser.parse_args()
    if not args.paths and not args.checkpoint:
        parser.error("Provide at least one lineage_fitness.csv path/directory, or --checkpoint, or both.")

    csv_paths = _find_csvs(args.paths) if args.paths else []
    checkpoint_paths = [Path(p).expanduser() for p in args.checkpoint]
    weights, offspring, lifespan, obs_dim, n_censored = _load(csv_paths, checkpoint_paths)
    rate = offspring / np.maximum(lifespan, 1.0)
    names = CHANNEL_NAMES[:obs_dim]

    censored_note = f" + {n_censored} right-censored (still alive) from {len(checkpoint_paths)} checkpoint(s)" if checkpoint_paths else ""
    print(f"Loaded {len(offspring)} agent lifetimes from {len(csv_paths)} CSV file(s){censored_note}, obs_dim={obs_dim}.")
    print(f"offspring_count: mean={offspring.mean():.3f}, max={offspring.max():.0f}")
    print()

    corr_offspring = [_pearson(weights[:, i], offspring) for i in range(obs_dim)]
    corr_rate = [_pearson(weights[:, i], rate) for i in range(obs_dim)]
    ols_offspring = _ols_standardized(weights, offspring)
    ols_rate = _ols_standardized(weights, rate)
    evolved_magnitude = np.abs(weights).mean(axis=0)

    order = sorted(range(obs_dim), key=lambda i: -abs(corr_offspring[i]))

    header = f"{'channel':<14}{'evolved |w|':>12}{'corr(offspr)':>14}{'corr(rate)':>12}{'ols(offspr)':>13}{'ols(rate)':>11}"
    print(header)
    print("-" * len(header))
    for i in order:
        print(
            f"{names[i]:<14}{evolved_magnitude[i]:>12.3f}{corr_offspring[i]:>14.3f}"
            f"{corr_rate[i]:>12.3f}{ols_offspring[i]:>13.3f}{ols_rate[i]:>11.3f}"
        )

    print()
    print("'evolved |w|' = mean |eval_weight| evolution actually assigned this channel (the reward it built).")
    print("'corr'/'ols'  = how well that channel predicts REALIZED fitness -- what a reward chasing fitness")
    print("                directly would have weighted instead. A channel evolution weighted heavily that")
    print("                is NOT the top fitness predictor is the proximate/ultimate divergence the paper")
    print("                predicts (e.g. energy_norm/health_norm -- dense, immediate -- over rare/delayed cues).")


if __name__ == "__main__":
    main()
