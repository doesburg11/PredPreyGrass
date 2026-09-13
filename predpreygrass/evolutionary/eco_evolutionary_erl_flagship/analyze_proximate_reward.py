"""Proximate-vs-ultimate reward analysis (Singh, Lewis, Barto & Sorg 2010), ported
from eco_evolutionary_erl_baldwin/analyze_proximate_reward.py onto Trial 13's data.

Their claim: evolution optimizes a REWARD function for reproductive fitness, but
the reward it finds need not resemble fitness itself. This script tests that
against Trial 13 data: each prey's `genome.eval_weights` IS its evolved,
lifetime-fixed reward function, and `offspring_count` is its realized fitness. If
evolution converged on a proximate substitute, the evolved weights should load
heavily on dense per-step channels (`energy_norm`, proximity channels) rather than
on whichever channel correlates best with actual offspring_count.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.analyze_proximate_reward \\
        ~/simulation_results/erl_results/ERL_FLAGSHIP_2026-.../lineage_fitness.csv

    # or point at a directory to combine every lineage_fitness.csv found under it:
    python -m ...analyze_proximate_reward ~/simulation_results/erl_results/erl_flagship_study/

    # include a run's still-alive prey (right-censored) via its final checkpoint --
    # lineage_fitness.csv deliberately never logs survivors itself (see
    # run_trial13_simulation.py):
    python -m ...analyze_proximate_reward .../lineage_fitness.csv \\
        --checkpoint .../checkpoints/checkpoint_step_100000.pkl
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import FEATURE_NAMES

CHANNEL_NAMES = FEATURE_NAMES  # 8 channels, see features.py


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
            f"{source} has obs_dim={this_dim}, but earlier data had obs_dim={obs_dim}."
        )
    return this_dim


def _array_from_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Returns (eval_weights [n, obs_dim], offspring_count [n], lifespan [n], obs_dim).

    Parses straight into numpy arrays via `np.loadtxt` rather than building a Python
    list of per-row objects first -- see erl_baldwin/analyze_proximate_reward.py's
    module docstring for why (an earlier row-by-row version there got OOM-killed at
    real multi-seed scale, ~150M rows / 89GB RSS). Same pattern from the start here.
    """
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    weight_cols = sorted(
        (c for c in header if c.startswith("eval_weight_")),
        key=lambda c: int(c.split("_")[-1]),
    )
    obs_dim = len(weight_cols)
    usecols = [header.index(c) for c in weight_cols] + [header.index("offspring_count"), header.index("lifespan")]
    arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols, dtype=np.float64, ndmin=2)
    return arr[:, :obs_dim], arr[:, obs_dim], arr[:, obs_dim + 1], obs_dim


def _rows_from_checkpoint(path: Path) -> tuple[list, int]:
    """Currently-alive prey in a checkpoint, as right-censored lineage rows -- see
    metrics.lineage_record's `censored` semantics. Population here is bounded by
    what a single flagship ecology can hold concurrently (small, per this module's
    README.md), so the plain-Python-object version is fine -- unlike the CSV path
    above, this never scales to hundreds of millions of rows."""
    from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.checkpoint import load_checkpoint

    payload = load_checkpoint(path)
    registry = payload["registry"]
    current_step = payload["current_step"]
    rows = [
        (list(state.genome.eval_weights), float(state.offspring_count), float(current_step - state.born_step))
        for state in registry.values()
    ]
    obs_dim = len(rows[0][0]) if rows else len(CHANNEL_NAMES)
    return rows, obs_dim


def _load(
    csv_paths: list[Path], checkpoint_paths: list[Path]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Returns (eval_weights [n, obs_dim], offspring_count [n], lifespan [n], obs_dim, n_censored)."""
    weights_parts: list[np.ndarray] = []
    offspring_parts: list[np.ndarray] = []
    lifespan_parts: list[np.ndarray] = []
    obs_dim = None
    n_censored = 0
    for path in csv_paths:
        w, o, l, this_dim = _array_from_csv(path)
        obs_dim = _check_dim(obs_dim, this_dim, path)
        weights_parts.append(w)
        offspring_parts.append(o)
        lifespan_parts.append(l)
    for path in checkpoint_paths:
        rows, this_dim = _rows_from_checkpoint(path)
        obs_dim = _check_dim(obs_dim, this_dim, path)
        if rows:
            weights_parts.append(np.array([r[0] for r in rows]))
            offspring_parts.append(np.array([r[1] for r in rows]))
            lifespan_parts.append(np.array([r[2] for r in rows]))
            n_censored += len(rows)
    if not weights_parts:
        raise ValueError(f"No data rows found in: {csv_paths + checkpoint_paths}")
    weights = np.concatenate(weights_parts, axis=0)
    offspring = np.concatenate(offspring_parts, axis=0)
    lifespan = np.concatenate(lifespan_parts, axis=0)
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
        help="checkpoint_step_*.pkl file(s) whose currently-alive prey are included "
             "as right-censored rows.",
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
    print(f"Loaded {len(offspring)} prey lifetimes from {len(csv_paths)} CSV file(s){censored_note}, obs_dim={obs_dim}.")
    print(f"offspring_count: mean={offspring.mean():.3f}, max={offspring.max():.0f}")
    print()

    corr_offspring = [_pearson(weights[:, i], offspring) for i in range(obs_dim)]
    corr_rate = [_pearson(weights[:, i], rate) for i in range(obs_dim)]
    ols_offspring = _ols_standardized(weights, offspring)
    ols_rate = _ols_standardized(weights, rate)
    evolved_magnitude = np.abs(weights).mean(axis=0)

    order = sorted(range(obs_dim), key=lambda i: -abs(corr_offspring[i]))

    header = f"{'channel':<20}{'evolved |w|':>12}{'corr(offspr)':>14}{'corr(rate)':>12}{'ols(offspr)':>13}{'ols(rate)':>11}"
    print(header)
    print("-" * len(header))
    for i in order:
        print(
            f"{names[i]:<20}{evolved_magnitude[i]:>12.3f}{corr_offspring[i]:>14.3f}"
            f"{corr_rate[i]:>12.3f}{ols_offspring[i]:>13.3f}{ols_rate[i]:>11.3f}"
        )

    print()
    print("'evolved |w|' = mean |eval_weight| evolution actually assigned this channel (the reward it built).")
    print("'corr'/'ols'  = how well that channel predicts REALIZED fitness -- what a reward chasing fitness")
    print("                directly would have weighted instead. A channel evolution weighted heavily that")
    print("                is NOT the top fitness predictor is the proximate/ultimate divergence the paper")
    print("                predicts.")


if __name__ == "__main__":
    main()
