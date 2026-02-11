"""
Create a child RLlib checkpoint by mixing two parent checkpoints.

Example:
  python make_child_checkpoint.py \\
    --parent-a /path/to/checkpoint_000009 \\
    --parent-b /path/to/checkpoint_000012 \\
    --out-dir  /path/to/child_checkpoint_000000 \\
    --alpha 0.5 --sigma 0.01
"""
import argparse
import pickle
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create a child checkpoint from two parents.")
    parser.add_argument("--parent-a", required=True, help="Parent A checkpoint directory.")
    parser.add_argument("--parent-b", required=True, help="Parent B checkpoint directory.")
    parser.add_argument("--out-dir", required=True, help="Output checkpoint directory.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Crossover mix (A weight).")
    parser.add_argument("--sigma", type=float, default=0.01, help="Mutation stddev.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for mutation.")
    parser.add_argument(
        "--policy",
        action="append",
        default=None,
        help="Restrict to specific policy id(s). Can be provided multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite out-dir if it already exists.",
    )
    return parser.parse_args()


def _resolve_checkpoint(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _rl_module_dir(checkpoint_dir: Path) -> Path:
    rl_module_dir = checkpoint_dir / "learner_group" / "learner" / "rl_module"
    if not rl_module_dir.is_dir():
        raise FileNotFoundError(f"rl_module dir not found under: {checkpoint_dir}")
    return rl_module_dir


def _load_state(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _save_state(path: Path, state) -> None:
    with path.open("wb") as f:
        pickle.dump(state, f)


def _mix_value(a, b, alpha: float, sigma: float, rng: np.random.Generator):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch for key: {a.shape} vs {b.shape}")
        out = a * alpha + b * (1.0 - alpha)
        if sigma > 0.0:
            noise = rng.normal(0.0, sigma, size=a.shape).astype(out.dtype, copy=False)
            out = out + noise
        return out.astype(a.dtype, copy=False)

    if isinstance(a, (float, int, np.floating, np.integer)) and isinstance(
        b, (float, int, np.floating, np.integer)
    ):
        out = float(a) * alpha + float(b) * (1.0 - alpha)
        if sigma > 0.0:
            out += float(rng.normal(0.0, sigma))
        return out

    return a


def _select_policies(parent_a_dir: Path, parent_b_dir: Path, allowed: Optional[Iterable[str]]):
    a_policies = {p.name for p in parent_a_dir.iterdir() if p.is_dir()}
    b_policies = {p.name for p in parent_b_dir.iterdir() if p.is_dir()}
    common = sorted(a_policies & b_policies)
    if allowed:
        allowed_set = set(allowed)
        common = [p for p in common if p in allowed_set]
    if not common:
        raise ValueError("No matching policies found between parents.")
    return common


def make_child_checkpoint(
    parent_a: Path,
    parent_b: Path,
    out_dir: Path,
    *,
    alpha: float,
    sigma: float,
    seed: Optional[int],
    policies: Optional[Iterable[str]],
    overwrite: bool,
) -> None:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"out-dir already exists: {out_dir}")
        shutil.rmtree(out_dir)

    shutil.copytree(parent_a, out_dir)
    rng = np.random.default_rng(seed)

    a_module_dir = _rl_module_dir(parent_a)
    b_module_dir = _rl_module_dir(parent_b)
    child_module_dir = _rl_module_dir(out_dir)

    policy_ids = _select_policies(a_module_dir, b_module_dir, policies)

    for pid in policy_ids:
        a_state_path = a_module_dir / pid / "module_state.pkl"
        b_state_path = b_module_dir / pid / "module_state.pkl"
        child_state_path = child_module_dir / pid / "module_state.pkl"

        if not a_state_path.exists() or not b_state_path.exists():
            raise FileNotFoundError(f"Missing module_state.pkl for policy {pid}")

        state_a = _load_state(a_state_path)
        state_b = _load_state(b_state_path)

        if state_a.keys() != state_b.keys():
            raise ValueError(f"State dict mismatch for policy {pid}")

        child_state = OrderedDict()
        for key in state_a.keys():
            child_state[key] = _mix_value(state_a[key], state_b[key], alpha, sigma, rng)

        _save_state(child_state_path, child_state)


def main():
    args = parse_args()
    parent_a = _resolve_checkpoint(args.parent_a)
    parent_b = _resolve_checkpoint(args.parent_b)
    out_dir = Path(args.out_dir).expanduser().resolve()

    make_child_checkpoint(
        parent_a,
        parent_b,
        out_dir,
        alpha=args.alpha,
        sigma=args.sigma,
        seed=args.seed,
        policies=args.policy,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
