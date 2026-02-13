#!/usr/bin/env python3
"""
Parameter sweep for cooperation model (public sharing), with optional adaptive refinement.

- Averages mean_coop over N successful runs for each (COOP_COST, P0)
- Produces a heatmap over COOP_COST vs P0
- Adaptive mode refines the grid around the best (lowest mean_coop) stable cells
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import statistics as stats
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def frange(start: float, stop: float, step: float) -> List[float]:
    vals = []
    x = start
    while x <= stop + 1e-9:
        vals.append(round(x, 10))
        x += step
    return vals


@dataclass(frozen=True)
class CellResult:
    i: int
    j: int
    cost: float
    p0: float
    successes: int
    mean: float


def _run_cell(
    i: int,
    j: int,
    cost: float,
    p0: float,
    successes_target: int,
    max_attempts: int,
    tail_window: int,
    steps: int,
    seed_base: int,
) -> CellResult:
    # Import inside worker so each process has its own module state
    import predpreygrass.ecology.emerging_cooperation as eco

    eco.ANIMATE = False
    eco.RESTART_ON_EXTINCTION = False
    eco.STEPS = steps
    eco.COOP_COST = cost
    eco.P0 = p0

    successes = 0
    means: List[float] = []

    for attempt in range(max_attempts):
        seed = seed_base + attempt
        with contextlib.redirect_stdout(io.StringIO()):
            (
                pred_hist,
                prey_hist,
                mean_coop_hist,
                var_coop_hist,
                preds_snaps,
                preys_snaps,
                preds_final,
                success,
                extinction_step,
            ) = eco.run_sim(seed_override=seed)

        if success and mean_coop_hist:
            tail_n = min(tail_window, len(mean_coop_hist))
            tail_mean = sum(mean_coop_hist[-tail_n:]) / tail_n
            means.append(tail_mean)
            successes += 1
            if successes >= successes_target:
                break

    mean = stats.mean(means) if means else float("nan")
    return CellResult(i=i, j=j, cost=cost, p0=p0, successes=successes, mean=mean)


def run_grid(
    coop_vals: List[float],
    p0_vals: List[float],
    args: argparse.Namespace,
    round_idx: int,
) -> Tuple[List[CellResult], np.ndarray, np.ndarray]:
    heat = np.full((len(p0_vals), len(coop_vals)), np.nan, dtype=float)
    counts = np.zeros((len(p0_vals), len(coop_vals)), dtype=int)

    jobs = []
    for i, p0 in enumerate(p0_vals):
        for j, cost in enumerate(coop_vals):
            seed_base = args.seed + round_idx * 100000 + i * 1000 + j * 100
            jobs.append((i, j, cost, p0, seed_base))

    worker_args = [
        (i, j, cost, p0, args.successes, args.max_attempts, args.tail_window, args.steps, seed_base)
        for i, j, cost, p0, seed_base in jobs
    ]

    results: List[CellResult] = []
    if args.workers == 1:
        for a in worker_args:
            results.append(_run_cell(*a))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run_cell, *a) for a in worker_args]
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda r: (r.i, r.j))
    for r in results:
        counts[r.i, r.j] = r.successes
        heat[r.i, r.j] = r.mean
        print(
            f"P0={r.p0:.2f} COOP_COST={r.cost:.4f} "
            f"success={r.successes}/{args.successes} mean_coop={r.mean:.3f}"
        )

    return results, heat, counts


def pick_refine_bounds(
    results: List[CellResult],
    coop_vals: List[float],
    p0_vals: List[float],
    args: argparse.Namespace,
    step_cost: float,
    step_p0: float,
    base_bounds: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float, float, float] | None:
    min_successes = math.ceil(args.min_success_rate * args.successes)
    candidates = [r for r in results if r.successes >= min_successes and not math.isnan(r.mean)]

    if not candidates:
        candidates = [r for r in results if r.successes > 0 and not math.isnan(r.mean)]

    if not candidates:
        return None

    candidates.sort(key=lambda r: r.mean)
    top = candidates[: args.top_k]

    min_cost = min(r.cost for r in top)
    max_cost = max(r.cost for r in top)
    min_p0 = min(r.p0 for r in top)
    max_p0 = max(r.p0 for r in top)

    span_cost = args.refine_span_mult * step_cost
    span_p0 = args.refine_span_mult * step_p0

    new_cost_min = min_cost - span_cost
    new_cost_max = max_cost + span_cost
    new_p0_min = min_p0 - span_p0
    new_p0_max = max_p0 + span_p0

    if args.clamp_to_initial:
        base_cost_min, base_cost_max, base_p0_min, base_p0_max = base_bounds
        new_cost_min = max(base_cost_min, new_cost_min)
        new_cost_max = min(base_cost_max, new_cost_max)
        new_p0_min = max(base_p0_min, new_p0_min)
        new_p0_max = min(base_p0_max, new_p0_max)

    new_step_cost = max(step_cost * args.refine_step_factor, args.min_step)
    new_step_p0 = max(step_p0 * args.refine_step_factor, args.min_step)

    # Ensure ranges are valid
    if new_cost_max - new_cost_min < new_step_cost * 0.5:
        center = 0.5 * (new_cost_min + new_cost_max)
        new_cost_min = center - new_step_cost
        new_cost_max = center + new_step_cost
    if new_p0_max - new_p0_min < new_step_p0 * 0.5:
        center = 0.5 * (new_p0_min + new_p0_max)
        new_p0_min = center - new_step_p0
        new_p0_max = center + new_step_p0

    return new_cost_min, new_cost_max, new_step_cost, new_p0_min, new_p0_max, new_step_p0


def save_heatmap(
    heat: np.ndarray,
    coop_vals: List[float],
    p0_vals: List[float],
    title: str,
    outfile: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#cccccc")
    im = ax.imshow(
        heat,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[min(coop_vals), max(coop_vals), min(p0_vals), max(p0_vals)],
        cmap=cmap,
    )
    ax.set_xlabel("COOP_COST")
    ax.set_ylabel("P0")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean coop")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved heatmap to {outfile}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coop-min", type=float, default=0.15)
    ap.add_argument("--coop-max", type=float, default=0.20)
    ap.add_argument("--coop-step", type=float, default=0.01)
    ap.add_argument("--p0-min", type=float, default=0.05)
    ap.add_argument("--p0-max", type=float, default=0.30)
    ap.add_argument("--p0-step", type=float, default=0.01)
    ap.add_argument("--successes", type=int, default=10)
    ap.add_argument("--max-attempts", type=int, default=100)
    ap.add_argument("--tail-window", type=int, default=200)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--outfile", type=str, default="./src/predpreygrass/ecology/coop_cost_p0_heatmap.png")

    # Adaptive refinement
    ap.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-success-rate", type=float, default=1.0)
    ap.add_argument("--refine-span-mult", type=float, default=2.0)
    ap.add_argument("--refine-step-factor", type=float, default=0.5)
    ap.add_argument("--min-step", type=float, default=0.0025)
    ap.add_argument("--save-all", action="store_true", default=True)
    ap.add_argument("--clamp-to-initial", action="store_true", default=True)
    args = ap.parse_args()

    base_bounds = (args.coop_min, args.coop_max, args.p0_min, args.p0_max)

    coop_min, coop_max, coop_step = args.coop_min, args.coop_max, args.coop_step
    p0_min, p0_max, p0_step = args.p0_min, args.p0_max, args.p0_step

    rounds = args.rounds if args.adaptive else 1

    for r in range(rounds):
        coop_vals = frange(coop_min, coop_max, coop_step)
        p0_vals = frange(p0_min, p0_max, p0_step)

        print(
            f"\n=== Round {r + 1}/{rounds} ===\n"
            f"COOP_COST range: [{coop_min:.4f}, {coop_max:.4f}] step {coop_step:.4f}\n"
            f"P0 range:        [{p0_min:.4f}, {p0_max:.4f}] step {p0_step:.4f}\n"
        )

        results, heat, counts = run_grid(coop_vals, p0_vals, args, r)

        title = f"Mean coop (avg over {args.successes} successes; tail {args.tail_window})"
        if rounds > 1:
            base, ext = os.path.splitext(args.outfile)
            out = f"{base}_r{r + 1}{ext}"
        else:
            out = args.outfile

        if out:
            save_heatmap(heat, coop_vals, p0_vals, title, out)

        if not args.adaptive or r == rounds - 1:
            break

        refined = pick_refine_bounds(
            results,
            coop_vals,
            p0_vals,
            args,
            coop_step,
            p0_step,
            base_bounds,
        )
        if refined is None:
            print("No successful cells found; cannot refine further.")
            break

        (coop_min, coop_max, coop_step, p0_min, p0_max, p0_step) = refined


if __name__ == "__main__":
    main()
