"""Companion to polymorphism_check.py, testing a DIFFERENT question. That script
asked whether a two-strategy split SPONTANEOUSLY EMERGES from a neutral, randomly
initialized population -- answer: not within ~260 generations; any detected
substructure stayed small (~0.1-0.2 units) and didn't grow, far short of the
4.0-unit split between `avoider` and `anti_adaptive` that showed real, opposite
fitness advantages in positive_control.py's SEGREGATED single-genotype comparison.

This script asks whether an ALREADY-ESTABLISHED split, once it exists, is
MAINTAINED by selection -- a genuinely different population-genetics question
(stability/invasion fitness vs. origination). Founders are seeded 50/50 from the
two extreme genomes directly (`--mixed-eval-weights-a/-b`, driver.py's
`mixed_founder_weights`), mutation stays ON (real evolution continues from there),
at the same 20x learning rate already confirmed to make reward-genome content
matter for fitness. If the split is truly evolutionarily stable, both clusters
should persist near their starting values and near a roughly balanced population
share across many generations. If one strategy is actually fitter once both
compete for the same food/space (not just fitter in isolation), its cluster's
population share should grow toward 1.0 while the other's shrinks toward 0 (even
if some residual statistical "bimodality" persists near the end from a shrinking
minority plus ongoing mutation).

Reuses polymorphism_check.py's null-calibrated, n_init=10, information-criterion
bimodality test unchanged (see that module's docstring for the Codex-reviewed
rigor notes) -- only the founder-seeding and what's reported (population SHARE
per cluster over time, not just whether bimodality is detected at all) differ.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.polymorphism_maintenance_check \\
        --steps 20000 --seeds 10 --checkpoint-every 2000
"""

import argparse
import math
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.checkpoint import load_checkpoint
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import FEATURE_NAMES
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.polymorphism_check import (
    DELTA_BIC_THRESHOLD,
    MIN_LIVE_AGENTS,
    bic_bimodality,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.positive_control import GENOMES
from predpreygrass.global_config import ERL_RESULTS_DIR

PRED_PROX_IDX = FEATURE_NAMES.index("predator_proximity")
# Midpoint between avoider's -2.0 and anti_adaptive's +2.0 -- classifies each live
# agent to whichever founder cluster it started closer to (mutation can move an
# individual descendant's weight over generations, so this is an approximate,
# not exact, lineage label; good enough for tracking aggregate population share).
CLASSIFY_MIDPOINT = (GENOMES["avoider"][PRED_PROX_IDX] + GENOMES["anti_adaptive"][PRED_PROX_IDX]) / 2.0


def run_one(seed: int, steps: int, checkpoint_every: int, lr_multiplier: float, out_root: Path) -> dict:
    out_dir = out_root / f"seed_{seed}"
    cmd = [
        sys.executable, "-m",
        "predpreygrass.evolutionary.eco_evolutionary_erl_flagship.run_trial13_simulation",
        "--steps", str(steps),
        "--seed", str(seed),
        "--log-every", str(max(checkpoint_every, 2000)),
        "--checkpoint-every", str(checkpoint_every),
        "--out-dir", str(out_dir),
        "--lr-multiplier", str(lr_multiplier),
        "--mixed-eval-weights-a=" + ",".join(str(w) for w in GENOMES["avoider"]),
        "--mixed-eval-weights-b=" + ",".join(str(w) for w in GENOMES["anti_adaptive"]),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        return {"seed": seed, "ok": False, "stderr_tail": f"subprocess.run raised: {e}"}
    if result.returncode != 0:
        return {"seed": seed, "ok": False, "stderr_tail": result.stderr[-2000:]}

    rng = np.random.default_rng(seed)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pkl"), key=lambda p: int(p.stem.split("_")[-1]))
    per_checkpoint = []
    for ckpt_path in checkpoints:
        step = int(ckpt_path.stem.split("_")[-1])
        try:
            payload = load_checkpoint(ckpt_path)
            registry = payload["registry"]
            if len(registry) < MIN_LIVE_AGENTS:
                continue
            weights = np.array([s.genome.eval_weights[PRED_PROX_IDX] for s in registry.values()])
            bic_result = bic_bimodality(weights, seed, rng)
            bic_result["step"] = step
            bic_result["mean_generation"] = float(np.mean([s.generation for s in registry.values()]))
            # Population share: fraction of the LIVE population whose predator_proximity
            # weight is still closer to avoider's founder value than anti_adaptive's --
            # the direct "is one side winning" signal, independent of whether the GMM
            # formally calls the distribution bimodal at this checkpoint.
            share_avoider = float(np.mean(weights < CLASSIFY_MIDPOINT))
            bic_result["share_avoider_side"] = share_avoider
            per_checkpoint.append(bic_result)
        except Exception:
            return {
                "seed": seed, "ok": False,
                "stderr_tail": f"Exception analyzing {ckpt_path}:\n{traceback.format_exc()}",
            }

    return {"seed": seed, "ok": True, "checkpoints": per_checkpoint}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--lr-multiplier", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.seeds < 1 or args.steps < 1 or args.checkpoint_every < 1:
        raise ValueError("--seeds, --steps, and --checkpoint-every must all be >= 1.")
    if not (math.isfinite(args.lr_multiplier) and args.lr_multiplier >= 0):
        raise ValueError(f"--lr-multiplier must be finite and >= 0, got {args.lr_multiplier}.")

    import os
    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(args.out_dir) if args.out_dir else ERL_RESULTS_DIR / f"ERL_FLAGSHIP_polymorphism_maintenance_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Launching {args.seeds} seeds (50/50 avoider/anti_adaptive founders), {args.steps} steps each, "
          f"lr_multiplier={args.lr_multiplier}, checkpoint every {args.checkpoint_every} steps, "
          f"{workers} parallel workers.")
    print(f"Output: {out_root}")

    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, seed, args.steps, args.checkpoint_every, args.lr_multiplier, out_root): seed
            for seed in range(1, args.seeds + 1)
        }
        done = 0
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                row = fut.result()
            except Exception:
                row = {"seed": seed, "ok": False, "stderr_tail": traceback.format_exc()}
            results.append(row)
            done += 1
            print(f"[{done}/{args.seeds}] seed={seed}: {'OK' if row['ok'] else 'FAILED'}")

    elapsed = time.time() - start
    print(f"\nAll runs finished in {elapsed:.0f}s.\n")

    ok_results = [r for r in results if r["ok"]]
    all_steps = sorted({c["step"] for r in ok_results for c in r["checkpoints"]})

    print(f"{'step':<10}{'mean_gen':<12}{'n_eligible':<12}{'frac_bimodal':<14}{'null_fp_rate':<14}"
          f"{'mean_share_avoider':<20}{'share_range'}")
    for step in all_steps:
        rows_at_step = [c for r in ok_results for c in r["checkpoints"] if c["step"] == step]
        if not rows_at_step:
            continue
        n = len(rows_at_step)
        bimodal = [c for c in rows_at_step if c["delta_bic"] > DELTA_BIC_THRESHOLD]
        null_bimodal = [c for c in rows_at_step if c["null_delta_bic"] > DELTA_BIC_THRESHOLD]
        frac_bimodal = len(bimodal) / n
        null_fp_rate = len(null_bimodal) / n
        shares = [c["share_avoider_side"] for c in rows_at_step]
        mean_gen = sum(c["mean_generation"] for c in rows_at_step) / n
        print(f"{step:<10}{mean_gen:<12.1f}{n:<12}{frac_bimodal:<14.2f}{null_fp_rate:<14.2f}"
              f"{np.mean(shares):<20.2f}[{np.min(shares):.2f}, {np.max(shares):.2f}]")

    print(
        "\nshare_avoider_side = fraction of the live population still closer to avoider's founder "
        "predator_proximity weight (-2.0) than anti_adaptive's (+2.0). Founders start at ~0.50 by "
        "construction (50/50 seeding). If the split is evolutionarily stable, this should stay near "
        "0.50 across checkpoints; if one strategy is actually fitter once both compete for the same "
        "food/space (not just fitter in isolation, per positive_control.py's segregated comparison), "
        "it should trend toward 0.0 or 1.0 over generations. frac_bimodal/null_fp_rate are the same "
        "null-calibrated GMM/BIC check as polymorphism_check.py -- see that module for the rigor notes."
    )

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\n{len(failed)} run(s) failed. First failure's stderr tail:")
        print(failed[0]["stderr_tail"])


if __name__ == "__main__":
    main()
