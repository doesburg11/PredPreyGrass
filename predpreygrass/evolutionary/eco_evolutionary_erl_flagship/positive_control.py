"""Positive-control check for Trial 13's "no optimal reward function found" result
(README.md's Status section) -- diagnostic, not a training/evolution run.

The Hunt-test result says evolved `eval_weights` show no reproducible directional
selection. Two very different things could explain that:

  (a) no fixed reward-weight vector actually produces a meaningfully different
      realized fitness in this ecology (the action network's 1-step-REINFORCE
      behavior doesn't track eval_weights strongly enough for selection to have
      anything real to act on), or
  (b) real fitness differences exist between reward-weight vectors, but evolution
      just hasn't found/kept them (a search or drift-dominance problem instead).

This script tells them apart directly: run a handful of hand-picked, EXTREME,
FIXED eval_weights vectors (mutation_rate=0.0, so every descendant keeps the exact
founder vector forever -- no evolution, no drift, just "does this fixed reward
function behave differently") across multiple seeds, and compare realized
population-level fitness (survival, total births, final population). If even an
obviously-good vector (forager/avoider) doesn't outperform an obviously-bad one
(anti_adaptive), the bottleneck is (a): the action side, not the reward side.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.positive_control \\
        --steps 20000 --seeds 10
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.checkpoint import load_checkpoint
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.config import config_env_flagship
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import FEATURE_NAMES
from predpreygrass.global_config import ERL_RESULTS_DIR

N_FOUNDER_PREY = config_env_flagship["n_initial_active_prey"]

# FEATURE_NAMES order: energy_norm, predator_dx, predator_dy, predator_proximity,
# food_dx, food_dy, food_proximity, local_grass_density. dx/dy left at 0 everywhere
# (direction-of-approach isn't obviously good or bad on its own -- proximity is the
# channel with an unambiguous sign); food_proximity/predator_proximity are 1.0 at
# the agent's own cell, 0.0 at the observation-window edge (features.py).
GENOMES = {
    "forager":       [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0],
    "avoider":       [0.5, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0],
    "balanced":      [1.0, 0.0, 0.0, -1.5, 0.0, 0.0, 1.5, 0.5],
    "inert":         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "anti_adaptive": [-1.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1.5, -0.5],
}
assert all(len(w) == len(FEATURE_NAMES) for w in GENOMES.values())

FINISHED_RE = re.compile(r"Finished at step (\d+) in ([\d.]+)s \(([\d.]+) steps/sec\)")
EXTINCTION_RE = re.compile(r"Prey population extinction at step (\d+)")
FINAL_POP_RE = re.compile(r"Final population: \{'prey': (\d+), 'predator': (\d+)\}")


def run_one(genome_name: str, weights: list[float], seed: int, steps: int, out_root: Path) -> dict:
    out_dir = out_root / genome_name / f"seed_{seed}"
    cmd = [
        sys.executable, "-m",
        "predpreygrass.evolutionary.eco_evolutionary_erl_flagship.run_trial13_simulation",
        "--steps", str(steps),
        "--seed", str(seed),
        "--log-every", "2000",
        "--checkpoint-every", "0",
        "--out-dir", str(out_dir),
        # "=" form, not a separate argv token: argparse misreads a leading "-" (from a
        # negative weight, e.g. anti_adaptive's -1.0) as a new flag otherwise.
        "--fixed-eval-weights=" + ",".join(str(w) for w in weights),
        "--mutation-rate", "0.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout
    row = {"genome": genome_name, "seed": seed, "ok": result.returncode == 0}
    if not row["ok"]:
        row["stderr_tail"] = result.stderr[-2000:]
        return row

    m = FINISHED_RE.search(stdout)
    final_step = int(m.group(1)) if m else None
    m = EXTINCTION_RE.search(stdout)
    extinct = m is not None
    m = FINAL_POP_RE.search(stdout)
    final_prey = int(m.group(1)) if m else None
    final_predator = int(m.group(2)) if m else None

    if final_step is None or final_prey is None or final_predator is None:
        # Exit code 0 but the expected summary lines weren't found (stdout format drifted,
        # or a truncated capture) -- surface as a failed row instead of silently carrying
        # None into later arithmetic (a real bug a Codex review caught: the original version
        # let this crash the final aggregation with a TypeError instead of reporting it).
        row["ok"] = False
        row["stderr_tail"] = (
            "Run exited 0 but expected summary lines were not found in stdout.\n"
            f"stdout tail:\n{stdout[-2000:]}"
        )
        return row

    row["final_step"] = final_step
    row["extinct"] = extinct
    row["final_prey"] = final_prey
    row["final_predator"] = final_predator

    lineage_csv = out_dir / "lineage_fitness.csv"
    max_gen, deaths_recorded = 0, 0
    if lineage_csv.exists():
        with open(lineage_csv) as f:
            for r in csv.DictReader(f):
                deaths_recorded += 1
                max_gen = max(max_gen, int(r["generation"]))

    # max_gen above only sees deaths (lineage_fitness.csv deliberately never logs
    # survivors, see run_trial13_simulation.py) -- if the deepest lineage is still
    # alive at the end, this undercounts it. Also check the final checkpoint's live
    # registry (a Codex review caught this gap in an earlier version).
    checkpoint_path = out_dir / "checkpoints" / f"checkpoint_step_{final_step}.pkl"
    if checkpoint_path.exists():
        payload = load_checkpoint(checkpoint_path)
        survivor_gens = [s.generation for s in payload["registry"].values()]
        if survivor_gens:
            max_gen = max(max_gen, max(survivor_gens))

    row["deaths_recorded"] = deaths_recorded
    row["total_offspring"] = deaths_recorded + final_prey - N_FOUNDER_PREY  # excludes the founders themselves
    row["max_generation"] = max_gen
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=20000, help="Step budget per run (extinction ends it earlier).")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per genome (seeds 1..N).")
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker cap (default: nproc - 2).")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    import os
    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(args.out_dir) if args.out_dir else ERL_RESULTS_DIR / f"ERL_FLAGSHIP_positive_control_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        (genome_name, weights, seed)
        for genome_name, weights in GENOMES.items()
        for seed in range(1, args.seeds + 1)
    ]
    print(f"Launching {len(jobs)} runs ({len(GENOMES)} genomes x {args.seeds} seeds), "
          f"{args.steps} steps each, {workers} parallel workers.")
    print(f"Output: {out_root}")

    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, name, weights, seed, args.steps, out_root): (name, seed)
            for name, weights, seed in jobs
        }
        done = 0
        for fut in as_completed(futures):
            name, seed = futures[fut]
            row = fut.result()
            results.append(row)
            done += 1
            status = "OK" if row["ok"] else "FAILED"
            print(f"[{done}/{len(jobs)}] {name} seed={seed}: {status}")

    elapsed = time.time() - start
    print(f"\nAll runs finished in {elapsed:.0f}s.\n")

    print(f"{'genome':<16}{'n':<4}{'mean_final_step':<18}{'frac_extinct':<14}"
          f"{'mean_offspring':<17}{'mean_final_prey':<16}{'mean_max_gen'}")
    for name in GENOMES:
        rows = [r for r in results if r["genome"] == name and r["ok"]]
        n = len(rows)
        if n == 0:
            print(f"{name:<16}{'0 (all failed)'}")
            continue
        mean_step = sum(r["final_step"] for r in rows) / n
        frac_ext = sum(1 for r in rows if r["extinct"]) / n
        mean_offspring = sum(r["total_offspring"] for r in rows) / n
        mean_prey = sum(r["final_prey"] for r in rows) / n
        mean_gen = sum(r["max_generation"] for r in rows) / n
        print(f"{name:<16}{n:<4}{mean_step:<18.0f}{frac_ext:<14.2f}{mean_offspring:<17.1f}{mean_prey:<16.1f}{mean_gen:.1f}")

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\n{len(failed)} run(s) failed. First failure's stderr tail:")
        print(failed[0]["stderr_tail"])


if __name__ == "__main__":
    main()
