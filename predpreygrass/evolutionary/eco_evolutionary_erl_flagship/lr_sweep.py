"""Follow-up to behavior_diagnostic.py's finding: reward-genome content never reaches
realized behavior, traced to the action network's within-lifetime learning being weak
(lr_positive=0.05, lr_negative=0.02) and short-lived (one uninherited lifetime, no
credit assignment beyond a single step).

This script tests the cheapest possible fix first, before any bigger redesign
(multi-step credit assignment, longer lifetimes, non-frozen action-network
inheritance): does simply raising the learning rate let the reward genome reach
behavior? Sweeps `lr_positive`/`lr_negative` by a multiplier (1x = current default)
across the starkest available contrast -- `avoider` (punished for predator
proximity) vs. `anti_adaptive` (rewarded for it) -- and reports whether
behavioral divergence appears at any multiplier tested.

If it never does, regardless of how high the multiplier goes, that rules out
"too weak/slow" as the explanation and points at the deeper structural issue
instead: a single step's reinforcement (e_now - prev_eval) may just be too
noisy a signal to shape a useful policy at all, no matter how hard it's
amplified -- pointing toward multi-step credit assignment or a longer/
inherited learning horizon as the real next step, not a config tweak.

Statistics: with exactly two genomes, both run on the SAME seed range (a
deliberate common-random-number pairing -- matched environment/predator-policy
draws per seed), a Wilcoxon signed-rank test on paired per-seed fractions is
used instead of an unpaired Mann-Whitney (a Codex review caught that an
earlier version discarded this pairing and used the wrong, less powerful
test). Seeds with zero encounters in EITHER genome for a given multiplier are
dropped from that multiplier's pairing (can't pair what didn't occur). A
Holm-Bonferroni correction is applied across the multipliers tested, since
each is a separate significance test and interpreting the first p<0.05 as
"divergence appears" without correction inflates the false-positive rate
(also a Codex finding).

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.lr_sweep \\
        --steps 5000 --seeds 5 --multipliers 1,5,20,100
"""

import argparse
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy import stats

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.behavior_diagnostic import run_one


def holm_correct(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    original order of `pvals`."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adjusted = [None] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * pvals[i]))
        adjusted[i] = running_max
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--genomes", type=str, default="avoider,anti_adaptive",
                         help="Comma-separated genome names (default: the starkest contrast). "
                              "The paired significance test only runs for exactly 2 genomes.")
    parser.add_argument("--multipliers", type=str, default="1,5,20,100",
                         help="Comma-separated lr_positive/lr_negative multipliers to sweep.")
    args = parser.parse_args()
    genome_names = args.genomes.split(",")
    multipliers = [float(x) for x in args.multipliers.split(",")]
    for mult in multipliers:
        if not (math.isfinite(mult) and mult >= 0):
            raise ValueError(f"--multipliers must all be finite and >= 0, got {mult}.")

    jobs = [
        (mult, name, seed)
        for mult in multipliers
        for name in genome_names
        for seed in range(1, args.seeds + 1)
    ]
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"Running {len(jobs)} runs ({len(multipliers)} multipliers x {len(genome_names)} genomes x "
          f"{args.seeds} seeds) on {workers} workers.")

    # (multiplier, genome, seed) -> fraction of chosen actions pointing TOWARD the predator,
    # keyed by seed (not just appended to a list) so genome pairs stay matched per seed.
    per_cell: dict[tuple[float, str], dict[int, float]] = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, name, seed, args.steps, mult): (mult, name, seed)
            for mult, name, seed in jobs
        }
        done = 0
        for fut in as_completed(futures):
            mult, name, seed = futures[fut]
            counts = fut.result()
            done += 1
            if counts.get("pred_encounters", 0) > 0:
                per_cell[(mult, name)][seed] = counts["pred_toward"] / counts["pred_encounters"]
            if done % 10 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}]")

    paired = len(genome_names) == 2
    raw_pvals = []
    rows = []
    for mult in multipliers:
        row = {"mult": mult}
        for name in genome_names:
            vals = list(per_cell[(mult, name)].values())
            row[name] = 100 * sum(vals) / len(vals) if vals else float("nan")
        if paired:
            a, b = genome_names
            common_seeds = sorted(set(per_cell[(mult, a)]) & set(per_cell[(mult, b)]))
            xa = [per_cell[(mult, a)][s] for s in common_seeds]
            xb = [per_cell[(mult, b)][s] for s in common_seeds]
            row["n_paired"] = len(common_seeds)
            if len(common_seeds) >= 2 and any(xa[i] != xb[i] for i in range(len(common_seeds))):
                try:
                    _, p = stats.wilcoxon(xa, xb)
                    row["p_raw"] = p
                except ValueError as e:
                    row["p_raw"] = None
                    row["error"] = str(e)
            else:
                row["p_raw"] = None
        rows.append(row)
        if paired and row.get("p_raw") is not None:
            raw_pvals.append(row["p_raw"])

    if paired and raw_pvals:
        adjusted = holm_correct(raw_pvals)
        it = iter(adjusted)
        for row in rows:
            if row.get("p_raw") is not None:
                row["p_holm"] = next(it)

    header = f"{'lr_multiplier':<15}" + "".join(f"{name + ' %toward':<20}" for name in genome_names)
    if paired:
        header += f"{'n_paired':<10}{'p_raw (Wilcoxon)':<20}{'p_holm'}"
    print("\n" + header)
    for row in rows:
        line = f"{row['mult']:<15}" + "".join(f"{row[name]:<20.1f}" for name in genome_names)
        if paired:
            n_paired = row.get("n_paired", 0)
            p_raw = row.get("p_raw")
            p_holm = row.get("p_holm")
            p_raw_str = f"{p_raw:.4f}" if p_raw is not None else row.get("error", "n/a")
            p_holm_str = f"{p_holm:.4f}" if p_holm is not None else "n/a"
            line += f"{n_paired:<10}{p_raw_str:<20}{p_holm_str}"
        print(line)

    print(
        "\n%toward = fraction of chosen actions whose move vector pointed TOWARD the predator "
        "when one was visible (see behavior_diagnostic.py). p_raw is a paired Wilcoxon "
        "signed-rank test on matched per-seed fractions (same seed range reused across "
        "genomes); p_holm is the Holm-Bonferroni-corrected p-value across all multipliers "
        "tested, controlling the family-wise false-positive rate. If behavioral divergence "
        "between genomes is never significant (p_holm) at any multiplier, the bottleneck is "
        "not learning-rate magnitude."
    )


if __name__ == "__main__":
    main()
