"""Analyze the "mixed" carnivore competition test (resident seed network vs. a
mutant variant, 50/50 start, type inherited). Written before the results.

Per seed (runs that reached the budget), from the time series:
  - mutant frequency at the immigration switch and at the end
  - relative per-capita reproduction after the switch:
    (mutant births / mutant carnivore-steps) / (resident births / resident carnivore-steps)

A variant is favored by within-population selection if its log birth ratio is
> 0 across seeds (Wilcoxon signed-rank) and its frequency rises. The "identical"
tag (mutant == resident) is the neutral yardstick: its ratio should sit at ~1.

PRIMARY readout (fixed 2026-09-29 before the rich-input run, after the first
competition test showed selection acts mostly during the warm-up): the whole
trajectory from the 50/50 start -- mutant frequency at the switch compared with
the "identical" tag (Mann-Whitney), and the warm-up birth ratio (500 -> switch,
Wilcoxon). A variant beats the resident if its frequency at the switch is
higher than identical's with p < 0.05 / (number of variants). The post-switch
numbers are secondary.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.analyze_invasion \\
        --out-dir ~/simulation_results/erl_results/coevo_invasion
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SWITCH_STEP = 20_000


def per_seed(out_dir: Path, r: dict, switch: int):
    path = out_dir / "timeseries" / f"{r['tag']}_{r['strategy']}_seed{r['seed']}.csv"
    with open(path) as f:
        rows = {int(x["step"]): x for x in csv.DictReader(f)}
    end = max(rows)
    if switch not in rows or r["end_reason"] != "budget":
        return None
    a, b = rows[switch], rows[end]

    def freq(x):
        return int(x["carn_mutants"]) / max(int(x["carnivore_count"]), 1)

    def rate(kind):
        births = int(b[f"{kind}_births"]) - int(a[f"{kind}_births"])
        steps = int(b[f"{kind}_steps"]) - int(a[f"{kind}_steps"])
        return births / steps if steps else float("nan")

    resident = rate("resident")
    # A type that has died out (no births after the switch) leaves the ratio undefined.
    return freq(a), freq(b), rate("mutant") / resident if resident else float("nan")


def whole_trajectory(out_dir: Path, r: dict, switch: int):
    """Mutant frequency at 500 and at the switch, and the warm-up birth ratio (500 -> switch)."""
    path = out_dir / "timeseries" / f"{r['tag']}_{r['strategy']}_seed{r['seed']}.csv"
    with open(path) as f:
        rows = {int(x["step"]): x for x in csv.DictReader(f)}
    if 500 not in rows or switch not in rows:
        return None
    a, b = rows[500], rows[switch]
    freq = lambda x: int(x["carn_mutants"]) / max(int(x["carnivore_count"]), 1)
    d = {k: int(b[k]) - int(a[k]) for k in ("mutant_births", "mutant_steps", "resident_births", "resident_steps")}
    ratio = float("nan")
    if d["mutant_steps"] and d["resident_steps"] and d["resident_births"]:
        ratio = (d["mutant_births"] / d["mutant_steps"]) / (d["resident_births"] / d["resident_steps"])
    return freq(a), freq(b), ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--switch", type=int, default=SWITCH_STEP)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser()
    results = [json.loads(line) for line in (out_dir / "results.jsonl").read_text().splitlines() if line.strip()]

    print(f"{'mutant':22s} {'n':>3s} {'freq@switch':>11s} {'freq@end':>9s} {'rises':>6s} "
          f"{'birth ratio (median)':>21s} {'Wilcoxon p':>11s}")
    for tag in sorted({r["tag"] for r in results}):
        rows = [x for x in (per_seed(out_dir, r, args.switch) for r in results if r["tag"] == tag) if x]
        if not rows:
            print(f"{tag:22s} no complete runs")
            continue
        f0, f1, ratio = map(np.array, zip(*rows))
        log_ratio = np.log(ratio[np.isfinite(ratio) & (ratio > 0)])
        p = wilcoxon(log_ratio).pvalue if len(log_ratio) >= 5 else float("nan")
        print(f"{tag:22s} {len(rows):3d} {f0.mean():11.2f} {f1.mean():9.2f} {int((f1 > f0).sum()):3d}/{len(rows):<2d} "
              f"{np.median(ratio):21.3f} {p:11.3g}")
    from scipy.stats import mannwhitneyu

    print(f"\nPRIMARY (whole trajectory to the switch at {args.switch})")
    traj = {}
    for tag in sorted({r["tag"] for r in results}):
        rows = [x for x in (whole_trajectory(out_dir, r, args.switch) for r in results if r["tag"] == tag) if x]
        if rows:
            traj[tag] = np.array(rows)
    base = next((t for t in traj if t.startswith("identical")), None)
    n_variants = len(traj) - (base is not None)
    for tag, arr in traj.items():
        lr = np.log(arr[:, 2][np.isfinite(arr[:, 2]) & (arr[:, 2] > 0)])
        p_ratio = wilcoxon(lr).pvalue if len(lr) >= 5 else float("nan")
        line = (f"  {tag:22s} n={len(arr)} freq 500 {arr[:, 0].mean():.2f} -> switch {arr[:, 1].mean():.2f} | "
                f"warm-up birth ratio {np.nanmedian(arr[:, 2]):.3f} (p={p_ratio:.2g})")
        if base and tag != base:
            p = mannwhitneyu(arr[:, 1], traj[base][:, 1]).pvalue
            beats = p < 0.05 / max(n_variants, 1) and arr[:, 1].mean() > traj[base][:, 1].mean()
            line += f" | freq vs identical p={p:.2g} {'BEATS RESIDENT' if beats else ''}"
        print(line)

    n_ext = {}
    for r in results:
        n_ext.setdefault(r["tag"], []).append(r["end_reason"])
    print("\nend reasons:", {t: {k: v.count(k) for k in set(v)} for t, v in n_ext.items()})


if __name__ == "__main__":
    main()
