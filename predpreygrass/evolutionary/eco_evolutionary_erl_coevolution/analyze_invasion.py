"""Analyze the "mixed" carnivore competition test (resident seed network vs. a
mutant variant, 50/50 start, type inherited). Written before the results.

Per seed (runs that reached the budget), from the time series:
  - mutant frequency at the immigration switch and at the end
  - relative per-capita reproduction after the switch:
    (mutant births / mutant carnivore-steps) / (resident births / resident carnivore-steps)

A variant is favored by within-population selection if its log birth ratio is
> 0 across seeds (Wilcoxon signed-rank) and its frequency rises. The "identical"
tag (mutant == resident) is the neutral yardstick: its ratio should sit at ~1.

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

    return freq(a), freq(b), rate("mutant") / rate("resident")


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
    n_ext = {}
    for r in results:
        n_ext.setdefault(r["tag"], []).append(r["end_reason"])
    print("\nend reasons:", {t: {k: v.count(k) for k in set(v)} for t, v in n_ext.items()})


if __name__ == "__main__":
    main()
