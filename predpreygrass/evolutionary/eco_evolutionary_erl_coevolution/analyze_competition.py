"""Confirmatory carnivore competition test with delayed type assignment
(`mixed_assign_step`). Written and fixed before the results (2026-09-29).

From the assignment step (50/50 split of living carnivores after the warm-up)
to the end of each run, per type: net per-capita growth rate
= 1,000 * (births - deaths) / carnivore-steps. Mortality is included, unlike
the birth-only ratio of analyze_invasion.py.

PRIMARY: the per-seed difference (mutant - resident) of net growth rate,
compared between the variant tag and the "identical" tag (Mann-Whitney,
two-sided, p < 0.05). SECONDARY: mutant frequency at the end, and fixation
counts (mutant share 1.0 or 0.0 at the end).

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.analyze_competition \\
        --out-dir ~/simulation_results/erl_results/coevo_competition --assign-step 20000
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon


def per_seed(out_dir: Path, r: dict, assign: int):
    path = out_dir / "timeseries" / f"{r['tag']}_{r['strategy']}_seed{r['seed']}.csv"
    with open(path) as f:
        rows = {int(x["step"]): x for x in csv.DictReader(f)}
    end = max(rows)
    if assign not in rows or end <= assign:
        return None
    a, b = rows[assign], rows[end]
    if int(a["carn_mutants"]) == 0 or int(a["carn_mutants"]) == int(a["carnivore_count"]):
        return None  # split didn't produce both types

    def net_rate(kind):
        births = int(b[f"{kind}_births"]) - int(a[f"{kind}_births"])
        deaths = int(b[f"{kind}_deaths"]) - int(a[f"{kind}_deaths"])
        steps = int(b[f"{kind}_steps"]) - int(a[f"{kind}_steps"])
        return 1000 * (births - deaths) / steps if steps else float("nan")

    end_freq = int(b["carn_mutants"]) / int(b["carnivore_count"]) if int(b["carnivore_count"]) else float("nan")
    return net_rate("mutant") - net_rate("resident"), end_freq


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--assign-step", type=int, default=20_000)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser()
    results = [json.loads(line) for line in (out_dir / "results.jsonl").read_text().splitlines() if line.strip()]

    data = {}
    for tag in sorted({r["tag"] for r in results}):
        rows = [x for x in (per_seed(out_dir, r, args.assign_step) for r in results if r["tag"] == tag) if x]
        if rows:
            data[tag] = np.array(rows)
    base = next((t for t in data if t.startswith("identical")), None)

    print(f"net growth rate difference, mutant - resident (per 1,000 carnivore-steps), from step {args.assign_step}")
    for tag, arr in data.items():
        diff, freq = arr[:, 0], arr[:, 1]
        diff = diff[np.isfinite(diff)]
        within = wilcoxon(diff).pvalue if len(diff) >= 5 else float("nan")
        line = (f"  {tag:22s} n={len(arr)} median diff {np.median(diff):+.3f} (vs 0: p={within:.2g}) | "
                f"end freq {np.nanmean(freq):.2f}, fixed {int((freq == 1).sum())}, lost {int((freq == 0).sum())}")
        if base and tag != base:
            p = mannwhitneyu(diff, data[base][:, 0][np.isfinite(data[base][:, 0])]).pvalue
            line += f" | PRIMARY vs identical p={p:.3g} -> {'MUTANT BETTER' if p < 0.05 and np.median(diff) > np.median(data[base][:, 0]) else 'no difference' if p >= 0.05 else 'MUTANT WORSE'}"
        print(line)
    ends = {}
    for r in results:
        ends.setdefault(r["tag"], []).append(r["end_reason"])
    print("end reasons:", {t: {k: v.count(k) for k in set(v)} for t, v in ends.items()})


if __name__ == "__main__":
    main()
