"""Persistence-trait landscape check (step 2c), fixed before the results (2026-09-30).

Founders get wide variation on the persistence weights (previous move i ->
action i, N(0, 3)). Per seed, from carnivore_lineage/ (rich_memory layout), the
persist trait = mean of those four weights per carnivore:

  - mean persist of carnivores born in the first 2k steps vs. the last 10k
  - upper-tail share (persist > +2) early vs. late  -> selection FOR persistence
  - lower-tail share (persist < -2) early vs. late  -> purging of anti-persistence
  - standardized selection gradient of offspring count on persist
    (dead carnivores born after the switch)

Reading, stated in advance: selection favors persistence if the late mean ends
clearly above 0 AND the upper-tail share grows, in significantly more step2
seeds than chance (sign test) and not in the neutral control. A mean that only
climbs from below toward 0, with the lower tail shrinking but the upper tail not
growing, means purging, not selection for persistence.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.analyze_persist \\
        --lineage-dir ~/simulation_results/erl_results/coevo_step2c/carnivore_lineage --real-tag X --neutral-tag Y
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, mannwhitneyu

PREV = 16  # rich_memory: previous-move rows 16-19
EARLY_END = 2_000
LATE_WINDOW = 10_000
SWITCH = 20_000
TAIL = 2.0


def seed_stats(path: Path) -> dict | None:
    born, persist, offspring, censored = [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            born.append(int(r["born_step"]))
            persist.append(np.mean([float(r[f"w{PREV + i}_{i}"]) for i in range(4)]))
            offspring.append(int(r["offspring_count"]))
            censored.append(int(r["censored"]))
    born, persist, offspring, censored = map(np.array, (born, persist, offspring, censored))
    end = born.max()
    early, late = persist[born < EARLY_END], persist[born >= end - LATE_WINDOW]
    if len(early) < 20 or len(late) < 20:
        return None
    sel = (born >= SWITCH) & (censored == 0)
    p, o = persist[sel], offspring[sel]
    gradient = float(np.cov(p, o)[0, 1] / p.var() / max(o.mean(), 1e-9)) if len(p) > 50 and p.var() > 0 else np.nan
    return {
        "early_mean": early.mean(), "late_mean": late.mean(),
        "early_upper": (early > TAIL).mean(), "late_upper": (late > TAIL).mean(),
        "early_lower": (early < -TAIL).mean(), "late_lower": (late < -TAIL).mean(),
        "gradient": gradient,
    }


def summarize(lineage_dir: Path, tag: str) -> list[dict]:
    return [s for s in (seed_stats(p) for p in sorted(lineage_dir.glob(f"{tag}_ERL_seed*.csv"))) if s]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lineage-dir", required=True)
    parser.add_argument("--real-tag", required=True)
    parser.add_argument("--neutral-tag", required=True)
    args = parser.parse_args()
    d = Path(args.lineage_dir).expanduser()
    groups = {"step2": summarize(d, args.real_tag), "neutral": summarize(d, args.neutral_tag)}
    for name, rows in groups.items():
        if not rows:
            print(f"{name}: no seeds")
            continue
        n = len(rows)
        col = lambda k: np.array([r[k] for r in rows])
        up = int((col("late_mean") > col("early_mean")).sum())
        tail_up = int((col("late_upper") > col("early_upper")).sum())
        g = col("gradient")[np.isfinite(col("gradient"))]
        print(f"{name:8s} n={n} persist mean {col('early_mean').mean():+.2f} -> {col('late_mean').mean():+.2f} "
              f"(rises in {up}/{n}, one-sided p={binomtest(up, n, alternative='greater').pvalue:.2g}) | "
              f"upper tail >{TAIL:+.0f}: {col('early_upper').mean():.1%} -> {col('late_upper').mean():.1%} "
              f"(grows in {tail_up}/{n}, one-sided p={binomtest(tail_up, n, alternative='greater').pvalue:.2g}) | "
              f"lower tail: {col('early_lower').mean():.1%} -> {col('late_lower').mean():.1%} | "
              f"gradient median {np.median(g):+.3f} (positive {int((g > 0).sum())}/{len(g)})")
    if groups["step2"] and groups["neutral"]:
        a = [r["late_mean"] - r["early_mean"] for r in groups["step2"]]
        b = [r["late_mean"] - r["early_mean"] for r in groups["neutral"]]
        print(f"mean change step2 vs neutral: Mann-Whitney p={mannwhitneyu(a, b).pvalue:.2g}")


if __name__ == "__main__":
    main()
