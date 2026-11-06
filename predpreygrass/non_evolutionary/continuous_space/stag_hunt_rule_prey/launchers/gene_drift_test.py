"""Test 2 of RESULTS.md, section 14: did the rabbits evolve beyond drift? For
each run, each lineage's mean gene in its pool at the checkpoint
(gene_pools.load_pools); evolving vs neutral lineages, per gene, two-sided
Mann-Whitney U. Passes if at least one gene has p < 0.025 (0.05, Bonferroni
over the two genes). Run from the repository root:

    python predpreygrass/non_evolutionary/continuous_space/stag_hunt_rule_prey/launchers/gene_drift_test.py \\
        EVOLVING_CHECKPOINT EVOLVING_POOL_DIR NEUTRAL_CHECKPOINT NEUTRAL_POOL_DIR [OUT]
"""

import json
import math
import sys

import numpy as np
from scipy.stats import mannwhitneyu

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import GENES
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.launchers.gene_pools import load_pools

ALPHA = 0.05 / len(GENES)
BASE = {"flee": 120.0, "breeding": 8.0}  # the base phenotypes (config_env)


def lineage_means(pools):
    return np.array([np.mean(np.array(pool, dtype=float), axis=0) for pool in pools])


def main():
    evolving = lineage_means(load_pools(sys.argv[2], sys.argv[1]))
    neutral = lineage_means(load_pools(sys.argv[4], sys.argv[3]))
    result = {"lineages": {"evolving": len(evolving), "neutral": len(neutral)}, "alpha": ALPHA, "genes": {}}
    for i, gene in enumerate(GENES):
        test = mannwhitneyu(evolving[:, i], neutral[:, i], alternative="two-sided")
        row = {
            "evolving_mean": float(evolving[:, i].mean()),
            "neutral_mean": float(neutral[:, i].mean()),
            "evolving_lineages": evolving[:, i].tolist(),
            "neutral_lineages": neutral[:, i].tolist(),
            "evolving_phenotype_ratio": math.exp(float(evolving[:, i].mean())),
            "neutral_phenotype_ratio": math.exp(float(neutral[:, i].mean())),
            "u": float(test.statistic),
            "p": float(test.pvalue),
        }
        result["genes"][gene] = row
        print(
            f"{gene:9s} evolving {row['evolving_mean']:+.3f} (x{row['evolving_phenotype_ratio']:.2f} of base {BASE[gene]:g})"
            f"  neutral {row['neutral_mean']:+.3f} (x{row['neutral_phenotype_ratio']:.2f})  U {row['u']:.0f}  p {row['p']:.2g}"
        )
    result["passed"] = any(row["p"] < ALPHA for row in result["genes"].values())
    print(f"lineages {len(evolving)} vs {len(neutral)}; test 2 passed (p < {ALPHA} for a gene): {result['passed']}")
    if len(sys.argv) > 5:
        with open(sys.argv[5], "w") as handle:
            json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
