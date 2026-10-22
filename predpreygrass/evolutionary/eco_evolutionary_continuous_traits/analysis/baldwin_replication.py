"""Replication of the Baldwin-style branches (launchers/baldwin_replication.py).

Every arm starts from CT-05's iteration-1,500 policies and pooled gene pool
and runs 1,500 iterations' worth of environment steps on 7 workers or
islands, 3 seeds per arm. The measure is the one of baldwin_branches.py: per
species and gene, the change of the mean gene (genomes alive in all workers)
from the start to the mean of the last 300 iterations' snapshots.

Pre-specified test (RESULTS.md section 4): the predator-learning effect on
predator speed, ((both_learn + prey_frozen) - (both_frozen + pred_frozen)) / 2,
with each arm's mean over its seeds. Runs are the unit of replication.
Reported for every gene, with the predator-speed row the confirmatory one:

- the effect and a 95% t interval from the pooled within-arm variance;
- a randomization p value: arm labels shuffled across the runs of the four
  arms (two-sided, and one-sided in the pre-specified direction, > 0).

neutral_frozen (genes without effect, base bodies) gives the spread of the
same change under pure drift, as a reference only: its ecology differs from
the expressed arms.

Runs that have not finished are skipped and listed.

    python -m predpreygrass.evolutionary.eco_evolutionary_continuous_traits.analysis.baldwin_replication
"""

import json

import numpy as np
from scipy import stats

from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.analysis.baldwin_branches import (
    ITERATIONS,
    LATE,
    SNAPSHOT,
    START,
    trajectory,
)
from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.analysis.pilot import MODULE, SPECIES, TRAITS
from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.checkpoints import load_gene_pool

RUNS = MODULE / "runs"
ARMS = ("both_learn", "prey_frozen", "both_frozen", "pred_frozen")
# Signs of each arm in the two main effects (predators learn in both_learn
# and prey_frozen; prey learn in both_learn and pred_frozen).
PREDATOR_LEARNING = {"both_learn": 1, "prey_frozen": 1, "both_frozen": -1, "pred_frozen": -1}
PREY_LEARNING = {"both_learn": 1, "pred_frozen": 1, "both_frozen": -1, "prey_frozen": -1}
SEEDS = (1, 2, 3)
PERMUTATIONS = 20_000


def finished(run_dir):
    return (run_dir / "checkpoint" / "gene_pools.json").is_file() and (
        run_dir / "checkpoint" / f"iter_{ITERATIONS - SNAPSHOT:06d}" / "gene_pools.json"
    ).is_file()


def change(run_dir, start):
    """{species_trait: change} from the start to the last LATE iterations."""
    values = trajectory(run_dir)
    late = LATE // SNAPSHOT
    return {
        f"{species}_{trait}": float(values[species][-late:, k].mean() - start[species][k])
        for species in SPECIES
        for k, trait in enumerate(TRAITS)
    }


def effect(groups, signs):
    return sum(sign * np.mean(groups[arm]) for arm, sign in signs.items()) / 2


def test(groups, signs, rng):
    """Effect, pooled-variance 95% t interval, randomization p values."""
    estimate = effect(groups, signs)
    sizes = {arm: len(groups[arm]) for arm in signs}
    df = sum(n - 1 for n in sizes.values())
    pooled = sum(np.var(groups[arm], ddof=1) * (sizes[arm] - 1) for arm in signs) / df
    se = 0.5 * np.sqrt(pooled * sum(1 / n for n in sizes.values()))
    half = stats.t.ppf(0.975, df) * se
    values = np.concatenate([groups[arm] for arm in signs])
    labels = np.concatenate([[arm] * sizes[arm] for arm in signs])
    null = []
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(labels)
        null.append(effect({arm: values[shuffled == arm] for arm in signs}, signs))
    null = np.asarray(null)
    return {
        "effect": float(estimate),
        "ci95": [float(estimate - half), float(estimate + half)],
        "se": float(se),
        "df": int(df),
        "p_two_sided": float((np.sum(np.abs(null) >= abs(estimate) - 1e-12) + 1) / (PERMUTATIONS + 1)),
        "p_positive": float((np.sum(null >= estimate - 1e-12) + 1) / (PERMUTATIONS + 1)),
    }


def main():
    start_pool = load_gene_pool(START)
    start = {species: start_pool[species].mean(0) for species in SPECIES}
    changes, missing = {}, []
    for arm in ARMS + ("neutral_frozen",):
        for seed in SEEDS:
            run = RUNS / f"blrep_{arm}_s{seed}"
            if finished(run):
                changes.setdefault(arm, {})[seed] = change(run, start)
            else:
                missing.append(run.name)

    genes = [f"{species}_{trait}" for species in SPECIES for trait in TRAITS]
    output = {"runs": changes, "missing": missing}
    if all(len(changes.get(arm, {})) >= 2 for arm in ARMS):
        rng = np.random.default_rng(0)
        output["effects"] = {}
        for gene in genes:
            groups = {arm: np.array([c[gene] for c in changes[arm].values()]) for arm in ARMS}
            output["effects"][gene] = {
                "predator_learning": test(groups, PREDATOR_LEARNING, rng),
                "prey_learning": test(groups, PREY_LEARNING, rng),
                "arm_means": {arm: float(groups[arm].mean()) for arm in ARMS},
            }
        output["confirmatory"] = {"gene": "predator_speed", **output["effects"]["predator_speed"]["predator_learning"]}
    if len(changes.get("neutral_frozen", {})) >= 2:
        output["neutral_sd"] = {
            gene: float(np.std([c[gene] for c in changes["neutral_frozen"].values()], ddof=1)) for gene in genes
        }

    path = MODULE / "analysis" / "results" / "baldwin_replication.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"written {path}")
    if missing:
        print("not finished:", ", ".join(missing))
    for gene, result in output.get("effects", {}).items():
        for name in ("predator_learning", "prey_learning"):
            r = result[name]
            print(
                f"{gene:16s} {name:17s} {r['effect']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] "
                f"p2={r['p_two_sided']:.3f} p+={r['p_positive']:.3f}"
            )


if __name__ == "__main__":
    main()
