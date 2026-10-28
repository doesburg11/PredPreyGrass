"""Pre-specified test 1 of the pilot (DESIGN_fov_tradeoff.md in
eco_evolutionary_continuous_traits): the change in the mean field-of-view
gene from start to end (mean over the last 300 iterations), negative for
predators and non-negative for prey in the trade-off run, rising for both in
the sigma 0 control.

End: the gene-pool snapshots saved with every checkpoint (every 10
iterations; the last one, at the final iteration, in checkpoint/ itself),
not TensorBoard, whose trait means are EMA-smoothed. A snapshot pools the
genomes of every simulation worker as pooled_gene_pool does (the living, or
a worker's founder pool or recent births for a species with none alive).

Start: the founders of every worker's first episode, rebuilt exactly. RLlib
seeds only a worker's first reset, with seed + worker index (workers 1 to
num_env_runners), and founders are drawn at reset from the run's
environment config, so resetting a fresh environment with that seed draws
the same founders. The change from the first snapshot (iteration 10) is
reported too, as a check on how much happened before it.

Per worker the same change is computed from that worker's genomes alone
(from its own founders, each worker weighted equally): workers are islands
that share the policy, so not independent replicates, but their signs show
whether a mean is carried by a few islands.

Also the whole trajectory of every gene (for cycles, Olson et al. 2016) and
the population sizes. Writes results/pilot_summary.json and
assets/pilot_traits.png.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.analyze_pilot
"""

import json
import re
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.checkpoints import GENE_POOLS, checkpoint_env_config, run_settings
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.environment import (
    SPECIES,
    TRAITS,
    PredPreyGrassContinuous,
    pooled_gene_pool,
)

MODULE = Path(__file__).resolve().parent
RUNS = MODULE / "runs"
ARMS = ("tradeoff", "control")
LAST = 300  # iterations at the end that make up "end"
FOV = TRAITS.index("fov")


def snapshots(arm):
    """[(iteration, gene_pools.json path)] of a pilot arm, in order, the final
    checkpoint's included."""
    checkpoint = RUNS / f"pilot_{arm}" / "checkpoint"
    found = [(int(re.search(r"iter_(\d+)", str(p)).group(1)), p) for p in checkpoint.glob(f"iter_*/{GENE_POOLS}")]
    if (checkpoint / GENE_POOLS).is_file():
        found.append((run_settings(checkpoint)["config_ppo"]["iterations"], checkpoint / GENE_POOLS))
    return sorted(found)


def founders(arm):
    """[{species: (n, genes) array}] per worker: the founders of its first
    episode."""
    checkpoint = RUNS / f"pilot_{arm}" / "checkpoint"
    ppo = run_settings(checkpoint)["config_ppo"]
    env = PredPreyGrassContinuous(checkpoint_env_config(checkpoint))
    out = []
    for worker in range(1, ppo["num_env_runners"] + 1):
        env.reset(seed=ppo["seed"] + worker)
        out.append({species: env.trait_values(species) for species in SPECIES})
    return out


def worker_genomes(entry):
    """A worker's genomes of one species, as pooled_gene_pool takes them."""
    if entry.get("frozen") and entry["pool"]:
        return entry["pool"]
    return entry["alive"] or entry["pool"] or entry["archive"]


def mean_or_nan(genomes):
    return np.mean(genomes, axis=0) if len(genomes) else np.full(len(TRAITS), np.nan)


def analyse(arm):
    shots = snapshots(arm)
    start = founders(arm)
    iterations = np.array([i for i, _ in shots])
    end = iterations > iterations.max() - LAST
    out = {"iterations": iterations.tolist(), "end_snapshots": int(end.sum()), "species": {}}
    for species in SPECIES:
        genes, counts, workers = [], [], []
        for _, path in shots:
            pool = pooled_gene_pool(path)[species]
            genes.append(mean_or_nan([] if pool is None else pool))
            counts.append(0 if pool is None else len(pool))
            workers.append([mean_or_nan(worker_genomes(w[species])) for w in json.loads(path.read_text())])
        genes, workers = np.array(genes), np.array(workers)  # (snapshots, genes), (snapshots, workers, genes)
        start_mean = np.concatenate([w[species] for w in start]).mean(axis=0)
        start_workers = np.array([w[species].mean(axis=0) for w in start])
        end_mean = np.nanmean(genes[end], axis=0)
        per_worker = np.nanmean(workers[end, :, FOV], axis=0) - start_workers[:, FOV]
        out["species"][species] = {
            "fov_change": float(end_mean[FOV] - start_mean[FOV]),
            "fov_change_from_first_snapshot": float(end_mean[FOV] - genes[0, FOV]),
            "fov_change_per_worker": per_worker.tolist(),
            "start_mean": dict(zip(TRAITS, start_mean.tolist())),
            "first_snapshot": dict(zip(TRAITS, genes[0].tolist())),
            "end_mean": dict(zip(TRAITS, end_mean.tolist())),
            "fov_min_max": [float(np.nanmin(genes[:, FOV])), float(np.nanmax(genes[:, FOV]))],
            "count_end_mean": float(np.mean(np.array(counts)[end])),
            "trajectory": {t: genes[:, k].tolist() for k, t in enumerate(TRAITS)},
            "count": counts,
        }
    return out


def verdict(results):
    trade = results["tradeoff"]["species"]
    control = results["control"]["species"]
    return {
        "tradeoff_predator_negative": trade["predator"]["fov_change"] < 0,
        "tradeoff_prey_non_negative": trade["prey"]["fov_change"] >= 0,
        "control_both_rise": control["predator"]["fov_change"] > 0 and control["prey"]["fov_change"] > 0,
    }


def plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(TRAITS) + 1, 2, figsize=(11, 11), sharex=True)
    colors = {"predator": "#c0392b", "prey": "#2471a3"}
    for col, arm in enumerate(ARMS):
        res = results[arm]
        x = res["iterations"]
        for row, trait in enumerate(TRAITS):
            ax = axes[row, col]
            for species in SPECIES:
                r = res["species"][species]
                ax.plot([0] + x, [r["start_mean"][trait]] + r["trajectory"][trait], color=colors[species], label=species, lw=1.2)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_ylabel(f"{trait} gene")
            if row == 0:
                ax.set_title(f"{arm} (sigma {'0.3' if arm == 'tradeoff' else '0'})")
                ax.legend(frameon=False, fontsize=8)
        ax = axes[-1, col]
        for species in SPECIES:
            ax.plot(x, res["species"][species]["count"], color=colors[species], lw=1.0)
        ax.set_ylabel("alive (all workers)")
        ax.set_xlabel("iteration")
    for row in range(len(TRAITS)):
        lo = min(axes[row, c].get_ylim()[0] for c in range(2))
        hi = max(axes[row, c].get_ylim()[1] for c in range(2))
        for c in range(2):
            axes[row, c].set_ylim(lo, hi)
    for c, arm in enumerate(ARMS):
        last = max(results[arm]["iterations"])
        for row in range(len(TRAITS)):
            axes[row, c].axvspan(last - LAST, last, color="grey", alpha=0.1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    results = {arm: analyse(arm) for arm in ARMS}
    results["test_1"] = verdict(results)
    (MODULE / "results").mkdir(exist_ok=True)
    (MODULE / "results" / "pilot_summary.json").write_text(json.dumps(results) + "\n")
    plot(results, MODULE / "assets" / "pilot_traits.png")
    for arm in ARMS:
        print(f"{arm}: {results[arm]['end_snapshots']} end snapshots, iterations {results[arm]['iterations'][0]}..{results[arm]['iterations'][-1]}")
        for species in SPECIES:
            r = results[arm]["species"][species]
            print(
                f"  {species:8s} fov start {r['start_mean']['fov']:+.4f} first {r['first_snapshot']['fov']:+.4f} end {r['end_mean']['fov']:+.4f}"
                f"  change {r['fov_change']:+.4f} (from first snapshot {r['fov_change_from_first_snapshot']:+.4f})"
                f"  per worker {np.round(r['fov_change_per_worker'], 3).tolist()}  alive {r['count_end_mean']:.0f}"
            )
    print("test 1:", results["test_1"])


if __name__ == "__main__":
    main()
