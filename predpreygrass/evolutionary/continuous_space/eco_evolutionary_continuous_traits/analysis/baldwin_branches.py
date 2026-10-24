"""Baldwin-style branches from CT-05 at iteration 1,500: does body evolution
change course when behaviour stops learning?

BL-1: both species learning; BL-2: both frozen (evolve_frozen.py); BL-3:
predators frozen; BL-4: prey frozen. All start from CT-05's iteration-1,500
policies and pooled gene pool and run 1,500 iterations' worth of
environment steps (4,800 decisions each) on 7 simulation workers or
islands. Measures, per species and gene:

- the mean gene of the genomes alive in all workers, from the gene pools
  saved every 10 iterations (the same measure for every branch);
- its change from the start to the mean of the last 300 iterations'
  snapshots;
- 2 x 2 effects on that change: predator learning ((BL-1 + BL-4) - (BL-2 +
  BL-3)) / 2, prey learning ((BL-1 + BL-3) - (BL-2 + BL-4)) / 2;
- a rough drift reference (not a calibrated null): the same change (one
  snapshot to the mean of the snapshots 1,210–1,500 iterations later), for
  genes without effect, in the saved gene pools of the neutral runs CT-02
  and CT-03 (14 and 10 workers), over overlapping windows starting every
  100 iterations, and four such changes drawn at random and combined like
  the 2 x 2 effects. It pools traits, species and runs, and the branches'
  7 workers probably drift more, so it likely understates drift;
- the selection differential averaged over the branch (RLlib branches:
  logged per iteration; BL-2: per episode).

Writes analysis/results/baldwin_branches.json and the README figure.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.analysis.baldwin_branches
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.analysis.pilot import (
    AXIS,
    GRID,
    INK,
    INK_2,
    MODULE,
    MUTED,
    REPO,
    SPECIES,
    TRAITS,
    TRAIT_NAMES,
    scalars,
)
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import load_gene_pool

RUNS = MODULE / "runs"
START = RUNS / "eco_evolutionary_continuous_traits_2026-10-16_20-53-30_161103" / "checkpoint" / "iter_001500"
BRANCHES = {
    "BL-1": ("eco_evolutionary_continuous_traits_2026-10-18_06-18-25_230954", "both learning"),
    "BL-2": ("bl2_frozen_both", "both frozen"),
    "BL-3": ("eco_evolutionary_continuous_traits_2026-10-18_06-18-45_270476", "predators frozen"),
    "BL-4": ("eco_evolutionary_continuous_traits_2026-10-18_06-19-05_426648", "prey frozen"),
}
LEARNS = {"BL-1": (1, 1), "BL-2": (0, 0), "BL-3": (0, 1), "BL-4": (1, 0)}  # (predators, prey)
NEUTRAL = {
    "CT-02": "eco_evolutionary_continuous_traits_2026-10-15_21-31-42_894279",
    "CT-03": "eco_evolutionary_continuous_traits_2026-10-16_20-52-50_092027",
}
ITERATIONS = 1500
SNAPSHOT = 10
LATE = 300  # iterations at the end averaged for the end state
COLORS = {"BL-1": "#2a78d6", "BL-2": "#eb6834", "BL-3": "#1baf7a", "BL-4": "#6250d6"}


def trajectory(run_dir):
    """{species: (snapshots, genes) array of mean genes}, iterations 10 ...
    1,500 (train.py writes the last to checkpoint/gene_pools.json)."""
    out = {species: [] for species in SPECIES}
    for n in range(SNAPSHOT, ITERATIONS + 1, SNAPSHOT):
        folder = run_dir / "checkpoint" / f"iter_{n:06d}"
        if not (folder / "gene_pools.json").is_file():
            if n != ITERATIONS:
                raise FileNotFoundError(folder / "gene_pools.json")
            folder = run_dir / "checkpoint"  # train.py's final snapshot
        pool = load_gene_pool(folder)
        for species in SPECIES:
            out[species].append(pool[species].mean(0))
    return {species: np.array(values) for species, values in out.items()}


def frozen_selection(run_dir):
    """BL-2's selection differentials averaged over its episodes."""
    values = {f"{species}_{trait}": [] for species in SPECIES for trait in TRAITS}
    for path in sorted((run_dir / "islands").glob("island_*.jsonl")):
        for line in path.read_text().splitlines():
            stats = json.loads(line)["stats"]
            for species in SPECIES:
                for trait in TRAITS:
                    value = stats[species].get(f"{trait}_selection")
                    if value is not None:
                        values[f"{species}_{trait}"].append(value)
    return {key: float(np.mean(v)) for key, v in values.items()}


def neutral_changes():
    """Changes of neutral genes measured like the branches' changes: from one
    gene-pool snapshot to the mean of the snapshots 1,200-1,500 iterations
    later (iterations 1,210-1,500 of the window), windows starting every
    100 iterations, CT-02 and CT-03."""
    changes = []
    for folder in NEUTRAL.values():
        means = {species: [] for species in SPECIES}
        for n in range(SNAPSHOT, 3000, SNAPSHOT):
            pool = load_gene_pool(RUNS / folder / "checkpoint" / f"iter_{n:06d}")
            for species in SPECIES:
                means[species].append(pool[species].mean(0))
        for species in SPECIES:
            series = np.array(means[species])
            for start in range(0, len(series) - ITERATIONS // SNAPSHOT, 100 // SNAPSHOT):
                end = series[start + (ITERATIONS - LATE) // SNAPSHOT + 1 : start + ITERATIONS // SNAPSHOT + 1].mean(0)
                changes.extend(end - series[start])
    return np.array(changes)


def drift_yardstick(rng):
    """Spread of neutral changes, and of the 2 x 2 effect form
    ((a + b) - (c + d)) / 2 built from four randomly drawn neutral changes."""
    changes = neutral_changes()
    effects = np.abs((rng.choice(changes, (10_000, 4)) * [1, 1, -1, -1]).sum(1) / 2)
    summary = lambda v: {"median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "p95": float(np.quantile(v, 0.95)), "max": float(v.max())}
    return {"n_changes": len(changes), "abs_change": summary(np.abs(changes)), "abs_effect": summary(effects)}


def plot(start, trajectories, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True, facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.09, hspace=0.38, wspace=0.22)
    fig.text(0.08, 0.955, "Does body evolution change course when behaviour stops learning?", fontsize=15, weight="bold")
    fig.text(0.08, 0.915, "Four branches from CT-05 at iteration 1,500 (costs ×0.25): mean gene of the living animals in all workers, "
             "saved every 10 iterations", fontsize=10.5, color=INK_2)
    x = np.arange(SNAPSHOT, ITERATIONS + 1, SNAPSHOT)
    for row, species in enumerate(SPECIES):
        for col, trait in enumerate(TRAITS):
            ax = axes[row, col]
            for branch, values in trajectories.items():
                y = values[species][:, col]
                ax.plot(x, y, color=COLORS[branch], lw=0.8, alpha=0.25)
                smooth = np.convolve(y, np.ones(10) / 10, mode="valid")  # 100 iterations
                ax.plot(x[9:], smooth, color=COLORS[branch], lw=2.2, label=f"{branch} · {BRANCHES[branch][1]}")
            ax.axhline(start[species][col], color=AXIS, lw=1, ls=(0, (4, 3)), zorder=0)
            ax.set_title(f"{species.capitalize()} · {TRAIT_NAMES[trait]}", fontsize=10.5, weight="bold", loc="left")
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
            ax.tick_params(colors=MUTED, length=0)
            if col == 0:
                ax.set_ylabel("Mean gene (log-ratio)")
            if row == 1:
                ax.set_xlabel("Iterations after the branch point")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.075, 0.895), ncol=4, frameon=False, fontsize=10)
    fig.text(0.08, 0.015, "Thin: snapshots; thick: 100-iteration moving mean. Dashed: the starting value (CT-05, iteration 1,500). "
             "One run per branch.", fontsize=9, color=INK_2)
    fig.savefig(path, dpi=150, facecolor="white")


def main():
    start_pool = load_gene_pool(START)
    start = {species: start_pool[species].mean(0) for species in SPECIES}
    trajectories = {branch: trajectory(RUNS / folder) for branch, (folder, _) in BRANCHES.items()}
    late = LATE // SNAPSHOT
    change = {
        branch: {species: (values[species][-late:].mean(0) - start[species]) for species in SPECIES}
        for branch, values in trajectories.items()
    }
    effects = {}
    for species in SPECIES:
        for k, trait in enumerate(TRAITS):
            c = {branch: change[branch][species][k] for branch in BRANCHES}
            effects[f"{species}_{trait}"] = {
                "predator_learning": float(((c["BL-1"] + c["BL-4"]) - (c["BL-2"] + c["BL-3"])) / 2),
                "prey_learning": float(((c["BL-1"] + c["BL-3"]) - (c["BL-2"] + c["BL-4"])) / 2),
                "interaction": float((c["BL-1"] - c["BL-3"]) - (c["BL-4"] - c["BL-2"])),
            }
    selection = {}
    for branch, (folder, _) in BRANCHES.items():
        if branch == "BL-2":
            selection[branch] = frozen_selection(RUNS / folder)
        else:
            values = scalars(RUNS / folder)
            selection[branch] = {
                f"{species}_{trait}": float(values[f"{species}_{trait}_selection"].mean()) for species in SPECIES for trait in TRAITS
            }
    output = {
        "start": {species: start[species].tolist() for species in SPECIES},
        "branches": {branch: {"run": folder, "learning": label} for branch, (folder, label) in BRANCHES.items()},
        "change_to_last_300_iterations": {b: {s: v.tolist() for s, v in d.items()} for b, d in change.items()},
        "effects": effects,
        "drift_yardstick": drift_yardstick(np.random.default_rng(20261018)),
        "selection_mean": selection,
        "end_mean": {b: {s: trajectories[b][s][-late:].mean(0).tolist() for s in SPECIES} for b in BRANCHES},
    }
    out = MODULE / "analysis" / "results" / "baldwin_branches.json"
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    figure = REPO / "assets" / "images" / "readme" / "continuous_traits_baldwin_branches.png"
    plot(start, trajectories, figure)
    print(json.dumps({k: output[k] for k in ("start", "effects", "drift_yardstick")}, indent=1))
    for b in BRANCHES:
        print(b, {s: np.round(change[b][s], 3).tolist() for s in SPECIES})
    print(out)
    print(figure)


if __name__ == "__main__":
    main()
