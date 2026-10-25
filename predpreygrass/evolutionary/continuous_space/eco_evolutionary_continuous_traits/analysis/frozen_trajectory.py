"""Why did predator speed fall in the frozen arms? (RESULTS.md section 7)

Two parts, both from mechanism_speed.py's episodes:

1. Births against survival, branch point (condition set "main", same
   prey): per speed value, births per founder's lineage, members alive at
   the end per founder, and the pooled share of the lineage's members
   (founder plus births) alive at the end, under the frozen and the learned policies.
2. The pre-specified test of condition set "frozen_trajectory": the slope of
   relative lineage fitness (members alive at the end) on founder speed for
   the frozen branch-point predator policy with each frozen arm's gene pools
   and prey policy at iterations 500, 1,000 and 1,500. Primary: the mean
   slope over the 6 runs at iteration 1,500 minus the branch-point frozen
   slope on the same seeds, with a run-level 95% t interval and a bootstrap
   over seeds (paired, all conditions resampled together). Also the slopes
   with births, and the mean genes of each gene pool.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.analysis.frozen_trajectory
"""

import json

import numpy as np

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.analysis.mechanism_speed import interval, run_level, slope
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.analysis.pilot import MODULE
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import load_gene_pool
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.mechanism_speed import (
    BRANCH_POINT,
    FROZEN_RUNS,
    LEARNED,
    SPEEDS,
    TRAJECTORY_ITERATIONS,
    conditions,
    output_dir,
)

BOOTSTRAP = 20000
MEASURES = ("alive", "births")


def load(directory, name, episodes=None):
    rows = [json.loads(line) for line in (directory / f"{name}.jsonl").read_text().splitlines()]
    if episodes is not None:
        rows = [r for r in rows if r["seed"] < episodes]
    out = {key: np.array([r[key] for r in rows], dtype=float) for key in ("alive", "births", "speed", "prey_alive")}
    out["seed"] = np.array([r["seed"] for r in rows])
    return out


def births_and_survival(names):
    """Per speed value, mean over the conditions in names."""
    rows = {}
    for v in SPEEDS:
        births, alive, share = [], [], []
        for name in names:
            d = load(output_dir("main"), name)
            m = np.isclose(d["speed"], v)
            births.append(d["births"][m].mean())
            alive.append(d["alive"][m].mean())
            share.append(d["alive"][m].sum() / (1 + d["births"][m]).sum())
        rows[str(v)] = {"births": float(np.mean(births)), "alive": float(np.mean(alive)), "share_alive": float(np.mean(share))}
    return rows


def relative_at(d, v):
    """Mean relative fitness (members alive at the end over the episode's
    mean) of founders with speed gene v, episodes with a mean above zero."""
    mean = d["alive"].mean(axis=1, keepdims=True)
    keep = mean[:, 0] > 0
    relative = d["alive"][keep] / mean[keep]
    return relative[np.isclose(d["speed"][keep], v)].mean()


def mean_genes(checkpoint):
    pool = load_gene_pool(checkpoint)
    return {species: [float(x) for x in pool[species].mean(axis=0)] for species in ("predator", "prey")}


def main():
    split = {
        "frozen": births_and_survival(["frozen__same_prey"]),
        "learned_mean_of_runs": births_and_survival([f"{run}__same_prey" for run in LEARNED]),
    }
    trajectory = output_dir("frozen_trajectory")
    names = list(conditions("frozen_trajectory"))
    data = {name: load(trajectory, name) for name in names}
    episodes = len(data[names[0]]["seed"])
    base = load(output_dir("main"), "frozen__same_prey", episodes)
    for name, d in data.items():
        if not np.array_equal(d["seed"], base["seed"]) or not np.allclose(d["speed"], base["speed"]):
            raise ValueError(f"{name}: seeds or speed assignments differ from the branch point's")
    runs = list(FROZEN_RUNS)
    everything = np.arange(episodes)
    rng = np.random.default_rng(0)
    samples = [rng.integers(0, episodes, episodes) for _ in range(BOOTSTRAP)]

    output = {
        "episodes": episodes,
        "bootstrap": BOOTSTRAP,
        "speeds": list(SPEEDS),
        "conditions": {name: list(map(str, v)) for name, v in conditions("frozen_trajectory").items()},
        "births_and_survival_branch_point": split,
        "measures": {},
    }
    for measure in MEASURES:
        slopes = {name: slope(d, measure, everything) for name, d in data.items()}
        base_slope = slope(base, measure, everything)
        final = [slopes[f"frozen__{run}_it1500"] - base_slope for run in runs]

        def primary(index):
            b = slope(base, measure, index)
            return np.mean([slope(data[f"frozen__{run}_it1500"], measure, index) - b for run in runs])

        output["measures"][measure] = {
            "branch_point_slope": base_slope,
            "slopes": slopes,
            "mean_slope_by_iteration": {
                str(it): float(np.mean([slopes[f"frozen__{run}_it{it}"] for run in runs])) for it in TRAJECTORY_ITERATIONS
            },
            "by_arm_it1500_minus_branch_point": {
                arm: float(np.mean([slopes[f"frozen__{run}_it1500"] - base_slope for run in runs if run.startswith(arm)]))
                for arm in ("both_frozen", "pred_frozen")
            },
            "primary": {
                "estimate": float(np.mean(final)),
                "ci95_bootstrap": interval([primary(index) for index in samples]),
                "run_level": run_level(final),
            },
        }
    output["relative_fitness_by_speed_alive"] = {
        f"{arm}_it{it}": {
            str(v): float(np.mean([relative_at(data[f"frozen__{arm}_s{seed}_it{it}"], v) for seed in (1, 2, 3)])) for v in SPEEDS
        }
        for arm in ("both_frozen", "pred_frozen")
        for it in TRAJECTORY_ITERATIONS
    }
    output["prey_alive_at_end"] = {name: float(d["prey_alive"].mean()) for name, d in data.items()}
    output["mean_genes"] = {"branch_point": mean_genes(BRANCH_POINT)}
    for name, (_, _, pool, _) in conditions("frozen_trajectory").items():
        output["mean_genes"][name] = mean_genes(pool)
    path = MODULE / "analysis" / "results" / "frozen_trajectory.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"written {path}")
    for label, rows in split.items():
        print(f"\n{label}: speed -> births, alive, share alive")
        for v, r in rows.items():
            print(f"  {float(v):+.1f}: {r['births']:.2f} {r['alive']:.2f} {r['share_alive']:.3f}")
    for measure, m in output["measures"].items():
        p = m["primary"]
        print(f"\n{measure}: branch point {m['branch_point_slope']:+.3f}; mean by iteration", {k: round(v, 3) for k, v in m["mean_slope_by_iteration"].items()})
        print("  by arm (it1500 - branch point):", {k: round(v, 3) for k, v in m["by_arm_it1500_minus_branch_point"].items()})
        print(f"  primary (it1500 - branch point): {p['estimate']:+.3f} bootstrap [{p['ci95_bootstrap'][0]:+.3f}, {p['ci95_bootstrap'][1]:+.3f}]"
              f" run level [{p['run_level']['ci95'][0]:+.3f}, {p['run_level']['ci95'][1]:+.3f}], {p['run_level']['positive']}/6 > 0")
        for run in runs:
            print(f"  {run:16s}", "  ".join(f"it{it} {m['slopes'][f'frozen__{run}_it{it}']:+.2f}" for it in TRAJECTORY_ITERATIONS))
    print("\npredator genes (speed, vision, fov) and prey speed of the pools at it1500:")
    for run in runs:
        g = output["mean_genes"][f"frozen__{run}_it1500"]
        print(f"  {run:16s} pred {[round(x, 2) for x in g['predator']]} prey speed {g['prey'][0]:+.2f}  prey alive {output['prey_alive_at_end'][f'frozen__{run}_it1500']:.1f}")
    print("  branch point     pred", [round(x, 2) for x in output["mean_genes"]["branch_point"]["predator"]], f"prey speed {output['mean_genes']['branch_point']['prey'][0]:+.2f}  prey alive {base['prey_alive'].mean():.1f}")


if __name__ == "__main__":
    main()
