"""Test 2, the coevolution branches (DESIGN_fov_tradeoff.md in
eco_evolutionary_continuous_traits, addendum of 2026-10-29;
launchers/coevolution.py).

From the gene-pool snapshots (every 10 iterations, the final one in
checkpoint/ itself; end = mean of the snapshots over the last 300
iterations, as in test 1):

- Predator: mean field-of-view gene. C_pred = end in `both` minus end in
  `prey_frozen`, per branch point; supported if |C_pred| >= 0.05 with the
  same sign at both.
- Prey: mean expressed field of view, min(240 exp(gene), 360) degrees
  averaged over genomes. C_prey = end in `both` minus end in
  `predator_frozen`; supported if |C_prey| >= 10 degrees with the same sign
  at both. Secondary (interpretive): the same contrast on the mean
  expressed field over iterations 0 to 300 (10 degrees) and on the end
  gene (0.05).
- Manipulation check: a frozen species' pool is the branch point's, so its
  genes are reported from the living animals at each snapshot.
- Guard (test 3): both species coexist in at least 45 of 50 episodes of
  final_check.py, results/final_check_coevolution_<source>_<arm>_summary.json;
  a contrast with an arm that fails, or has no check yet, is flagged.
- Reported with each contrast: per-worker values (each worker's end in arm
  a minus arm b's pooled end; workers share the policy and are not
  independent replicates) and the largest threshold at which it would be
  supported (None when the signs differ). Test 2 overall: reciprocal if
  both primary contrasts are supported, one-sided if one is, else none.

Arms that have not finished are skipped. Writes
results/coevolution_summary.json and assets/coevolution_traits.png.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.analyze_coevolution
"""

import json
import re
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.analyze_pilot import worker_genomes
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.checkpoints import GENE_POOLS, run_settings
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.environment import SPECIES, TRAITS, pooled_gene_pool
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.launchers.coevolution import ARMS, SOURCES, branch_point

MODULE = Path(__file__).resolve().parent
RUNS = MODULE / "runs"
LAST = 300
EARLY = 300  # iterations 0 to EARLY: the prey's secondary, uncapped measure
FOV = TRAITS.index("fov")
PREY_BASE_FIELD = 240.0
THRESHOLDS = {"predator_gene": 0.05, "prey_field": 10.0, "prey_gene": 0.05}
GUARD = 45  # of 50 episodes coexisting


def prey_field(genomes):
    """Mean expressed prey field of view in degrees."""
    genomes = np.asarray(genomes, dtype=float)
    return float(np.mean(np.minimum(PREY_BASE_FIELD * np.exp(genomes[:, FOV]), 360.0))) if len(genomes) else float("nan")


def gene(genomes):
    genomes = np.asarray(genomes, dtype=float)
    return float(genomes[:, FOV].mean()) if len(genomes) else float("nan")


def snapshots(run):
    checkpoint = run / "checkpoint"
    found = [(int(re.search(r"iter_(\d+)", str(p)).group(1)), p) for p in checkpoint.glob(f"iter_*/{GENE_POOLS}")]
    final = checkpoint / GENE_POOLS
    if not final.is_file():
        return None  # not finished
    found.append((run_settings(checkpoint)["config_ppo"]["iterations"], final))
    return sorted(found)


def analyse_run(source, arm):
    shots = snapshots(RUNS / f"coevolution_{source}_{arm}")
    if shots is None:
        return None
    iterations = np.array([i for i, _ in shots])
    end = iterations > iterations.max() - LAST
    early = iterations <= EARLY
    series = {"predator_gene": [], "prey_field": [], "prey_gene": [], "alive_predator_gene": [], "alive_prey_field": [], "alive_prey_gene": []}
    workers = {"predator_gene": [], "prey_field": [], "prey_gene": []}
    for _, path in shots:
        pooled = pooled_gene_pool(path)
        raw = json.loads(path.read_text())
        series["predator_gene"].append(gene(pooled["predator"]))
        series["prey_field"].append(prey_field(pooled["prey"]))
        series["prey_gene"].append(gene(pooled["prey"]))
        alive = {s: [g for w in raw for g in w[s]["alive"]] for s in SPECIES}
        series["alive_predator_gene"].append(gene(alive["predator"]))
        series["alive_prey_field"].append(prey_field(alive["prey"]))
        series["alive_prey_gene"].append(gene(alive["prey"]))
        workers["predator_gene"].append([gene(worker_genomes(w["predator"])) for w in raw])
        workers["prey_field"].append([prey_field(worker_genomes(w["prey"])) for w in raw])
        workers["prey_gene"].append([gene(worker_genomes(w["prey"])) for w in raw])
    series = {k: np.array(v) for k, v in series.items()}
    start = pooled_gene_pool(branch_point(source) / GENE_POOLS)
    return {
        "iterations": iterations.tolist(),
        "start": {"predator_gene": gene(start["predator"]), "prey_field": prey_field(start["prey"]), "prey_gene": gene(start["prey"])},
        "end": {k: float(np.nanmean(v[end])) for k, v in series.items()},
        "end_sd": {k: float(np.nanstd(v[end])) for k, v in series.items()},
        # iteration 0 is the branch point's pool
        "early_prey_field": float(np.nanmean(np.r_[prey_field(start["prey"]), series["prey_field"][early]])),
        "end_workers": {k: np.nanmean(np.array(v)[end], axis=0).tolist() for k, v in workers.items()},
        "trajectory": {k: v.tolist() for k, v in series.items()},
        "guard": guard(source, arm),
    }


def guard(source, arm):
    path = MODULE / "results" / f"final_check_coevolution_{source}_{arm}_summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())["summary"]
    return {"coexisting": summary["coexisting"], "episodes": summary["episodes"], "passed": summary["coexisting"] >= GUARD}


def contrast(results, a, b, key, early=False):
    """Per branch point: value in arm a minus in arm b, and per worker of arm a
    against arm b's pooled value (None if either arm is missing)."""
    values, workers = {}, {}
    for source in SOURCES:
        ra, rb = results[source].get(a), results[source].get(b)
        if ra is None or rb is None:
            values[source] = workers[source] = None
        elif early:
            values[source], workers[source] = ra["early_prey_field"] - rb["early_prey_field"], None
        else:
            values[source] = ra["end"][key] - rb["end"][key]
            workers[source] = [w - rb["end"][key] for w in ra["end_workers"][key]]
    return values, workers


def supported(values, threshold):
    if any(v is None for v in values.values()):
        return None
    signs = {np.sign(v) for v in values.values()}
    return all(abs(v) >= threshold for v in values.values()) and len(signs) == 1


def largest_threshold(values):
    """The largest threshold at which the contrast would be supported."""
    if any(v is None for v in values.values()) or len({np.sign(v) for v in values.values()}) != 1:
        return None
    return min(abs(v) for v in values.values())


def flagged(results, arms):
    """True if an arm of the contrast failed the guard or has no check yet."""
    return any((r := results[source].get(arm)) is None or r["guard"] is None or not r["guard"]["passed"] for source in SOURCES for arm in arms)


def verdict(results):
    tests = {
        "predator_responds": (("both", "prey_frozen"), "predator_gene", False, THRESHOLDS["predator_gene"]),
        "prey_responds": (("both", "predator_frozen"), "prey_field", False, THRESHOLDS["prey_field"]),
        "prey_secondary_early_field": (("both", "predator_frozen"), None, True, THRESHOLDS["prey_field"]),
        "prey_secondary_end_gene": (("both", "predator_frozen"), "prey_gene", False, THRESHOLDS["prey_gene"]),
    }
    out = {}
    for name, (arms, key, early, threshold) in tests.items():
        values, workers = contrast(results, *arms, key, early=early)
        out[name] = {
            "arms": arms,
            "contrast": values,
            "per_worker": workers,
            "threshold": threshold,
            "supported": supported(values, threshold),
            "supported_up_to_threshold": largest_threshold(values),
            "guard_flagged": flagged(results, arms),
        }
    primary = [out["predator_responds"]["supported"], out["prey_responds"]["supported"]]
    out["coevolution"] = None if None in primary else {2: "reciprocal", 1: "one-sided", 0: "none detected"}[sum(primary)]
    return out


def plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"both": "#333333", "prey_frozen": "#2471a3", "predator_frozen": "#c0392b"}
    rows = (("predator_gene", "predator fov gene"), ("prey_field", "prey field (degrees)"))
    fig, axes = plt.subplots(len(rows), len(SOURCES), figsize=(11, 7), sharex=True)
    for col, source in enumerate(SOURCES):
        for row, (key, label) in enumerate(rows):
            ax = axes[row, col]
            for arm in ARMS:
                r = results[source].get(arm)
                if r is None:
                    continue
                ax.plot([0] + r["iterations"], [r["start"][key]] + r["trajectory"][key], color=colors[arm], lw=1.0, label=arm)
                ax.plot(r["iterations"], r["trajectory"][f"alive_{key}"], color=colors[arm], lw=0.6, alpha=0.4, ls=":")
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"branch point: {source}, iteration 10")
                ax.legend(frameon=False, fontsize=8)
        axes[-1, col].set_xlabel("iteration after branching (dotted: living animals)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    results = {source: {arm: analyse_run(source, arm) for arm in ARMS} for source in SOURCES}
    summary = {"runs": results, "test_2": verdict(results)}
    (MODULE / "results").mkdir(exist_ok=True)
    (MODULE / "results" / "coevolution_summary.json").write_text(json.dumps(summary) + "\n")
    plot(results, MODULE / "assets" / "coevolution_traits.png")
    for source in SOURCES:
        print(source)
        for arm in ARMS:
            r = results[source][arm]
            if r is None:
                print(f"  {arm:16s} not finished")
                continue
            e, sd = r["end"], r["end_sd"]
            print(
                f"  {arm:16s} predator gene {r['start']['predator_gene']:+.3f} -> {e['predator_gene']:+.3f} (sd {sd['predator_gene']:.3f}, alive {e['alive_predator_gene']:+.3f})"
                f"  prey field {r['start']['prey_field']:.0f} -> {e['prey_field']:.0f} (alive {e['alive_prey_field']:.0f}; 0-300 {r['early_prey_field']:.0f})"
                f"  prey gene {e['prey_gene']:+.3f} (alive {e['alive_prey_gene']:+.3f})"
            )
            print(f"  {'':16s} per worker predator {np.round(r['end_workers']['predator_gene'], 3).tolist()}")
            g = r["guard"]
            status = "no check yet" if g is None else f"{g['coexisting']} of {g['episodes']} coexisting, passed {g['passed']}"
            print(f"  {'':16s} guard {status}")
    for name, t in summary["test_2"].items():
        if name == "coevolution":
            print(f"coevolution: {t}")
            continue
        values = {k: (None if v is None else round(v, 4)) for k, v in t["contrast"].items()}
        upto = t["supported_up_to_threshold"]
        print(
            f"{name}: {values} threshold {t['threshold']} supported {t['supported']}"
            f" (supported up to {'-' if upto is None else round(upto, 4)}; guard flagged {t['guard_flagged']})"
        )
        for source, workers in (t["per_worker"] or {}).items():
            if workers is not None:
                print(f"  {source} per worker {np.round(workers, 3).tolist()}")


if __name__ == "__main__":
    main()
