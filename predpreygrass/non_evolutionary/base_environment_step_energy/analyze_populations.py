"""
Six-seed population comparison, base_environment vs base_environment_step_energy (see RESULTS.md section 19).

Reads each run's Ray Tune result.json under ~/simulation_results/ray_results (seed 42 uses the
pre-existing runs, seeds 43-47 the multi-seed runs) and prints:
  - the population table for training iterations 201-300 (predator extinction rate, predators, prey,
    episode length; averaged over iterations that logged a completed episode)
  - tests across the six seeds (unpaired and paired, one-sided)
  - each seed's base_environment_step_energy trajectory by training window (is a struggling seed recovering?)
  - the clustering result recomputed without seed 45 (uses clustering_results.json)

Usage: python analyze_populations.py
"""
import glob
import json
import math
import os

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

RR = os.path.expanduser("~/simulation_results/ray_results")
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 43, 44, 45, 46, 47]


def result_path(kind, seed):
    if seed == 42:
        if kind == "base":
            return (f"{RR}/base_v_drive_2026-09-06/PPO_BASE_ENVIRONMENT_SEED42_2026-09-05_18-55-45/"
                    "PPO_PredPreyGrass_a2fe1_00000_0_2026-09-05_18-55-45/result.json")
        return glob.glob(f"{RR}/PPO_STEP_ENERGY_ADDITIVE_PREDEASE_CONFIRM500_SEED42/PPO_*/result.json")[0]
    name = (f"PPO_BASE_ENVIRONMENT_SEED{seed}_MULTISEED" if kind == "base"
            else f"PPO_STEP_ENERGY_RUNB_SEED{seed}_MULTISEED")
    return glob.glob(f"{RR}/{name}/PPO_*/result.json")[0]


def get(rec, key):
    v = rec.get("env_runners", {}).get(f"ecology/{key}")
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def window(kind, seed, lo, hi):
    rows = [json.loads(line) for line in open(result_path(kind, seed))]
    rows = [r for r in rows if lo < r["training_iteration"] <= hi and get(r, "episode_length") is not None]
    f = lambda k: float(np.mean([get(r, k) for r in rows]))
    return {"ext": 100 * f("extinct_predator"), "pred": f("final_num_predators"),
            "prey": f("final_num_prey"), "len": f("episode_length")}


T = {(k, s): window(k, s, 200, 300) for k in ("base", "step") for s in SEEDS}
print("POPULATIONS, training iterations 201-300")
print("| seed | base: extinction | pred | prey | len | step_energy: extinction | pred | prey | len |")
for s in SEEDS:
    b, t = T[("base", s)], T[("step", s)]
    print(f"| {s} | {b['ext']:.0f}% | {b['pred']:.1f} | {b['prey']:.1f} | {b['len']:.0f} | "
          f"{t['ext']:.0f}% | {t['pred']:.1f} | {t['prey']:.1f} | {t['len']:.0f} |")
m = lambda k, f: np.mean([T[(k, s)][f] for s in SEEDS])
print(f"| mean | {m('base', 'ext'):.0f}% | {m('base', 'pred'):.1f} | {m('base', 'prey'):.1f} | {m('base', 'len'):.0f} | "
      f"{m('step', 'ext'):.0f}% | {m('step', 'pred'):.1f} | {m('step', 'prey'):.1f} | {m('step', 'len'):.0f} |")

for f, alt, label in (("pred", "greater", "predators, base > step_energy"), ("prey", "less", "prey, base < step_energy")):
    b = np.array([T[("base", s)][f] for s in SEEDS])
    t = np.array([T[("step", s)][f] for s in SEEDS])
    n_dir = int((b > t).sum()) if alt == "greater" else int((b < t).sum())
    print(f"{label}: {n_dir}/6 seeds; unpaired one-sided p={mannwhitneyu(b, t, alternative=alt).pvalue:.4f}; "
          f"paired p={wilcoxon(b, t, alternative=alt).pvalue:.4f}; base {b.min():.1f}-{b.max():.1f}, step_energy {t.min():.1f}-{t.max():.1f}")

print("\nbase_environment_step_energy trajectory: predator extinction % / predators, by training window")
print("seed | 1-100 | 101-200 | 201-250 | 251-300")
for s in SEEDS:
    cells = []
    for lo, hi in ((0, 100), (100, 200), (200, 250), (250, 300)):
        w = window("step", s, lo, hi)
        cells.append(f"{w['ext']:3.0f}% / {w['pred']:4.1f}")
    print(f"{s}   | " + " | ".join(cells))

d = json.load(open(os.path.join(HERE, "clustering_results.json")))["per_seed"]
keep = [s for s in SEEDS if s != 45]
b = np.array([np.mean(d[str(s)]["base"]) for s in keep])
t = np.array([np.mean(d[str(s)]["step_energy_runB"]) for s in keep])
print("\nCLUSTERING R without seed 45 (5 pairs): base %.3f, step_energy %.3f, difference %.3f; pairs in predicted direction %d/5"
      % (b.mean(), t.mean(), (b - t).mean(), int((b > t).sum())))
print("  paired Wilcoxon one-sided p=%.4f (floor for 5 pairs is 0.0313); unpaired Mann-Whitney one-sided p=%.4f"
      % (wilcoxon(b, t, alternative="greater").pvalue, mannwhitneyu(b, t, alternative="greater").pvalue))
