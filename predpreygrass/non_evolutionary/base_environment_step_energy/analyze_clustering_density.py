"""
Density-adjusted clustering analysis (see RESULTS.md section 17). Reads clustering_density_data/*.json
(written by evaluate_clustering_density.py) and prints:
  - a random-placement null: expected Clark-Evans R for N predators placed on distinct cells of the
    25x25 grid (includes edge effects; R is not 1.0 for random placement on a bounded grid)
  - per-policy observed R, mean predator count N, and excess = R - R_random(N)
  - the density-adjusted paired/unpaired test across the six seed pairs
  - the within-policy dependence of R on N, and the seed-42 dose-response table

Usage: python analyze_clustering_density.py
"""
import json
import math
import os

import numpy as np
from scipy.stats import linregress, mannwhitneyu, spearmanr, wilcoxon

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clustering_density_data")
G = 25
SEEDS = [42, 43, 44, 45, 46, 47]
rng = np.random.default_rng(0)
cells = np.array([(x, y) for x in range(G) for y in range(G)], float)


def null_R(N, trials=3000):
    obs = []
    for _ in range(trials):
        pts = cells[rng.choice(G * G, N, replace=False)]
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        obs.append(d.min(1).mean())
    return np.mean(obs) / (0.5 * math.sqrt(G * G / N))


Ns = list(range(6, 27, 2))
tab = {N: null_R(N) for N in Ns}
R0 = lambda N: np.interp(N, Ns, [tab[k] for k in Ns])
print("NULL: expected R for random placement (no shared cells, edge effects included)")
print("  N :", " ".join(f"{N:5d}" for N in Ns))
print("  R0:", " ".join(f"{tab[N]:5.3f}" for N in Ns))


def load(kind, seed):
    return json.load(open(os.path.join(DATA, f"dose_{kind}_{seed}.json")))


rows = {}
print("\nPER POLICY (30 episodes each): R, mean predators N, excess = R - R0(N)")
for kind, seeds in (("base", SEEDS), ("step", SEEDS), ("runA", [42]), ("freerest", [42])):
    for s in seeds:
        d = load(kind, s)
        R = np.array([x["R"] for x in d])
        N = np.array([x["n_pred"] for x in d])
        ex = R - np.array([R0(n) for n in N])
        rows[(kind, s)] = (R.mean(), N.mean(), ex.mean(), R, N)
        print(f"  {kind:4s} s{s}: R={R.mean():.3f} N={N.mean():5.2f} excess={ex.mean():+.3f}")

be = np.array([rows[("base", s)][2] for s in SEEDS])
se = np.array([rows[("step", s)][2] for s in SEEDS])
print("\nDENSITY-ADJUSTED TEST across the six seed pairs (excess clustering vs random at own density)")
print(f"  base excess {be.mean():+.3f} | base_environment_step_energy excess {se.mean():+.3f}")
print(f"  paired Wilcoxon one-sided p={wilcoxon(be, se, alternative='greater').pvalue:.4f}; "
      f"unpaired Mann-Whitney one-sided p={mannwhitneyu(be, se, alternative='greater').pvalue:.4f}; "
      f"pairs base>base_environment_step_energy: {(be > se).sum()}/6")
print("  predator N ranges: base %.1f-%.1f, base_environment_step_energy %.1f-%.1f (no overlap)" % (
    min(rows[("base", s)][1] for s in SEEDS), max(rows[("base", s)][1] for s in SEEDS),
    min(rows[("step", s)][1] for s in SEEDS), max(rows[("step", s)][1] for s in SEEDS)))

print("\nWITHIN-POLICY dependence of R on predator count (episode level, policy-demeaned)")
for kind in ("base", "step"):
    xs, ys = [], []
    for s in SEEDS:
        _, _, _, R, N = rows[(kind, s)]
        xs += list(N - N.mean())
        ys += list(R - R.mean())
    lr = linregress(xs, ys)
    print(f"  {kind}: dR/dN = {lr.slope:+.4f} per predator (p={lr.pvalue:.3g}, r={lr.rvalue:+.2f})")
print("  across all 14 policies, Spearman(R, N) = %.2f" % spearmanr(
    [v[1] for v in rows.values()], [v[0] for v in rows.values()])[0])

print("\nSEED-42 DOSE-RESPONSE (one policy per level; move cost = extra cost of moving over resting)")
for k, label, gap in (("base", "base_environment", 0.00), ("step", "base_environment_step_energy", 0.08),
                      ("runA", "run A", 0.10), ("freerest", "earlier free-resting run (rest 0)", 0.15)):
    v = rows[(k, 42)]
    print(f"  gap {gap:.2f}  {label:32s} R={v[0]:.3f} N={v[1]:.1f} excess={v[2]:+.3f}")
print("  base_environment_step_energy seed-to-seed R range at the SAME setting: %.3f - %.3f" % (
    min(rows[("step", s)][0] for s in SEEDS), max(rows[("step", s)][0] for s in SEEDS)))
