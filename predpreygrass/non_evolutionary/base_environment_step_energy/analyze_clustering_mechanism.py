"""
Analysis of the birth-and-dispersal mechanism measurements (see RESULTS.md section 18).
Reads clustering_mechanism_data/ (evaluate_clustering_mechanism.py) and clustering_density_data/
(per-episode predator counts, evaluate_clustering_density.py) and prints:
  - per-policy movement, birth, dispersal and adjacency measures
  - base_environment vs run B across the six seeds, one-sided in the direction the mechanism predicts
  - Spearman correlation of each measure with clustering R across the 12 policies
  - the mean parent-offspring dispersal profile by offspring age
  - adjacency measured against a random-placement null at each episode's own predator count
    (raw adjacency is density-dependent, so it is only interpretable relative to that null)

Usage: python analyze_clustering_mechanism.py
"""
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
MECH = os.path.join(HERE, "clustering_mechanism_data")
DENS = os.path.join(HERE, "clustering_density_data")
AGES = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]
SEEDS = [42, 43, 44, 45, 46, 47]
G = 25


def episodes(kind, seed):
    return json.load(open(os.path.join(MECH, f"mech_{kind}_{seed}.json")))


def policy(kind, seed):
    eps = episodes(kind, seed)
    m = {k: np.mean([e[k] for e in eps]) for k in ("R", "noop_frac", "moved_frac", "mean_disp", "adj_frac")}
    m["birth_rate"] = 1000 * sum(e["births"] for e in eps) / sum(e["pred_steps"] for e in eps)  # per 1000 predator-steps
    for a in AGES:
        c = sum(e["age_cnt"][str(a)] for e in eps)
        m[f"d{a}"] = sum(e["age_sum"][str(a)] for e in eps) / c if c else np.nan
    return m


P = {(k, s): policy(k, s) for k in ("base", "step") for s in SEEDS}
for k in ("runA", "freerest"):
    P[(k, 42)] = policy(k, 42)

print("PER POLICY (mean over 30 episodes); birth = births per 1000 predator-steps; adj = share of predators with a neighbour within 1.5 cells")
print(f"{'':12s} {'R':>6} {'noop':>6} {'moved':>6} {'disp':>6} {'birth':>6} {'adj':>6} {'d(10)':>6} {'d(30)':>6} {'d(50)':>6}")
for (k, s), m in P.items():
    print(f"{k + ' s' + str(s):12s} {m['R']:6.3f} {m['noop_frac']:6.3f} {m['moved_frac']:6.3f} {m['mean_disp']:6.3f} "
          f"{m['birth_rate']:6.2f} {m['adj_frac']:6.3f} {m['d10']:6.2f} {m['d30']:6.2f} {m['d50']:6.2f}")


def compare(key, predicted):
    b = np.array([P[("base", s)][key] for s in SEEDS])
    st = np.array([P[("step", s)][key] for s in SEEDS])
    alt = "greater" if predicted == "lower" else "less"  # test that base is greater when run B is predicted lower
    p_unp = mannwhitneyu(b, st, alternative=alt).pvalue
    p_pair = wilcoxon(b, st, alternative=alt).pvalue
    n_dir = int((b > st).sum()) if predicted == "lower" else int((b < st).sum())
    print(f"  {key:11s} base {b.mean():7.3f}  run B {st.mean():7.3f}  predicted run B {predicted:6s}: "
          f"{n_dir}/6 seeds, unpaired p={p_unp:.4f}, paired p={p_pair:.4f}")


print("\nBASE vs RUN B across the six seeds (one-sided, direction predicted by the mechanism)")
print(" movement:")
for k, d in (("noop_frac", "higher"), ("moved_frac", "lower"), ("mean_disp", "lower")):
    compare(k, d)
print(" offspring dispersal (parent-offspring distance at a given age):")
for a in (5, 10, 20, 30, 50):
    compare(f"d{a}", "lower")
print(" births:")
compare("birth_rate", "higher")

keys = [(k, s) for k in ("base", "step") for s in SEEDS]
Rv = [P[x]["R"] for x in keys]
print("\nSPEARMAN of each measure with R across the 12 base + run B policies (negative = more of the measure, more clustering)")
for key in ("noop_frac", "moved_frac", "mean_disp", "birth_rate", "adj_frac", "d10", "d30", "d50"):
    r, p = spearmanr([P[x][key] for x in keys], Rv)
    print(f"  {key:11s} rho={r:+.2f} (p={p:.3f})")
print("  within run B only (6 policies):")
for key in ("noop_frac", "mean_disp", "birth_rate", "d30"):
    r, p = spearmanr([P[("step", s)][key] for s in SEEDS], [P[("step", s)]["R"] for s in SEEDS])
    print(f"  {key:11s} rho={r:+.2f} (p={p:.3f})")

print("\nDISPERSAL PROFILE: mean parent-offspring distance (cells) by offspring age, mean over seeds")
print("  age      " + " ".join(f"{a:6d}" for a in AGES))
for k in ("base", "step"):
    print(f"  {k:8s} " + " ".join(f"{np.mean([P[(k, s)][f'd{a}'] for s in SEEDS]):6.2f}" for a in AGES))

# adjacency against a random-placement null at each episode's own predator count
rng = np.random.default_rng(1)
cells = np.array([(x, y) for x in range(G) for y in range(G)], float)


def null_adj(N, trials=2000):
    c = 0.0
    for _ in range(trials):
        pts = cells[rng.choice(G * G, N, replace=False)]
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        c += (d.min(1) <= 1.5).mean()
    return c / trials


Ns = list(range(4, 31, 2))
tab = {N: null_adj(N) for N in Ns}
A0 = lambda N: np.interp(N, Ns, [tab[k] for k in Ns])
print("\nADJACENCY vs RANDOM PLACEMENT at the episode's own predator count")
print("  random expectation of adj_frac by N:", " ".join(f"N={N}:{tab[N]:.3f}" for N in Ns[::2]))
ex, obs_m, rnd_m = {}, {}, {}
for k in ("base", "step"):
    for s in SEEDS:
        m = episodes(k, s)
        d = json.load(open(os.path.join(DENS, f"dose_{k}_{s}.json")))
        assert len(m) == len(d) and all(abs(a["R"] - b["R"]) < 1e-9 for a, b in zip(m, d)), "episode alignment"
        obs = np.array([e["adj_frac"] for e in m])
        null = np.array([A0(x["n_pred"]) for x in d])
        ex[(k, s)] = (obs - null).mean()
        obs_m[(k, s)], rnd_m[(k, s)] = obs.mean(), null.mean()
b = np.array([ex[("base", s)] for s in SEEDS])
st = np.array([ex[("step", s)] for s in SEEDS])
print("  observed / random-expected: base %.3f / %.3f (x%.2f) | run B %.3f / %.3f (x%.2f)" % (
    np.mean([obs_m[("base", s)] for s in SEEDS]), np.mean([rnd_m[("base", s)] for s in SEEDS]),
    np.mean([obs_m[("base", s)] for s in SEEDS]) / np.mean([rnd_m[("base", s)] for s in SEEDS]),
    np.mean([obs_m[("step", s)] for s in SEEDS]), np.mean([rnd_m[("step", s)] for s in SEEDS]),
    np.mean([obs_m[("step", s)] for s in SEEDS]) / np.mean([rnd_m[("step", s)] for s in SEEDS])))
print("  excess adjacency: base %+.3f | run B %+.3f; run B > base in %d/6 seeds; unpaired one-sided p=%.4f; paired p=%.4f" % (
    b.mean(), st.mean(), (st > b).sum(),
    mannwhitneyu(st, b, alternative="greater").pvalue, wilcoxon(st, b, alternative="greater").pvalue))
r, p = spearmanr([ex[k] for k in keys], Rv)
print("  Spearman(excess adjacency, R) across 12 policies: rho=%+.2f (p=%.3f) -- largely definitional: adjacent pairs feed directly into R" % (r, p))
