"""ERL vs. L prey against evolving vs. fixed carnivores -- a 2x2, fixed before
the results (2026-09-30).

Cells (same seeds, 60k steps, 150x150, rich_memory carnivores with wide founder
persistence variation, N(0, 3)):
  prey ERL x carnivore evolving  = step 2c `step2` runs (reused)
  prey ERL x carnivore fixed     = step 2c `step2_neutral` runs (reused; every
                                   carnivore acts with the canonical seed network)
  prey L   x carnivore evolving  = new
  prey L   x carnivore fixed     = new

Prediction: L cannot change its innate goals, so it should fall behind a moving
target. PRIMARY: ERL vs. L against EVOLVING carnivores -- prey extinctions and
"any collapse" (prey or carnivores extinct), Fisher exact. SECONDARY: the same
against fixed carnivores, and the interaction (is L's excess collapse rate
larger against evolving carnivores?), by a permutation test shuffling prey labels
within each carnivore condition. n=20 per cell makes the interaction a pilot.
Exploratory: the carnivore persist trait's late mean under each prey type.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact


def load(out_dir: Path, tag: str) -> list[dict]:
    rows = [json.loads(line) for line in (out_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    return [r for r in rows if r["tag"] == tag]


def outcomes(rows: list[dict]) -> dict:
    prey = np.array([r["end_reason"] == "agent_extinct" for r in rows])
    carn = np.array([r["end_reason"] == "carnivore_extinct" for r in rows])
    return {"n": len(rows), "prey": prey, "carn": carn, "collapse": prey | carn}


def late_persist(out_dir: Path, r: dict, window: int = 10_000) -> float:
    path = out_dir / "carnivore_lineage" / f"{r['tag']}_{r['strategy']}_seed{r['seed']}.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    end = max(int(x["born_step"]) for x in rows)
    vals = [np.mean([float(x[f"w{16 + i}_{i}"]) for i in range(4)]) for x in rows if int(x["born_step"]) >= end - window]
    return float(np.mean(vals)) if vals else float("nan")


def interaction_p(erl_evo, l_evo, erl_fix, l_fix, n_perm=20_000, seed=0) -> tuple[float, float]:
    def stat(a, b, c, d):
        return (b.mean() - a.mean()) - (d.mean() - c.mean())

    observed = stat(erl_evo, l_evo, erl_fix, l_fix)
    rng = np.random.default_rng(seed)
    evo, fix = np.concatenate([erl_evo, l_evo]), np.concatenate([erl_fix, l_fix])
    hits = 0
    for _ in range(n_perm):
        pe, pf = rng.permutation(evo), rng.permutation(fix)
        s = stat(pe[: len(erl_evo)], pe[len(erl_evo):], pf[: len(erl_fix)], pf[len(erl_fix):])
        hits += abs(s) >= abs(observed) - 1e-12
    return observed, hits / n_perm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--erl-dir", required=True, help="step 2c output dir")
    parser.add_argument("--erl-evolving-tag", required=True)
    parser.add_argument("--erl-fixed-tag", required=True)
    parser.add_argument("--l-dir", required=True)
    parser.add_argument("--l-evolving-tag", required=True)
    parser.add_argument("--l-fixed-tag", required=True)
    args = parser.parse_args()
    erl_dir, l_dir = Path(args.erl_dir).expanduser(), Path(args.l_dir).expanduser()
    cells = {
        ("ERL", "evolving"): (erl_dir, load(erl_dir, args.erl_evolving_tag)),
        ("ERL", "fixed"): (erl_dir, load(erl_dir, args.erl_fixed_tag)),
        ("L", "evolving"): (l_dir, load(l_dir, args.l_evolving_tag)),
        ("L", "fixed"): (l_dir, load(l_dir, args.l_fixed_tag)),
    }
    out = {k: outcomes(rows) for k, (_, rows) in cells.items()}

    print(f"{'prey':4s} {'carnivores':10s} {'n':>3s} {'prey extinct':>13s} {'carn extinct':>13s} {'any collapse':>13s} {'late persist':>13s}")
    for (prey, carn), (d, rows) in cells.items():
        o = out[(prey, carn)]
        persist = np.nanmean([late_persist(d, r) for r in rows if r["end_step"] > 20_000])
        print(f"{prey:4s} {carn:10s} {o['n']:3d} {int(o['prey'].sum()):13d} {int(o['carn'].sum()):13d} "
              f"{int(o['collapse'].sum()):13d} {persist:13.2f}")

    for carn, label in (("evolving", "PRIMARY"), ("fixed", "secondary")):
        a, b = out[("ERL", carn)], out[("L", carn)]
        for key in ("prey", "collapse"):
            p = fisher_exact([[int(a[key].sum()), a["n"] - int(a[key].sum())],
                              [int(b[key].sum()), b["n"] - int(b[key].sum())]]).pvalue
            print(f"  {label:9s} vs {carn:8s} carnivores, {key:8s}: ERL {int(a[key].sum())}/{a['n']} vs L "
                  f"{int(b[key].sum())}/{b['n']}, Fisher p={p:.3g}")
    diff, p = interaction_p(*(out[k]["collapse"].astype(float) for k in
                              [("ERL", "evolving"), ("L", "evolving"), ("ERL", "fixed"), ("L", "fixed")]))
    print(f"  secondary interaction (L's excess collapse rate, evolving minus fixed): {diff:+.2f}, permutation p={p:.3g}")


if __name__ == "__main__":
    main()
