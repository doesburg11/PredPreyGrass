"""Evaluate step 2's pre-registered pass/fail criteria (README.md, "Step 2").

Compares a `step2` tag (carnivore genome expressed) against a `step2_neutral`
tag (neutral marker: same inheritance, genome not expressed) over the same
seeds, from study.py's output directory:

  1. Coexistence: among runs with prey alive at the immigration switch, the
     fraction that keep their carnivores to the budget (pass: >= 90%).
  2. Hunting improves by selection: per-seed change in kill rate (kills per
     1,000 carnivore-steps, last 10k steps minus first 5k) is larger under
     step2 than under the neutral control (Mann-Whitney across seeds).
  3. Selection, not drift, in the genome: per-generation mean of each trait
     (`pursuit` = mean prey-signal_i -> action_i weight, `avoid` = mean
     blocked_i -> action_i weight) from carnivore_lineage/, fit with the Hunt
     Stasis/URW/GRW model selection. Pass: step2 seeds agree on the direction
     of change (binomial test on the sign of the net change) and the neutral
     control does not.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.analyze_step2 \\
        --out-dir ~/simulation_results/erl_results/coevo_step2_pilot --real-tag step2_X --neutral-tag step2neutral_X
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, mannwhitneyu

from predpreygrass.evolutionary.model_selection import fit_all_models
from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.world import CARN_OBS_LAYOUTS, N_ACTIONS

SWITCH_STEP = 20_000
EARLY_WINDOW = 5_000
LATE_WINDOW = 10_000
MIN_PER_GENERATION = 5  # generations with fewer lineage rows are dropped from the trait series

def traits_for(layout_name: str) -> dict[str, list[str]]:
    """Lineage columns per trait for a carnivore input layout: `pursuit` = the
    first prey channel's i -> i weights, `avoid` = blocked i -> i, and for
    "rich_memory" `persist` = previous move i -> i."""
    layout = CARN_OBS_LAYOUTS[layout_name]
    first_prey = next(iter(layout["prey"].values()))
    traits = {
        "pursuit": [f"w{first_prey + i}_{i}" for i in range(N_ACTIONS)],
        "avoid": [f"w{layout['block'] + i}_{i}" for i in range(N_ACTIONS)],
    }
    if "prev" in layout:
        traits["persist"] = [f"w{layout['prev'] + i}_{i}" for i in range(N_ACTIONS)]
    return traits


TRAITS = traits_for("basic")


def load_results(out_dir: Path, tag: str) -> list[dict]:
    rows = [json.loads(line) for line in (out_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    return [r for r in rows if r["tag"] == tag]


def coexistence(results: list[dict], switch: int) -> tuple[int, int]:
    established = [r for r in results if r["end_step"] > switch]
    return sum(r["end_reason"] == "budget" for r in established), len(established)


def kill_rate_change(out_dir: Path, r: dict, early: int, late: int) -> float | None:
    """Kills per 1,000 carnivore-steps, last LATE_WINDOW steps minus first EARLY_WINDOW.
    Only for runs that reached the budget (both windows fully observed)."""
    if r["end_reason"] != "budget":
        return None
    path = out_dir / "timeseries" / f"{r['tag']}_{r['strategy']}_seed{r['seed']}.csv"
    with open(path) as f:
        rows = [(int(x["step"]), int(x["carnivore_kills"]), int(x["carnivore_steps"])) for x in csv.DictReader(f)]
    by_step = {s: (k, c) for s, k, c in rows}
    budget = r["budget"]

    def rate(start, end):
        k0, c0 = by_step.get(start, (0, 0)) if start else (0, 0)
        k1, c1 = by_step[end]
        return 1000 * (k1 - k0) / max(c1 - c0, 1)

    return rate(budget - late, budget) - rate(0, early)


def trait_series(out_dir: Path, r: dict, trait: str):
    """Per-generation (mean, var, n) of a trait over every carnivore that lived."""
    path = out_dir / "carnivore_lineage" / f"{r['tag']}_{r['strategy']}_seed{r['seed']}.csv"
    per_gen: dict[int, list[float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            value = np.mean([float(row[col]) for col in TRAITS[trait]])
            per_gen.setdefault(int(row["generation"]), []).append(value)
    gens = sorted(g for g, vals in per_gen.items() if len(vals) >= MIN_PER_GENERATION)
    if len(gens) < 10:
        return None
    return (
        np.array(gens),
        np.array([np.mean(per_gen[g]) for g in gens]),
        np.array([np.var(per_gen[g], ddof=1) for g in gens]),
        np.array([len(per_gen[g]) for g in gens]),
    )


def trait_summary(out_dir: Path, results: list[dict], trait: str) -> dict:
    net, best, mstep_signs = [], [], []
    for r in results:
        series = trait_series(out_dir, r, trait)
        if series is None:
            continue
        gens, mean, var, n = series
        fits = fit_all_models(gens, mean, var, n)
        net.append(mean[-1] - mean[0])
        best.append(fits[0].model)
        grw = next(f for f in fits if f.model == "GRW")
        mstep_signs.append(np.sign(grw.params["mstep"]))
    n_pos = sum(x > 0 for x in net)
    return {
        "n": len(net),
        "net_changes": net,
        "positive": n_pos,
        "sign_p": binomtest(n_pos, len(net), 0.5).pvalue if net else float("nan"),
        "best_models": {m: best.count(m) for m in ("Stasis", "URW", "GRW")},
        "median_abs_change": float(np.median(np.abs(net))) if net else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--real-tag", required=True)
    parser.add_argument("--neutral-tag", required=True)
    parser.add_argument("--switch", type=int, default=SWITCH_STEP)
    parser.add_argument("--early", type=int, default=EARLY_WINDOW)
    parser.add_argument("--late", type=int, default=LATE_WINDOW)
    parser.add_argument("--layout", default="basic", choices=sorted(CARN_OBS_LAYOUTS),
                        help="carnivore_obs layout of the runs (sets which lineage columns form each trait)")
    args = parser.parse_args()
    global TRAITS
    TRAITS = traits_for(args.layout)
    out_dir = Path(args.out_dir).expanduser()
    groups = {"step2": load_results(out_dir, args.real_tag), "neutral": load_results(out_dir, args.neutral_tag)}

    print("== 1. Coexistence after the switch (pass: >= 90% for step2)")
    for name, results in groups.items():
        kept, established = coexistence(results, args.switch)
        share = kept / established if established else float("nan")
        verdict = ("PASS" if share >= 0.9 else "FAIL") if name == "step2" else ""
        print(f"  {name:8s} {kept}/{established} established runs keep carnivores ({share:.0%}) {verdict}")

    print("\n== 2. Kill-rate change, last 10k minus first 5k (per 1,000 carnivore-steps)")
    print("   (reported only: dropped as a selection test 2026-09-29 -- per-capita kill rate tracks ecology; README)")
    changes = {}
    for name, results in groups.items():
        changes[name] = [c for c in (kill_rate_change(out_dir, r, args.early, args.late) for r in results)
                         if c is not None]
        vals = np.array(changes[name])
        print(f"  {name:8s} n={len(vals)} median {np.median(vals):+.2f}  mean {vals.mean():+.2f}")
    if changes["step2"] and changes["neutral"]:
        p = mannwhitneyu(changes["step2"], changes["neutral"], alternative="two-sided").pvalue
        higher = np.median(changes["step2"]) > np.median(changes["neutral"])
        verdict = "PASS" if p < 0.05 and higher else "FAIL"
        print(f"  step2 vs neutral: Mann-Whitney p={p:.3g}, step2 {'higher' if higher else 'not higher'} -> {verdict}")

    print("\n== 3. Genome change: selection vs. drift (Hunt model selection per seed)")
    for trait in TRAITS:
        print(f"  trait {trait}:")
        for name, results in groups.items():
            s = trait_summary(out_dir, results, trait)
            print(f"    {name:8s} n={s['n']} net change positive in {s['positive']}/{s['n']} seeds "
                  f"(sign test p={s['sign_p']:.3g}), median |change| {s['median_abs_change']:.3f}, "
                  f"best model {s['best_models']}")
    print("  pass: step2 sign test p < 0.05 for a trait while neutral is not.")


if __name__ == "__main__":
    main()
