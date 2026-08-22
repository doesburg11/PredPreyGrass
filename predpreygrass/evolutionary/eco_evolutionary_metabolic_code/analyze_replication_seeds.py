"""
Trial 7 analysis: aggregate multi-seed replication runs (real satiation-throttle
config vs. neutral-drift control) and compare combinatorial-genome selection
signal between the two groups -- same real-vs-control replication methodology
as eco_evolutionary_investment's R7, adapted to this module's metrics.

Headline metric: mean_wrong_loci (lower = more WRONG loci purged by selection --
these are the loci an individual can never resolve within its own lifetime, so a
real decline below the founder expectation, exceeding the neutral control, is the
actual Darwin-signal test). Secondary metric: fraction_solved (higher = more of
the live population has achieved a full genome match this life).

Unlike eco_evolutionary_investment's founder-mean-0.35 trait, this design has a
directional prediction (selection should push mean_wrong_loci DOWN, not just away
from a neutral center), so the Mann-Whitney test here is one-sided on the raw
values (real < control for mean_wrong_loci; real > control for fraction_solved),
not on |deviation from founder|.

Expects experiment directories under ~/simulation_results/ray_results/ named:
  PPO_ECO_EVOLUTION_METABOLIC_CODE_SEED<seed>_<timestamp>                (real)
  PPO_ECO_EVOLUTION_METABOLIC_CODE_NEUTRAL_CONTROL_SEED<seed>_<timestamp> (control)

as produced by tune_ppo_metabolic_code.py / tune_ppo_metabolic_code_neutral_control.py
when run with --seed. Missing runs are reported, not treated as an error, so this
can be run before all seeds have finished.

Caveat printed alongside every result: with only ~3 runs per group, this is a
directional check, not a well-powered significance test -- Mann-Whitney U with
n=3 vs n=3 cannot reach conventional significance thresholds even in the best
case. Treat the p-values as a rough indicator, not proof either way.

Usage:
    python predpreygrass/evolutionary/eco_evolutionary_metabolic_code/analyze_replication_seeds.py
"""

import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from predpreygrass.global_config import RAY_RESULTS_DIR
FOUNDER_MEAN_WRONG_LOCI = 3.0  # L=10 * p_wrong=0.3, see config_env_eco_evolutionary.py

REAL_PATTERN = re.compile(r"^PPO_ECO_EVOLUTION_METABOLIC_CODE_SEED(\d+)_")
CONTROL_PATTERN = re.compile(r"^PPO_ECO_EVOLUTION_METABOLIC_CODE_NEUTRAL_CONTROL_SEED(\d+)_")

METRICS = ("mean_wrong_loci", "fraction_solved")


def find_seed_runs(pattern: re.Pattern) -> dict[int, Path]:
    """Return {seed: result.json path} for the most recent run of each seed."""
    matches: dict[int, tuple[float, Path]] = {}
    for exp_dir in RAY_RESULTS_DIR.glob("PPO_ECO_EVOLUTION_METABOLIC_CODE*"):
        m = pattern.match(exp_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        result_jsons = list(exp_dir.glob("*/result.json"))
        if not result_jsons:
            continue
        result_json = result_jsons[0]
        mtime = result_json.stat().st_mtime
        if seed not in matches or mtime > matches[seed][0]:
            matches[seed] = (mtime, result_json)
    return {seed: path for seed, (_, path) in matches.items()}


def load_series(result_json: Path, key: str) -> list[float]:
    values = []
    with open(result_json) as f:
        for line in f:
            d = json.loads(line)
            er = d.get("env_runners", {}) or {}
            v = er.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                values.append(float(v))
    return values


def summarize_run(result_json: Path) -> dict:
    """Q1/Q5 mean for predator and prey mean_wrong_loci / fraction_solved."""
    out = {}
    for species in ("predator", "prey"):
        species_out = {}
        any_metric = False
        for metric in METRICS:
            series = load_series(result_json, f"live_haystack/{species}_{metric}")
            if not series:
                continue
            any_metric = True
            n = len(series)
            q = max(n // 5, 1)
            species_out[metric] = {
                "n_points": n,
                "q1_mean": float(np.mean(series[:q])),
                "q5_mean": float(np.mean(series[-q:])),
            }
        out[species] = species_out if any_metric else None
    return out


def main():
    real_runs = find_seed_runs(REAL_PATTERN)
    control_runs = find_seed_runs(CONTROL_PATTERN)

    print("=== Runs found ===")
    print(f"Real (satiation-throttle):  seeds {sorted(real_runs)} ({len(real_runs)} found)")
    print(f"Neutral control:            seeds {sorted(control_runs)} ({len(control_runs)} found)")
    print()

    real_summaries = {seed: summarize_run(p) for seed, p in sorted(real_runs.items())}
    control_summaries = {seed: summarize_run(p) for seed, p in sorted(control_runs.items())}

    print("=== Per-run summary (Q5 = final-quintile mean) ===")
    header = f"{'group':<8}{'seed':<6}{'species':<10}{'metric':<18}{'n':<6}{'Q1':<8}{'Q5':<8}"
    print(header)
    for group_name, summaries in (("real", real_summaries), ("control", control_summaries)):
        for seed, s in summaries.items():
            for species in ("predator", "prey"):
                metrics = s.get(species)
                if not metrics:
                    print(f"{group_name:<8}{seed:<6}{species:<10} no data")
                    continue
                for metric in METRICS:
                    r = metrics.get(metric)
                    if r is None:
                        print(f"{group_name:<8}{seed:<6}{species:<10}{metric:<18} no data")
                        continue
                    print(
                        f"{group_name:<8}{seed:<6}{species:<10}{metric:<18}{r['n_points']:<6}"
                        f"{r['q1_mean']:<8.4f}{r['q5_mean']:<8.4f}"
                    )
    print()

    print("=== Mann-Whitney U: real vs. control (final-quintile Q5 mean) ===")
    print("CAVEAT: n=3 vs n=3 (or fewer) has very limited power -- cannot reach")
    print("conventional significance thresholds even in the best case. Read the")
    print("direction and effect size, not the p-value alone.")
    print("Founder expectation for mean_wrong_loci: "
          f"{FOUNDER_MEAN_WRONG_LOCI:.1f} (L=10, p_wrong=0.3).\n")

    directional_alternative = {
        # Selection should push WRONG-loci count DOWN (real < control).
        "mean_wrong_loci": "less",
        # Selection should push solved-fraction UP (real > control).
        "fraction_solved": "greater",
    }

    for species in ("predator", "prey"):
        for metric in METRICS:
            real_vals = [
                s[species][metric]["q5_mean"]
                for s in real_summaries.values()
                if s.get(species) and metric in s[species]
            ]
            control_vals = [
                s[species][metric]["q5_mean"]
                for s in control_summaries.values()
                if s.get(species) and metric in s[species]
            ]
            if len(real_vals) < 2 or len(control_vals) < 2:
                print(f"{species:<10} {metric:<18} not enough runs yet "
                      f"(real n={len(real_vals)}, control n={len(control_vals)})")
                continue
            alternative = directional_alternative[metric]
            stat, p = mannwhitneyu(real_vals, control_vals, alternative=alternative)
            print(
                f"{species:<10} {metric:<18} real={np.mean(real_vals):.4f} "
                f"(n={len(real_vals)})  control={np.mean(control_vals):.4f} (n={len(control_vals)})  "
                f"U={stat:.1f}  p(real {'<' if alternative == 'less' else '>'} control)={p:.3f}"
            )


if __name__ == "__main__":
    main()
