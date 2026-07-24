"""
Trial 6 (population scaling) analysis: aggregate the scaled multi-seed
replication runs (real satiation-throttle config vs. neutral-drift control)
and compare offspring_investment_fraction drift magnitude between the two
groups -- the scaled counterpart of R7's analyze_replication_seeds.py.

Expects experiment directories under ~/ray_results/ named:
  PPO_ECO_EVOLUTION_INVESTMENT_SCALED_SEED<seed>_<timestamp>                (real)
  PPO_ECO_EVOLUTION_INVESTMENT_SCALED_NEUTRAL_CONTROL_SEED<seed>_<timestamp> (control)

as produced by tune_ppo_investment_scaled.py / tune_ppo_investment_neutral_control_scaled.py
when run with --seed. Missing runs are reported, not treated as an error, so this
can be run before all seeds have finished.

Caveat printed alongside every result: with only ~3 runs per group, this is a
directional check, not a well-powered significance test -- Mann-Whitney U with
n=3 vs n=3 cannot reach conventional significance thresholds even in the best
case. Treat the p-values as a rough indicator, not proof either way.

Usage:
    python predpreygrass/evolutionary/eco_evolutionary_investment/analyze_replication_seeds_scaled.py
"""

import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

RAY_RESULTS_DIR = Path("~/ray_results").expanduser()
FOUNDER_MEAN = 0.35

REAL_PATTERN = re.compile(r"^PPO_ECO_EVOLUTION_INVESTMENT_SCALED_SEED(\d+)_")
CONTROL_PATTERN = re.compile(r"^PPO_ECO_EVOLUTION_INVESTMENT_SCALED_NEUTRAL_CONTROL_SEED(\d+)_")


def find_seed_runs(pattern: re.Pattern) -> dict[int, Path]:
    """Return {seed: result.json path} for the most recent run of each seed."""
    matches: dict[int, tuple[float, Path]] = {}
    for exp_dir in RAY_RESULTS_DIR.glob("PPO_ECO_EVOLUTION_INVESTMENT_SCALED*"):
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
    """Q1/Q5 mean and max |deviation from founder| for predator and prey live investment fraction."""
    out = {}
    for species in ("predator", "prey"):
        series = load_series(result_json, f"live_investment/{species}_offspring_investment_fraction_mean")
        if not series:
            out[species] = None
            continue
        n = len(series)
        q = max(n // 5, 1)
        q1_mean = float(np.mean(series[:q]))
        q5_mean = float(np.mean(series[-q:]))
        max_abs_dev = float(np.max(np.abs(np.array(series) - FOUNDER_MEAN)))
        out[species] = {
            "n_points": n,
            "q1_mean": q1_mean,
            "q5_mean": q5_mean,
            "net_change": q5_mean - q1_mean,
            "final_dev_from_founder": q5_mean - FOUNDER_MEAN,
            "max_abs_dev_from_founder": max_abs_dev,
        }
    return out


def main():
    real_runs = find_seed_runs(REAL_PATTERN)
    control_runs = find_seed_runs(CONTROL_PATTERN)

    print("=== Runs found (SCALED) ===")
    print(f"Real (satiation-throttle):  seeds {sorted(real_runs)} ({len(real_runs)} found)")
    print(f"Neutral control:            seeds {sorted(control_runs)} ({len(control_runs)} found)")
    print()

    real_summaries = {seed: summarize_run(p) for seed, p in sorted(real_runs.items())}
    control_summaries = {seed: summarize_run(p) for seed, p in sorted(control_runs.items())}

    print("=== Per-run summary ===")
    header = f"{'group':<8}{'seed':<6}{'species':<10}{'n':<6}{'Q1':<8}{'Q5':<8}{'net_chg':<9}{'|dev_final|':<12}{'max|dev|':<9}"
    print(header)
    for group_name, summaries in (("real", real_summaries), ("control", control_summaries)):
        for seed, s in summaries.items():
            for species in ("predator", "prey"):
                r = s.get(species)
                if r is None:
                    print(f"{group_name:<8}{seed:<6}{species:<10} no data")
                    continue
                print(
                    f"{group_name:<8}{seed:<6}{species:<10}{r['n_points']:<6}"
                    f"{r['q1_mean']:<8.4f}{r['q5_mean']:<8.4f}{r['net_change']:<+9.4f}"
                    f"{abs(r['final_dev_from_founder']):<12.4f}{r['max_abs_dev_from_founder']:<9.4f}"
                )
    print()

    print("=== Mann-Whitney U: real vs. control drift magnitude ===")
    print("CAVEAT: n=3 vs n=3 (or fewer) has very limited power -- cannot reach")
    print("conventional significance thresholds even in the best case. Read the")
    print("direction and effect size, not the p-value alone.\n")

    for species in ("predator", "prey"):
        for metric_name, metric_key in (
            ("final |deviation from founder|", "final_dev_from_founder"),
            ("max |deviation from founder|", "max_abs_dev_from_founder"),
        ):
            real_vals = [
                abs(s[species][metric_key]) if metric_key == "final_dev_from_founder" else s[species][metric_key]
                for s in real_summaries.values() if s.get(species) is not None
            ]
            control_vals = [
                abs(s[species][metric_key]) if metric_key == "final_dev_from_founder" else s[species][metric_key]
                for s in control_summaries.values() if s.get(species) is not None
            ]
            if len(real_vals) < 2 or len(control_vals) < 2:
                print(f"{species:<10} {metric_name:<32} not enough runs yet "
                      f"(real n={len(real_vals)}, control n={len(control_vals)})")
                continue
            stat, p = mannwhitneyu(real_vals, control_vals, alternative="greater")
            print(
                f"{species:<10} {metric_name:<32} real={np.mean(real_vals):.4f} "
                f"(n={len(real_vals)})  control={np.mean(control_vals):.4f} (n={len(control_vals)})  "
                f"U={stat:.1f}  p(real>control)={p:.3f}"
            )


if __name__ == "__main__":
    main()
