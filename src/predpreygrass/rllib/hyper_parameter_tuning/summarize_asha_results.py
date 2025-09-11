"""Summarize ASHA + Optuna PPO experiment outputs.

Looks for experiment directories named PPO_ASHA_* in the default ray_results path
(~/Dropbox/02_marl_results/predpreygrass_results/ray_results/) and aggregates:
  - stop reason counts
  - basic iteration statistics
  - top-N trials by score_pred
  - distribution of rung_index (if present in predator_final.csv)
  - fraction of trials reaching predator_100_hits

Usage (from repo root):
  python -m predpreygrass.rllib.hyper_parameter_tuning.summarize_asha_results \
      --base ~/Dropbox/02_marl_results/predpreygrass_results/ray_results \
      --top 10

Optional flags:
  --pattern "PPO_ASHA_2025-09-11*"  # restrict to subset
  --csv-out summary.csv             # write flattened per-trial table
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics as st
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class TrialRow:
    experiment: str
    trial_name: str
    iteration: int
    progress_ratio: float | None
    score_pred: float | None
    lr: float | None
    num_epochs: int | None
    stop_reason: str
    rung_index: int | None
    rung_pruned_at: str | None
    asha_pruned: int | None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="~/Dropbox/02_marl_results/predpreygrass_results/ray_results", help="Base ray results directory")
    p.add_argument("--pattern", default="PPO_ASHA_*", help="Glob for experiment dirs under base")
    p.add_argument("--top", type=int, default=5, help="Show top-N trials by score_pred")
    p.add_argument("--csv-out", default=None, help="Optional path to write flattened per-trial CSV")
    p.add_argument("--verbose", action="store_true", help="Print matched experiment dirs and missing CSVs")
    return p.parse_args()


def load_trials(base: str, pattern: str, verbose: bool = False) -> List[TrialRow]:
    base = os.path.expanduser(base)
    rows: List[TrialRow] = []
    matched = sorted(glob.glob(os.path.join(base, pattern)))
    if verbose:
        print(f"[verbose] Matched {len(matched)} experiment directories under {base} (pattern={pattern})")
        for d in matched:
            print(f"  - {os.path.basename(d)}")
    for exp_dir in matched:
        final_path = os.path.join(exp_dir, "predator_final.csv")
        if not os.path.exists(final_path):
            if verbose:
                print(f"[verbose] Missing predator_final.csv in {exp_dir}")
            continue
        with open(final_path) as f:
            r = csv.DictReader(f)
            for line in r:
                def to_float(v):
                    if v is None or v == "" or v == "nan":
                        return None
                    try:
                        return float(v)
                    except ValueError:
                        return None
                def to_int(v):
                    if v is None or v == "" or v == "nan":
                        return None
                    try:
                        return int(v)
                    except ValueError:
                        return None

                rows.append(
                    TrialRow(
                        experiment=os.path.basename(exp_dir),
                        trial_name=line.get("trial_name", ""),
                        iteration=to_int(line.get("iteration")) or -1,
                        progress_ratio=to_float(line.get("progress_ratio")),
                        score_pred=to_float(line.get("score_pred")),
                        lr=to_float(line.get("lr")),
                        num_epochs=to_int(line.get("num_epochs")),
                        stop_reason=line.get("stop_reason", ""),
                        rung_index=to_int(line.get("rung_index")),
                        rung_pruned_at=line.get("rung_pruned_at"),
                        asha_pruned=to_int(line.get("asha_pruned")),
                    )
                )
    return rows


def load_hits(base: str, pattern: str, verbose: bool = False) -> set[str]:
    base = os.path.expanduser(base)
    hit_trials = set()
    for exp_dir in glob.glob(os.path.join(base, pattern)):
        hp = os.path.join(exp_dir, "predator_100_hits.csv")
        if not os.path.exists(hp):
            if verbose:
                print(f"[verbose] No predator_100_hits.csv in {exp_dir}")
            continue
        with open(hp) as f:
            r = csv.DictReader(f)
            for line in r:
                hit_trials.add(line.get("trial_name", ""))
    return hit_trials


def summarize(rows: List[TrialRow], hit_trials: set[str], top: int):
    if not rows:
        print("No predator_final.csv rows found.")
        return

    stop_counts = Counter(r.stop_reason for r in rows)
    asha_count = sum(1 for r in rows if r.stop_reason == "asha_early_stop")
    completed_count = stop_counts.get("completed", 0)

    its = [r.iteration for r in rows if r.iteration >= 0]
    scores = [r.score_pred for r in rows if r.score_pred is not None and not math.isnan(r.score_pred)]

    print("=== Trial Outcome Summary ===")
    print(f"Total trials: {len(rows)}")
    for reason, c in stop_counts.most_common():
        print(f"  {reason:15s}: {c}")
    print(f"  (asha_pruned flag rows: {asha_count})")

    if its:
        print("Iterations: min={:.0f} p25={:.0f} median={:.0f} p75={:.0f} max={:.0f}".format(
            min(its),
            sorted(its)[len(its)//4],
            sorted(its)[len(its)//2],
            sorted(its)[(len(its)*3)//4],
            max(its),
        ))
    if scores:
        print("Score_pred: mean={:.3f} median={:.3f} max={:.3f}".format(
            st.mean(scores),
            sorted(scores)[len(scores)//2],
            max(scores),
        ))

    # Rung distribution
    rung_vals = [r.rung_index for r in rows if r.rung_index is not None and r.rung_index >= 0]
    if rung_vals:
        rc = Counter(rung_vals)
        print("Rung index distribution:")
        for k in sorted(rc):
            print(f"  rung {k}: {rc[k]}")

    # Hit fraction
    hit_fraction = len(hit_trials) / len({r.trial_name for r in rows}) if rows else 0.0
    print(f"Trials reaching predator_100 threshold: {len(hit_trials)} ({hit_fraction:.1%})")

    # Top trials
    ranked = [r for r in rows if r.score_pred is not None]
    ranked.sort(key=lambda r: (r.score_pred, r.iteration), reverse=True)
    print(f"\nTop {min(top, len(ranked))} trials by score_pred:")
    for r in ranked[:top]:
        print(
            f"  {r.experiment}/{r.trial_name} score={r.score_pred:.3f} iter={r.iteration} "
            f"reason={r.stop_reason} rung={r.rung_index}"
        )


def maybe_write_csv(rows: List[TrialRow], path: str | None):
    if not path:
        return
    fieldnames = [
        "experiment","trial_name","iteration","progress_ratio","score_pred","lr","num_epochs",
        "stop_reason","rung_index","rung_pruned_at","asha_pruned"
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "experiment": r.experiment,
                "trial_name": r.trial_name,
                "iteration": r.iteration,
                "progress_ratio": r.progress_ratio if r.progress_ratio is not None else "",
                "score_pred": r.score_pred if r.score_pred is not None else "",
                "lr": r.lr if r.lr is not None else "",
                "num_epochs": r.num_epochs if r.num_epochs is not None else "",
                "stop_reason": r.stop_reason,
                "rung_index": r.rung_index if r.rung_index is not None else "",
                "rung_pruned_at": r.rung_pruned_at or "",
                "asha_pruned": r.asha_pruned if r.asha_pruned is not None else "",
            })
    print(f"Wrote per-trial CSV: {path}")


def main():
    args = parse_args()
    rows = load_trials(args.base, args.pattern, verbose=args.verbose)
    hit_trials = load_hits(args.base, args.pattern, verbose=args.verbose)
    summarize(rows, hit_trials, args.top)
    maybe_write_csv(rows, args.csv_out)


if __name__ == "__main__":
    main()
