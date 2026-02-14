#!/usr/bin/env python3
"""Summarize cooperation/defection metrics plus payoff vs join-cost burden.

Scans eval folders for defection_metrics_aggregate.json, then augments per-run
metrics with reward summaries and join-costs-per-predator to estimate the payoff
of joining vs defecting.

Configuration lives below; no CLI arguments are used.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path


RAY_RESULTS_DIR = (
    "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/"
    "stag_hunt_forward_view_nature_nurture/ray_results/join_cost_0.02"
)

# Edit these settings directly.
OUTPUT_PER_RUN_CSV = "eval_payoff_per_run.csv"
OUTPUT_PER_EVAL_CSV = "eval_payoff_summary.csv"
OUTPUT_BY_SCAV_CSV = "eval_payoff_by_scavenger.csv"
OUTPUT_REPORT_MD = "eval_payoff_report.md"
INCLUDE_ALL_EVALS = True
SCAVENGER_FILTERS = [0.0, 0.1, 0.2, 0.3, 0.4]
EVAL_NAME_FILTER = None  # e.g. "eval_10_runs" to only include matching eval dirs


def _load_json(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_eval_timestamp(name: str):
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(0), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def _find_eval_aggregate_paths(run_dir: Path):
    results = []
    for agg_path in run_dir.rglob("defection_metrics_aggregate.json"):
        if agg_path.parent.name == "summary_data":
            eval_dir = agg_path.parent.parent
        else:
            eval_dir = agg_path.parent
        results.append((eval_dir, agg_path))
    return results


def _select_latest_eval(evals):
    def _score(item):
        eval_dir, _ = item
        ts = _parse_eval_timestamp(eval_dir.name)
        if ts:
            return ts.timestamp()
        try:
            return eval_dir.stat().st_mtime
        except OSError:
            return 0.0

    return max(evals, key=_score)


def _float_matches(value, targets, tol=1e-6):
    if value is None:
        return False
    for target in targets:
        if abs(value - target) <= tol:
            return True
    return False


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values):
    items = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not items:
        return None
    return sum(items) / len(items)


def _parse_reward_summary(path: Path):
    rewards = {}
    total_reward = None
    if not path.is_file():
        return rewards, total_reward
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Total Reward"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    total_reward = _to_float(parts[1].strip())
                continue
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            name = name.strip()
            value = _to_float(value.strip())
            if value is None:
                continue
            rewards[name] = value
    return rewards, total_reward


def _parse_join_costs(path: Path):
    rows = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("predator_id")
            if not pid:
                continue
            rows[pid] = {
                "join_cost_total": _to_float(row.get("join_cost_total")) or 0.0,
                "join_cost_events": int(float(row.get("join_cost_events") or 0)),
                "attempt_cost_total": _to_float(row.get("attempt_cost_total")) or 0.0,
                "attempt_cost_events": int(float(row.get("attempt_cost_events") or 0)),
                "total_hunt_cost": _to_float(row.get("total_hunt_cost")),
                "hunt_events": int(float(row.get("hunt_events") or 0)),
                "join_cost_per_event": _to_float(row.get("join_cost_per_event")) or 0.0,
            }
    return rows


def _compute_payoff_metrics(reward_summary_path: Path, join_cost_path: Path):
    rewards, total_reward = _parse_reward_summary(reward_summary_path)
    join_costs = _parse_join_costs(join_cost_path)

    predator_rewards = {k: v for k, v in rewards.items() if "predator" in k}
    predators_total = len(predator_rewards)

    joiner_rewards = []
    non_joiner_rewards = []
    joiner_net_rewards = []
    non_joiner_net_rewards = []
    joiner_net_rewards_attempt = []
    non_joiner_net_rewards_attempt = []
    joiner_net_rewards_total = []
    non_joiner_net_rewards_total = []
    join_cost_total = 0.0
    attempt_cost_total = 0.0
    total_hunt_cost_total = 0.0
    joiner_attempt_cost_total = 0.0
    non_joiner_attempt_cost_total = 0.0
    joiner_total_cost_total = 0.0
    non_joiner_total_cost_total = 0.0
    joiner_count = 0

    for pid, reward in predator_rewards.items():
        cost_info = join_costs.get(
            pid,
            {
                "join_cost_total": 0.0,
                "join_cost_events": 0,
                "attempt_cost_total": 0.0,
                "attempt_cost_events": 0,
                "total_hunt_cost": None,
                "hunt_events": 0,
                "join_cost_per_event": 0.0,
            },
        )
        join_cost_total_pid = float(cost_info.get("join_cost_total") or 0.0)
        join_cost_events = int(cost_info.get("join_cost_events") or 0)
        attempt_cost_total_pid = float(cost_info.get("attempt_cost_total") or 0.0)
        total_hunt_cost_pid = cost_info.get("total_hunt_cost")
        if total_hunt_cost_pid is None:
            total_hunt_cost_pid = join_cost_total_pid + attempt_cost_total_pid

        join_cost_total += join_cost_total_pid
        attempt_cost_total += attempt_cost_total_pid
        total_hunt_cost_total += total_hunt_cost_pid

        if join_cost_events > 0:
            joiner_count += 1
            joiner_rewards.append(reward)
            joiner_net_rewards.append(reward - join_cost_total_pid)
            joiner_net_rewards_attempt.append(reward - attempt_cost_total_pid)
            joiner_net_rewards_total.append(reward - total_hunt_cost_pid)
            joiner_attempt_cost_total += attempt_cost_total_pid
            joiner_total_cost_total += total_hunt_cost_pid
        else:
            non_joiner_rewards.append(reward)
            non_joiner_net_rewards.append(reward - join_cost_total_pid)
            non_joiner_net_rewards_attempt.append(reward - attempt_cost_total_pid)
            non_joiner_net_rewards_total.append(reward - total_hunt_cost_pid)
            non_joiner_attempt_cost_total += attempt_cost_total_pid
            non_joiner_total_cost_total += total_hunt_cost_pid

    non_joiner_count = predators_total - joiner_count

    return {
        "reward_total": total_reward,
        "predators_with_rewards": predators_total,
        "joiners_count": joiner_count,
        "non_joiners_count": non_joiner_count,
        "join_cost_total_from_csv": join_cost_total,
        "attempt_cost_total_from_csv": attempt_cost_total,
        "total_hunt_cost_total_from_csv": total_hunt_cost_total,
        "reward_mean_predator": _mean(predator_rewards.values()),
        "reward_mean_joiner": _mean(joiner_rewards),
        "reward_mean_non_joiner": _mean(non_joiner_rewards),
        "reward_minus_cost_mean_joiner": _mean(joiner_net_rewards),
        "reward_minus_cost_mean_non_joiner": _mean(non_joiner_net_rewards),
        "reward_minus_attempt_cost_mean_joiner": _mean(joiner_net_rewards_attempt),
        "reward_minus_attempt_cost_mean_non_joiner": _mean(non_joiner_net_rewards_attempt),
        "reward_minus_total_cost_mean_joiner": _mean(joiner_net_rewards_total),
        "reward_minus_total_cost_mean_non_joiner": _mean(non_joiner_net_rewards_total),
        "reward_diff_joiner_vs_non_joiner": (
            _mean(joiner_rewards) - _mean(non_joiner_rewards)
            if joiner_rewards and non_joiner_rewards
            else None
        ),
        "reward_minus_cost_diff_joiner_vs_non_joiner": (
            _mean(joiner_net_rewards) - _mean(non_joiner_net_rewards)
            if joiner_net_rewards and non_joiner_net_rewards
            else None
        ),
        "reward_minus_attempt_cost_diff_joiner_vs_non_joiner": (
            _mean(joiner_net_rewards_attempt) - _mean(non_joiner_net_rewards_attempt)
            if joiner_net_rewards_attempt and non_joiner_net_rewards_attempt
            else None
        ),
        "reward_minus_total_cost_diff_joiner_vs_non_joiner": (
            _mean(joiner_net_rewards_total) - _mean(non_joiner_net_rewards_total)
            if joiner_net_rewards_total and non_joiner_net_rewards_total
            else None
        ),
        "join_cost_mean_joiner": (
            join_cost_total / joiner_count if joiner_count else None
        ),
        "join_cost_mean_all_predators": (
            join_cost_total / predators_total if predators_total else None
        ),
        "attempt_cost_mean_joiner": (
            joiner_attempt_cost_total / joiner_count if joiner_count else None
        ),
        "attempt_cost_mean_non_joiner": (
            non_joiner_attempt_cost_total / non_joiner_count if non_joiner_count else None
        ),
        "attempt_cost_mean_all_predators": (
            attempt_cost_total / predators_total if predators_total else None
        ),
        "total_hunt_cost_mean_joiner": (
            joiner_total_cost_total / joiner_count if joiner_count else None
        ),
        "total_hunt_cost_mean_non_joiner": (
            non_joiner_total_cost_total / non_joiner_count if non_joiner_count else None
        ),
        "total_hunt_cost_mean_all_predators": (
            total_hunt_cost_total / predators_total if predators_total else None
        ),
    }


def _extract_run_metrics(run_entry):
    keys = [
        "join_decision_rate",
        "defect_decision_rate",
        "join_steps",
        "defect_steps",
        "total_predator_steps",
        "coop_capture_rate",
        "solo_capture_rate",
        "coop_captures",
        "solo_captures",
        "captures_successful",
        "free_rider_share",
        "coop_free_rider_rate",
        "coop_free_rider_presence_rate",
        "free_riders_total",
        "joiners_total",
        "coop_free_riders_total",
        "coop_participants_total",
        "coop_captures_with_free_riders",
        "team_capture_failure_rate",
        "team_capture_attempts",
        "team_capture_failures",
        "team_capture_successes",
        "team_capture_coop_failure_rate",
        "team_capture_coop_attempts",
        "team_capture_coop_failures",
        "team_capture_coop_successes",
        "team_capture_solo_failure_rate",
        "team_capture_solo_attempts",
        "team_capture_solo_failures",
        "team_capture_solo_successes",
        "team_capture_mammoth_failure_rate",
        "team_capture_mammoth_attempts",
        "team_capture_mammoth_failures",
        "team_capture_mammoth_successes",
        "team_capture_rabbit_failure_rate",
        "team_capture_rabbit_attempts",
        "team_capture_rabbit_failures",
        "team_capture_rabbit_successes",
        "join_cost_total",
        "join_cost_events",
        "predators_with_join_cost",
        "predators_total",
        "join_cost_per_event",
        "join_cost_per_predator",
        "join_cost_per_predator_all",
        "attempt_cost_total",
        "attempt_cost_events",
        "predators_with_attempt_cost",
        "attempt_cost_per_event",
        "attempt_cost_per_predator",
        "attempt_cost_per_predator_all",
    ]
    payload = {}
    for key in keys:
        if key in run_entry:
            payload[key] = run_entry.get(key)
    return payload


def _mean_rows(rows, keys):
    summary = {}
    for key in keys:
        summary[key] = _mean([row.get(key) for row in rows])
    return summary


def _write_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(path: Path, scav_rows):
    def _fmt(value):
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = []
    lines.append("# Eval Payoff Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(
        "Note: `reward_minus_cost_*` subtracts join cost from rewards. "
        "If the reward already includes the join cost penalty, this will double-count the penalty. "
        "Use it as a counterfactual lens only if unsure."
    )
    lines.append(
        "Note: `reward_minus_attempt_cost_*` subtracts attempt cost; "
        "`reward_minus_total_cost_*` subtracts join + attempt cost."
    )
    lines.append("")

    header = [
        "scavenger",
        "n_eval_dirs",
        "n_runs",
        "join_rate",
        "defect_rate",
        "free_rider_share",
        "joiner_reward",
        "non_joiner_reward",
        "reward_diff",
        "joiner_reward_minus_cost",
        "non_joiner_reward_minus_cost",
        "reward_minus_cost_diff",
        "join_cost_mean_joiner",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in sorted(scav_rows, key=lambda r: float(r["scavenger_fraction"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("scavenger_fraction")),
                    _fmt(row.get("n_eval_dirs")),
                    _fmt(row.get("n_runs_total")),
                    _fmt(row.get("join_decision_rate")),
                    _fmt(row.get("defect_decision_rate")),
                    _fmt(row.get("free_rider_share")),
                    _fmt(row.get("reward_mean_joiner")),
                    _fmt(row.get("reward_mean_non_joiner")),
                    _fmt(row.get("reward_diff_joiner_vs_non_joiner")),
                    _fmt(row.get("reward_minus_cost_mean_joiner")),
                    _fmt(row.get("reward_minus_cost_mean_non_joiner")),
                    _fmt(row.get("reward_minus_cost_diff_joiner_vs_non_joiner")),
                    _fmt(row.get("join_cost_mean_joiner")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("Attempt/Total Cost Impact")
    lines.append("")
    header = [
        "scavenger",
        "attempt_cost_mean_joiner",
        "attempt_cost_mean_non_joiner",
        "total_hunt_cost_mean_joiner",
        "total_hunt_cost_mean_non_joiner",
        "reward_minus_attempt_cost_diff",
        "reward_minus_total_cost_diff",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in sorted(scav_rows, key=lambda r: float(r["scavenger_fraction"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("scavenger_fraction")),
                    _fmt(row.get("attempt_cost_mean_joiner")),
                    _fmt(row.get("attempt_cost_mean_non_joiner")),
                    _fmt(row.get("total_hunt_cost_mean_joiner")),
                    _fmt(row.get("total_hunt_cost_mean_non_joiner")),
                    _fmt(row.get("reward_minus_attempt_cost_diff_joiner_vs_non_joiner")),
                    _fmt(row.get("reward_minus_total_cost_diff_joiner_vs_non_joiner")),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ray_results = Path(RAY_RESULTS_DIR)
    output_per_run = ray_results / OUTPUT_PER_RUN_CSV

    scavenger_filters = SCAVENGER_FILTERS if SCAVENGER_FILTERS else None

    per_run_rows = []

    for run_dir in sorted(ray_results.iterdir()):
        if not run_dir.is_dir():
            continue

        eval_paths = _find_eval_aggregate_paths(run_dir)
        if EVAL_NAME_FILTER:
            eval_paths = [item for item in eval_paths if EVAL_NAME_FILTER in item[0].name]
        if not eval_paths:
            continue
        if not INCLUDE_ALL_EVALS:
            eval_paths = [_select_latest_eval(eval_paths)]

        for eval_dir, agg_path in eval_paths:
            config_env = _load_json(eval_dir / "config_env.json") or {}
            scavenger_fraction = config_env.get("team_capture_scavenger_fraction")
            if scavenger_filters is not None and not _float_matches(scavenger_fraction, scavenger_filters):
                continue

            aggregate = _load_json(agg_path)
            if not aggregate:
                continue

            per_run_stats = aggregate.get("per_run_stats", {})
            run_entries = per_run_stats.get("per_run", [])
            summary_dir = eval_dir / "summary_data"

            for idx, run_entry in enumerate(run_entries):
                run_num = run_entry.get("run") or (idx + 1)
                seed = run_entry.get("seed")
                reward_summary_path = summary_dir / f"reward_summary_{run_num}.txt"
                join_costs_path = summary_dir / f"join_costs_per_predator_{run_num}.csv"

                payoff_metrics = _compute_payoff_metrics(reward_summary_path, join_costs_path)

                row = {
                    "experiment_dir": run_dir.name,
                    "eval_dir": eval_dir.name,
                    "scavenger_fraction": scavenger_fraction,
                    "join_cost": config_env.get("team_capture_join_cost"),
                    "max_steps": config_env.get("max_steps"),
                    "run": run_num,
                    "seed": seed,
                    "eval_path": str(eval_dir.relative_to(ray_results)),
                }
                row.update(_extract_run_metrics(run_entry))
                row.update(payoff_metrics)
                per_run_rows.append(row)

    if not per_run_rows:
        print("No evaluation metrics found. Make sure defection_metrics_aggregate.json exists.")
        return

    per_run_columns = [
        "experiment_dir",
        "eval_dir",
        "scavenger_fraction",
        "join_cost",
        "max_steps",
        "run",
        "seed",
        "join_decision_rate",
        "defect_decision_rate",
        "join_steps",
        "defect_steps",
        "total_predator_steps",
        "coop_capture_rate",
        "solo_capture_rate",
        "coop_captures",
        "solo_captures",
        "captures_successful",
        "free_rider_share",
        "coop_free_rider_rate",
        "coop_free_rider_presence_rate",
        "free_riders_total",
        "joiners_total",
        "coop_free_riders_total",
        "coop_participants_total",
        "coop_captures_with_free_riders",
        "team_capture_failure_rate",
        "team_capture_attempts",
        "team_capture_failures",
        "team_capture_successes",
        "team_capture_coop_failure_rate",
        "team_capture_coop_attempts",
        "team_capture_coop_failures",
        "team_capture_coop_successes",
        "team_capture_solo_failure_rate",
        "team_capture_solo_attempts",
        "team_capture_solo_failures",
        "team_capture_solo_successes",
        "team_capture_mammoth_failure_rate",
        "team_capture_mammoth_attempts",
        "team_capture_mammoth_failures",
        "team_capture_mammoth_successes",
        "team_capture_rabbit_failure_rate",
        "team_capture_rabbit_attempts",
        "team_capture_rabbit_failures",
        "team_capture_rabbit_successes",
        "join_cost_total",
        "join_cost_events",
        "predators_with_join_cost",
        "predators_total",
        "join_cost_per_event",
        "join_cost_per_predator",
        "join_cost_per_predator_all",
        "attempt_cost_total",
        "attempt_cost_events",
        "predators_with_attempt_cost",
        "attempt_cost_per_event",
        "attempt_cost_per_predator",
        "attempt_cost_per_predator_all",
        "reward_total",
        "predators_with_rewards",
        "joiners_count",
        "non_joiners_count",
        "join_cost_total_from_csv",
        "attempt_cost_total_from_csv",
        "total_hunt_cost_total_from_csv",
        "reward_mean_predator",
        "reward_mean_joiner",
        "reward_mean_non_joiner",
        "reward_minus_cost_mean_joiner",
        "reward_minus_cost_mean_non_joiner",
        "reward_minus_attempt_cost_mean_joiner",
        "reward_minus_attempt_cost_mean_non_joiner",
        "reward_minus_total_cost_mean_joiner",
        "reward_minus_total_cost_mean_non_joiner",
        "reward_diff_joiner_vs_non_joiner",
        "reward_minus_cost_diff_joiner_vs_non_joiner",
        "reward_minus_attempt_cost_diff_joiner_vs_non_joiner",
        "reward_minus_total_cost_diff_joiner_vs_non_joiner",
        "join_cost_mean_joiner",
        "join_cost_mean_all_predators",
        "attempt_cost_mean_joiner",
        "attempt_cost_mean_non_joiner",
        "attempt_cost_mean_all_predators",
        "total_hunt_cost_mean_joiner",
        "total_hunt_cost_mean_non_joiner",
        "total_hunt_cost_mean_all_predators",
        "eval_path",
    ]

    per_run_rows.sort(
        key=lambda r: (
            r.get("scavenger_fraction") if r.get("scavenger_fraction") is not None else 999.0,
            r.get("experiment_dir") or "",
            r.get("eval_dir") or "",
            r.get("run") or 0,
        )
    )

    _write_csv(output_per_run, per_run_rows, per_run_columns)
    print(f"Wrote {len(per_run_rows)} rows to {output_per_run}")

    per_eval_rows = []
    grouped = {}
    for row in per_run_rows:
        key = (row.get("experiment_dir"), row.get("eval_dir"))
        grouped.setdefault(key, []).append(row)

    eval_columns = [
        "experiment_dir",
        "eval_dir",
        "scavenger_fraction",
        "join_cost",
        "max_steps",
        "n_runs",
    ]
    mean_columns = [
        "join_decision_rate",
        "defect_decision_rate",
        "coop_capture_rate",
        "solo_capture_rate",
        "free_rider_share",
        "coop_free_rider_rate",
        "coop_free_rider_presence_rate",
        "team_capture_failure_rate",
        "team_capture_coop_failure_rate",
        "team_capture_solo_failure_rate",
        "join_cost_total",
        "join_cost_total_from_csv",
        "attempt_cost_total_from_csv",
        "total_hunt_cost_total_from_csv",
        "join_cost_events",
        "join_cost_per_predator",
        "join_cost_per_predator_all",
        "attempt_cost_total",
        "attempt_cost_events",
        "predators_with_attempt_cost",
        "attempt_cost_per_event",
        "attempt_cost_per_predator",
        "attempt_cost_per_predator_all",
        "predators_with_rewards",
        "joiners_count",
        "non_joiners_count",
        "reward_mean_predator",
        "reward_mean_joiner",
        "reward_mean_non_joiner",
        "reward_minus_cost_mean_joiner",
        "reward_minus_cost_mean_non_joiner",
        "reward_minus_attempt_cost_mean_joiner",
        "reward_minus_attempt_cost_mean_non_joiner",
        "reward_minus_total_cost_mean_joiner",
        "reward_minus_total_cost_mean_non_joiner",
        "reward_diff_joiner_vs_non_joiner",
        "reward_minus_cost_diff_joiner_vs_non_joiner",
        "reward_minus_attempt_cost_diff_joiner_vs_non_joiner",
        "reward_minus_total_cost_diff_joiner_vs_non_joiner",
        "join_cost_mean_joiner",
        "join_cost_mean_all_predators",
        "attempt_cost_mean_joiner",
        "attempt_cost_mean_non_joiner",
        "attempt_cost_mean_all_predators",
        "total_hunt_cost_mean_joiner",
        "total_hunt_cost_mean_non_joiner",
        "total_hunt_cost_mean_all_predators",
    ]

    for (experiment_dir, eval_dir), rows in grouped.items():
        base_row = rows[0]
        summary = {
            "experiment_dir": experiment_dir,
            "eval_dir": eval_dir,
            "scavenger_fraction": base_row.get("scavenger_fraction"),
            "join_cost": base_row.get("join_cost"),
            "max_steps": base_row.get("max_steps"),
            "n_runs": len(rows),
        }
        summary.update(_mean_rows(rows, mean_columns))
        per_eval_rows.append(summary)

    per_eval_rows.sort(
        key=lambda r: (
            r.get("scavenger_fraction") if r.get("scavenger_fraction") is not None else 999.0,
            r.get("experiment_dir") or "",
            r.get("eval_dir") or "",
        )
    )

    output_per_eval = ray_results / OUTPUT_PER_EVAL_CSV
    _write_csv(output_per_eval, per_eval_rows, eval_columns + mean_columns)
    print(f"Wrote {len(per_eval_rows)} eval summaries to {output_per_eval}")

    by_scav = {}
    for row in per_eval_rows:
        key = row.get("scavenger_fraction")
        by_scav.setdefault(key, []).append(row)

    by_scav_rows = []
    for scavenger_fraction, rows in by_scav.items():
        summary = {
            "scavenger_fraction": scavenger_fraction,
            "n_eval_dirs": len(rows),
            "n_runs_total": sum(r.get("n_runs", 0) for r in rows),
        }
        summary.update(_mean_rows(rows, mean_columns))
        by_scav_rows.append(summary)

    by_scav_rows.sort(key=lambda r: float(r["scavenger_fraction"]) if r.get("scavenger_fraction") is not None else 999.0)
    output_by_scav = ray_results / OUTPUT_BY_SCAV_CSV
    _write_csv(output_by_scav, by_scav_rows, ["scavenger_fraction", "n_eval_dirs", "n_runs_total"] + mean_columns)
    print(f"Wrote {len(by_scav_rows)} scavenger summaries to {output_by_scav}")

    report_path = ray_results / OUTPUT_REPORT_MD
    _write_report(report_path, by_scav_rows)
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
