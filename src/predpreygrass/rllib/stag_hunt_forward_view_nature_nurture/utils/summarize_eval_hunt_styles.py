#!/usr/bin/env python3
"""Summarize reproductive reward (and net of hunt costs) by hunt style.

Classifies predators into dominant hunt styles based on event logs:
- solo_hunter: more solo join events than cooperative join events
- group_hunter: more cooperative join events than solo join events
- free_rider: no join events, but free-ride events present
- mixed_joiner: tie between solo and coop join events

Outputs per-run and per-scavenger summaries.
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


RAY_RESULTS_DIR = os.getenv(
    "RAY_RESULTS_DIR",
    "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/"
    "stag_hunt_forward_view_nature_nurture/ray_results/join_cost_0.02",
)
OUTPUT_PER_RUN_CSV = "eval_hunt_style_per_run.csv"
OUTPUT_BY_SCAV_CSV = "eval_hunt_style_by_scavenger.csv"
OUTPUT_REPORT_MD = "eval_hunt_style_report.md"
EVAL_NAME_FILTER = None  # e.g. "eval_10_runs" to only include matching eval dirs

RUN_RE = re.compile(r"agent_event_log_(\d+)\.json$")


def _load_json(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value):
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_reward_summary(path: Path):
    rewards = {}
    if not path.is_file():
        return rewards
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("Total Reward"):
                continue
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            name = name.strip()
            value = _to_float(value.strip())
            rewards[name] = value
    return rewards


def _extract_run_num(path: Path) -> int | None:
    match = RUN_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def _find_eval_dirs(root: Path):
    eval_dirs = defaultdict(set)
    for log_path in root.rglob("summary_data/agent_event_log_*.json"):
        run_num = _extract_run_num(log_path)
        if run_num is None:
            continue
        eval_dir = log_path.parent.parent
        if EVAL_NAME_FILTER and EVAL_NAME_FILTER not in eval_dir.name:
            continue
        eval_dirs[eval_dir].add(run_num)
    return eval_dirs


def _classify_agent(counts: dict) -> str:
    solo = counts.get("solo_join_events", 0)
    coop = counts.get("coop_join_events", 0)
    free = counts.get("free_ride_events", 0)
    join_total = solo + coop

    if join_total == 0:
        if free > 0:
            return "free_rider"
        return "no_hunt"
    if coop > solo:
        return "group_hunter"
    if solo > coop:
        return "solo_hunter"
    return "mixed_joiner"


def _analyze_event_log(event_log: dict):
    per_agent = {}
    for aid, record in event_log.items():
        if "predator" not in aid:
            continue
        counts = {
            "solo_join_events": 0,
            "coop_join_events": 0,
            "free_ride_events": 0,
            "join_cost_total": 0.0,
            "attempt_cost_total": 0.0,
        }
        for evt in record.get("eating_events", []):
            join_hunt = bool(evt.get("join_hunt", False))
            team_capture = bool(evt.get("team_capture", False))
            if join_hunt:
                if team_capture:
                    counts["coop_join_events"] += 1
                else:
                    counts["solo_join_events"] += 1
            else:
                counts["free_ride_events"] += 1
            counts["join_cost_total"] += _to_float(evt.get("join_cost", 0.0))
            counts["attempt_cost_total"] += _to_float(evt.get("attempt_cost", 0.0))
        for evt in record.get("failed_eating_events", []):
            join_hunt = bool(evt.get("join_hunt", False))
            team_capture = bool(evt.get("team_capture", False))
            if join_hunt:
                if team_capture:
                    counts["coop_join_events"] += 1
                else:
                    counts["solo_join_events"] += 1
            else:
                counts["free_ride_events"] += 1
            counts["join_cost_total"] += _to_float(evt.get("join_cost", 0.0))
            counts["attempt_cost_total"] += _to_float(evt.get("attempt_cost", 0.0))
        counts["total_hunt_cost"] = counts["join_cost_total"] + counts["attempt_cost_total"]
        counts["group"] = _classify_agent(counts)
        per_agent[aid] = counts
    return per_agent


def _mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def _write_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary_rows: list[dict]):
    def _fmt(value):
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    lines = []
    lines.append("# Hunt Style Reproductive Payoff Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(
        "Groups are based on dominant event type: `group_hunter` (more coop than solo join events), "
        "`solo_hunter` (more solo than coop join events), `free_rider` (no join events but free-ride events present)."
    )
    lines.append("")
    header = [
        "scavenger",
        "group_hunters_n",
        "solo_hunters_n",
        "free_riders_n",
        "group_net_reward",
        "solo_net_reward",
        "free_net_reward",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("scavenger_fraction")),
                    str(int(row.get("group_hunter_count", 0))),
                    str(int(row.get("solo_hunter_count", 0))),
                    str(int(row.get("free_rider_count", 0))),
                    _fmt(row.get("group_hunter_net_reward_mean")),
                    _fmt(row.get("solo_hunter_net_reward_mean")),
                    _fmt(row.get("free_rider_net_reward_mean")),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    root = Path(RAY_RESULTS_DIR)
    eval_dirs = _find_eval_dirs(root)
    if not eval_dirs:
        print("No eval dirs found.")
        return

    per_run_rows = []
    agg = {}

    for eval_dir, runs in sorted(eval_dirs.items()):
        config_env = _load_json(eval_dir / "config_env.json") or {}
        scavenger_fraction = config_env.get("team_capture_scavenger_fraction")
        summary_dir = eval_dir / "summary_data"

        for run_num in sorted(runs):
            reward_summary_path = summary_dir / f"reward_summary_{run_num}.txt"
            event_log_path = summary_dir / f"agent_event_log_{run_num}.json"
            if not reward_summary_path.is_file() or not event_log_path.is_file():
                continue

            rewards = _parse_reward_summary(reward_summary_path)
            event_log = _load_json(event_log_path) or {}
            per_agent = _analyze_event_log(event_log)

            group_rewards = defaultdict(list)
            group_net_rewards = defaultdict(list)
            group_counts = defaultdict(int)

            for aid, stats in per_agent.items():
                if aid not in rewards:
                    continue
                reward = _to_float(rewards.get(aid, 0.0))
                net_reward = reward - _to_float(stats.get("total_hunt_cost", 0.0))
                group = stats.get("group", "no_hunt")
                if group == "no_hunt":
                    continue
                group_rewards[group].append(reward)
                group_net_rewards[group].append(net_reward)
                group_counts[group] += 1

                # aggregate by scavenger fraction
                agg_key = scavenger_fraction
                bucket = agg.setdefault(
                    agg_key,
                    {
                        "group_hunter": {"count": 0, "reward_sum": 0.0, "net_sum": 0.0},
                        "solo_hunter": {"count": 0, "reward_sum": 0.0, "net_sum": 0.0},
                        "free_rider": {"count": 0, "reward_sum": 0.0, "net_sum": 0.0},
                    },
                )
                if group in bucket:
                    bucket[group]["count"] += 1
                    bucket[group]["reward_sum"] += reward
                    bucket[group]["net_sum"] += net_reward

            per_run_rows.append(
                {
                    "eval_dir": eval_dir.name,
                    "scavenger_fraction": scavenger_fraction,
                    "run": run_num,
                    "group_hunter_count": group_counts.get("group_hunter", 0),
                    "solo_hunter_count": group_counts.get("solo_hunter", 0),
                    "free_rider_count": group_counts.get("free_rider", 0),
                    "group_hunter_reward_mean": _mean(group_rewards.get("group_hunter", [])),
                    "solo_hunter_reward_mean": _mean(group_rewards.get("solo_hunter", [])),
                    "free_rider_reward_mean": _mean(group_rewards.get("free_rider", [])),
                    "group_hunter_net_reward_mean": _mean(group_net_rewards.get("group_hunter", [])),
                    "solo_hunter_net_reward_mean": _mean(group_net_rewards.get("solo_hunter", [])),
                    "free_rider_net_reward_mean": _mean(group_net_rewards.get("free_rider", [])),
                }
            )

    per_run_columns = [
        "eval_dir",
        "scavenger_fraction",
        "run",
        "group_hunter_count",
        "solo_hunter_count",
        "free_rider_count",
        "group_hunter_reward_mean",
        "solo_hunter_reward_mean",
        "free_rider_reward_mean",
        "group_hunter_net_reward_mean",
        "solo_hunter_net_reward_mean",
        "free_rider_net_reward_mean",
    ]

    per_run_path = root / OUTPUT_PER_RUN_CSV
    _write_csv(per_run_path, per_run_rows, per_run_columns)
    print(f"Wrote {len(per_run_rows)} rows to {per_run_path}")

    summary_rows = []
    for scavenger_fraction, groups in sorted(agg.items(), key=lambda item: float(item[0] or 0.0)):
        def _mean_from(group, key):
            count = group["count"]
            if not count:
                return 0.0
            return group[key] / count

        row = {
            "scavenger_fraction": scavenger_fraction,
            "group_hunter_count": groups["group_hunter"]["count"],
            "solo_hunter_count": groups["solo_hunter"]["count"],
            "free_rider_count": groups["free_rider"]["count"],
            "group_hunter_reward_mean": _mean_from(groups["group_hunter"], "reward_sum"),
            "solo_hunter_reward_mean": _mean_from(groups["solo_hunter"], "reward_sum"),
            "free_rider_reward_mean": _mean_from(groups["free_rider"], "reward_sum"),
            "group_hunter_net_reward_mean": _mean_from(groups["group_hunter"], "net_sum"),
            "solo_hunter_net_reward_mean": _mean_from(groups["solo_hunter"], "net_sum"),
            "free_rider_net_reward_mean": _mean_from(groups["free_rider"], "net_sum"),
        }
        summary_rows.append(row)

    summary_cols = [
        "scavenger_fraction",
        "group_hunter_count",
        "solo_hunter_count",
        "free_rider_count",
        "group_hunter_reward_mean",
        "solo_hunter_reward_mean",
        "free_rider_reward_mean",
        "group_hunter_net_reward_mean",
        "solo_hunter_net_reward_mean",
        "free_rider_net_reward_mean",
    ]
    summary_path = root / OUTPUT_BY_SCAV_CSV
    _write_csv(summary_path, summary_rows, summary_cols)
    print(f"Wrote {len(summary_rows)} rows to {summary_path}")

    report_path = root / OUTPUT_REPORT_MD
    _write_report(report_path, summary_rows)
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
