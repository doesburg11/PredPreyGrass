#!/usr/bin/env python3
"""Summarize team-capture failure metrics from an evaluation directory.

Edit EVAL_DIR below or pass the evaluation directory as the first argument.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


RAY_RESULTS_DIR = (
    "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/"
    "stag_hunt_forward_view/ray_results"
)
CHECKPOINT_ROOT = (
    "STAG_HUNT_FORWARD_VIEW_JOIN_COST_0_02_SCAVENGER_0_1_2026-01-25_14-20-20/"
    "PPO_PredPreyGrass_99161_00000_0_2026-01-25_14-20-20"
)
CHECKPOINT_NR = "checkpoint_000099"
EVAL_DIR_NAME = "eval_10_runs_STAG_HUNT_FORWARD_VIEW_2026-02-03_10-42-43"
EVAL_DIR = str(Path(RAY_RESULTS_DIR) / CHECKPOINT_ROOT / CHECKPOINT_NR / EVAL_DIR_NAME)


def _load_json(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_defection_metrics(eval_dir: Path):
    summary_dir = eval_dir / "summary_data"
    per_run = []
    aggregate = None

    if summary_dir.is_dir():
        files = sorted(summary_dir.glob("defection_metrics_*.json"))
        for fpath in files:
            data = _load_json(fpath)
            if data is None:
                continue
            if "aggregate" in fpath.name:
                aggregate = data
            else:
                per_run.append(data)

    if aggregate is None:
        aggregate = _load_json(summary_dir / "defection_metrics_aggregate.json")
    if aggregate is None:
        aggregate = _load_json(eval_dir / "defection_metrics_aggregate.json")

    return aggregate, per_run


def _fmt_pct(value):
    return f"{value * 100:.2f}%"


def _summarize_rates(per_run_metrics, key):
    values = []
    for metrics in per_run_metrics:
        failures = metrics.get("capture_failures", {})
        if key in failures:
            values.append(failures[key])
    if not values:
        return None
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }

def _format_rate_line(label, stats):
    if stats is None:
        return f"{label:32} n/a"
    return (
        f"{label:32} mean {_fmt_pct(stats['mean'])} |"
        f" min {_fmt_pct(stats['min'])} | max {_fmt_pct(stats['max'])}"
    )

def _format_aggregate_counts_lines(failures: dict):
    def _fmt_rate(key):
        value = failures.get(key)
        return _fmt_pct(value) if isinstance(value, (int, float)) else "n/a"

    lines = [
        f"team_capture_attempts: {failures.get('team_capture_attempts', 'n/a')} "
        f"(success {failures.get('team_capture_successes', 'n/a')}, "
        f"fail {failures.get('team_capture_failures', 'n/a')}), "
        f"failure rate {_fmt_rate('team_capture_failure_rate')}"
    ]
    lines.append(
        f"team_capture_coop_attempts: {failures.get('team_capture_coop_attempts', 'n/a')} "
        f"(success {failures.get('team_capture_coop_successes', 'n/a')}, "
        f"fail {failures.get('team_capture_coop_failures', 'n/a')}), "
        f"failure rate {_fmt_rate('team_capture_coop_failure_rate')}"
    )
    lines.append(
        f"team_capture_solo_attempts: {failures.get('team_capture_solo_attempts', 'n/a')} "
        f"(success {failures.get('team_capture_solo_successes', 'n/a')}, "
        f"fail {failures.get('team_capture_solo_failures', 'n/a')}), "
        f"failure rate {_fmt_rate('team_capture_solo_failure_rate')}"
    )
    lines.append(
        f"team_capture_mammoth_attempts: {failures.get('team_capture_mammoth_attempts', 'n/a')} "
        f"(success {failures.get('team_capture_mammoth_successes', 'n/a')}, "
        f"fail {failures.get('team_capture_mammoth_failures', 'n/a')}), "
        f"failure rate {_fmt_rate('team_capture_mammoth_failure_rate')}"
    )
    lines.append(
        f"team_capture_rabbit_attempts: {failures.get('team_capture_rabbit_attempts', 'n/a')} "
        f"(success {failures.get('team_capture_rabbit_successes', 'n/a')}, "
        f"fail {failures.get('team_capture_rabbit_failures', 'n/a')}), "
        f"failure rate {_fmt_rate('team_capture_rabbit_failure_rate')}"
    )
    return lines


def build_capture_failure_summary(eval_dir: Path):
    aggregate, per_run = _collect_defection_metrics(eval_dir)
    payload = {"eval_dir": str(eval_dir), "per_run": None, "aggregate": None}
    lines = [f"Eval dir: {eval_dir}"]

    if aggregate is None and not per_run:
        lines.append("No defection_metrics JSON files found.")
        payload["error"] = "no_defection_metrics"
        return payload, "\n".join(lines) + "\n"

    per_run_with_failures = [
        metrics for metrics in per_run if isinstance(metrics, dict) and "capture_failures" in metrics
    ]
    if not per_run_with_failures:
        lines.append("No capture_failures found in per-run metrics. Re-run eval with updated script.")
        payload["per_run"] = {"n_runs": 0, "rates": {}}
    else:
        title = f"Failure Rates Across {len(per_run_with_failures)} Runs"
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))
        rates = {}
        for label, key in [
            ("Overall failure rate", "team_capture_failure_rate"),
            ("Coop failure rate", "team_capture_coop_failure_rate"),
            ("Solo failure rate", "team_capture_solo_failure_rate"),
            ("Mammoth failure rate", "team_capture_mammoth_failure_rate"),
            ("Rabbit failure rate", "team_capture_rabbit_failure_rate"),
        ]:
            stats = _summarize_rates(per_run_with_failures, key)
            rates[key] = stats
            lines.append(_format_rate_line(label, stats))
        payload["per_run"] = {"n_runs": len(per_run_with_failures), "rates": rates}

    if aggregate:
        failures = aggregate.get("capture_failures", {})
        if failures:
            title = "Aggregate Counts"
            lines.append("")
            lines.append(title)
            lines.append("-" * len(title))
            lines.extend(_format_aggregate_counts_lines(failures))
            payload["aggregate"] = failures
        else:
            lines.append("")
            lines.append("No capture_failures found in aggregate metrics.")
            payload["aggregate"] = None

    return payload, "\n".join(lines) + "\n"


def main():
    eval_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(EVAL_DIR)
    _, summary_text = build_capture_failure_summary(eval_dir)
    print(summary_text, end="")


if __name__ == "__main__":
    main()
