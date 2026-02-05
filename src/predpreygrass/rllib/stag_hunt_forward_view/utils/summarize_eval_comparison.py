#!/usr/bin/env python3
"""Collect cooperation/defection/free-rider/failed-capture metrics across eval runs.

Scans ray_results for defection_metrics_aggregate.json and writes a comparison CSV
plus an averaged summary CSV grouped by scavenger fraction.

Configuration lives below; no CLI arguments are used.
"""
from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path


RAY_RESULTS_DIR = os.getenv(
    "RAY_RESULTS_DIR",
    "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/"
    "stag_hunt_forward_view/ray_results/join_cost_0.02",
)

# Edit these settings directly.
OUTPUT_CSV = "eval_comparison.csv"
OUTPUT_SUMMARY_CSV = "eval_comparison_summary.csv"
OUTPUT_REPORT_MD = "eval_comparison_report.md"
OUTPUT_PLOTS_DIR = "eval_comparison_summary_plots"
INCLUDE_ALL_EVALS = True
SCAVENGER_FILTERS = [0.0, 0.1, 0.2, 0.3, 0.4]
EVAL_NAME_FILTER = None  # e.g. "eval_10_runs" to only include matching eval dirs
REQUIRE_N_RUNS = 30  # only include evals with this many runs in the summary; set None to disable
REQUIRE_FAILURE_METRICS = True  # skip evals without failure metrics in the summary


def _repo_root() -> Path:
    # This file lives at repo_root/src/predpreygrass/rllib/stag_hunt_forward_view/utils/...
    return Path(__file__).resolve().parents[5]


def _assets_plots_dir() -> Path:
    return _repo_root() / "assets" / OUTPUT_PLOTS_DIR


def _load_json(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_eval_timestamp(name: str):
    match = re.search(r"(\\d{4}-\\d{2}-\\d{2})_(\\d{2}-\\d{2}-\\d{2})", name)
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


def _extract_metrics(aggregate):
    join_defect = aggregate.get("join_defect", {})
    capture = aggregate.get("capture_outcomes", {})
    failures = aggregate.get("capture_failures", {})
    join_costs = aggregate.get("join_costs", {})
    n_runs = aggregate.get("n_runs")
    if n_runs is None:
        n_runs = aggregate.get("per_run_stats", {}).get("n_runs")

    join_cost_total = join_costs.get("join_cost_total")
    captures_successful = capture.get("captures_successful")
    coop_captures = capture.get("coop_captures")

    return {
        "n_runs": n_runs,
        "join_decision_rate": join_defect.get("join_decision_rate"),
        "defect_decision_rate": join_defect.get("defect_decision_rate"),
        "join_steps": join_defect.get("join_steps"),
        "defect_steps": join_defect.get("defect_steps"),
        "total_predator_steps": join_defect.get("total_predator_steps"),
        "coop_capture_rate": capture.get("coop_capture_rate"),
        "solo_capture_rate": capture.get("solo_capture_rate"),
        "coop_captures": capture.get("coop_captures"),
        "solo_captures": capture.get("solo_captures"),
        "captures_successful": capture.get("captures_successful"),
        "free_rider_share": capture.get("free_rider_share"),
        "coop_free_rider_rate": capture.get("coop_free_rider_rate"),
        "coop_free_rider_presence_rate": capture.get("coop_free_rider_presence_rate"),
        "free_riders_total": capture.get("free_riders_total"),
        "joiners_total": capture.get("joiners_total"),
        "coop_free_riders_total": capture.get("coop_free_riders_total"),
        "coop_participants_total": capture.get("coop_participants_total"),
        "coop_captures_with_free_riders": capture.get("coop_captures_with_free_riders"),
        "team_capture_failure_rate": failures.get("team_capture_failure_rate"),
        "team_capture_attempts": failures.get("team_capture_attempts"),
        "team_capture_failures": failures.get("team_capture_failures"),
        "team_capture_successes": failures.get("team_capture_successes"),
        "team_capture_coop_failure_rate": failures.get("team_capture_coop_failure_rate"),
        "team_capture_coop_attempts": failures.get("team_capture_coop_attempts"),
        "team_capture_coop_failures": failures.get("team_capture_coop_failures"),
        "team_capture_coop_successes": failures.get("team_capture_coop_successes"),
        "team_capture_solo_failure_rate": failures.get("team_capture_solo_failure_rate"),
        "team_capture_solo_attempts": failures.get("team_capture_solo_attempts"),
        "team_capture_solo_failures": failures.get("team_capture_solo_failures"),
        "team_capture_solo_successes": failures.get("team_capture_solo_successes"),
        "team_capture_mammoth_failure_rate": failures.get("team_capture_mammoth_failure_rate"),
        "team_capture_mammoth_attempts": failures.get("team_capture_mammoth_attempts"),
        "team_capture_mammoth_failures": failures.get("team_capture_mammoth_failures"),
        "team_capture_mammoth_successes": failures.get("team_capture_mammoth_successes"),
        "team_capture_rabbit_failure_rate": failures.get("team_capture_rabbit_failure_rate"),
        "team_capture_rabbit_attempts": failures.get("team_capture_rabbit_attempts"),
        "team_capture_rabbit_failures": failures.get("team_capture_rabbit_failures"),
        "team_capture_rabbit_successes": failures.get("team_capture_rabbit_successes"),
        "join_cost_total": join_costs.get("join_cost_total"),
        "join_cost_events": join_costs.get("join_cost_events"),
        "predators_with_join_cost": join_costs.get("predators_with_join_cost"),
        "predators_total": join_costs.get("predators_total"),
        "join_cost_per_event": join_costs.get("join_cost_per_event"),
        "join_cost_per_predator": join_costs.get("join_cost_per_predator"),
        "join_cost_per_predator_all": join_costs.get("join_cost_per_predator_all"),
        "join_cost_per_successful_capture": (
            (join_cost_total / captures_successful)
            if join_cost_total is not None and captures_successful
            else None
        ),
        "join_cost_per_coop_capture": (
            (join_cost_total / coop_captures)
            if join_cost_total is not None and coop_captures
            else None
        ),
    }


def _safe_div(num, denom):
    return num / denom if denom else 0.0


def _sum_metric(rows, key):
    total = 0.0
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            total += value
    return total


def _summarize_group(rows):
    join_steps = _sum_metric(rows, "join_steps")
    defect_steps = _sum_metric(rows, "defect_steps")
    total_pred_steps = _sum_metric(rows, "total_predator_steps")
    coop_captures = _sum_metric(rows, "coop_captures")
    solo_captures = _sum_metric(rows, "solo_captures")
    captures_successful = _sum_metric(rows, "captures_successful")
    free_riders_total = _sum_metric(rows, "free_riders_total")
    joiners_total = _sum_metric(rows, "joiners_total")
    coop_free_riders_total = _sum_metric(rows, "coop_free_riders_total")
    coop_participants_total = _sum_metric(rows, "coop_participants_total")
    coop_captures_with_free_riders = _sum_metric(rows, "coop_captures_with_free_riders")
    team_capture_attempts = _sum_metric(rows, "team_capture_attempts")
    team_capture_failures = _sum_metric(rows, "team_capture_failures")
    team_capture_successes = _sum_metric(rows, "team_capture_successes")
    team_capture_coop_attempts = _sum_metric(rows, "team_capture_coop_attempts")
    team_capture_coop_failures = _sum_metric(rows, "team_capture_coop_failures")
    team_capture_coop_successes = _sum_metric(rows, "team_capture_coop_successes")
    team_capture_solo_attempts = _sum_metric(rows, "team_capture_solo_attempts")
    team_capture_solo_failures = _sum_metric(rows, "team_capture_solo_failures")
    team_capture_solo_successes = _sum_metric(rows, "team_capture_solo_successes")
    team_capture_mammoth_attempts = _sum_metric(rows, "team_capture_mammoth_attempts")
    team_capture_mammoth_failures = _sum_metric(rows, "team_capture_mammoth_failures")
    team_capture_mammoth_successes = _sum_metric(rows, "team_capture_mammoth_successes")
    team_capture_rabbit_attempts = _sum_metric(rows, "team_capture_rabbit_attempts")
    team_capture_rabbit_failures = _sum_metric(rows, "team_capture_rabbit_failures")
    team_capture_rabbit_successes = _sum_metric(rows, "team_capture_rabbit_successes")
    join_cost_total = _sum_metric(rows, "join_cost_total")
    join_cost_events = _sum_metric(rows, "join_cost_events")
    predators_with_join_cost = _sum_metric(rows, "predators_with_join_cost")
    predators_total = _sum_metric(rows, "predators_total")

    return {
        "n_eval_dirs": len(rows),
        "n_runs_total": _sum_metric(rows, "n_runs"),
        "join_steps": join_steps,
        "defect_steps": defect_steps,
        "total_predator_steps": total_pred_steps,
        "join_decision_rate": _safe_div(join_steps, total_pred_steps),
        "defect_decision_rate": _safe_div(defect_steps, total_pred_steps),
        "coop_captures": coop_captures,
        "solo_captures": solo_captures,
        "captures_successful": captures_successful,
        "coop_capture_rate": _safe_div(coop_captures, captures_successful),
        "solo_capture_rate": _safe_div(solo_captures, captures_successful),
        "free_riders_total": free_riders_total,
        "joiners_total": joiners_total,
        "free_rider_share": _safe_div(free_riders_total, joiners_total + free_riders_total),
        "coop_free_riders_total": coop_free_riders_total,
        "coop_participants_total": coop_participants_total,
        "coop_free_rider_rate": _safe_div(coop_free_riders_total, coop_participants_total),
        "coop_captures_with_free_riders": coop_captures_with_free_riders,
        "coop_free_rider_presence_rate": _safe_div(coop_captures_with_free_riders, coop_captures),
        "team_capture_attempts": team_capture_attempts,
        "team_capture_failures": team_capture_failures,
        "team_capture_successes": team_capture_successes,
        "team_capture_failure_rate": _safe_div(team_capture_failures, team_capture_attempts),
        "team_capture_coop_attempts": team_capture_coop_attempts,
        "team_capture_coop_failures": team_capture_coop_failures,
        "team_capture_coop_successes": team_capture_coop_successes,
        "team_capture_coop_failure_rate": _safe_div(team_capture_coop_failures, team_capture_coop_attempts),
        "team_capture_solo_attempts": team_capture_solo_attempts,
        "team_capture_solo_failures": team_capture_solo_failures,
        "team_capture_solo_successes": team_capture_solo_successes,
        "team_capture_solo_failure_rate": _safe_div(team_capture_solo_failures, team_capture_solo_attempts),
        "team_capture_mammoth_attempts": team_capture_mammoth_attempts,
        "team_capture_mammoth_failures": team_capture_mammoth_failures,
        "team_capture_mammoth_successes": team_capture_mammoth_successes,
        "team_capture_mammoth_failure_rate": _safe_div(team_capture_mammoth_failures, team_capture_mammoth_attempts),
        "team_capture_rabbit_attempts": team_capture_rabbit_attempts,
        "team_capture_rabbit_failures": team_capture_rabbit_failures,
        "team_capture_rabbit_successes": team_capture_rabbit_successes,
        "team_capture_rabbit_failure_rate": _safe_div(team_capture_rabbit_failures, team_capture_rabbit_attempts),
        "join_cost_total": join_cost_total,
        "join_cost_events": join_cost_events,
        "predators_with_join_cost": predators_with_join_cost,
        "predators_total": predators_total,
        "join_cost_per_event": _safe_div(join_cost_total, join_cost_events),
        "join_cost_per_predator": _safe_div(join_cost_total, predators_with_join_cost),
        "join_cost_per_predator_all": _safe_div(join_cost_total, predators_total),
        "join_cost_per_successful_capture": _safe_div(join_cost_total, captures_successful),
        "join_cost_per_coop_capture": _safe_div(join_cost_total, coop_captures),
    }


def _plot_series(plots_dir: Path, scav, series, title, ylabel, filename, labels=None, ylim=(0, 1)):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    plt.figure(figsize=(6, 4))
    if labels is None:
        labels = [None] * len(series)
    for values, label in zip(series, labels):
        if any(v is None for v in values):
            return None
        plt.plot(scav, values, marker="o", label=label)
    plt.title(title)
    plt.xlabel("Scavenger Fraction")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    if any(label for label in labels):
        plt.legend()
    plt.tight_layout()
    out_path = plots_dir / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _generate_plots(summary_rows, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        return []

    rows_sorted = sorted(summary_rows, key=lambda r: float(r["scavenger_fraction"]))
    scav = [float(r["scavenger_fraction"]) for r in rows_sorted]

    def _series(key):
        return [r.get(key) for r in rows_sorted]

    outputs = []

    def _add_plot(title, path):
        if path is not None:
            outputs.append((title, path))

    _add_plot(
        "Join vs Defect Decision Rate",
        _plot_series(
            plots_dir,
            scav,
            [_series("join_decision_rate"), _series("defect_decision_rate")],
            "Join vs Defect Decision Rate",
            "Rate",
            "join_defect_rate.png",
            labels=["join", "defect"],
        ),
    )
    _add_plot(
        "Coop vs Solo Capture Rate",
        _plot_series(
            plots_dir,
            scav,
            [_series("coop_capture_rate"), _series("solo_capture_rate")],
            "Coop vs Solo Capture Rate",
            "Rate",
            "coop_solo_capture_rate.png",
            labels=["coop", "solo"],
        ),
    )
    _add_plot(
        "Free Rider Share",
        _plot_series(
            plots_dir,
            scav,
            [_series("free_rider_share")],
            "Free Rider Share",
            "Share",
            "free_rider_share.png",
            labels=["free_rider_share"],
            ylim=(0, max(_series("free_rider_share") or [0.05]) * 1.2),
        ),
    )
    _add_plot(
        "Team Capture Failure Rate",
        _plot_series(
            plots_dir,
            scav,
            [_series("team_capture_failure_rate")],
            "Team Capture Failure Rate",
            "Failure Rate",
            "team_capture_failure_rate.png",
            labels=["overall"],
        ),
    )
    _add_plot(
        "Failure Rate: Coop vs Solo",
        _plot_series(
            plots_dir,
            scav,
            [
                _series("team_capture_coop_failure_rate"),
                _series("team_capture_solo_failure_rate"),
            ],
            "Failure Rate: Coop vs Solo",
            "Failure Rate",
            "team_capture_failure_rate_coop_solo.png",
            labels=["coop", "solo"],
        ),
    )
    _add_plot(
        "Failure Rate: Mammoth vs Rabbit",
        _plot_series(
            plots_dir,
            scav,
            [
                _series("team_capture_mammoth_failure_rate"),
                _series("team_capture_rabbit_failure_rate"),
            ],
            "Failure Rate: Mammoth vs Rabbit",
            "Failure Rate",
            "team_capture_failure_rate_prey.png",
            labels=["mammoth", "rabbit"],
        ),
    )
    _add_plot(
        "Free Rider Rates in Coop Captures",
        _plot_series(
            plots_dir,
            scav,
            [
                _series("coop_free_rider_rate"),
                _series("coop_free_rider_presence_rate"),
            ],
            "Free Rider Rates in Coop Captures",
            "Rate",
            "coop_free_rider_rates.png",
            labels=["free_rider_rate", "presence_rate"],
        ),
    )
    _add_plot(
        "Join Cost per Predator",
        _plot_series(
            plots_dir,
            scav,
            [
                _series("join_cost_per_predator"),
                _series("join_cost_per_predator_all"),
            ],
            "Join Cost per Predator",
            "Total Join Cost",
            "join_cost_per_predator.png",
            labels=["predators_with_cost", "all_predators"],
            ylim=None,
        ),
    )
    _add_plot(
        "Join Cost per Successful Capture",
        _plot_series(
            plots_dir,
            scav,
            [
                _series("join_cost_per_successful_capture"),
                _series("join_cost_per_coop_capture"),
            ],
            "Join Cost per Successful Capture",
            "Total Join Cost",
            "join_cost_per_capture.png",
            labels=["per_success", "per_coop_capture"],
            ylim=None,
        ),
    )

    return outputs


def _write_report(report_path: Path, summary_rows, plots, summary_csv_path: Path, detail_csv_path: Path):
    def _fmt(value):
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = []
    lines.append("# Eval Comparison Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Source files:")
    lines.append(f"- `{detail_csv_path}`")
    lines.append(f"- `{summary_csv_path}`")
    lines.append("")
    lines.append("Summary (aggregated across eval dirs)")
    lines.append("")
    header = [
        "scavenger",
        "n_eval_dirs",
        "n_runs_total",
        "join_rate",
        "defect_rate",
        "coop_capture_rate",
        "free_rider_share",
        "failure_rate",
        "coop_fail_rate",
        "solo_fail_rate",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in sorted(summary_rows, key=lambda r: float(r["scavenger_fraction"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("scavenger_fraction")),
                    _fmt(row.get("n_eval_dirs")),
                    _fmt(row.get("n_runs_total")),
                    _fmt(row.get("join_decision_rate")),
                    _fmt(row.get("defect_decision_rate")),
                    _fmt(row.get("coop_capture_rate")),
                    _fmt(row.get("free_rider_share")),
                    _fmt(row.get("team_capture_failure_rate")),
                    _fmt(row.get("team_capture_coop_failure_rate")),
                    _fmt(row.get("team_capture_solo_failure_rate")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("Failure rate by prey type")
    lines.append("")
    header = ["scavenger", "mammoth_fail_rate", "rabbit_fail_rate"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in sorted(summary_rows, key=lambda r: float(r["scavenger_fraction"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("scavenger_fraction")),
                    _fmt(row.get("team_capture_mammoth_failure_rate")),
                    _fmt(row.get("team_capture_rabbit_failure_rate")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("Join cost summary")
    lines.append("")
    header = [
        "scavenger",
        "join_cost_total",
        "join_cost_events",
        "predators_with_cost",
        "predators_total",
        "join_cost_per_event",
        "join_cost_per_predator",
        "join_cost_per_predator_all",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in sorted(summary_rows, key=lambda r: float(r["scavenger_fraction"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("scavenger_fraction")),
                    _fmt(row.get("join_cost_total")),
                    _fmt(row.get("join_cost_events")),
                    _fmt(row.get("predators_with_join_cost")),
                    _fmt(row.get("predators_total")),
                    _fmt(row.get("join_cost_per_event")),
                    _fmt(row.get("join_cost_per_predator")),
                    _fmt(row.get("join_cost_per_predator_all")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("Join cost per successful capture")
    lines.append("")
    header = ["scavenger", "join_cost_per_successful_capture", "join_cost_per_coop_capture"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in sorted(summary_rows, key=lambda r: float(r["scavenger_fraction"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("scavenger_fraction")),
                    _fmt(row.get("join_cost_per_successful_capture")),
                    _fmt(row.get("join_cost_per_coop_capture")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("Plots")
    lines.append("")
    if plots:
        for title, path in plots:
            try:
                rel_path = path.relative_to(report_path.parent)
            except ValueError:
                rel_path = Path(os.path.relpath(path, report_path.parent))
            lines.append(f"### {title}")
            lines.append(f"![{title}]({rel_path})")
            lines.append("")
    else:
        lines.append("Plot generation skipped (matplotlib not available).")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ray_results = Path(RAY_RESULTS_DIR)
    output_path = ray_results / OUTPUT_CSV

    scavenger_filters = SCAVENGER_FILTERS if SCAVENGER_FILTERS else None

    rows = []
    for run_dir in sorted(ray_results.iterdir()):
        if not run_dir.is_dir():
            continue
        run_config = _load_json(run_dir / "run_config.json") or {}
        config_env = run_config.get("config_env", {})
        scavenger_fraction = config_env.get("team_capture_scavenger_fraction")
        if scavenger_filters is not None and not _float_matches(scavenger_fraction, scavenger_filters):
            continue

        eval_paths = _find_eval_aggregate_paths(run_dir)
        if EVAL_NAME_FILTER:
            eval_paths = [item for item in eval_paths if EVAL_NAME_FILTER in item[0].name]
        if not eval_paths:
            continue
        if not INCLUDE_ALL_EVALS:
            eval_paths = [_select_latest_eval(eval_paths)]

        for eval_dir, agg_path in eval_paths:
            aggregate = _load_json(agg_path)
            if not aggregate:
                continue
            metrics = _extract_metrics(aggregate)
            checkpoint_dir = ""
            if eval_dir.parent.name.startswith("checkpoint_"):
                checkpoint_dir = eval_dir.parent.name

            row = {
                "experiment_dir": run_dir.name,
                "eval_dir": eval_dir.name,
                "checkpoint_dir": checkpoint_dir,
                "scavenger_fraction": scavenger_fraction,
                "join_cost": config_env.get("team_capture_join_cost"),
                "max_steps": config_env.get("max_steps"),
                "eval_path": str(eval_dir.relative_to(ray_results)),
            }
            row.update(metrics)
            rows.append(row)

    if not rows:
        print("No evaluation metrics found. Make sure defection_metrics_aggregate.json exists.")
        return

    columns = [
        "experiment_dir",
        "eval_dir",
        "checkpoint_dir",
        "scavenger_fraction",
        "join_cost",
        "max_steps",
        "n_runs",
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
        "join_cost_per_successful_capture",
        "join_cost_per_coop_capture",
        "eval_path",
    ]

    rows.sort(
        key=lambda r: (
            r.get("scavenger_fraction") if r.get("scavenger_fraction") is not None else 999.0,
            r.get("experiment_dir") or "",
            r.get("eval_dir") or "",
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output_path}")

    summary_rows = []
    for row in rows:
        if REQUIRE_N_RUNS is not None and row.get("n_runs") != REQUIRE_N_RUNS:
            continue
        if REQUIRE_FAILURE_METRICS and row.get("team_capture_attempts") is None:
            continue
        summary_rows.append(row)

    if not summary_rows:
        print("No rows eligible for summary CSV (check REQUIRE_N_RUNS/REQUIRE_FAILURE_METRICS).")
        return

    grouped = {}
    for row in summary_rows:
        key = (
            row.get("scavenger_fraction"),
            row.get("join_cost"),
            row.get("max_steps"),
        )
        grouped.setdefault(key, []).append(row)

    summary_output = output_path.parent / OUTPUT_SUMMARY_CSV
    summary_columns = [
        "scavenger_fraction",
        "join_cost",
        "max_steps",
        "n_eval_dirs",
        "n_runs_total",
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
        "join_cost_per_successful_capture",
        "join_cost_per_coop_capture",
    ]

    summary_payload = []
    for key, group_rows in grouped.items():
        scavenger_fraction, join_cost, max_steps = key
        summary = _summarize_group(group_rows)
        summary_row = {
            "scavenger_fraction": scavenger_fraction,
            "join_cost": join_cost,
            "max_steps": max_steps,
        }
        summary_row.update(summary)
        summary_payload.append(summary_row)

    summary_payload.sort(
        key=lambda r: (
            r.get("scavenger_fraction") if r.get("scavenger_fraction") is not None else 999.0,
            r.get("join_cost") if r.get("join_cost") is not None else 999.0,
        )
    )

    with summary_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_columns)
        writer.writeheader()
        for row in summary_payload:
            writer.writerow(row)

    print(f"Wrote {len(summary_payload)} summary rows to {summary_output}")

    plots_dir = _assets_plots_dir()
    plots = _generate_plots(summary_payload, plots_dir)
    report_path = Path(__file__).resolve().parents[1] / OUTPUT_REPORT_MD
    _write_report(report_path, summary_payload, plots, summary_output, output_path)
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
