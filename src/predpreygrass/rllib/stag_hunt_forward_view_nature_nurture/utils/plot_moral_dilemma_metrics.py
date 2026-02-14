#!/usr/bin/env python3
"""Plot moral-dilemma metrics from a single evaluation directory.

Edit EVAL_DIR below. This script takes no arguments.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RAY_RESULTS_DIR = (
    "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/"
    "stag_hunt_forward_view_nature_nurture/ray_results"
)
CHECKPOINT_ROOT = (
    "STAG_HUNT_FORWARD_VIEW_JOIN_COST_0.02_SCAVENGER_0.1_2026-01-25_14-20-20/"
    "PPO_PredPreyGrass_99161_00000_0_2026-01-25_14-20-20"
)
CHECKPOINT_NR = "checkpoint_000099"
EVAL_DIR_NAME = "eval_multiple_runs_STAG_HUNT_FORWARD_VIEW_2026-01-28_00-06-36"
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


def _collect_event_logs(eval_dir: Path):
    logs = []
    for fpath in sorted(eval_dir.glob("agent_event_log_*.json")):
        data = _load_json(fpath)
        if isinstance(data, dict):
            logs.append(data)
    return logs


def _ensure_visuals_dir(eval_dir: Path) -> Path:
    visuals = eval_dir / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)
    return visuals


def _save_fig(fig, visuals_dir: Path, filename: str):
    out_path = visuals_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_bar(values, labels, title, ylabel, visuals_dir: Path, filename: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#2f6f9f", "#9f2f2f", "#6f9f2f", "#9f6f2f"][: len(values)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values + [0.05]) * 1.2)
    for idx, val in enumerate(values):
        ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    return _save_fig(fig, visuals_dir, filename)


def _plot_box(data, labels, title, ylabel, visuals_dir: Path, filename: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return _save_fig(fig, visuals_dir, filename)


def _plot_scatter(xs, ys, xlabel, ylabel, title, visuals_dir: Path, filename: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return _save_fig(fig, visuals_dir, filename)


def _extract_rate(per_run, key_path, default=None):
    node = per_run
    for key in key_path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _first_key(node, keys):
    for key in keys:
        if isinstance(node, dict) and key in node:
            return node[key]
    return None


def _extract_any(per_run, key_paths, default=None):
    for key_path in key_paths:
        value = _extract_rate(per_run, key_path, default=None)
        if value is not None:
            return value
    return default


def plot_aggregate_metrics(aggregate, visuals_dir: Path, event_log_metrics=None):
    if not aggregate:
        return []

    outputs = []
    join_defect = aggregate.get("join_defect", {})
    capture = aggregate.get("capture_outcomes", {})

    join_rate = _first_key(join_defect, ["join_decision_rate", "join_rate"])
    defect_rate = _first_key(join_defect, ["defect_decision_rate", "defect_rate"])
    if join_rate is not None and defect_rate is not None:
        outputs.append(
            _plot_bar(
                [join_rate, defect_rate],
                ["join_decision_rate", "defect_decision_rate"],
                "Join vs Defect Decision Rate (aggregate)",
                "rate",
                visuals_dir,
                "moral_join_defect_rate.png",
            )
        )

    coop_rate = _first_key(capture, ["coop_capture_rate", "coop_rate"])
    solo_rate = _first_key(capture, ["solo_capture_rate", "solo_rate"])
    if coop_rate is not None and solo_rate is not None:
        outputs.append(
            _plot_bar(
                [coop_rate, solo_rate],
                ["coop_capture_rate", "solo_capture_rate"],
                "Coop vs Solo Capture Rate (aggregate)",
                "rate",
                visuals_dir,
                "moral_coop_solo_rate.png",
            )
        )

    coop_defection_rate = _first_key(capture, ["coop_free_rider_rate", "coop_defection_rate"])
    coop_free_rider_presence_rate = capture.get("coop_free_rider_presence_rate")
    if event_log_metrics:
        coop_defection_rate = event_log_metrics.get("coop_free_rider_rate", coop_defection_rate)
        coop_free_rider_presence_rate = event_log_metrics.get(
            "coop_free_rider_presence_rate", coop_free_rider_presence_rate
        )
    if coop_defection_rate is not None or coop_free_rider_presence_rate is not None:
        values = []
        labels = []
        if coop_defection_rate is not None:
            values.append(coop_defection_rate)
            labels.append("coop_free_rider_rate")
        if coop_free_rider_presence_rate is not None:
            values.append(coop_free_rider_presence_rate)
            labels.append("coop_free_rider_presence")
        outputs.append(
            _plot_bar(
                values,
                labels,
                "Defection in Coop Captures (aggregate)",
                "rate",
                visuals_dir,
                "moral_coop_defection_rate.png",
            )
        )

    joiners_total = capture.get("joiners_total")
    free_riders_total = capture.get("free_riders_total")
    if joiners_total is not None and free_riders_total is not None:
        outputs.append(
            _plot_bar(
                [joiners_total, free_riders_total],
                ["joiners_total", "free_riders_total"],
                "Joiners vs Free Riders (aggregate)",
                "count",
                visuals_dir,
                "moral_joiners_vs_free_riders.png",
            )
        )

    return outputs


def plot_per_run_metrics(per_runs, visuals_dir: Path):
    if not per_runs:
        return []
    outputs = []

    join_rates = []
    defect_rates = []
    coop_defection_rates = []
    coop_free_rider_presence_rates = []
    coop_rates = []

    for run in per_runs:
        jr = _extract_any(run, [["join_defect", "join_decision_rate"], ["join_defect", "join_rate"]])
        dr = _extract_any(run, [["join_defect", "defect_decision_rate"], ["join_defect", "defect_rate"]])
        fr = _extract_any(run, [["capture_outcomes", "coop_free_rider_rate"], ["capture_outcomes", "coop_defection_rate"]])
        fpr = _extract_rate(run, ["capture_outcomes", "coop_free_rider_presence_rate"])
        cr = _extract_any(run, [["capture_outcomes", "coop_capture_rate"], ["capture_outcomes", "coop_rate"]])
        if jr is not None:
            join_rates.append(jr)
        if dr is not None:
            defect_rates.append(dr)
        if fr is not None:
            coop_defection_rates.append(fr)
        if fpr is not None:
            coop_free_rider_presence_rates.append(fpr)
        if cr is not None:
            coop_rates.append(cr)

    if join_rates and defect_rates:
        outputs.append(
            _plot_box(
                [join_rates, defect_rates],
                ["join_decision_rate", "defect_decision_rate"],
                "Join/Defect Decision Rate per Run",
                "rate",
                visuals_dir,
                "moral_join_defect_per_run.png",
            )
        )

    if coop_defection_rates or coop_free_rider_presence_rates:
        data = []
        labels = []
        if coop_defection_rates:
            data.append(coop_defection_rates)
            labels.append("coop_free_rider_rate")
        if coop_free_rider_presence_rates:
            data.append(coop_free_rider_presence_rates)
            labels.append("coop_free_rider_presence")
        outputs.append(
            _plot_box(
                data,
                labels,
                "Defection in Coop Captures per Run",
                "rate",
                visuals_dir,
                "moral_coop_defection_per_run.png",
            )
        )

    if join_rates and coop_defection_rates:
        outputs.append(
            _plot_scatter(
                join_rates,
                coop_defection_rates,
                "join_decision_rate",
                "coop_free_rider_rate",
                "Join Decision Rate vs Coop Free-Rider Rate",
                visuals_dir,
                "moral_join_vs_coop_defection_scatter.png",
            )
        )

    return outputs


def plot_event_based_metrics(event_logs, visuals_dir: Path):
    if not event_logs:
        return [], None

    join_net = []
    free_net = []
    join_fail = []

    for log in event_logs:
        for _, record in log.items():
            if not isinstance(record, dict):
                continue
            for evt in record.get("eating_events", []) or []:
                if not isinstance(evt, dict):
                    continue
                bite = float(evt.get("bite_size", 0.0))
                join_cost = float(evt.get("join_cost", 0.0))
                if evt.get("join_hunt") is True:
                    join_net.append(bite - join_cost)
                elif evt.get("join_hunt") is False:
                    free_net.append(bite)
            for evt in record.get("failed_eating_events", []) or []:
                if not isinstance(evt, dict):
                    continue
                before = evt.get("energy_before")
                after = evt.get("energy_after")
                if before is not None and after is not None and evt.get("join_hunt") is True:
                    join_fail.append(float(after) - float(before))

    outputs = []
    if join_net or free_net:
        data = []
        labels = []
        if join_net:
            data.append(join_net)
            labels.append("join_net_gain")
        if free_net:
            data.append(free_net)
            labels.append("free_rider_gain")
        outputs.append(
            _plot_box(
                data,
                labels,
                "Net Energy Gain (Event Logs)",
                "energy",
                visuals_dir,
                "moral_energy_gain_event_log.png",
            )
        )

    if join_fail:
        outputs.append(
            _plot_box(
                [join_fail],
                ["join_fail_delta"],
                "Failed Attempt Energy Delta (Joiners)",
                "energy",
                visuals_dir,
                "moral_join_fail_delta.png",
            )
        )

    coop_captures = 0
    coop_participants_total = 0
    coop_free_riders_total = 0
    coop_captures_with_free_riders = 0

    for log in event_logs:
        captures = {}
        for agent_id, record in log.items():
            if not isinstance(record, dict) or "predator" not in agent_id:
                continue
            for evt in record.get("eating_events", []) or []:
                resource_id = evt.get("id_resource") or evt.get("id_eaten")
                if not resource_id or "prey" not in str(resource_id):
                    continue
                key = (evt.get("t"), resource_id)
                entry = captures.setdefault(key, {"joiners": set(), "free_riders": set()})
                if evt.get("join_hunt", True):
                    entry["joiners"].add(agent_id)
                else:
                    entry["free_riders"].add(agent_id)

        for entry in captures.values():
            joiners = len(entry["joiners"])
            if joiners <= 1:
                continue
            free_riders = len(entry["free_riders"])
            coop_captures += 1
            coop_participants_total += joiners + free_riders
            coop_free_riders_total += free_riders
            if free_riders > 0:
                coop_captures_with_free_riders += 1

    coop_free_rider_rate = (
        coop_free_riders_total / coop_participants_total if coop_participants_total else None
    )
    coop_free_rider_presence_rate = (
        coop_captures_with_free_riders / coop_captures if coop_captures else None
    )
    event_log_metrics = {
        "coop_captures": coop_captures,
        "coop_participants_total": coop_participants_total,
        "coop_free_riders_total": coop_free_riders_total,
        "coop_free_rider_rate": coop_free_rider_rate,
        "coop_free_rider_presence_rate": coop_free_rider_presence_rate,
    }

    return outputs, event_log_metrics


def main():
    eval_dir = Path(EVAL_DIR)
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

    visuals_dir = _ensure_visuals_dir(eval_dir)

    aggregate, per_runs = _collect_defection_metrics(eval_dir)
    outputs = []
    event_logs = _collect_event_logs(eval_dir)
    event_outputs, event_metrics = plot_event_based_metrics(event_logs, visuals_dir)

    outputs += plot_aggregate_metrics(aggregate, visuals_dir, event_metrics)
    outputs += plot_per_run_metrics(per_runs, visuals_dir)
    outputs += event_outputs

    if outputs:
        print("Saved plots:")
        for path in outputs:
            print(f"- {path}")
    else:
        print("No plots generated (missing metrics or logs).")


if __name__ == "__main__":
    main()
