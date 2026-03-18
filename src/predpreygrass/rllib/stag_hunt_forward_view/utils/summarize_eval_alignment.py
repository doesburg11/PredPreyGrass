#!/usr/bin/env python3
"""Summarize deliberate vs coincidental catch alignment from eval logs.

Uses per-step facing + predator/prey positions to classify each predator event as
deliberate (within distance + angle thresholds) or coincidental.

Outputs a per-scavenger summary CSV and a short Markdown report.
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
    "stag_hunt_forward_view/ray_results/join_cost_0.02"
)

OUTPUT_BY_SCAV_CSV = "eval_alignment_summary.csv"
OUTPUT_REPORT_MD = "eval_alignment_report.md"
INCLUDE_ALL_EVALS = True
SCAVENGER_FILTERS = [0.0, 0.1, 0.2, 0.3, 0.4]
EVAL_NAME_FILTER = None  # e.g. "eval_10_runs" to only include matching eval dirs

# Keep in sync with extract_predator_trajectory.py
DELIBERATE_MAX_DISTANCE = 3.0  # grid cells
DELIBERATE_MAX_ANGLE_DEG = 45.0


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


def _float_matches(value, targets, tol=1e-6):
    if value is None:
        return False
    for target in targets:
        if abs(value - target) <= tol:
            return True
    return False


def _parse_scavenger_from_path(path: Path):
    text = str(path)
    match = re.search(r"SCAVENGER_([0-9_]+)_\d{4}-", text)
    if not match:
        match = re.search(r"SCAVENGER_([0-9_]+)(?:/|_)", text)
    if not match:
        return None
    raw = match.group(1).rstrip("_")
    try:
        return float(raw.replace("_", "."))
    except ValueError:
        return None


def _discover_eval_dirs(ray_results: Path):
    eval_dirs = []
    for summary_dir in ray_results.rglob("summary_data"):
        if not summary_dir.is_dir():
            continue
        if list(summary_dir.glob("per_step_agent_data_*.json")):
            eval_dirs.append(summary_dir.parent)
    return eval_dirs


def _select_latest_eval(evals):
    def _score(eval_dir: Path):
        ts = _parse_eval_timestamp(eval_dir.name)
        if ts:
            return ts.timestamp()
        try:
            return eval_dir.stat().st_mtime
        except OSError:
            return 0.0

    return max(evals, key=_score)


def _iter_run_numbers(summary_dir: Path):
    run_nums = []
    for path in summary_dir.glob("agent_event_log_*.json"):
        match = re.match(r"agent_event_log_(\d+)\.json", path.name)
        if not match:
            continue
        run_nr = int(match.group(1))
        if (summary_dir / f"per_step_agent_data_{run_nr}.json").is_file():
            run_nums.append(run_nr)
    return sorted(run_nums)


def _is_predator(agent_id: str):
    return agent_id.startswith("type_1_predator") or agent_id.startswith("type_2_predator")


def _alignment_for_event(predator_pos, facing, prey_pos):
    if not predator_pos or not facing or not prey_pos:
        return None
    if len(predator_pos) < 2 or len(facing) < 2 or len(prey_pos) < 2:
        return None
    fx, fy = facing[0], facing[1]
    if fx is None or fy is None:
        return None
    facing_mag = math.hypot(fx, fy)
    if facing_mag == 0:
        return None
    dx = prey_pos[0] - predator_pos[0]
    dy = prey_pos[1] - predator_pos[1]
    dist = math.hypot(dx, dy)
    dot = dx * fx + dy * fy
    if dist == 0:
        angle_deg = 0.0
    else:
        cos_val = dot / (dist * facing_mag)
        cos_val = max(-1.0, min(1.0, cos_val))
        angle_deg = math.degrees(math.acos(cos_val))
    deliberate = dist <= DELIBERATE_MAX_DISTANCE and angle_deg <= DELIBERATE_MAX_ANGLE_DEG
    return {
        "distance": dist,
        "angle_deg": angle_deg,
        "deliberate": bool(deliberate),
    }


def _init_counts():
    return {
        "attempts_total": 0,
        "attempts_scored": 0,
        "attempts_deliberate": 0,
        "attempts_coincidental": 0,
        "attempts_unknown": 0,
        "success_total": 0,
        "success_deliberate": 0,
        "success_coincidental": 0,
        "failure_total": 0,
        "failure_deliberate": 0,
        "failure_coincidental": 0,
    }


def _merge_counts(dst, src):
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + value


def _safe_div(num, den):
    if not den:
        return None
    return num / den


def _summarize_eval_dir(eval_dir: Path):
    summary_dir = eval_dir / "summary_data"
    run_nums = _iter_run_numbers(summary_dir)
    counts = _init_counts()
    for run_nr in run_nums:
        per_step_path = summary_dir / f"per_step_agent_data_{run_nr}.json"
        event_log_path = summary_dir / f"agent_event_log_{run_nr}.json"
        per_step = _load_json(per_step_path) or []
        event_log = _load_json(event_log_path) or {}
        if not per_step or not event_log:
            continue
        for agent_id, record in event_log.items():
            if not _is_predator(agent_id):
                continue
            for event_type, key in (("eat", "eating_events"), ("failed", "failed_eating_events")):
                for evt in record.get(key, []):
                    t = evt.get("t")
                    if t is None or t >= len(per_step):
                        continue
                    step_snapshot = per_step[int(t)]
                    predator_state = step_snapshot.get(agent_id)
                    if not predator_state:
                        continue
                    prey_id = evt.get("id_resource") or evt.get("id_eaten")
                    prey_state = step_snapshot.get(prey_id) if prey_id else None
                    predator_pos = evt.get("position_consumer") or predator_state.get("position")
                    prey_pos = evt.get("position_resource") or (
                        prey_state.get("position") if prey_state else None
                    )
                    alignment = _alignment_for_event(
                        predator_pos,
                        predator_state.get("facing"),
                        prey_pos,
                    )
                    counts["attempts_total"] += 1
                    if alignment is None:
                        counts["attempts_unknown"] += 1
                        continue
                    counts["attempts_scored"] += 1
                    if alignment["deliberate"]:
                        counts["attempts_deliberate"] += 1
                    else:
                        counts["attempts_coincidental"] += 1
                    if event_type == "eat":
                        counts["success_total"] += 1
                        if alignment["deliberate"]:
                            counts["success_deliberate"] += 1
                        else:
                            counts["success_coincidental"] += 1
                    else:
                        counts["failure_total"] += 1
                        if alignment["deliberate"]:
                            counts["failure_deliberate"] += 1
                        else:
                            counts["failure_coincidental"] += 1
    return run_nums, counts


def main():
    ray_results = Path(os.environ.get("RAY_RESULTS_DIR", RAY_RESULTS_DIR))
    if not ray_results.is_dir():
        raise SystemExit(f"Missing ray results dir: {ray_results}")

    eval_dirs = _discover_eval_dirs(ray_results)
    if EVAL_NAME_FILTER:
        eval_dirs = [d for d in eval_dirs if EVAL_NAME_FILTER in d.name]
    if not INCLUDE_ALL_EVALS:
        by_scav = {}
        for d in eval_dirs:
            scav = _parse_scavenger_from_path(d)
            if scav is None:
                continue
            by_scav.setdefault(scav, []).append(d)
        eval_dirs = [(_select_latest_eval(v)) for v in by_scav.values() if v]

    rows = []
    by_scav = {}
    for eval_dir in sorted(eval_dirs):
        scav = _parse_scavenger_from_path(eval_dir)
        if scav is None:
            continue
        if SCAVENGER_FILTERS and not _float_matches(scav, SCAVENGER_FILTERS):
            continue
        run_nums, counts = _summarize_eval_dir(eval_dir)
        entry = {
            "scavenger": scav,
            "eval_dir": str(eval_dir),
            "n_runs": len(run_nums),
            **counts,
        }
        rows.append(entry)
        by_scav.setdefault(scav, {"n_eval_dirs": 0, "n_runs_total": 0, "counts": _init_counts()})
        by_scav[scav]["n_eval_dirs"] += 1
        by_scav[scav]["n_runs_total"] += len(run_nums)
        _merge_counts(by_scav[scav]["counts"], counts)

    summary_rows = []
    for scav, payload in sorted(by_scav.items()):
        counts = payload["counts"]
        attempts_scored = counts["attempts_scored"]
        attempts_deliberate = counts["attempts_deliberate"]
        attempts_coincidental = counts["attempts_coincidental"]
        summary_rows.append(
            {
                "scavenger": scav,
                "n_eval_dirs": payload["n_eval_dirs"],
                "n_runs_total": payload["n_runs_total"],
                **counts,
                "deliberate_share": _safe_div(attempts_deliberate, attempts_scored),
                "unknown_share": _safe_div(counts["attempts_unknown"], counts["attempts_total"]),
                "deliberate_success_rate": _safe_div(
                    counts["success_deliberate"], attempts_deliberate
                ),
                "coincidental_success_rate": _safe_div(
                    counts["success_coincidental"], attempts_coincidental
                ),
            }
        )

    output_dir = ray_results
    summary_path = output_dir / OUTPUT_BY_SCAV_CSV
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    report_path = output_dir / OUTPUT_REPORT_MD
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Alignment Summary (Deliberate vs Coincidental)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Deliberate = distance <= 3 and angle <= 45° relative to facing.\n\n")
        f.write("| scavenger | n_eval_dirs | n_runs_total | deliberate_share | deliberate_success_rate | coincidental_success_rate | unknown_share |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for row in summary_rows:
            f.write(
                "| {scavenger:.4f} | {n_eval_dirs} | {n_runs_total} | {deliberate_share:.4f} | "
                "{deliberate_success_rate:.4f} | {coincidental_success_rate:.4f} | {unknown_share:.4f} |\n".format(
                    **row
                )
            )

    print(f"Wrote {len(summary_rows)} rows to {summary_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
