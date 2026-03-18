#!/usr/bin/env python3
"""Extract a single predator's trajectory into CSV/JSON from eval logs.

Defaults can be edited below, or you can pass args:
  python extract_predator_trajectory.py <eval_dir> <run_nr> <predator_id> [output_dir]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


# Defaults (edit if you want)
RAY_RESULTS_DIR = (
    "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/"
    "sexual_reproduction/ray_results"
)
EVAL_DIR = (
    "/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/"
    "sexual_reproduction/ray_results/..."
)
RUN_NR = 1
PREDATOR_ID = "type_1_predator_0"
OUTPUT_DIR = None  # default: eval_dir/summary_data
WRITE_JSON = True
WRITE_CSV = True

# "Deliberate hunt" alignment thresholds
DELIBERATE_MAX_DISTANCE = 3.0  # grid cells
DELIBERATE_MAX_ANGLE_DEG = 45.0


def _load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_eval_dir(run_nr: int) -> Path | None:
    ray_results = Path(RAY_RESULTS_DIR)
    if not ray_results.is_dir():
        return None
    candidates = []
    pattern = f"per_step_agent_data_{run_nr}.json"
    for per_step_path in ray_results.rglob(pattern):
        if per_step_path.parent.name != "summary_data":
            continue
        eval_dir = per_step_path.parent.parent
        try:
            mtime = per_step_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, eval_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _build_event_index(agent_event_log: dict, predator_id: str):
    record = agent_event_log.get(predator_id, {})
    events_by_t: dict[int, list[dict]] = {}

    def _add_event(evt, etype):
        t = evt.get("t")
        if t is None:
            return
        payload = {
            "type": etype,
            "t": t,
            "prey_id": evt.get("id_resource") or evt.get("id_eaten"),
            "team_capture": bool(evt.get("team_capture")),
            "join_cost": evt.get("join_cost", 0.0) or 0.0,
            "predator_list": evt.get("predator_list", []) or [],
            "free_riders": evt.get("free_riders", []) or [],
        }
        events_by_t.setdefault(int(t), []).append(payload)

    for evt in record.get("eating_events", []):
        _add_event(evt, "eat")
    for evt in record.get("failed_eating_events", []):
        _add_event(evt, "failed")

    meta = {
        "predator_id": predator_id,
        "birth_step": record.get("birth_step"),
        "parent_id": record.get("parent_id"),
        "death_step": record.get("death_step"),
        "death_cause": record.get("death_cause"),
    }
    return meta, events_by_t


def _summarize_events(events: list[dict]):
    if not events:
        return {
            "eat_events": 0,
            "failed_events": 0,
            "prey_ids": "",
            "join_cost_total": 0.0,
            "team_capture_any": False,
            "max_helpers": 0,
            "max_free_riders": 0,
        }

    prey_ids = []
    join_cost_total = 0.0
    team_capture_any = False
    max_helpers = 0
    max_free_riders = 0
    eat_events = 0
    failed_events = 0
    for evt in events:
        if evt.get("type") == "eat":
            eat_events += 1
        elif evt.get("type") == "failed":
            failed_events += 1
        pid = evt.get("prey_id")
        if pid:
            prey_ids.append(str(pid))
        join_cost_total += float(evt.get("join_cost", 0.0) or 0.0)
        team_capture_any = team_capture_any or bool(evt.get("team_capture"))
        helpers = len(evt.get("predator_list") or [])
        free_riders = len(evt.get("free_riders") or [])
        max_helpers = max(max_helpers, helpers)
        max_free_riders = max(max_free_riders, free_riders)

    return {
        "eat_events": eat_events,
        "failed_events": failed_events,
        "prey_ids": ";".join(sorted(set(prey_ids))),
        "join_cost_total": join_cost_total,
        "team_capture_any": team_capture_any,
        "max_helpers": max_helpers,
        "max_free_riders": max_free_riders,
    }


def _alignment_from_events(
    events: list[dict],
    predator_pos: tuple | list | None,
    facing: tuple | list | None,
    step_snapshot: dict,
):
    alignment = {
        "alignment_prey_id": None,
        "alignment_event_type": None,
        "alignment_prey_dx": None,
        "alignment_prey_dy": None,
        "alignment_prey_distance": None,
        "alignment_facing_dot": None,
        "alignment_facing_mag": None,
        "alignment_cos": None,
        "alignment_angle_deg": None,
        "alignment_prey_in_front": None,
        "alignment_any_in_front": None,
        "alignment_candidates": 0,
        "alignment_distance_ok": None,
        "alignment_angle_ok": None,
        "alignment_deliberate": None,
    }
    if not events or not predator_pos or not facing:
        return alignment
    if len(predator_pos) < 2 or len(facing) < 2:
        return alignment
    fx, fy = facing[0], facing[1]
    if fx is None or fy is None:
        return alignment
    facing_mag = math.hypot(fx, fy)
    if facing_mag == 0:
        return alignment

    candidates = []
    any_in_front = False
    for evt in events:
        prey_id = evt.get("prey_id")
        if not prey_id:
            continue
        prey_state = step_snapshot.get(prey_id)
        if not prey_state:
            continue
        prey_pos = prey_state.get("position")
        if not prey_pos or len(prey_pos) < 2:
            continue
        dx = prey_pos[0] - predator_pos[0]
        dy = prey_pos[1] - predator_pos[1]
        dist = math.hypot(dx, dy)
        dot = dx * fx + dy * fy
        if dist == 0:
            cos_val = 1.0
            angle_deg = 0.0
        else:
            cos_val = dot / (dist * facing_mag)
            cos_val = max(-1.0, min(1.0, cos_val))
            angle_deg = math.degrees(math.acos(cos_val))
        in_front = dot > 0 or dist == 0
        any_in_front = any_in_front or in_front
        candidates.append((angle_deg, dist, prey_id, evt.get("type"), dx, dy, dot, in_front, cos_val))

    if not candidates:
        return alignment

    candidates.sort(key=lambda item: (item[0], item[1]))
    angle_deg, dist, prey_id, event_type, dx, dy, dot, in_front, cos_val = candidates[0]
    distance_ok = dist <= DELIBERATE_MAX_DISTANCE
    angle_ok = angle_deg <= DELIBERATE_MAX_ANGLE_DEG
    alignment.update(
        {
            "alignment_prey_id": prey_id,
            "alignment_event_type": event_type,
            "alignment_prey_dx": dx,
            "alignment_prey_dy": dy,
            "alignment_prey_distance": dist,
            "alignment_facing_dot": dot,
            "alignment_facing_mag": facing_mag,
            "alignment_cos": cos_val,
            "alignment_angle_deg": angle_deg,
            "alignment_prey_in_front": in_front,
            "alignment_any_in_front": any_in_front,
            "alignment_candidates": len(candidates),
            "alignment_distance_ok": distance_ok,
            "alignment_angle_ok": angle_ok,
            "alignment_deliberate": bool(distance_ok and angle_ok),
        }
    )
    return alignment


def _row_from_step(step_idx: int, step_data: dict, events: list[dict], step_snapshot: dict):
    pos = step_data.get("position") or (None, None)
    facing = step_data.get("facing") or (None, None)
    summary = _summarize_events(events)
    alignment = _alignment_from_events(events, pos, facing, step_snapshot)
    row = {
        "step": step_idx,
        "position": [pos[0], pos[1]] if len(pos) > 1 else None,
        "energy": step_data.get("energy"),
        "energy_decay": step_data.get("energy_decay"),
        "energy_eating": step_data.get("energy_eating"),
        "energy_reproduction": step_data.get("energy_reproduction"),
        "age": step_data.get("age"),
        "offspring_count": step_data.get("offspring_count"),
        "join_hunt": step_data.get("join_hunt"),
        "facing_x": facing[0] if len(facing) > 0 else None,
        "facing_y": facing[1] if len(facing) > 1 else None,
        **alignment,
        **summary,
    }
    return row


def extract(eval_dir: Path, run_nr: int, predator_id: str, output_dir: Path):
    summary_dir = eval_dir / "summary_data"
    per_step_path = summary_dir / f"per_step_agent_data_{run_nr}.json"
    event_log_path = summary_dir / f"agent_event_log_{run_nr}.json"

    per_step = _load_json(per_step_path)
    agent_event_log = _load_json(event_log_path)

    meta, events_by_t = _build_event_index(agent_event_log, predator_id)
    rows = []
    for step_idx, step_data in enumerate(per_step):
        if predator_id not in step_data:
            continue
        events = events_by_t.get(step_idx, [])
        rows.append(_row_from_step(step_idx, step_data[predator_id], events, step_data))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trajectory_{predator_id}_run_{run_nr}"

    if WRITE_JSON:
        payload = {
            "meta": meta,
            "rows": rows,
        }
        (output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2))

    if WRITE_CSV:
        import csv

        if rows:
            fields = list(rows[0].keys())
        else:
            fields = []
        with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    if len(sys.argv) >= 4:
        eval_dir = Path(sys.argv[1]).expanduser().resolve()
        run_nr = int(sys.argv[2])
        predator_id = sys.argv[3]
        output_dir = Path(sys.argv[4]).expanduser().resolve() if len(sys.argv) >= 5 else None
    else:
        eval_dir = Path(EVAL_DIR).expanduser().resolve()
        run_nr = RUN_NR
        predator_id = PREDATOR_ID
        output_dir = None

    if (str(eval_dir).endswith("/...") or not eval_dir.exists()):
        fallback = _find_latest_eval_dir(run_nr)
        if fallback is None:
            raise FileNotFoundError(
                f"No eval directories with per_step_agent_data_{run_nr}.json found under {RAY_RESULTS_DIR}."
            )
        eval_dir = fallback

    if output_dir is None:
        output_dir = eval_dir / "summary_data"

    extract(eval_dir, run_nr, predator_id, output_dir)
    print(f"Wrote trajectory files to {output_dir}")


if __name__ == "__main__":
    main()
