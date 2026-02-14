"""
Sweep nature+nurture coupling knobs with random-policy rollouts.

This script runs short ablations over `team_capture_nature_weight` and writes:
- Per-run metrics CSV.
- Condition-level summary CSV (mean/std).

Default conditions:
- `nurture_only`: `coop_trait_enabled=False`.
- `trait_on_w{w}` for each weight in `--weights` with `coop_trait_enabled=True`.

Optional:
- `trait_only_forced_join`: all predators forced to `join_hunt=1`, so variation is
  driven by trait dynamics and ecology rather than join/defect decisions.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics

import numpy as np

from predpreygrass.rllib.stag_hunt_forward_view_nature_nurture.config.config_env_stag_hunt_forward_view import (
    config_env,
)
from predpreygrass.rllib.stag_hunt_forward_view_nature_nurture.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.stag_hunt_forward_view_nature_nurture.utils.defection_metrics import (
    aggregate_capture_outcomes_from_event_log,
    aggregate_join_choices,
)


def _parse_weights(raw: str) -> list[float]:
    weights: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        w = float(token)
        weights.append(min(max(w, 0.0), 1.0))
    return sorted(set(weights))


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _force_join_action(sampled_action, join_value: int = 1):
    join_value = 1 if int(join_value) != 0 else 0
    if isinstance(sampled_action, dict):
        out = dict(sampled_action)
        out["join_hunt"] = join_value
        if "move" not in out and "action" in out:
            out["move"] = out["action"]
        return out

    if isinstance(sampled_action, np.ndarray):
        if sampled_action.shape == ():
            return np.array([int(sampled_action), join_value], dtype=np.int64)
        arr = sampled_action.astype(np.int64, copy=True)
        if arr.size >= 2:
            arr[1] = join_value
            return arr
        if arr.size == 1:
            return np.array([int(arr[0]), join_value], dtype=np.int64)
        return np.array([0, join_value], dtype=np.int64)

    if isinstance(sampled_action, (tuple, list)):
        if len(sampled_action) >= 2:
            return (int(sampled_action[0]), join_value)
        if len(sampled_action) == 1:
            return (int(sampled_action[0]), join_value)
        return (0, join_value)

    return (int(sampled_action), join_value)


def run_episode(
    cfg: dict,
    *,
    steps: int,
    seed: int,
    force_predator_join: int | None,
    condition: str,
    run_idx: int,
) -> dict:
    env = PredPreyGrass(cfg)
    env.reset(seed=seed)
    info_all_list: list[dict] = []

    for _ in range(steps):
        action_dict = {}
        for agent_id in env.agents:
            sampled = env.action_spaces[agent_id].sample()
            if force_predator_join is not None and "predator" in agent_id:
                sampled = _force_join_action(sampled, force_predator_join)
            action_dict[agent_id] = sampled

        _, _, terms, truncs, infos = env.step(action_dict)
        info_all_list.append(infos.get("__all__", {}))
        if terms.get("__all__") or truncs.get("__all__"):
            break

    join_stats = aggregate_join_choices(env.per_step_agent_data)
    capture_stats = aggregate_capture_outcomes_from_event_log(env.agent_event_log)

    trait_values = list(env.predator_cooperation_trait.values())
    trait_mean = float(np.mean(trait_values)) if trait_values else 0.0
    trait_var = float(np.var(trait_values)) if trait_values else 0.0

    final_global_info = info_all_list[-1] if info_all_list else {}
    predators_final = len(env.predator_positions)
    prey_final = len(env.prey_positions)

    row = {
        "condition": condition,
        "run_idx": int(run_idx),
        "seed": int(seed),
        "steps_completed": int(env.current_step),
        "predators_final": int(predators_final),
        "prey_final": int(prey_final),
        "coexistence_final": int(predators_final > 0 and prey_final > 0),
        "join_decision_rate": float(join_stats.get("join_decision_rate", 0.0)),
        "defect_decision_rate": float(join_stats.get("defect_decision_rate", 0.0)),
        "captures_successful": int(capture_stats.get("captures_successful", 0)),
        "solo_capture_rate": float(capture_stats.get("solo_capture_rate", 0.0)),
        "coop_capture_rate": float(capture_stats.get("coop_capture_rate", 0.0)),
        "free_rider_share": float(capture_stats.get("free_rider_share", 0.0)),
        "team_capture_attempts": int(final_global_info.get("team_capture_attempts", 0)),
        "team_capture_avg_success_prob": float(final_global_info.get("team_capture_avg_success_prob", 0.0)),
        "predator_mean_coop_trait_final": trait_mean,
        "predator_var_coop_trait_final": trait_var,
    }

    env.close()
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    metric_keys = [
        "coexistence_final",
        "join_decision_rate",
        "defect_decision_rate",
        "captures_successful",
        "solo_capture_rate",
        "coop_capture_rate",
        "free_rider_share",
        "team_capture_attempts",
        "team_capture_avg_success_prob",
        "predator_mean_coop_trait_final",
        "predator_var_coop_trait_final",
        "steps_completed",
    ]

    summary_rows: list[dict] = []
    for condition, cond_rows in sorted(grouped.items()):
        out = {
            "condition": condition,
            "runs": len(cond_rows),
        }
        for key in metric_keys:
            vals = [float(r[key]) for r in cond_rows]
            out[f"{key}_mean"] = _mean(vals)
            out[f"{key}_std"] = _std(vals)
        summary_rows.append(out)

    return summary_rows


def build_conditions(weights: list[float], include_forced_join_baseline: bool) -> list[dict]:
    conditions: list[dict] = [
        {
            "name": "nurture_only",
            "coop_trait_enabled": False,
            "nature_weight": 0.0,
            "force_join": None,
        }
    ]

    for weight in weights:
        conditions.append(
            {
                "name": f"trait_on_w{weight:.2f}",
                "coop_trait_enabled": True,
                "nature_weight": float(weight),
                "force_join": None,
            }
        )

    if include_forced_join_baseline:
        conditions.append(
            {
                "name": "trait_only_forced_join",
                "coop_trait_enabled": True,
                "nature_weight": 1.0,
                "force_join": 1,
            }
        )

    return conditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep nature+nurture coupling in stag_hunt_forward_view_nature_nurture")
    parser.add_argument("--weights", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=int(config_env.get("max_steps", 1000)))
    parser.add_argument("--base-seed", type=int, default=int(config_env.get("seed", 41) or 41))
    parser.add_argument("--success-model", type=str, default=str(config_env.get("team_capture_success_model", "hybrid")))
    parser.add_argument("--include-forced-join-baseline", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ablation_results" / "nature_weight_sweep",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be >= 1")
    if args.steps <= 0:
        raise ValueError("--steps must be >= 1")

    weights = _parse_weights(args.weights)
    if not weights:
        raise ValueError("No valid --weights provided")

    conditions = build_conditions(weights, args.include_forced_join_baseline)

    rows: list[dict] = []
    for cidx, condition in enumerate(conditions):
        for ridx in range(args.runs):
            run_seed = int(args.base_seed + cidx * 10_000 + ridx)
            cfg = dict(config_env)
            cfg["max_steps"] = max(int(cfg.get("max_steps", args.steps)), int(args.steps))
            cfg["seed"] = run_seed
            cfg["team_capture_success_model"] = str(args.success_model)
            cfg["coop_trait_enabled"] = bool(condition["coop_trait_enabled"])
            cfg["team_capture_nature_weight"] = float(condition["nature_weight"])

            row = run_episode(
                cfg,
                steps=int(args.steps),
                seed=run_seed,
                force_predator_join=condition["force_join"],
                condition=str(condition["name"]),
                run_idx=ridx,
            )
            rows.append(row)

    summary_rows = summarize(rows)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_csv = out_dir / "nature_weight_runs.csv"
    summary_csv = out_dir / "nature_weight_summary.csv"
    write_csv(run_csv, rows)
    write_csv(summary_csv, summary_rows)

    print(f"Wrote per-run metrics: {run_csv}")
    print(f"Wrote summary metrics: {summary_csv}")
    print("")
    print("Condition summary (means)")
    for row in summary_rows:
        print(
            f"- {row['condition']}: "
            f"coexist={row['coexistence_final_mean']:.2f}, "
            f"coop_capture={row['coop_capture_rate_mean']:.3f}, "
            f"join_rate={row['join_decision_rate_mean']:.3f}, "
            f"trait_mean={row['predator_mean_coop_trait_final_mean']:.3f}, "
            f"success_prob={row['team_capture_avg_success_prob_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
