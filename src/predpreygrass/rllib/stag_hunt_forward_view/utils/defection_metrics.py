"""
Defection metrics helper for stag_hunt_defection.

Runs a short rollout (random policy) and summarizes:
- Join vs defect choices per predator-step.
- Solo vs cooperative captures (successful only).
- Free-rider exposure on successful captures.
"""

from __future__ import annotations

from predpreygrass.rllib.stag_hunt_forward_view.config.config_env_stag_hunt_forward_view import config_env
from predpreygrass.rllib.stag_hunt_forward_view.predpreygrass_rllib_env import PredPreyGrass


DEFAULT_STEPS = int(config_env.get("max_steps", 200))
DEFAULT_SEED = config_env.get("seed", 0)


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def aggregate_join_choices(per_step_agent_data: list[dict]) -> dict:
    join = 0
    defect = 0
    for step in per_step_agent_data:
        for agent_id, data in step.items():
            if "predator" not in agent_id:
                continue
            if data.get("join_hunt", True):
                join += 1
            else:
                defect += 1
    total = join + defect
    return {
        "join_steps": join,
        "defect_steps": defect,
        "total_predator_steps": total,
        "join_decision_rate": _safe_div(join, total),
        "defect_decision_rate": _safe_div(defect, total),
    }


def aggregate_capture_outcomes(info_all_list: list[dict]) -> dict:
    solo = 0
    coop = 0
    joiners_total = 0
    free_riders_total = 0
    coop_participants_total = 0
    coop_free_riders_total = 0
    coop_captures_with_free_riders = 0
    for info in info_all_list:
        if "team_capture_last_helpers" not in info:
            continue
        helpers = int(info.get("team_capture_last_helpers", 0))
        if helpers <= 0:
            continue
        free_riders = int(info.get("team_capture_last_free_riders", 0))
        if helpers == 1:
            solo += 1
        else:
            coop += 1
            coop_participants_total += helpers + free_riders
            coop_free_riders_total += free_riders
            if free_riders > 0:
                coop_captures_with_free_riders += 1
        joiners_total += helpers
        free_riders_total += free_riders
    captures = solo + coop
    coop_free_rider_rate = _safe_div(coop_free_riders_total, coop_participants_total)
    coop_free_rider_presence_rate = _safe_div(coop_captures_with_free_riders, coop)
    return {
        "captures_successful": captures,
        "solo_captures": solo,
        "coop_captures": coop,
        "solo_capture_rate": _safe_div(solo, captures),
        "coop_capture_rate": _safe_div(coop, captures),
        "joiners_total": joiners_total,
        "free_riders_total": free_riders_total,
        "free_rider_share": _safe_div(free_riders_total, joiners_total + free_riders_total),
        "coop_participants_total": coop_participants_total,
        "coop_free_riders_total": coop_free_riders_total,
        "coop_free_rider_rate": coop_free_rider_rate,
        "coop_captures_with_free_riders": coop_captures_with_free_riders,
        "coop_free_rider_presence_rate": coop_free_rider_presence_rate,
    }


def aggregate_capture_outcomes_from_event_log(event_log: dict) -> dict:
    captures = {}
    for agent_id, record in event_log.items():
        if "predator" not in agent_id:
            continue
        for evt in record.get("eating_events", []):
            resource_id = evt.get("id_resource") or evt.get("id_eaten")
            if not resource_id or "prey" not in str(resource_id):
                continue
            key = (evt.get("t"), resource_id)
            entry = captures.setdefault(key, {"joiners": set(), "free_riders": set()})
            if evt.get("join_hunt", True):
                entry["joiners"].add(agent_id)
            else:
                entry["free_riders"].add(agent_id)

    solo = 0
    coop = 0
    joiners_total = 0
    free_riders_total = 0
    coop_participants_total = 0
    coop_free_riders_total = 0
    coop_captures_with_free_riders = 0
    for entry in captures.values():
        joiners = len(entry["joiners"])
        if joiners <= 0:
            continue
        if joiners == 1:
            solo += 1
        else:
            coop += 1
            coop_participants_total += joiners + len(entry["free_riders"])
            coop_free_riders_total += len(entry["free_riders"])
            if entry["free_riders"]:
                coop_captures_with_free_riders += 1
        joiners_total += joiners
        free_riders_total += len(entry["free_riders"])
    captures_successful = solo + coop
    coop_free_rider_rate = _safe_div(coop_free_riders_total, coop_participants_total)
    coop_free_rider_presence_rate = _safe_div(coop_captures_with_free_riders, coop)
    return {
        "captures_successful": captures_successful,
        "solo_captures": solo,
        "coop_captures": coop,
        "solo_capture_rate": _safe_div(solo, captures_successful),
        "coop_capture_rate": _safe_div(coop, captures_successful),
        "joiners_total": joiners_total,
        "free_riders_total": free_riders_total,
        "free_rider_share": _safe_div(free_riders_total, joiners_total + free_riders_total),
        "coop_participants_total": coop_participants_total,
        "coop_free_riders_total": coop_free_riders_total,
        "coop_free_rider_rate": coop_free_rider_rate,
        "coop_captures_with_free_riders": coop_captures_with_free_riders,
        "coop_free_rider_presence_rate": coop_free_rider_presence_rate,
    }


def compute_opportunity_preference_metrics(per_step_agent_data: list[dict]) -> dict:
    def _pos_tuple(pos):
        if hasattr(pos, "tolist"):
            pos = pos.tolist()
        return tuple(int(x) for x in pos)

    def _has_neighbor(center, positions):
        cx, cy = center
        for px, py in positions:
            if max(abs(px - cx), abs(py - cy)) <= 1:
                return True
        return False

    buckets = {
        "any_prey": {"predator_steps": 0, "join_steps": 0},
        "mammoth_available": {"predator_steps": 0, "join_steps": 0},
        "rabbit_available": {"predator_steps": 0, "join_steps": 0},
        "mammoth_only": {"predator_steps": 0, "join_steps": 0},
        "rabbit_only": {"predator_steps": 0, "join_steps": 0},
        "both_available": {"predator_steps": 0, "join_steps": 0},
    }

    for step in per_step_agent_data:
        mammoths = []
        rabbits = []
        for agent_id, data in step.items():
            if "prey" not in agent_id:
                continue
            pos = data.get("position")
            if pos is None:
                continue
            if "type_1_prey" in agent_id:
                mammoths.append(_pos_tuple(pos))
            elif "type_2_prey" in agent_id:
                rabbits.append(_pos_tuple(pos))

        if not mammoths and not rabbits:
            continue

        for agent_id, data in step.items():
            if "predator" not in agent_id:
                continue
            pos = data.get("position")
            if pos is None:
                continue
            center = _pos_tuple(pos)
            has_mammoth = _has_neighbor(center, mammoths) if mammoths else False
            has_rabbit = _has_neighbor(center, rabbits) if rabbits else False
            if not (has_mammoth or has_rabbit):
                continue
            joined = bool(data.get("join_hunt", True))

            buckets["any_prey"]["predator_steps"] += 1
            if joined:
                buckets["any_prey"]["join_steps"] += 1

            if has_mammoth:
                buckets["mammoth_available"]["predator_steps"] += 1
                if joined:
                    buckets["mammoth_available"]["join_steps"] += 1
            if has_rabbit:
                buckets["rabbit_available"]["predator_steps"] += 1
                if joined:
                    buckets["rabbit_available"]["join_steps"] += 1
            if has_mammoth and has_rabbit:
                buckets["both_available"]["predator_steps"] += 1
                if joined:
                    buckets["both_available"]["join_steps"] += 1
            elif has_mammoth:
                buckets["mammoth_only"]["predator_steps"] += 1
                if joined:
                    buckets["mammoth_only"]["join_steps"] += 1
            elif has_rabbit:
                buckets["rabbit_only"]["predator_steps"] += 1
                if joined:
                    buckets["rabbit_only"]["join_steps"] += 1

    for stats in buckets.values():
        stats["join_decision_rate"] = _safe_div(stats["join_steps"], stats["predator_steps"])

    return buckets


def run_rollout(steps: int, seed: int | None) -> tuple[PredPreyGrass, list[dict]]:
    cfg = dict(config_env)
    cfg["max_steps"] = max(int(cfg.get("max_steps", steps)), steps)
    env = PredPreyGrass(cfg)
    env.reset(seed=seed)
    info_all_list: list[dict] = []

    for _ in range(steps):
        actions = {aid: env.action_spaces[aid].sample() for aid in env.agents}
        _, _, terms, truncs, infos = env.step(actions)
        info_all_list.append(infos.get("__all__", {}))
        if terms.get("__all__") or truncs.get("__all__"):
            break

    return env, info_all_list


def main() -> None:
    env, info_all_list = run_rollout(DEFAULT_STEPS, DEFAULT_SEED)
    join_stats = aggregate_join_choices(env.per_step_agent_data)
    capture_stats = aggregate_capture_outcomes_from_event_log(env.agent_event_log)

    print("Join/Defect (per predator-step)")
    print(f"  join_steps={join_stats['join_steps']}")
    print(f"  defect_steps={join_stats['defect_steps']}")
    print(f"  join_decision_rate={join_stats['join_decision_rate']:.3f}")
    print(f"  defect_decision_rate={join_stats['defect_decision_rate']:.3f}")
    print("")
    print("Capture Outcomes (successful only)")
    print(f"  captures={capture_stats['captures_successful']}")
    print(f"  solo_captures={capture_stats['solo_captures']}")
    print(f"  coop_captures={capture_stats['coop_captures']}")
    print(f"  solo_capture_rate={capture_stats['solo_capture_rate']:.3f}")
    print(f"  coop_capture_rate={capture_stats['coop_capture_rate']:.3f}")
    print("")
    print("Free-Rider Exposure (successful only)")
    print(f"  joiners_total={capture_stats['joiners_total']}")
    print(f"  free_riders_total={capture_stats['free_riders_total']}")
    print(f"  free_rider_share={capture_stats['free_rider_share']:.3f}")


if __name__ == "__main__":
    main()
