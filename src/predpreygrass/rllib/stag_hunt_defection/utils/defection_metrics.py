"""
Defection metrics helper for stag_hunt_defection.

Runs a short rollout (random policy) and summarizes:
- Join vs defect choices per predator-step.
- Solo vs cooperative captures (successful only).
- Free-rider exposure on successful captures.
"""

from __future__ import annotations

from predpreygrass.rllib.stag_hunt_defection.config.config_env_stag_hunt_defection import config_env
from predpreygrass.rllib.stag_hunt_defection.predpreygrass_rllib_env import PredPreyGrass


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
        "join_rate": _safe_div(join, total),
        "defect_rate": _safe_div(defect, total),
    }


def aggregate_capture_outcomes(info_all_list: list[dict]) -> dict:
    solo = 0
    coop = 0
    joiners_total = 0
    free_riders_total = 0
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
        joiners_total += helpers
        free_riders_total += free_riders
    captures = solo + coop
    return {
        "captures_successful": captures,
        "solo_captures": solo,
        "coop_captures": coop,
        "solo_rate": _safe_div(solo, captures),
        "coop_rate": _safe_div(coop, captures),
        "joiners_total": joiners_total,
        "free_riders_total": free_riders_total,
        "free_rider_rate": _safe_div(free_riders_total, joiners_total + free_riders_total),
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
    for entry in captures.values():
        joiners = len(entry["joiners"])
        if joiners <= 0:
            continue
        if joiners == 1:
            solo += 1
        else:
            coop += 1
        joiners_total += joiners
        free_riders_total += len(entry["free_riders"])
    captures_successful = solo + coop
    return {
        "captures_successful": captures_successful,
        "solo_captures": solo,
        "coop_captures": coop,
        "solo_rate": _safe_div(solo, captures_successful),
        "coop_rate": _safe_div(coop, captures_successful),
        "joiners_total": joiners_total,
        "free_riders_total": free_riders_total,
        "free_rider_rate": _safe_div(free_riders_total, joiners_total + free_riders_total),
    }


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
    print(f"  join_rate={join_stats['join_rate']:.3f}")
    print(f"  defect_rate={join_stats['defect_rate']:.3f}")
    print("")
    print("Capture Outcomes (successful only)")
    print(f"  captures={capture_stats['captures_successful']}")
    print(f"  solo_captures={capture_stats['solo_captures']}")
    print(f"  coop_captures={capture_stats['coop_captures']}")
    print(f"  solo_rate={capture_stats['solo_rate']:.3f}")
    print(f"  coop_rate={capture_stats['coop_rate']:.3f}")
    print("")
    print("Free-Rider Exposure (successful only)")
    print(f"  joiners_total={capture_stats['joiners_total']}")
    print(f"  free_riders_total={capture_stats['free_riders_total']}")
    print(f"  free_rider_rate={capture_stats['free_rider_rate']:.3f}")


if __name__ == "__main__":
    main()
