"""
Direct reciprocity metrics helper.

Summarizes:
- Share opportunity / share / refusal rates.
- Whether share decisions increase when a prior helper is available nearby.
- How often share events are reciprocal at the dyad level.
"""

from __future__ import annotations

from predpreygrass.direct_reciprocity.config.config_env_direct_reciprocity import config_env
from predpreygrass.direct_reciprocity.predpreygrass_rllib_env import PredPreyGrass


DEFAULT_STEPS = int(config_env.get("max_steps", 200))
DEFAULT_SEED = config_env.get("seed", 0)


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def _iter_share_opportunity_events(event_log: dict) -> list[tuple[int, str, dict]]:
    events: list[tuple[int, str, dict]] = []
    if not isinstance(event_log, dict):
        return events
    for agent_id, record in event_log.items():
        if "predator" not in agent_id or not isinstance(record, dict):
            continue
        for evt in record.get("eating_events", []) or []:
            if not isinstance(evt, dict):
                continue
            if "share_opportunity" not in evt:
                continue
            events.append((int(evt.get("t", 0)), agent_id, evt))
    events.sort(key=lambda item: (item[0], item[1]))
    return events


def aggregate_share_decisions_from_event_log(event_log: dict) -> dict:
    opportunities = 0
    share_events = 0
    refusals = 0
    shared_energy_total = 0.0

    for _, _, evt in _iter_share_opportunity_events(event_log):
        if not evt.get("share_opportunity", False):
            continue
        opportunities += 1
        shared_energy = float(evt.get("shared_energy", 0.0) or 0.0)
        recipient = evt.get("share_recipient")
        if recipient is not None and shared_energy > 0.0:
            share_events += 1
            shared_energy_total += shared_energy
        else:
            refusals += 1

    return {
        "share_opportunities": opportunities,
        "share_events": share_events,
        "share_refusals": refusals,
        "share_decision_rate": _safe_div(share_events, opportunities),
        "share_refusal_rate": _safe_div(refusals, opportunities),
        "shared_energy_total": shared_energy_total,
        "mean_shared_energy_per_share": _safe_div(shared_energy_total, share_events),
        "mean_shared_energy_per_opportunity": _safe_div(shared_energy_total, opportunities),
    }


def aggregate_direct_reciprocity_metrics(event_log: dict) -> dict:
    prior_shares: dict[tuple[str, str], int] = {}
    opportunities_with_prior_helper_available = 0
    opportunities_without_prior_helper_available = 0
    shares_when_prior_helper_available = 0
    shares_when_no_prior_helper_available = 0
    share_to_prior_helper_events = 0
    share_to_non_helper_events = 0
    reciprocal_share_events = 0
    reciprocal_dyads: set[frozenset[str]] = set()

    for _, donor, evt in _iter_share_opportunity_events(event_log):
        if not evt.get("share_opportunity", False):
            continue
        candidates = [str(cid) for cid in evt.get("share_candidates", []) or [] if cid is not None]
        recipient = evt.get("share_recipient")
        if recipient is not None:
            recipient = str(recipient)
        prior_helpers = [candidate for candidate in candidates if prior_shares.get((candidate, donor), 0) > 0]
        prior_helper_available = bool(prior_helpers)

        if prior_helper_available:
            opportunities_with_prior_helper_available += 1
        else:
            opportunities_without_prior_helper_available += 1

        if recipient is not None and float(evt.get("shared_energy", 0.0) or 0.0) > 0.0:
            if prior_helper_available:
                shares_when_prior_helper_available += 1
            else:
                shares_when_no_prior_helper_available += 1

            if prior_shares.get((recipient, donor), 0) > 0:
                share_to_prior_helper_events += 1
                reciprocal_share_events += 1
                reciprocal_dyads.add(frozenset((donor, recipient)))
            else:
                share_to_non_helper_events += 1

            prior_shares[(donor, recipient)] = prior_shares.get((donor, recipient), 0) + 1

    total_shares = share_to_prior_helper_events + share_to_non_helper_events
    total_opportunities = opportunities_with_prior_helper_available + opportunities_without_prior_helper_available
    return {
        "opportunities_total": total_opportunities,
        "opportunities_with_prior_helper_available": opportunities_with_prior_helper_available,
        "opportunities_without_prior_helper_available": opportunities_without_prior_helper_available,
        "shares_when_prior_helper_available": shares_when_prior_helper_available,
        "shares_when_no_prior_helper_available": shares_when_no_prior_helper_available,
        "share_rate_when_prior_helper_available": _safe_div(
            shares_when_prior_helper_available, opportunities_with_prior_helper_available
        ),
        "share_rate_when_no_prior_helper_available": _safe_div(
            shares_when_no_prior_helper_available, opportunities_without_prior_helper_available
        ),
        "share_to_prior_helper_events": share_to_prior_helper_events,
        "share_to_non_helper_events": share_to_non_helper_events,
        "share_to_prior_helper_rate": _safe_div(share_to_prior_helper_events, total_shares),
        "reciprocal_share_events": reciprocal_share_events,
        "reciprocal_share_rate": _safe_div(reciprocal_share_events, total_shares),
        "reciprocal_dyads": len(reciprocal_dyads),
    }


def compute_reciprocity_metrics(env: PredPreyGrass) -> dict:
    share_decisions = aggregate_share_decisions_from_event_log(getattr(env, "agent_event_log", {}))
    direct_reciprocity = aggregate_direct_reciprocity_metrics(getattr(env, "agent_event_log", {}))
    return {
        "steps": int(getattr(env, "current_step", 0)),
        "share_decisions": share_decisions,
        "direct_reciprocity": direct_reciprocity,
    }


def run_rollout(steps: int, seed: int | None) -> PredPreyGrass:
    cfg = dict(config_env)
    cfg["max_steps"] = max(int(cfg.get("max_steps", steps)), steps)
    env = PredPreyGrass(cfg)
    env.reset(seed=seed)

    for _ in range(steps):
        actions = {aid: env.action_spaces[aid].sample() for aid in env.agents}
        _, _, terms, truncs, _ = env.step(actions)
        if terms.get("__all__") or truncs.get("__all__"):
            break

    return env


def main() -> None:
    env = run_rollout(DEFAULT_STEPS, DEFAULT_SEED)
    metrics = compute_reciprocity_metrics(env)
    share_decisions = metrics["share_decisions"]
    direct_reciprocity = metrics["direct_reciprocity"]

    print("Share Decisions")
    print(f"  share_opportunities={share_decisions['share_opportunities']}")
    print(f"  share_events={share_decisions['share_events']}")
    print(f"  share_refusals={share_decisions['share_refusals']}")
    print(f"  share_decision_rate={share_decisions['share_decision_rate']:.3f}")
    print("")
    print("Direct Reciprocity")
    print(
        "  share_rate_when_prior_helper_available="
        f"{direct_reciprocity['share_rate_when_prior_helper_available']:.3f}"
    )
    print(
        "  share_rate_when_no_prior_helper_available="
        f"{direct_reciprocity['share_rate_when_no_prior_helper_available']:.3f}"
    )
    print(f"  reciprocal_share_rate={direct_reciprocity['reciprocal_share_rate']:.3f}")
    print(f"  reciprocal_dyads={direct_reciprocity['reciprocal_dyads']}")


if __name__ == "__main__":
    main()
