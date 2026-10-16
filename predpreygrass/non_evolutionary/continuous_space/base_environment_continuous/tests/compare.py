"""Shared by the equivalence and regression tests: the test cases, the
environment's full state, and an exact, serializable form of results."""

import hashlib
import json
import struct

import numpy as np

# name: (episodes, decisions per episode, first seed, setting overrides,
#        events that must occur). Random actions.
CASES = {
    # Many short episodes: resets, early births, catches and starvation.
    "short": (40, 120, 0, {}, ("births", "caught", "starved")),
    # Long episodes: grass collapse, extinction and the time limit.
    "long": (3, 1100, 1000, {}, ("collapses", "extinct", "time_limit")),
    # One physics step per decision.
    "repeat1": (8, 600, 2000, {"action_repeat": 1}, ("births", "caught")),
    # Small caps and ID pools: births blocked by the cap, pools used up.
    "caps": (
        10,
        400,
        3000,
        {
            "reproduction_max_predators": 8,
            "reproduction_max_prey": 12,
            "reproduction_predator_pool": 9,
            "reproduction_prey_pool": 20,
        },
        ("births", "pool_exhausted", "capped"),
    ),
}


def random_actions(rng, obs, terms=None, truncs=None):
    """The reference's actions: uniform over 48 for every agent in obs that
    is still acting, in sorted order."""
    terms, truncs = terms or {}, truncs or {}
    active = sorted(a for a in obs if not (terms.get(a) or truncs.get(a)))
    return {agent: int(rng.integers(48)) for agent in active}


def snapshot(env):
    """The full simulation state, including what only future steps use."""
    animals = {
        a.id: (a.position.x, a.position.y, a.velocity.x, a.velocity.y, a.acceleration.x, a.acceleration.y, a.orientation, a.energy)
        for a in env.predators + env.prey
    }
    grass = env.grass
    return {
        "animals": animals,
        "patches": [(p.position.x, p.position.y, p.ready_at, p.regrow, p.cluster) for p in grass.patches],
        "centres": [(c.x, c.y) for c in grass.centres],
        "collapsed": list(grass.collapsed),
        "grass_counters": (grass.tick, grass.total_consumed, grass.total_collapses),
        "grass_rng": grass.rng.getstate(),
        "birth_rng": env.birth_rng.getstate(),
        "next_id": dict(env.next_id),
        "time_step": env.time_step,
        "order": [a.id for a in env.entities],
        "active": sorted(env.active),
        "agents": sorted(env.agents),
        "births": dict(env.births),
    }


def exact(value):
    """A JSON-serializable form that tells apart float bits (-0.0 from 0.0,
    NaN payloads) and types (1 from 1.0, Python from NumPy scalars)."""
    if isinstance(value, dict):
        return {str(key): exact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [exact(item) for item in value]]
    if isinstance(value, np.ndarray):
        return ["array", str(value.dtype), list(value.shape), value.tobytes().hex()]
    if isinstance(value, (float, np.floating)):
        return [type(value).__name__, struct.pack("<d", float(value)).hex()]
    if value is None or isinstance(value, (bool, int, str, np.integer, np.bool_)):
        return [type(value).__name__, value if not isinstance(value, np.generic) else value.item()]
    raise TypeError(f"no exact form for {type(value)}")


def count_events(env, population, terms, truncs, infos, events):
    """Add one decision's events to the counts in events (a Counter):
    births, catches, starvation, extinction, the time limit, exhausted ID
    pools and births blocked by a population cap. population: the
    species' sizes before the decision."""
    for agent, info in infos.items():
        events["births"] += info.get("born", False)
        events["starved"] += info.get("starved", False)
        if agent.startswith("prey") and terms.get(agent) and not info.get("starved"):
            events["caught"] += 1
    if truncs["__all__"] and env.prey and env.predators:
        events["time_limit"] += 1
    if not env.prey or not env.predators:
        events["extinct"] += 1
    if any(env.next_id[s] >= env.pool[s] for s in env.pool):
        events["pool_exhausted"] += 1
    for species, animals in (("predator", env.predators), ("prey", env.prey)):
        if population[species] >= env.max_population[species] and any(a.energy >= env.threshold[species] for a in animals):
            events["capped"] += 1


def digest(value):
    """The first 64 bits of the SHA-256 of the exact form, in hex
    (dictionary key order does not matter)."""
    text = json.dumps(exact(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]
