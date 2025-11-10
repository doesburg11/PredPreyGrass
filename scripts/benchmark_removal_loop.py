"""Micro-benchmark for agent removal loop efficiency.

We simulate termination patterns on synthetic agent dictionaries to compare:
1. Old pattern: iterate over list copy and list.remove per terminated agent.
2. New pattern: collect terminated agents, bulk-pop from dicts, rebuild list via comprehension.

This does not import the environment; it emulates the data structures at scale to isolate the removal cost.
"""
from __future__ import annotations
import time
import statistics
import random

N_AGENTS = 5000  # scale of active agents
TERMINATION_FRACTION = 0.25  # fraction terminated each cycle
REPS = 50  # repetitions for timing stability
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Synthetic agents
base_agents = [f"type_1_predator_{i}" if i % 3 == 0 else (f"type_2_prey_{i}" if i % 3 == 1 else f"type_1_prey_{i}") for i in range(N_AGENTS)]

# Helper to prepare fresh structures per rep (simulate a step before removal)

def make_state():
    agents = list(base_agents)
    positions = {a: (i % 64, (i * 7) % 64) for i, a in enumerate(agents)}
    energies = {a: float((i * 13) % 100) for i, a in enumerate(agents)}
    predator_pos = {a: positions[a] for a in agents if "predator" in a}
    prey_pos = {a: positions[a] for a in agents if "prey" in a}
    unique_agents = {a: f"{a}_0" for a in agents}
    unique_stats = {f"{a}_0": {"lifetime": 0, "parent": None} for a in agents}
    terminations = {}
    truncations = {}
    # Mark a fraction as terminated
    terminated = set(random.sample(agents, int(len(agents) * TERMINATION_FRACTION)))
    for a in terminated:
        terminations[a] = True
        truncations[a] = False
    return {
        "agents": agents,
        "positions": positions,
        "energies": energies,
        "predator_pos": predator_pos,
        "prey_pos": prey_pos,
        "unique_agents": unique_agents,
        "unique_stats": unique_stats,
        "terminations": terminations,
        "truncations": truncations,
        "terminated": terminated,
    }


def old_pattern(state):
    agents = state["agents"]
    terminations = state["terminations"]
    positions = state["positions"]
    energies = state["energies"]
    predator_pos = state["predator_pos"]
    prey_pos = state["prey_pos"]
    unique_agents = state["unique_agents"]
    unique_stats = state["unique_stats"]
    death_stats = {}
    for agent in agents[:]:  # list copy
        if terminations.get(agent, False):
            uid = unique_agents[agent]
            death_stats[uid] = {
                **unique_stats[uid],
                "lifetime": 0,
                "parent": None,
            }
            agents.remove(agent)  # O(n) each
            del unique_agents[agent]
            del positions[agent]
            del energies[agent]
            if "predator" in agent:
                predator_pos.pop(agent, None)
            elif "prey" in agent:
                prey_pos.pop(agent, None)
    return len(death_stats)


def new_pattern(state):
    agents = state["agents"]
    terminations = state["terminations"]
    positions = state["positions"]
    energies = state["energies"]
    predator_pos = state["predator_pos"]
    prey_pos = state["prey_pos"]
    unique_agents = state["unique_agents"]
    unique_stats = state["unique_stats"]
    death_stats = {}
    to_remove = [a for a, t in terminations.items() if t and a in positions]
    if to_remove:
        to_remove_set = set(to_remove)
        for agent in to_remove:
            uid = unique_agents.get(agent)
            if uid is not None:
                death_stats[uid] = {
                    **unique_stats[uid],
                    "lifetime": 0,
                    "parent": None,
                }
            unique_agents.pop(agent, None)
            positions.pop(agent, None)
            energies.pop(agent, None)
            if agent in predator_pos:
                predator_pos.pop(agent, None)
            elif agent in prey_pos:
                prey_pos.pop(agent, None)
        agents[:] = [a for a in agents if a not in to_remove_set]
    return len(death_stats)


def time_fn(fn, make_state_fn, reps=REPS):
    timings = []
    total_removed = 0
    for _ in range(reps):
        state = make_state_fn()
        t0 = time.perf_counter()
        removed = fn(state)
        t1 = time.perf_counter()
        total_removed += removed
        timings.append((t1 - t0) * 1000.0)  # ms
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "removed_total": total_removed,
        "removed_per_rep": total_removed / reps,
    }

if __name__ == "__main__":
    old_stats = time_fn(old_pattern, make_state)
    new_stats = time_fn(new_pattern, make_state)
    speedup = old_stats["median_ms"] / new_stats["median_ms"] if new_stats["median_ms"] else float('inf')
    print("Removal Loop Benchmark")
    print(f"Agents: {N_AGENTS}  Termination fraction: {TERMINATION_FRACTION*100:.1f}%  Reps: {REPS}")
    print("Old pattern:   median={:.3f}ms mean={:.3f}ms range=({:.3f},{:.3f}) removed/rep={:.1f}".format(old_stats['median_ms'], old_stats['mean_ms'], old_stats['min_ms'], old_stats['max_ms'], old_stats['removed_per_rep']))
    print("New pattern:   median={:.3f}ms mean={:.3f}ms range=({:.3f},{:.3f}) removed/rep={:.1f}".format(new_stats['median_ms'], new_stats['mean_ms'], new_stats['min_ms'], new_stats['max_ms'], new_stats['removed_per_rep']))
    print(f"Speedup (median): {speedup:.2f}x")
