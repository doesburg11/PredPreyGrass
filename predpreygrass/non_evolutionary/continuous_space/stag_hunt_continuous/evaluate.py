"""Evaluation of trained (or untrained) policies: coexistence of the three
species, the hunt counters, and joint arrivals against a density-matched
chance level (DESIGN.md, 4).

Plays seeded episodes 0, 1, ... with actions sampled from the policies'
distributions as in training (not a replay of training's samples: one
torch generator per episode, seeded with the episode's seed, agents in
sorted order; a predator's movement and join heads sampled separately), in
parallel processes.

Chance level for joint arrivals: the positions of every predator and
mammoth at the end of each decision are recorded (as the environment's own
counter sees them, before the hunts). Each predator's whole track is then
shifted by its own random offset on the torus, the mammoths' tracks kept,
and encounter episodes and joint arrivals are counted again on the shifted
tracks with the environment's definitions (NULL_REPEATS shifts per
episode; the chance rate is the ratio of the expected totals). Shifted
tracks are not behaviourally consistent with their new surroundings, and
the shift also removes any grouping of predators, intended or not. This keeps the number of predators and each one's movement, and
breaks their relation to the mammoths and to each other. Predators that
seek mammoths independently of each other also arrive together more often
than shifted tracks do; that is what the solo-control arm and the
untrained policies are for (DESIGN.md, 5).

    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.evaluate \\
        <checkpoint dir | untrained> <episodes> [out.json] [--workers N]

With `untrained`, the policies are freshly initialised (seed 0) in the
world of config/config_env.py.
"""

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import (
    JOINT_WINDOW,
    MOVES,
    SPECIES,
    StagHuntContinuous,
    new_hunt_stats,
    species_of,
)

NULL_REPEATS = 20
_worker = {}


def sample_actions(modules, obs, generator):
    """Every agent's action, one batched forward pass per species. A
    predator's logits are its movement head (MOVES) and its join head (2)."""
    import torch
    from ray.rllib.core.columns import Columns

    actions = {}
    for species in SPECIES:
        agents = [agent for agent in obs if species_of(agent) == species]
        if not agents:
            continue
        batch = torch.from_numpy(np.stack([np.asarray(obs[agent], dtype=np.float32) for agent in agents]))
        with torch.inference_mode():
            logits = modules[species].forward_inference({Columns.OBS: batch})[Columns.ACTION_DIST_INPUTS]
        move = torch.multinomial(torch.softmax(logits[:, :MOVES], dim=-1), 1, generator=generator).squeeze(1)
        if species == "predator":
            join = torch.multinomial(torch.softmax(logits[:, MOVES:], dim=-1), 1, generator=generator).squeeze(1)
            actions.update({agent: np.array([m, j]) for agent, m, j in zip(agents, move.tolist(), join.tolist())})
        else:
            actions.update(zip(agents, move.tolist()))
    return actions


def outcome(env):
    """The species still alive at the end."""
    alive = [s for s in SPECIES if env.animals[s]]
    return "coexisting" if len(alive) == len(SPECIES) else "only_" + "_".join(alive) if alive else "none"


def joint_arrivals(predator_tracks, mammoth_tracks, decisions, width, height, touch, near):
    """(encounter episodes, touched episodes, joint arrivals) from tracks
    {id: {decision: (x, y)}}, with the environment's definitions."""

    def distance(a, b):
        dx = abs(a[0] - b[0]) % width
        dy = abs(a[1] - b[1]) % height
        return float(np.hypot(min(dx, width - dx), min(dy, height - dy)))

    episodes = touched = joint = 0
    for track in mammoth_tracks.values():
        open_episode = None
        for decision in decisions:
            if decision not in track:
                position = None
            else:
                position = track[decision]
            present = [] if position is None else [
                (pid, distance(p[decision], position)) for pid, p in predator_tracks.items() if decision in p
            ]
            close = [(pid, d) for pid, d in present if d <= near]
            if position is None or not close:
                if open_episode is not None:
                    first = sorted(open_episode.values())
                    touched += bool(first)
                    joint += any(b - a <= JOINT_WINDOW for a, b in zip(first, first[1:]))
                    open_episode = None
                continue
            if open_episode is None:
                open_episode = {}
                episodes += 1
            for pid, d in close:
                if d < touch:
                    open_episode.setdefault(pid, decision)
        if open_episode is not None:
            first = sorted(open_episode.values())
            touched += bool(first)
            joint += any(b - a <= JOINT_WINDOW for a, b in zip(first, first[1:]))
    return episodes, touched, joint


def shifted(tracks, rng, width, height):
    """Each track moved by its own uniform random offset on the torus."""
    out = {}
    for pid, track in tracks.items():
        dx, dy = rng.uniform(0, width), rng.uniform(0, height)
        out[pid] = {d: ((x + dx) % width, (y + dy) % height) for d, (x, y) in track.items()}
    return out


def play(env, modules, seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    predator_tracks, mammoth_tracks, decisions = {}, {}, []
    record = env._record

    def recording(joins):
        decisions.append(env.decision)
        for animal in env.predators:
            predator_tracks.setdefault(animal.id, {})[env.decision] = (animal.position.x, animal.position.y)
        for animal in env.mammoths:
            mammoth_tracks.setdefault(animal.id, {})[env.decision] = (animal.position.x, animal.position.y)
        record(joins)

    env._record = recording
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            obs, _, terms, truncs, _ = env.step(sample_actions(modules, {a: obs[a] for a in sorted(obs) if a in env.active}, generator))
            if terms["__all__"] or truncs["__all__"]:
                break
    finally:
        env._record = record
    touch = env.body["predator"][0] + env.body["mammoth"][0]
    near = env.bystander_radius
    real = joint_arrivals(predator_tracks, mammoth_tracks, decisions, env.width, env.height, touch, near)
    rng = random.Random(seed)
    null = Counter()
    for _ in range(NULL_REPEATS):
        tracks = shifted(predator_tracks, rng, env.width, env.height)
        e, t, j = joint_arrivals(tracks, mammoth_tracks, decisions, env.width, env.height, touch, near)
        null.update(encounter_episodes=e, touched_episodes=t, joint_arrivals=j)
    return {
        "seed": seed,
        "outcome": outcome(env),
        "physics_steps": env.time_step,
        "hunt_stats": env.episode_hunt_stats,
        "tracked": dict(zip(("encounter_episodes", "touched_episodes", "joint_arrivals"), real)),
        "null": {key: value / NULL_REPEATS for key, value in null.items()},
    }


def _init(checkpoint):
    import torch

    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.checkpoints import (
        checkpoint_env_config,
        load_modules,
        untrained_modules,
    )

    torch.set_num_threads(1)
    if checkpoint == "untrained":
        from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.config.config_env import config_env

        config = dict(config_env)
        _worker["modules"] = untrained_modules(config, seed=0)
    else:
        config = checkpoint_env_config(checkpoint)
        _worker["modules"] = load_modules(checkpoint)
    _worker["env"] = StagHuntContinuous(config)


def _play(seed):
    return play(_worker["env"], _worker["modules"], seed)


def summarise(rows):
    """Totals over episodes: outcomes, hunt counters, rates, and joint
    arrivals per encounter episode, tracked and at chance level."""
    outcomes = Counter(row["outcome"] for row in rows)
    totals = Counter()
    for row in rows:
        totals.update({key: value for key, value in (row["hunt_stats"] or new_hunt_stats()).items()})
    tracked = Counter()
    null = Counter()
    for row in rows:
        tracked.update(row["tracked"])
        null.update(row["null"])
    summary = {
        "episodes": len(rows),
        "outcomes": dict(outcomes),
        "coexisting": outcomes["coexisting"],
        "totals": dict(totals),
        "joint_arrival_rate": tracked["joint_arrivals"] / tracked["encounter_episodes"] if tracked["encounter_episodes"] else None,
        "joint_arrival_rate_chance": null["joint_arrivals"] / null["encounter_episodes"] if null["encounter_episodes"] else None,
        "tracked": dict(tracked),
        "chance": dict(null),
    }
    for key, denominator in (("contact_join_rate", "contact_decisions"), ("contact_with_partner_join_rate", "contact_with_partner"), ("contact_alone_join_rate", "contact_alone")):
        joins = "contact_joins" if key == "contact_join_rate" else f"{denominator}_joins"
        summary[key] = totals[joins] / totals[denominator] if totals[denominator] else None
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("episodes", type=int)
    parser.add_argument("out", nargs="?")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init, initargs=(args.checkpoint,)) as pool:
        rows = sorted(pool.map(_play, range(args.episodes)), key=lambda row: row["seed"])
    summary = {"checkpoint": str(args.checkpoint), **summarise(rows)}
    for key, value in summary.items():
        if key not in ("totals", "tracked", "chance"):
            print(f"{key}: {round(value, 3) if isinstance(value, float) else value}")
    print("tracked:", summary["tracked"], " chance:", {k: round(v, 2) for k, v in summary["chance"].items()})
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))


if __name__ == "__main__":
    main()
