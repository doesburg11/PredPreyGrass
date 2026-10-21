"""Measure predator restraint and group hunting in trained policies
(README, "Predator restraint and group hunting"; after Ritz et al. 2021,
"A Sustainable Ecosystem through Emergent Cooperation in Multi-Agent
Reinforcement Learning").

Plays seeded episodes of one checkpoint with sampled actions (as in
training), in parallel, and records for predators at least MIN_AGE
decisions old:

- restraint: how hard a predator goes for the nearest prey it sees (its
  chosen action's velocity toward that prey, as a share of top speed: the
  policy's intent, not realized motion, which steering and inertia lag), and
  how often it stops, by distance to the prey, by the number of prey it sees
  (0-3), by its own energy, and by the number of prey alive in the whole
  world. Holding back when prey are scarce, at the same distance and energy,
  is the restraint Ritz et al. call herding. The policy cannot see the
  global prey count, only up to 3 prey in its view cone, so any restraint
  has to work through local cues. These are observational associations:
  global prey count also tracks episode time, predator numbers and which
  predators are still alive;
- encounters: a predator with a prey within ENCOUNTER units at the start of
  a decision. Success means that this prey (the encountered one, not any
  prey) is caught by this predator within the decision; by global prey
  count, energy, and target distance. Joint pursuit: another predator
  within JOIN_RADIUS whose chosen action for this decision heads toward the
  same prey (also reported with its current velocity instead). By joint
  pursuit or not, the share of encounters in which the prey is caught by
  anyone, by this predator, or by another one. Joint pursuit is cooperation
  only if it raises the chance that the prey is caught at all; if it only
  shifts catches to the other predator, it is competition for the same
  prey;
- at every catch (descriptive, as Ritz et al.'s cooperation ratio): whether
  other predators are within 64, 96 and 200 units of the catcher, and
  whether one of them within JOIN_RADIUS is moving toward the caught prey.
  This has no fair null: a catch is measured at contact (within 32 units),
  after successful chases converge, so it is more social than any
  decision-start situation by construction. The encounter-level tests are
  the evidence.

Only predators at least MIN_AGE decisions old count, as focal predators,
catchers and helpers (newborns start next to their parent with a fixed
velocity).

The environment gives the whole catch to one predator (no sharing), so
nothing rewards group hunting directly.

Every number is written to analysis/results/<label>.json and .txt. Usage,
from the repository root:

    python -m predpreygrass.non_evolutionary.continuous_space.foraging_swarms.measure_predators \\
        <checkpoint> <label> [--episodes 50] [--seed 0] [--workers 25]
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import torch

from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.checkpoints import (
    checkpoint_env_config,
    load_modules,
)
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.environment import (
    DIRECTIONS,
    SPEEDS,
    PredPreyGrassContinuous,
)
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.geometry import Vector
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.measure_swarms import (
    MIN_AGE,
    layout,
    sample_actions,
)
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.tournament import classify

HERE = Path(__file__).resolve().parent
ENCOUNTER = 64.0  # prey this close at the start of a decision: an encounter
NEAR = (64.0, 96.0, 200.0)  # radii for "another predator nearby"
JOIN_RADIUS = 96.0
APPROACHING = 0.3  # velocity toward the prey, as a share of top speed: pursuing
DIST_EDGES = (50.0, 100.0, 150.0)  # bins 0-50, 50-100, 100-150, 150-200
PREY_EDGES = (5, 15, 30)  # global prey alive: <=5, 6-15, 16-30, >30
PREY_BIN_NAMES = ("<=5", "6-15", "16-30", ">30")
ENERGY_EDGES = (0.25, 0.4)  # own energy / cap (reproduction at 0.5): low, mid, high
BOOTSTRAP = 2000
LARGE_ENERGY = 6.0  # prey with at least this energy count as large (README, "Large prey": the K threshold)
STOP = SPEEDS.index(0.0)

_ANGLES = np.radians(np.arange(DIRECTIONS) * 360 / DIRECTIONS)
_UNITS = np.stack([np.sin(_ANGLES), -np.cos(_ANGLES)], axis=1)  # geometry.direction_vector


def bin_of(value, edges):
    return int(sum(value > edge for edge in edges))


def intended_velocity(action):
    """The velocity an action asks for, as a share of top speed."""
    return _UNITS[action % DIRECTIONS] * SPEEDS[action // DIRECTIONS]


def toward(vx, vy, dx, dy):
    """Component of (vx, vy) along the direction (dx, dy)."""
    norm = math.hypot(dx, dy)
    return (vx * dx + vy * dy) / norm if norm > 0 else 0.0


class Accumulator:
    def __init__(self):
        self.sum = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, name, value):
        self.sum[name] += float(value)
        self.n[name] += 1


def play(env, modules, seed):
    """One episode; returns its raw measurements."""
    generator = torch.Generator().manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    lay = layout(env)
    start = lay["prey"]
    n_slots = lay["count"]["prey"]
    acc = Accumulator()
    born_at: dict[str, int] = {}
    decision = 0
    catches_seen = []  # (catcher, prey x, y, predator snapshot), filled during a step
    move = env._move

    def recording_move(animal, action, catcher, aways, catches):
        if catcher is not None:
            snapshot = [(p.id, p.position.x, p.position.y, p.velocity.x, p.velocity.y, p.max_speed) for p in env.predators]
            catches_seen.append((catcher.id, animal.position.x, animal.position.y, snapshot, animal.id))
        move(animal, action, catcher, aways, catches)

    env._move = recording_move
    total_catches = 0
    try:
        while True:
            for p in env.predators:
                born_at.setdefault(p.id, decision)
            actions = sample_actions(modules, obs, generator)
            prey_alive = len(env.prey)
            g = bin_of(prey_alive, PREY_EDGES)
            encounters = []
            for p in env.predators:
                o = obs.get(p.id)
                if o is None or decision - born_at[p.id] < MIN_AGE:
                    continue
                a = int(actions[p.id])
                e = bin_of(float(o[-1]), ENERGY_EDGES)
                n_vis = sum(o[start + lay["slot"] * k] > 0 for k in range(n_slots))
                stopped = a // DIRECTIONS == STOP
                acc.add(f"stop|nvis{n_vis}", stopped)
                acc.add(f"stop|g{g}", stopped)
                if not n_vis:
                    continue
                reach = p.view_distance
                dx, dy, d = o[start + 1] * reach, o[start + 2] * reach, o[start + 3] * reach
                vx, vy = intended_velocity(a)
                approach = toward(vx, vy, dx, dy)
                db = bin_of(d, DIST_EDGES)
                acc.add(f"approach|d{db}|g{g}", approach)
                acc.add(f"approach|d{db}|e{e}", approach)
                acc.add(f"approach|d{db}|nvis{n_vis}", approach)
                acc.add(f"approach|d{db}|e{e}|g{g}", approach)
                acc.add(f"stop_seeing_prey|g{g}", stopped)
                acc.add(f"stop_seeing_prey|e{e}", stopped)
                px, py = p.position.x + dx, p.position.y + dy
                spot = Vector(px % env.width, py % env.height)
                target = min(env.prey, key=lambda q: env.torus.distance(spot, q.position))
                if env.torus.distance(spot, target.position) > 1.0:
                    acc.add("enc_target_ambiguous", 1)  # reconstruction failed: leave out
                    continue
                large = int(target.energy >= LARGE_ENERGY)
                acc.add(f"approach|d{db}|size{large}", approach)
                acc.add(f"stop_seeing_prey|size{large}", stopped)
                if d > ENCOUNTER:
                    continue
                # Encounter: who else is near, and is anyone else pursuing this prey?
                near = {r: False for r in NEAR}
                joint = joint_v = False
                for q in env.predators:
                    if q is p or decision - born_at[q.id] < MIN_AGE:
                        continue
                    ox, oy = env.torus.offset(p.position, q.position)
                    dist = math.hypot(ox, oy)
                    for r in NEAR:
                        near[r] |= dist <= r
                    if dist <= JOIN_RADIUS:
                        tx = (px - q.position.x + env.width / 2) % env.width - env.width / 2
                        ty = (py - q.position.y + env.height / 2) % env.height - env.height / 2
                        qa = actions.get(q.id)
                        if qa is not None:
                            joint |= toward(*intended_velocity(int(qa)), tx, ty) >= APPROACHING
                        joint_v |= toward(q.velocity.x, q.velocity.y, tx, ty) >= APPROACHING * q.max_speed
                encounters.append((p.id, g, e, bin_of(d, (32.0,)), near, joint, joint_v, target.id, large))

            catches_seen.clear()
            obs, _, terms, truncs, _ = env.step(actions)
            decision += 1

            caught_by = defaultdict(int)
            catcher_of = {}
            for catcher, x, y, snapshot, prey_id in catches_seen:
                caught_by[catcher] += 1
                catcher_of[prey_id] = catcher
                total_catches += 1
                if decision - 1 - born_at.get(catcher, decision - 1) < MIN_AGE:
                    acc.add("catch_by_young", 1)
                    continue
                cx, cy = next((sx, sy) for i, sx, sy, _, _, _ in snapshot if i == catcher)
                near = {r: False for r in NEAR}
                joint = False
                for i, sx, sy, svx, svy, top in snapshot:
                    if i == catcher or decision - 1 - born_at.get(i, decision - 1) < MIN_AGE:
                        continue
                    ox = (sx - cx + env.width / 2) % env.width - env.width / 2
                    oy = (sy - cy + env.height / 2) % env.height - env.height / 2
                    dist = math.hypot(ox, oy)
                    for r in NEAR:
                        near[r] |= dist <= r
                    if dist <= JOIN_RADIUS:
                        tx = (x - sx + env.width / 2) % env.width - env.width / 2
                        ty = (y - sy + env.height / 2) % env.height - env.height / 2
                        joint |= toward(svx, svy, tx, ty) >= APPROACHING * top
                for r in NEAR:
                    acc.add(f"catch_near{int(r)}", near[r])
                acc.add("catch_joint", joint)
                acc.add(f"catch|g{g}", 1)
            for pid, g_, e_, db_, near, joint, joint_v, target, large in encounters:
                by = catcher_of.get(target)
                success = by == pid
                j = int(joint)
                acc.add(f"enc_target_caught|joint{j}", by is not None)
                acc.add(f"enc_target_caught|joint{j}|d{db_}", by is not None)
                acc.add(f"enc_target_by_self|joint{j}", success)
                acc.add(f"enc_target_by_self|joint{j}|d{db_}", success)
                acc.add(f"enc_target_by_other|joint{j}", by is not None and by != pid)
                acc.add(f"enc_target_caught|jointv{int(joint_v)}", by is not None)
                acc.add(f"enc_success|g{g_}", success)
                acc.add(f"enc_success|e{e_}", success)
                acc.add(f"enc_success|e{e_}|g{g_}", success)
                for r in NEAR:
                    acc.add(f"enc_near{int(r)}", near[r])
                acc.add("enc_joint", joint)
                acc.add("enc_joint_v", joint_v)
                acc.add(f"enc_joint|size{large}", joint)
                acc.add(f"enc_target_caught|size{large}|joint{j}", by is not None)
                acc.add(f"enc_target_by_self|size{large}|joint{j}", success)
            if terms["__all__"] or truncs["__all__"]:
                break
    finally:
        env._move = move
    return {
        "seed": seed,
        "outcome": classify(len(env.predators), len(env.prey)),
        "catches": total_catches,
        "decisions": decision,
        "sums": dict(acc.sum),
        "counts": dict(acc.n),
    }


# --- Summaries -------------------------------------------------------------

# (name, key, key subtracted from it): differences with episode-bootstrap intervals.
INTERVALS = [
    ("joint_pursuit_share_large_minus_small", "enc_joint|size1", "enc_joint|size0"),
    ("large_prey_caught_by_anyone_joint_minus_alone", "enc_target_caught|size1|joint1", "enc_target_caught|size1|joint0"),
    ("encounter_prey_caught_by_anyone_joint_minus_alone", "enc_target_caught|joint1", "enc_target_caught|joint0"),
    ("  same, prey within 32", "enc_target_caught|joint1|d0", "enc_target_caught|joint0|d0"),
    ("  same, prey 32-64", "enc_target_caught|joint1|d1", "enc_target_caught|joint0|d1"),
    ("  same, joint by current velocity", "enc_target_caught|jointv1", "enc_target_caught|jointv0"),
    ("encounter_caught_by_self_joint_minus_alone", "enc_target_by_self|joint1", "enc_target_by_self|joint0"),
    ("encounter_success_low_energy_prey6-15_minus_16-30", "enc_success|e0|g1", "enc_success|e0|g2"),
    ("approach_d50-100_low_energy_prey6-15_minus_16-30", "approach|d1|e0|g1", "approach|d1|e0|g2"),
    ("encounter_success_prey<=5_minus_16-30", "enc_success|g0", "enc_success|g2"),
    ("encounter_success_prey6-15_minus_16-30", "enc_success|g1", "enc_success|g2"),
    ("approach_d50-100_prey<=5_minus_16-30", "approach|d1|g0", "approach|d1|g2"),
    ("approach_d50-100_prey6-15_minus_16-30", "approach|d1|g1", "approach|d1|g2"),
    ("approach_d0-50_prey6-15_minus_16-30", "approach|d0|g1", "approach|d0|g2"),
]


def episode_intervals(episodes, rng, resamples=BOOTSTRAP):
    """95% percentile intervals, resampling whole episodes. They describe
    these policies over episodes, not variation between training runs."""
    out = {}
    for name, a, b in INTERVALS:
        sums = np.array([[e["sums"].get(k, 0.0) for k in (a, b)] for e in episodes])
        counts = np.array([[e["counts"].get(k, 0) for k in (a, b)] for e in episodes])
        if (counts.sum(axis=0) == 0).any():
            continue

        def statistic(index):
            means = sums[index].sum(axis=0) / np.maximum(counts[index].sum(axis=0), 1)
            return means[0] - means[1]

        draws = [statistic(rng.integers(len(episodes), size=len(episodes))) for _ in range(resamples)]
        out[name] = {
            "estimate": float(statistic(np.arange(len(episodes)))),
            "low": float(np.percentile(draws, 2.5)),
            "high": float(np.percentile(draws, 97.5)),
            "n": [int(c) for c in counts.sum(axis=0)],
        }
    return out


def summarize(episodes):
    sums, counts = defaultdict(float), defaultdict(int)
    for e in episodes:
        for k, v in e["sums"].items():
            sums[k] += v
        for k, v in e["counts"].items():
            counts[k] += v
    outcomes = defaultdict(int)
    for e in episodes:
        outcomes[e["outcome"]] += 1
    return {
        "episodes": len(episodes),
        "outcomes": dict(outcomes),
        "coexisting": outcomes["time_limit"],
        "catches": sum(e["catches"] for e in episodes),
        "means": {k: sums[k] / counts[k] for k in sorted(sums) if counts[k]},
        "counts": dict(counts),
        "intervals": episode_intervals(episodes, np.random.default_rng(0)),
    }


def format_summary(label, s):
    m, c = s["means"], s["counts"]

    def cell(key, fmt="{:.3f}"):
        return f"{fmt.format(m[key])} ({c[key]})" if key in m else "-"

    lines = [f"== {label}: {s['episodes']} episodes, {s['coexisting']} coexisting {s['outcomes']}; {s['catches']} catches"]
    lines.append("   approach toward nearest visible prey (share of top speed), rows distance 0-50/50-100/100-150/150-200,")
    lines.append("   columns prey alive " + " / ".join(PREY_BIN_NAMES) + "  [n]:")
    for d in range(4):
        lines.append(f"     d{d}: " + "  ".join(cell(f"approach|d{d}|g{g}") for g in range(4)))
    lines.append("   approach at low energy (<0.25), by prey alive:")
    for d in range(4):
        lines.append(f"     d{d}: " + "  ".join(cell(f"approach|d{d}|e0|g{g}") for g in range(4)))
    lines.append("   approach by own energy low/mid/high (<0.25 / 0.25-0.4 / >0.4; reproduction at 0.5):")
    for d in range(2):
        lines.append(f"     d{d}: " + "  ".join(cell(f"approach|d{d}|e{e}") for e in range(3)))
    lines.append("   approach by prey in view 1/2/3:")
    for d in range(2):
        lines.append(f"     d{d}: " + "  ".join(cell(f"approach|d{d}|nvis{n}") for n in (1, 2, 3)))
    lines.append(f"   by prey size (large: energy >= {LARGE_ENERGY:g}), small / large:")
    lines.append("     joint pursuit share: " + "  ".join(cell(f"enc_joint|size{z}") for z in (0, 1)))
    for z, name in ((0, "small"), (1, "large")):
        lines.append(f"     {name}: caught by anyone alone / joint " + "  ".join(cell(f"enc_target_caught|size{z}|joint{j}") for j in (0, 1))
                     + "; by self alone / joint " + "  ".join(cell(f"enc_target_by_self|size{z}|joint{j}") for j in (0, 1)))
    lines.append("     approach d0/d1, small: " + "  ".join(cell(f"approach|d{d}|size0") for d in (0, 1))
                 + "  large: " + "  ".join(cell(f"approach|d{d}|size1") for d in (0, 1)))
    lines.append("     stop share seeing prey, small / large: " + "  ".join(cell(f"stop_seeing_prey|size{z}") for z in (0, 1)))
    lines.append("   stop share by prey in view 0-3: " + "  ".join(cell(f"stop|nvis{n}") for n in range(4)))
    lines.append("   stop share by prey alive:      " + "  ".join(cell(f"stop|g{g}") for g in range(4)))
    lines.append("   stop share seeing prey, by prey alive: " + "  ".join(cell(f"stop_seeing_prey|g{g}") for g in range(4)))
    lines.append("   stop share seeing prey, by energy:     " + "  ".join(cell(f"stop_seeing_prey|e{e}") for e in range(3)))
    lines.append("   encounter -> catch, by prey alive: " + "  ".join(cell(f"enc_success|g{g}") for g in range(4)))
    lines.append("   encounter -> catch, by energy:     " + "  ".join(cell(f"enc_success|e{e}") for e in range(3)))
    for j, name in ((0, "alone"), (1, "joint")):
        lines.append(f"   encounter ({name}): target caught by anyone {cell(f'enc_target_caught|joint{j}')}, by self "
                     f"{cell(f'enc_target_by_self|joint{j}')}, by another {cell(f'enc_target_by_other|joint{j}')}")
    lines.append(f"   encounters with joint pursuit: by action {cell('enc_joint')}, by current velocity {cell('enc_joint_v')}; "
                 f"ambiguous targets left out: {c.get('enc_target_ambiguous', 0)}")
    lines.append("   descriptive (no fair null), other predator near the catcher at catches: "
                 + "  ".join(f"within {int(r)} {cell(f'catch_near{int(r)}')}" for r in NEAR)
                 + f"  joint {cell('catch_joint')}; catches by young predators left out: {c.get('catch_by_young', 0)}")
    lines.append("   95% intervals over episodes:")
    for name, i in s["intervals"].items():
        lines.append(f"     {name}: {i['estimate']:.4f} ({i['low']:.4f} to {i['high']:.4f}) n={i['n']}")
    return "\n".join(lines)


_worker: dict = {}


def _init(checkpoint, env_config):
    torch.set_num_threads(1)
    _worker["env"] = PredPreyGrassContinuous(env_config)
    _worker["modules"] = load_modules(checkpoint)


def _episode(seed):
    return play(_worker["env"], _worker["modules"], seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("label")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(25, os.cpu_count() or 1))
    parser.add_argument("--out-dir", default=str(HERE / "analysis" / "results"))
    args = parser.parse_args()
    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    env_config = checkpoint_env_config(checkpoint)
    start = time.time()
    seeds = list(range(args.seed, args.seed + args.episodes))
    with get_context("spawn").Pool(min(args.workers, len(seeds)), initializer=_init, initargs=(checkpoint, env_config)) as pool:
        episodes = pool.map(_episode, seeds, chunksize=1)
    summary = summarize(episodes)
    summary.update(
        label=args.label,
        checkpoint=checkpoint,
        seeds=seeds,
        settings=dict(MIN_AGE=MIN_AGE, ENCOUNTER=ENCOUNTER, NEAR=NEAR, JOIN_RADIUS=JOIN_RADIUS, APPROACHING=APPROACHING,
                      DIST_EDGES=DIST_EDGES, PREY_EDGES=PREY_EDGES, ENERGY_EDGES=ENERGY_EDGES),
        per_episode=[{k: e[k] for k in ("seed", "outcome", "catches", "decisions")} for e in episodes],
        wall_seconds=round(time.time() - start, 1),
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(summary, indent=1) + "\n")
    text = format_summary(args.label, summary)
    (out / f"{args.label}.txt").write_text(text + "\n")
    print(text)
    print(f"({summary['wall_seconds']} s; written to {out / args.label}.json)")


if __name__ == "__main__":
    sys.exit(main())
