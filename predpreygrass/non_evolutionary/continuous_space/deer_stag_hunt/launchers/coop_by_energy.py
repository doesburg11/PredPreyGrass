"""Exploratory: cooperation by the deer's energy (the stag hunt inside the
touch rule). Under deer_catch_rule "touch", a middle deer (energy between
fawn and adult) can be caught alone with chance (adult - e) / (adult - fawn)
or by two touchers for certain; the expected gain per hunter is higher alone
below e = 3.5 and higher shared above it. Does the share of approaches with
a partner rise with the deer's energy, past chance, where pairing pays?

Approaches are counted as in coop_fair.py (a predator touching a deer it has
not touched in the last GAP physics steps), binned by the deer's energy at
the approach. Per bin:
  - near: another predator within the party radius of the deer (coop_fair's
    measure);
  - joint: another predator, ready to catch (not in a catch cooldown),
    touched the same deer within touch_window physics steps of the
    approach, before or after (closer to teaming up: under "touch", two
    ready touchers within the window catch the deer). Retrospective: it
    marks both the earlier and the later hunter. Under "party" it is only
    descriptive (young deer need no second toucher there).
Approaches, and so the near counts, match coop_fair.py's exactly for the
same checkpoint and seeds (fawn = first bin, middle = the middle bins,
adult = last bin): physical touches, cooldowns included.
Chance (for near): predator time with another predator within the radius.
Intervals: bootstrap over episodes of share minus chance (near) and of the
share itself (joint).

Run from the repository root:
    python predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/launchers/coop_by_energy.py \\
        CHECKPOINT EPISODES WORKERS [OUT.json] [--set KEY=JSON ...]
"""

import argparse
import bisect
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt import evaluate
from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.launchers.coop_fair import GAP

# Bin upper edges (energy <= edge); the last bin is adults (>= adult energy).
EDGES = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
ADULT = 5.0  # checked against each episode's world in count()
# Bins of RESULTS.md section 2.1's tests (bin b holds energies in
# (EDGES[b-1], EDGES[b]], the last middle bin (4.5, 5)).
LOW = (1, 2, 3)  # (2, 3.5]: alone pays more
HIGH = (5, 6)  # (4, 5): a partner pays more
TOP = 6  # (4.5, 5)
ADULT_BIN = 7  # >= 5
BOOTSTRAP = 5000


def bin_names(fawn, adult):
    names = [f"<={fawn:g} (fawn)"]
    lows = (fawn,) + EDGES[1:]
    for lo, hi in zip(lows, EDGES[1:] + (adult,)):
        names.append(f"{lo:g}-{hi:g}")
    names.append(f">={adult:g} (adult)")
    return names


def bin_of(energy, fawn, adult):
    """0 for a fawn, len(EDGES) + 1 for an adult, else the middle bin 1..len(EDGES)."""
    if energy >= adult:
        return len(EDGES) + 1
    if energy <= fawn:
        return 0
    return bisect.bisect_left(EDGES, energy)


def count(env, modules, seed):
    """Per-bin approach counts of one evaluated episode."""
    radius, window = env.party_radius, env.touch_window
    fawn, adult = env.fawn_energy, env.adult_energy
    assert fawn == EDGES[0], (fawn, EDGES)
    assert adult == ADULT, (adult, ADULT)
    nbins = len(EDGES) + 2
    totals = {"hunter_steps": 0, "with_partner": 0}
    approaches = []  # (bin, near, time, predator id, deer id)
    touches = defaultdict(list)  # deer id -> [(time, predator id)]
    last_touch = {}
    original_step, original_contacts = env._physics_step, env._contacts

    def step(moves):
        hunters = env.predators
        totals["hunter_steps"] += len(hunters)
        for h in hunters:
            if any(o is not h and env.torus.distance(h.position, o.position) <= radius for o in hunters):
                totals["with_partner"] += 1
        return original_step(moves)

    def contacts(animal):
        if animal.species == "deer":
            now = env.time_step
            near = [p for p in env.predators if env.torus.distance(animal.position, p.position) <= radius]
            for p in env.predators:
                if not env._touching(animal, p):
                    continue
                if p.catch_ready_at <= now:  # the world counts ready touches only
                    touches[animal.id].append((now, p.id))
                key = (p.id, animal.id)
                if now - last_touch.get(key, -GAP - 1) > GAP:
                    approaches.append((bin_of(animal.energy, fawn, adult), any(o is not p for o in near), now, p.id, animal.id))
                last_touch[key] = now
        return original_contacts(animal)

    env._physics_step, env._contacts = step, contacts
    try:
        evaluate.play(env, modules, seed)
    finally:
        env._physics_step, env._contacts = original_step, original_contacts

    row = dict(totals)
    for b in range(nbins):
        row[f"n{b}"] = row[f"near{b}"] = row[f"joint{b}"] = 0
    for b, near, t, pid, did in approaches:
        joint = any(abs(t2 - t) <= window and p2 != pid for t2, p2 in touches[did])
        row[f"n{b}"] += 1
        row[f"near{b}"] += near
        row[f"joint{b}"] += joint
    return row


def _count(seed):
    return count(evaluate._worker["env"], evaluate._worker["modules"], seed)


def pooled(rows, key, b):
    n = sum(r[f"n{b}"] for r in rows)
    return sum(r[f"{key}{b}"] for r in rows) / n if n else None


def chance_of(rows):
    steps = sum(r["hunter_steps"] for r in rows)
    return sum(r["with_partner"] for r in rows) / steps if steps else None


def interval(rows, fn, seed=0):
    rng = random.Random(seed)
    samples = sorted(v for v in (fn([rng.choice(rows) for _ in rows]) for _ in range(BOOTSTRAP)) if v is not None)
    if len(samples) < 0.95 * BOOTSTRAP:
        return None
    n = len(samples) - 1
    return [samples[round(0.025 * n)], samples[round(0.975 * n)]]


def pooled_group(rows, key, bins):
    n = sum(r[f"n{b}"] for r in rows for b in bins)
    return sum(r[f"{key}{b}"] for r in rows for b in bins) / n if n else None


def trend(rows):
    """RESULTS.md section 2.1, test A: near share of middle deer with energy
    in (4, 5) minus that with energy in (2, 3.5], or None."""
    high, low = pooled_group(rows, "near", HIGH), pooled_group(rows, "near", LOW)
    return None if high is None or low is None else high - low


def top_minus_chance(rows):
    """RESULTS.md section 2.1, test B: near share of middle deer with energy
    in (4.5, 5) minus chance, or None."""
    near, chance = pooled(rows, "near", TOP), chance_of(rows)
    return None if near is None or chance is None else near - chance


def joint_ratio(rows):
    """RESULTS.md section 2.1, test C: joint share of middle deer with energy
    in (4, 5) divided by joint share of adults, or None."""
    high, adult = pooled_group(rows, "joint", HIGH), pooled(rows, "joint", ADULT_BIN)
    return None if high is None or not adult else high / adult


def summarise(rows, names):
    chance = chance_of(rows)
    out = {
        "episodes": len(rows),
        "chance": chance,
        "trend": trend(rows),
        "trend_interval": interval(rows, trend),
        "top_minus_chance": top_minus_chance(rows),
        "top_minus_chance_interval": interval(rows, top_minus_chance),
        "joint_ratio": joint_ratio(rows),
        "joint_ratio_interval": interval(rows, joint_ratio),
        "bins": [],
    }
    for b, name in enumerate(names):
        near, joint = pooled(rows, "near", b), pooled(rows, "joint", b)
        out["bins"].append({
            "bin": name,
            "approaches": sum(r[f"n{b}"] for r in rows),
            "near": near,
            "near_over_chance": near / chance if near is not None and chance else None,
            "near_minus_chance_interval": interval(rows, lambda rs, b=b: None if pooled(rs, "near", b) is None or chance_of(rs) is None else pooled(rs, "near", b) - chance_of(rs)),
            "joint": joint,
            "joint_interval": interval(rows, lambda rs, b=b: pooled(rs, "joint", b)),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("episodes", type=int)
    parser.add_argument("workers", type=int)
    parser.add_argument("out", nargs="?")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("EPISODES must be at least 1")
    overrides = evaluate.parse_overrides(args.set)
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=evaluate._init, initargs=(args.checkpoint, overrides)) as pool:
        rows = list(pool.map(_count, range(args.episodes)))
    s = summarise(rows, bin_names(EDGES[0], ADULT))
    print(args.checkpoint, overrides or "")
    print(f"  chance (predator time with a partner near): {s['chance']:.3f}")
    print(f"  {'deer energy':16s} {'approaches':>10s} {'near':>6s} {'near/chance':>11s} {'near-chance 95%':>18s} {'joint':>6s} {'joint 95%':>16s}")
    for row in s["bins"]:
        def f(x, p=3):
            return "-" if x is None else f"{x:.{p}f}"
        iv = row["near_minus_chance_interval"]
        jv = row["joint_interval"]
        print(f"  {row['bin']:16s} {row['approaches']:10d} {f(row['near']):>6s} {f(row['near_over_chance'], 2):>11s} "
              f"{'-' if iv is None else f'[{iv[0]:+.3f},{iv[1]:+.3f}]':>18s} {f(row['joint']):>6s} "
              f"{'-' if jv is None else f'[{jv[0]:.3f},{jv[1]:.3f}]':>16s}")
    for label, key in (("A: near (4,5) - near (2,3.5]", "trend"), ("B: near (4.5,5) - chance", "top_minus_chance"), ("C: joint (4,5) / joint adult", "joint_ratio")):
        value, iv = s[key], s[f"{key}_interval"]
        print(f"  {label}: {'-' if value is None else f'{value:+.3f}'}, 95% interval {'-' if iv is None else f'[{iv[0]:+.3f},{iv[1]:+.3f}]'}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"checkpoint": args.checkpoint, "overrides": overrides, "summary": s, "episodes": rows}, handle, indent=1)


if __name__ == "__main__":
    main()
