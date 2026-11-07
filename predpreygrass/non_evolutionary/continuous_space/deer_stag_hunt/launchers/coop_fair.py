"""Cooperation against chance, counting each approach once (stag_hunt_continuous's
coop_fair.py with deer). An approach starts when a predator touches a deer it
has not touched in the last GAP physics steps; repeated touches within GAP
(retries after a failed attempt) belong to it. For each approach: was another
predator within the party radius of the deer, and was the deer an adult
(energy >= deer_adult_energy), a fawn (<= deer_fawn_energy) or in between?
Chance: the share of predator time with another predator within the party
radius. Predators in a catch cooldown count too.

Cooperation is shown if approaches to adults have a partner near more often
than approaches to fawns and than chance.

Run from the repository root:
    python predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/launchers/coop_fair.py \\
        CHECKPOINT EPISODES WORKERS [OUT.json] [--set KEY=JSON ...]
"""

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt import evaluate
from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.environment import KINDS

GAP = 64  # physics steps
BOOTSTRAP = 5000


def count(env, modules, seed):
    """Approach and partner counts of one evaluated episode."""
    radius = env.party_radius
    counts = {"hunter_steps": 0, "with_partner": 0}
    for kind in KINDS:
        counts[f"{kind}_approaches"] = 0
        counts[f"{kind}_with_partner"] = 0
    last_touch = {}
    original_step, original_contacts = env._physics_step, env._contacts

    def step(moves):
        hunters = env.predators
        counts["hunter_steps"] += len(hunters)
        for h in hunters:
            if any(o is not h and env.torus.distance(h.position, o.position) <= radius for o in hunters):
                counts["with_partner"] += 1
        return original_step(moves)

    def contacts(animal):
        if animal.species == "deer":
            now = env.time_step
            near = [p for p in env.predators if env.torus.distance(animal.position, p.position) <= radius]
            for p in env.predators:
                if not env._touching(animal, p):
                    continue
                key = (p.id, animal.id)
                if now - last_touch.get(key, -GAP - 1) > GAP:
                    kind = env.kind(animal)
                    counts[f"{kind}_approaches"] += 1
                    counts[f"{kind}_with_partner"] += any(o is not p for o in near)
                last_touch[key] = now
        return original_contacts(animal)

    env._physics_step, env._contacts = step, contacts
    try:
        evaluate.play(env, modules, seed)
    finally:
        env._physics_step, env._contacts = original_step, original_contacts
    return counts


def _count(seed):
    return count(evaluate._worker["env"], evaluate._worker["modules"], seed)


def adult_minus_chance(rows):
    """Pooled adult share minus pooled chance, or None."""
    approaches = sum(r["adult_approaches"] for r in rows)
    steps = sum(r["hunter_steps"] for r in rows)
    if not approaches or not steps:
        return None
    return sum(r["adult_with_partner"] for r in rows) / approaches - sum(r["with_partner"] for r in rows) / steps


def bootstrap_interval(rows, seed=0):
    """95% interval of adult share minus chance, resampling episodes
    (BOOTSTRAP resamples; None if under 95% of them have adult approaches)."""
    rng = random.Random(seed)
    samples = sorted(d for d in (adult_minus_chance([rng.choice(rows) for _ in rows]) for _ in range(BOOTSTRAP)) if d is not None)
    if len(samples) < 0.95 * BOOTSTRAP:
        return None
    n = len(samples) - 1
    return [samples[round(0.025 * n)], samples[round(0.975 * n)]]


def summarise(rows):
    t = {k: sum(r[k] for r in rows) for k in rows[0]}
    chance = t["with_partner"] / t["hunter_steps"] if t["hunter_steps"] else None
    share = {kind: t[f"{kind}_with_partner"] / t[f"{kind}_approaches"] if t[f"{kind}_approaches"] else None for kind in KINDS}
    adult_above = sum(
        bool(r["adult_approaches"] and r["fawn_approaches"] and r["adult_with_partner"] / r["adult_approaches"] > r["fawn_with_partner"] / r["fawn_approaches"])
        for r in rows
    )
    shown = None not in (share["adult"], share["fawn"], chance) and share["adult"] > share["fawn"] and share["adult"] > chance
    adult_above_chance = sum(bool(r["adult_approaches"] and r["hunter_steps"] and r["adult_with_partner"] / r["adult_approaches"] > r["with_partner"] / r["hunter_steps"]) for r in rows)
    return {
        "episodes": len(rows),
        "chance": chance,
        "share": share,
        "approaches": {k: t[f"{k}_approaches"] for k in KINDS},
        "adult_above_fawn_episodes": adult_above,
        "adult_above_chance_episodes": adult_above_chance,
        "cooperation_shown": shown,
        "adult_over_chance": share["adult"] / chance if share["adult"] is not None and chance else None,
        "adult_minus_chance": adult_minus_chance(rows),
        "adult_minus_chance_interval": bootstrap_interval(rows),
    }


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
    s = summarise(rows)
    print(args.checkpoint, overrides or "")
    chance = s["chance"]
    print(f"  chance (predator time with a partner near): {chance:.3f}" if chance is not None else "  chance: no predator steps")
    for kind in KINDS:
        share = s["share"][kind]
        text = "-" if share is None else f"{share:.3f}" + (f" ({share / chance:.2f}x chance)" if chance else "")
        print(f"  {kind:6s} approaches {s['approaches'][kind]:6d}, with a partner near {text}")
    print(f"  episodes with adult share above fawn share: {s['adult_above_fawn_episodes']} of {len(rows)}")
    print(f"  cooperation shown (adult > fawn and adult > chance): {s['cooperation_shown']}")
    print(f"  adult share / chance: {s['adult_over_chance']}; adult minus chance {s['adult_minus_chance']}, 95% interval {s['adult_minus_chance_interval']}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"checkpoint": args.checkpoint, "overrides": overrides, "summary": s, "episodes": rows}, handle, indent=1)


if __name__ == "__main__":
    main()
