"""Cooperation against chance with lean and fat rabbits, counting each approach
once, as stag_hunt_rule_prey/launchers/coop_fair.py. An approach starts when a
predator touches a rabbit it has not touched in the last GAP physics steps;
repeated touches within GAP (retries after a failed attempt) belong to it. For
each approach: was another predator within the party radius of the rabbit, and
was the rabbit fat (energy >= rabbit_fat_energy) or lean (<= rabbit_lean_energy)?
Chance: the share of predator time with another predator within the party
radius. Predators in a catch cooldown count too; touches of a rabbit inside a
burrow (which cannot be caught) do not.

Cooperation is shown if approaches to fat rabbits have a partner near more
often than approaches to lean rabbits and than chance (RESULTS.md, section 5).

Run from the repository root:
    python predpreygrass/non_evolutionary/continuous_space/stag_hunt_continuous/launchers/coop_fair.py \\
        CHECKPOINT EPISODES WORKERS [OUT.json] [--set KEY=JSON ...]

--set as in evaluate.py, e.g. --set rabbit_fat_energy=6.0 for policies trained
without the rule.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous import evaluate

GAP = 64  # physics steps
KINDS = ("lean", "middle", "fat")


def count(env, modules, seed):
    """Approach and partner counts of one evaluated episode."""
    if not env.fat_energy:
        raise SystemExit("rabbit_fat_energy is 0: no lean and fat rabbits (use --set)")
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
        if animal.species == "rabbit" and not animal.in_burrow:
            now = env.time_step
            near = [p for p in env.predators if env.torus.distance(animal.position, p.position) <= radius]
            for p in env.predators:
                if not env._touching(animal, p):
                    continue
                key = (p.id, animal.id)
                if now - last_touch.get(key, -GAP - 1) > GAP:
                    e = animal.energy
                    kind = "fat" if e >= env.fat_energy else "lean" if e <= env.lean_energy else "middle"
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


def summarise(rows):
    t = {k: sum(r[k] for r in rows) for k in rows[0]}
    chance = t["with_partner"] / t["hunter_steps"] if t["hunter_steps"] else None
    share = {kind: t[f"{kind}_with_partner"] / t[f"{kind}_approaches"] if t[f"{kind}_approaches"] else None for kind in KINDS}
    fat_above = sum(
        bool(r["fat_approaches"] and r["lean_approaches"] and r["fat_with_partner"] / r["fat_approaches"] > r["lean_with_partner"] / r["lean_approaches"])
        for r in rows
    )
    shown = None not in (share["fat"], share["lean"], chance) and share["fat"] > share["lean"] and share["fat"] > chance
    return {"episodes": len(rows), "chance": chance, "share": share, "approaches": {k: t[f"{k}_approaches"] for k in KINDS}, "fat_above_lean_episodes": fat_above, "cooperation_shown": shown}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("episodes", type=int)
    parser.add_argument("workers", type=int)
    parser.add_argument("out", nargs="?")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    args = parser.parse_args()
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
    print(f"  episodes with fat share above lean share: {s['fat_above_lean_episodes']} of {len(rows)}")
    print(f"  cooperation shown (fat > lean and fat > chance): {s['cooperation_shown']}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"checkpoint": args.checkpoint, "overrides": overrides, "summary": s, "episodes": rows}, handle, indent=1)


if __name__ == "__main__":
    main()
