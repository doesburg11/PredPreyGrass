"""Are calls followed by a partner? For worlds with hunter_calls (RESULTS.md,
section 8). At the end of each decision, a hunter is "alone at a fat rabbit"
if an open fat rabbit (energy >= rabbit_fat_energy, not in a burrow) is
within the party radius of the hunter and no other hunter is within the party
radius of that rabbit. A moment starts when a (hunter, rabbit) pair enters
that state (not again while it lasts), split by whether the hunter called in
that decision, and is followed for WINDOW decisions: did another hunter come
within the party radius of that rabbit, at any physics step? A catch of the rabbit by two or
more hunters counts as an arrival; the rabbit caught otherwise, starving,
entering a burrow or ceasing to be fat, or the hunter dying, ends the moment
uncounted (censored), as does the episode's end.

The test (section 8): pooled over episodes, arrivals after a call more
frequent than after silence, with a 95% bootstrap interval over episodes of
the difference above 0 (no interval if under 95% of resamples hold both
kinds). This is an association: the policy chooses when to call. Also: the
call rate at such moments against all other hunter decisions.

Run from the repository root:
    python predpreygrass/non_evolutionary/continuous_space/stag_hunt_continuous/launchers/calls.py \\
        CHECKPOINT EPISODES WORKERS [OUT.json] [--set KEY=JSON ...]
"""

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous import evaluate

WINDOW = 4  # decisions
BOOTSTRAP = 5000


def count(env, modules, seed):
    """Counts of one evaluated episode."""
    if not (env.hunter_calls and env.fat_energy):
        raise SystemExit("needs hunter_calls and rabbit_fat_energy (use --set)")
    radius = env.party_radius
    c = {"decisions": 0, "calls": 0, "alone_at_fat": 0, "alone_at_fat_calls": 0, "call_moments": 0, "call_arrivals": 0, "silent_moments": 0, "silent_arrivals": 0, "censored": 0}
    pending = []  # [hunter id, rabbit id, decisions left, called, partner arrived]
    active = set()  # (hunter id, rabbit id) alone at a fat rabbit at the last decision
    caught_together = set()  # rabbits caught by two or more hunters
    original_step, original_contacts = env.step, env._contacts

    def near(a, b):
        return env.torus.distance(a.position, b.position) <= radius

    def contacts(animal):
        # Every physics step: a partner reaching a followed rabbit at any
        # moment of a decision, and catches by two or more.
        if animal.species == "rabbit" and animal.energy >= env.fat_energy and not animal.in_burrow:
            for entry in pending:
                if entry[1] == animal.id and not entry[4]:
                    entry[4] = any(h.id != entry[0] and near(h, animal) for h in env.predators)
        catcher, aways = original_contacts(animal)
        if animal.species == "rabbit" and catcher and len(catcher) >= 2:
            caught_together.add(animal.id)
        return catcher, aways

    def close(entry, arrived):
        kind = "call" if entry[3] else "silent"
        c[f"{kind}_moments"] += 1
        c[f"{kind}_arrivals"] += arrived

    def step(action_dict):
        result = original_step(action_dict)
        hunters = {h.id: h for h in env.predators}
        rabbits = {r.id: r for r in env.rabbits}
        still = []
        for entry in pending:
            hunter_id, rabbit_id = entry[0], entry[1]
            rabbit = rabbits.get(rabbit_id)
            if rabbit_id in caught_together or entry[4]:
                close(entry, True)
            elif hunter_id not in hunters or rabbit is None or rabbit.in_burrow or rabbit.energy < env.fat_energy:
                c["censored"] += 1
            elif any(h.id != hunter_id and near(h, rabbit) for h in hunters.values()):
                close(entry, True)
            else:
                entry[2] -= 1
                if entry[2] == 0:
                    close(entry, False)
                else:
                    still.append(entry)
        pending[:] = still
        fat = [r for r in rabbits.values() if r.energy >= env.fat_energy and not r.in_burrow]
        now = set()
        for hunter in hunters.values():
            called = hunter.id in env.calls
            c["decisions"] += 1
            c["calls"] += called
            alone_at = [r for r in fat if near(r, hunter) and not any(o is not hunter and near(o, r) for o in hunters.values())]
            if alone_at:
                c["alone_at_fat"] += 1
                c["alone_at_fat_calls"] += called
            for r in alone_at:
                now.add((hunter.id, r.id))
                if (hunter.id, r.id) not in active:
                    pending.append([hunter.id, r.id, WINDOW, called, False])
        active.clear()
        active.update(now)
        return result

    env.step, env._contacts = step, contacts
    try:
        evaluate.play(env, modules, seed)
    finally:
        env.step, env._contacts = original_step, original_contacts
    c["censored"] += len(pending)
    return c


def _count(seed):
    return count(evaluate._worker["env"], evaluate._worker["modules"], seed)


def _rate(rows, kind):
    moments = sum(r[f"{kind}_moments"] for r in rows)
    return sum(r[f"{kind}_arrivals"] for r in rows) / moments if moments else None


def summarise(rows, seed=0):
    t = {k: sum(r[k] for r in rows) for k in rows[0]}
    after_call, after_silence = _rate(rows, "call"), _rate(rows, "silent")
    diff = None if None in (after_call, after_silence) else after_call - after_silence
    low = high = None
    valid = None
    if diff is not None:
        rng = random.Random(seed)
        samples = []
        for _ in range(BOOTSTRAP):
            draw = [rng.choice(rows) for _ in rows]
            a, b = _rate(draw, "call"), _rate(draw, "silent")
            if None not in (a, b):
                samples.append(a - b)
        samples.sort()
        valid = len(samples) / BOOTSTRAP
        if valid >= 0.95:
            n = len(samples) - 1
            low, high = samples[round(0.025 * n)], samples[round(0.975 * n)]
    other = t["decisions"] - t["alone_at_fat"]
    return {
        "episodes": len(rows),
        "call_rate_alone_at_fat": t["alone_at_fat_calls"] / t["alone_at_fat"] if t["alone_at_fat"] else None,
        "call_rate_otherwise": (t["calls"] - t["alone_at_fat_calls"]) / other if other else None,
        "moments": {"call": t["call_moments"], "silent": t["silent_moments"]},
        "arrival_after_call": after_call,
        "arrival_after_silence": after_silence,
        "difference": diff,
        "difference_interval": [low, high],
        "valid_resamples": valid,
        "censored": t.get("censored", 0),
        "calls_followed_by_partner": low is not None and low > 0,
    }


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
    print(json.dumps(s, indent=1))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"checkpoint": args.checkpoint, "overrides": overrides, "summary": s, "episodes": rows}, handle, indent=1)


if __name__ == "__main__":
    main()
