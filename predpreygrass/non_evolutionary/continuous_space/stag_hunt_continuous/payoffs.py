"""Does hunting mammoths pay the hunters, and do they join when it would? Seeded
evaluation episodes (same seeds and action sampling as evaluate.py, whole
episodes), with the hunters' energy accounted for separately:

- attempts: per hunt by party size, how many hunts, how many had enough summed
  energy to win under the energy rule, kills, and contributors' net energy
  (shares gained after the energy cap, less join and struggle costs);
- contacts: decisions of a hunter touching a mammoth and ready to hunt, alone
  or with other ready hunters touching it, by whether the energies of those
  touching it would win, and how often the hunter joined. Energies are from the
  end of the decision; the hunter chose from those at its start;
- budget: rabbit food, contributors' net hunt energy, scavenging and metabolic
  cost over all hunter steps, with an interval from resampling episodes;
- time: the same per hunter step by what the hunter was doing. A bout is a run of
  decisions within hunt_bystander_radius of any mammoth, gaps of at most
  BOUT_GAP decisions merged, in which the hunter touched one, counted with the
  decisions before it in which the distance to the nearest mammoth kept falling.

Bout groups are descriptive, not payoffs of a choice: a bout takes its group from
its largest party, approaches that never arrive stay in the rest, the nearest
mammoth can change during one, and all of it holds for the moments the policy
chose. Metabolic cost and rabbit food of a hunter's last decision before it
starves, and of the episode's last decision, are not counted. Births move energy
to the child and are not a cost here.

    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.payoffs \\
        <checkpoint dir> <episodes> [out.json] [--energy-ratio R] [--workers N]

--energy-ratio replaces the checkpoint's hunt_energy_ratio; such policies never
trained under it.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous import evaluate
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import PARTY_SIZES, StagHuntContinuous

BOUT_GAP = 2
GROUPS = ("elsewhere", "near", "touched_no_attempt", "attempted_alone", "attempted_in_party")
BOUTS = ("touched_no_attempt", "attempted_alone", "attempted_in_party")
ATTEMPT_COUNTERS = ("hunts", "enough_energy", "kills", "contributors", "contributor_net", "kill_contributors", "kill_net", "party_energy", "mammoth_energy")
CONTACT_COUNTERS = ("decisions", "joins", "enough", "enough_joins")
GROUP_COUNTERS = ("bouts", "physics_steps", "rabbit", "hunt_net", "scavenged", "metabolic", "approach_steps", "approach_rabbit", "approach_metabolic", "kill_credits", "group_kill_credits")
_worker = {}


def new_row():
    return {
        "attempts": {str(size): dict.fromkeys(ATTEMPT_COUNTERS, 0.0) for size in range(1, PARTY_SIZES + 1)},
        "contacts": {key: dict.fromkeys(CONTACT_COUNTERS, 0.0) for key in ("alone", "with_others")},
        "groups": {group: dict.fromkeys(GROUP_COUNTERS, 0.0) for group in GROUPS},
    }


def split_bouts(near, gap=BOUT_GAP):
    """[(first, last)] index pairs of runs of True in near, runs at most gap
    apart merged."""
    runs = []
    for index, value in enumerate(near):
        if not value:
            continue
        if runs and index - runs[-1][1] - 1 <= gap:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    return [tuple(run) for run in runs]


def approach_start(distances, first, earliest=0):
    """The earliest index from which the distance kept falling up to first, not
    before earliest."""
    start = first
    while start > earliest and distances[start - 1] > distances[start]:
        start -= 1
    return start


def bout_group(records):
    """A touched bout's group by its largest party, None if the hunter touched
    no mammoth."""
    if not any(r["touching"] for r in records):
        return None
    largest = max(r["party"] for r in records)
    if not largest:
        return "touched_no_attempt"
    return "attempted_in_party" if largest > 1 else "attempted_alone"


def add_life(row, records):
    """Adds one hunter's decision records, in order, to the row's groups."""
    groups = row["groups"]
    used = [False] * len(records)
    bouts = split_bouts([r["near"] for r in records])
    distances = [r["distance"] for r in records]
    previous_end = -1
    for first, last in bouts:
        part = records[first : last + 1]
        group = bout_group(part)
        if group is not None:
            totals = groups[group]
            totals["bouts"] += 1
            for r in part:
                totals["physics_steps"] += r["physics_steps"]
                totals["rabbit"] += r["rabbit"]
                totals["hunt_net"] += r["hunt_net"]
                totals["scavenged"] += r["scavenged"]
                totals["metabolic"] += r["metabolic"]
                totals["kill_credits"] += r["kill"]
                totals["group_kill_credits"] += r["group_kill"]
            for r in records[approach_start(distances, first, previous_end + 1) : first]:
                totals["approach_steps"] += r["physics_steps"]
                totals["approach_rabbit"] += r["rabbit"]
                totals["approach_metabolic"] += r["metabolic"]
            for index in range(approach_start(distances, first, previous_end + 1), last + 1):
                used[index] = True
        previous_end = last
    for index, r in enumerate(records):
        if used[index]:
            continue
        totals = groups["near" if r["near"] else "elsewhere"]
        totals["physics_steps"] += r["physics_steps"]
        totals["rabbit"] += r["rabbit"]
        totals["hunt_net"] += r["hunt_net"]
        totals["scavenged"] += r["scavenged"]
        totals["metabolic"] += r["metabolic"]


def play_episode(env, choose, seed):
    """Plays an episode with choose(obs) -> actions. Returns its attempts, its
    groups, and the energy changes the accounts did not explain (a check)."""
    row = new_row()
    lives = {}
    accounts = {}
    in_physics = [False]
    original = {"gain": env._gain, "physics": env._physics_step, "hunts": env._hunts}

    def account(hunter_id):
        return accounts.setdefault(hunter_id, {"rabbit": 0.0, "metabolic": 0.0, "physics_steps": 0, "hunt_net": 0.0, "start": None})

    def gaining(animal, amount):
        before = animal.energy
        original["gain"](animal, amount)
        if in_physics[0] and animal.species == "predator":
            account(animal.id)["rabbit"] += animal.energy - before

    def counting(moves):
        hunters = list(env.predators)
        in_physics[0] = True
        try:
            result = original["physics"](moves)
        finally:
            in_physics[0] = False
        for hunter in hunters:
            if hunter.alive:
                totals = account(hunter.id)
                totals["physics_steps"] += 1
                totals["metabolic"] += env.resting_cost["predator"] + env.speed_cost["predator"] * hunter.velocity.mag()
        return result

    def hunting(joins, terminateds, truncateds, infos):
        parties = env.hunt_parties(joins)
        decisions = {}
        touching = {}
        for hunter in env.predators:
            if hunter.id not in joins:
                continue
            nearest = min((env.torus.distance(hunter.position, m.position) for m in env.mammoths), default=float("inf"))
            touched = env._nearest_touched(hunter)
            decisions[hunter.id] = {
                "near": nearest <= env.bystander_radius,
                "distance": nearest,
                "touching": touched is not None,
                "party": 0,
                "kill": 0,
                "group_kill": 0,
                "hunter": hunter,
                "energy": hunter.energy,
            }
            if touched is not None and env.decision >= hunter.ready_at:
                touching.setdefault(id(touched), (touched, []))[1].append(hunter)
        for mammoth, hunters in touching.values():
            enough = sum(h.energy for h in hunters) > env.energy_ratio * mammoth.energy
            counters = row["contacts"]["alone" if len(hunters) == 1 else "with_others"]
            for hunter in hunters:
                join = bool(joins[hunter.id])
                counters["decisions"] += 1
                counters["joins"] += join
                counters["enough"] += enough
                counters["enough_joins"] += enough and join
        hunts = []
        for mammoth, (party, bystanders) in parties.items():
            hunts.append((mammoth, party, sum(p.energy for p in party), mammoth.energy, [p.energy for p in party]))
            for p in party:
                decisions[p.id]["party"] = len(party)
        contributing = {p.id for _, party, *_ in hunts for p in party}
        original["hunts"](joins, terminateds, truncateds, infos)
        for mammoth, party, party_energy, mammoth_energy, before in hunts:
            counters = row["attempts"][str(min(len(party), PARTY_SIZES))]
            killed = not mammoth.alive
            net = sum(p.energy - b for p, b in zip(party, before))
            counters["hunts"] += 1
            counters["enough_energy"] += party_energy > env.energy_ratio * mammoth_energy
            counters["kills"] += killed
            counters["contributors"] += len(party)
            counters["contributor_net"] += net
            counters["party_energy"] += party_energy
            counters["mammoth_energy"] += mammoth_energy
            if killed:
                counters["kill_contributors"] += len(party)
                counters["kill_net"] += net
                for p in party:
                    decisions[p.id]["kill"] = 1
                    decisions[p.id]["group_kill"] = int(len(party) > 1)
        for hunter_id, record in decisions.items():
            hunter = record.pop("hunter")
            totals = account(hunter_id)
            change = hunter.energy - record.pop("energy")
            record.update(
                rabbit=totals["rabbit"],
                metabolic=totals["metabolic"],
                physics_steps=totals["physics_steps"],
                hunt_net=change if hunter_id in contributing else 0.0,
                scavenged=0.0 if hunter_id in contributing else change,
            )
            lives.setdefault(hunter_id, []).append(record)
            step_records[hunter_id] = record
            accounts.pop(hunter_id)

    step_records = {}
    unexplained = 0.0
    env._gain, env._physics_step, env._hunts = gaining, counting, hunting
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            start = {hunter.id: hunter.energy for hunter in env.predators}
            step_records.clear()
            obs, _, terms, truncs, infos = env.step(choose({a: obs[a] for a in sorted(obs) if a in env.active}))
            for hunter in env.predators:
                record = step_records.get(hunter.id)
                if record is not None and hunter.id in start:
                    transferred = infos.get(hunter.id, {}).get("births", 0) * env.initial_energy["predator"]
                    unexplained += abs(hunter.energy - start[hunter.id] - (record["rabbit"] - record["metabolic"] + record["hunt_net"] + record["scavenged"] - transferred))
            if terms["__all__"] or truncs["__all__"]:
                break
    finally:
        del env._gain, env._physics_step, env._hunts
    for records in lives.values():
        add_life(row, records)
    row.update(seed=seed, outcome=evaluate.outcome(env), energy_ratio=env.energy_ratio, extinct_at=dict(env.extinct_at), physics_steps=env.time_step, unexplained_energy=unexplained)
    return row


def rate(totals, approach=True):
    """Net energy per 1000 hunter steps of a group's totals, scavenging
    apart, None if no steps."""
    steps = totals["physics_steps"] + (totals["approach_steps"] if approach else 0)
    if not steps:
        return None
    net = totals["rabbit"] + totals["hunt_net"] - totals["metabolic"]
    if approach:
        net += totals["approach_rabbit"] - totals["approach_metabolic"]
    return 1000 * net / steps


def merged(rows, groups):
    """Summed totals of the named groups over rows."""
    totals = dict.fromkeys(GROUP_COUNTERS, 0.0)
    for row in rows:
        for group in groups:
            for key in GROUP_COUNTERS:
                totals[key] += row["groups"][group][key]
    return totals


def budget(rows):
    """The hunters' energy over rows, in total and per 1000 hunter steps."""
    totals = merged(rows, GROUPS)
    steps = totals["physics_steps"] + totals["approach_steps"]
    amounts = {
        "rabbit": totals["rabbit"] + totals["approach_rabbit"],
        "hunt_net": totals["hunt_net"],
        "scavenged": totals["scavenged"],
        "metabolic": totals["metabolic"] + totals["approach_metabolic"],
    }
    return dict(amounts, hunter_steps=steps, **{f"{key}_per_1000_steps": 1000 * value / steps if steps else None for key, value in amounts.items()})


def hunt_net_interval(rows, draws=5000, seed=0):
    """95% interval of contributors' net hunt energy per 1000 hunter steps,
    resampling episodes."""
    rng = np.random.default_rng(seed)
    values = [budget([rows[i] for i in rng.integers(len(rows), size=len(rows))])["hunt_net_per_1000_steps"] for _ in range(draws)]
    return tuple(float(x) for x in np.percentile([v for v in values if v is not None], (2.5, 97.5)))


def summarise(rows):
    outcomes = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    attempts = {}
    for size in range(1, PARTY_SIZES + 1):
        totals = {key: sum(row["attempts"][str(size)][key] for row in rows) for key in ATTEMPT_COUNTERS}
        hunts = totals["hunts"]
        attempts[str(size)] = dict(
            totals,
            kill_rate=totals["kills"] / hunts if hunts else None,
            enough_energy_rate=totals["enough_energy"] / hunts if hunts else None,
            net_per_contributor=totals["contributor_net"] / totals["contributors"] if hunts else None,
            net_per_contributor_at_kill=totals["kill_net"] / totals["kill_contributors"] if totals["kill_contributors"] else None,
            mean_party_energy=totals["party_energy"] / hunts if hunts else None,
            mean_mammoth_energy=totals["mammoth_energy"] / hunts if hunts else None,
        )
    contacts = {}
    for key in ("alone", "with_others"):
        totals = {name: sum(row["contacts"][key][name] for row in rows) for name in CONTACT_COUNTERS}
        lacking = totals["decisions"] - totals["enough"]
        contacts[key] = dict(
            totals,
            enough_share=totals["enough"] / totals["decisions"] if totals["decisions"] else None,
            join_rate_enough=totals["enough_joins"] / totals["enough"] if totals["enough"] else None,
            join_rate_lacking=(totals["joins"] - totals["enough_joins"]) / lacking if lacking else None,
        )
    steps = sum(merged(rows, GROUPS)[key] for key in ("physics_steps", "approach_steps"))
    groups = {}
    for group in GROUPS:
        totals = merged(rows, (group,))
        spent = totals["physics_steps"] + totals["approach_steps"]
        groups[group] = dict(
            totals,
            time_share=spent / steps if steps else None,
            net_per_1000_steps=rate(totals),
            net_per_1000_steps_without_approach=rate(totals, approach=False),
            rabbit_per_1000_steps=1000 * (totals["rabbit"] + totals["approach_rabbit"]) / spent if spent else None,
        )
    return {
        "episodes": len(rows),
        "energy_ratio": rows[0]["energy_ratio"],
        "outcomes": outcomes,
        "unexplained_energy": sum(row["unexplained_energy"] for row in rows),
        "attempts": attempts,
        "contacts": contacts,
        "budget": budget(rows),
        "hunt_net_interval": hunt_net_interval(rows),
        "groups": groups,
    }


def _init(checkpoint, energy_ratio=None):
    import torch

    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.checkpoints import checkpoint_env_config, load_modules

    torch.set_num_threads(1)
    config = checkpoint_env_config(checkpoint)
    if energy_ratio is not None:
        config["hunt_energy_ratio"] = energy_ratio
    config["truncate_on_prey_extinction"] = False
    _worker["modules"] = load_modules(checkpoint)
    _worker["env"] = StagHuntContinuous(config)


def _play(seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    return play_episode(_worker["env"], lambda obs: evaluate.sample_actions(_worker["modules"], obs, generator), seed)


def show(number, width=8, digits=2):
    return " " * (width - 1) + "-" if number is None else f"{number:{width}.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("episodes", type=int)
    parser.add_argument("out", nargs="?")
    parser.add_argument("--energy-ratio", type=float)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be at least 1")
    if args.workers < 1:
        parser.error("workers must be at least 1")
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init, initargs=(args.checkpoint, args.energy_ratio)) as pool:
        rows = list(pool.map(_play, range(args.episodes)))
    summary = summarise(rows)
    print(f"outcomes {summary['outcomes']} unexplained energy {summary['unexplained_energy']:.3f}")
    print("party    hunts   enough    kills  net/hunter  at kill  party_E  mammoth_E")
    for size, a in summary["attempts"].items():
        print(f"{size:>5s} {a['hunts']:8.0f} {show(a['enough_energy_rate'])} {show(a['kill_rate'])}    {show(a['net_per_contributor'])} {show(a['net_per_contributor_at_kill'])} {show(a['mean_party_energy'])}   {show(a['mean_mammoth_energy'])}")
    print("contact        decisions  enough share  join if enough  join if lacking")
    for key, c in summary["contacts"].items():
        print(f"{key:12s} {c['decisions']:10.0f}      {show(c['enough_share'], digits=3)}        {show(c['join_rate_enough'], digits=3)}         {show(c['join_rate_lacking'], digits=3)}")
    b = summary["budget"]
    print(f"per 1000 hunter steps: rabbit {show(b['rabbit_per_1000_steps'])} hunt net {show(b['hunt_net_per_1000_steps'])} {summary['hunt_net_interval']} scavenged {show(b['scavenged_per_1000_steps'])} metabolic {show(b['metabolic_per_1000_steps'])}")
    print("group                  bouts  time share  net/1000  no approach  rabbit/1000  hunt_net")
    for group, g in summary["groups"].items():
        print(f"{group:20s} {g['bouts']:7.0f}    {show(g['time_share'], digits=3)}  {show(g['net_per_1000_steps'])}     {show(g['net_per_1000_steps_without_approach'])}     {show(g['rabbit_per_1000_steps'])} {g['hunt_net']:9.0f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"checkpoint": args.checkpoint, "summary": summary, "rows": rows}))


if __name__ == "__main__":
    main()
