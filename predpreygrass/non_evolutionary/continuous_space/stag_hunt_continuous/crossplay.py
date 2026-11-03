"""Cross-play: hunters, rabbits and mammoths from two checkpoints of the same
world, in every combination, to separate which species' learning goes with
a change between the checkpoints. Episodes as in evaluate.py (same seeds,
same action sampling, whole episodes), so the cells where all three come
from one checkpoint repeat that checkpoint's evaluation.

The policies never trained against the mixed partners: a cell shows how
these policies meet, not what would have been learned together, and a
population that cannot cope may collapse for that reason alone.

    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.crossplay \\
        <checkpoint A> <checkpoint B> <episodes> [out.json] [--workers N]
"""

import argparse
import itertools
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous import evaluate, prey_switching
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import SPECIES, StagHuntContinuous

SOURCES = ("A", "B")
# Rabbit-count bins of the printed rates and the paired contrasts.
SHOWN_BINS = ("11-20", "21-40")
_worker = {}


def cells():
    """Every assignment {species: source}, as tuples in SPECIES order."""
    return list(itertools.product(SOURCES, repeat=len(SPECIES)))


def cell_modules(modules, cell):
    """The policies of a cell, from modules {source: {species: module}}."""
    return {species: modules[source][species] for species, source in zip(SPECIES, cell)}


def same_world(config_a, config_b):
    """The worlds must match, or a cell would run a policy outside its own."""
    if config_a != config_b:
        differ = sorted(k for k in set(config_a) | set(config_b) if config_a.get(k) != config_b.get(k))
        raise ValueError(f"the checkpoints' worlds differ in {differ}")
    return config_a


def _init(checkpoint_a, checkpoint_b):
    import torch

    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.checkpoints import checkpoint_env_config, load_modules

    torch.set_num_threads(1)
    config = dict(same_world(checkpoint_env_config(checkpoint_a), checkpoint_env_config(checkpoint_b)))
    config["truncate_on_prey_extinction"] = False
    _worker["env"] = StagHuntContinuous(config)
    _worker["modules"] = {"A": load_modules(checkpoint_a), "B": load_modules(checkpoint_b)}


def _play(job):
    import torch

    cell, seed = job
    modules = cell_modules(_worker["modules"], cell)
    played = evaluate.play(_worker["env"], modules, seed)
    generator = torch.Generator().manual_seed(seed)
    measured = prey_switching.play_episode(_worker["env"], lambda obs: evaluate.sample_actions(modules, obs, generator), seed)
    same_episode(played, measured)
    return cell, played, measured


def same_episode(played, measured):
    """The two plays of a seed must be the same episode."""
    if played["outcome"] != measured["outcome"] or played["hunt_stats"]["rabbits_caught"] != measured["hunt_stats"]["rabbits_caught"]:
        raise ValueError(f"seed {played['seed']}: the two plays differ")


def paired_ratio(rows_a, rows_b, key, denominator, bin_index, seed=0, repeats=2000):
    """Ratio of a rate between two cells in a rabbit-count bin (key over
    denominator, pooled over episodes), with a 95% interval resampling the
    seeds of both together. Rows are in seed order."""
    def rate(rows, weights):
        total = sum(w * row["bins"][bin_index][denominator] for w, row in zip(weights, rows))
        return sum(w * row["bins"][bin_index][key] for w, row in zip(weights, rows)) / total if total else math.nan

    if [row["seed"] for row in rows_a] != [row["seed"] for row in rows_b]:
        raise ValueError("the cells' seeds differ")
    n = len(rows_a)
    one = [1] * n
    denominator_b = rate(rows_b, one)
    point = rate(rows_a, one) / denominator_b if denominator_b else math.nan
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        weights = rng.multinomial(n, [1 / n] * n)
        b = rate(rows_b, weights)
        samples.append(rate(rows_a, weights) / b if b else math.nan)
    samples = np.array(samples)
    samples = samples[np.isfinite(samples)]
    return point, [float(q) for q in np.quantile(samples, (0.025, 0.975))] if len(samples) else None


def summarise_cell(cell, played, measured):
    summary = evaluate.summarise(played)
    totals = summary["totals"]
    food = totals["mammoth_energy"] + totals["rabbit_energy"]
    alive = Counter()
    for row in played:
        for species in SPECIES:
            alive[species] += species not in row["extinct_at"]
    return {
        "cell": dict(zip(SPECIES, cell)),
        "episodes": summary["episodes"],
        "outcomes": summary["outcomes"],
        "coexisting": summary["coexisting"],
        "survived": dict(alive),
        "mammoth_food_share": totals["mammoth_energy"] / food if food else None,
        "rabbits_caught_per_episode": totals["rabbits_caught"] / summary["episodes"],
        "joint_arrival_rate": summary["joint_arrival_rate"],
        "joint_arrival_rate_chance": summary["joint_arrival_rate_chance"],
        "contact_with_partner_join_rate": summary["contact_with_partner_join_rate"],
        "contact_alone_join_rate": summary["contact_alone_join_rate"],
        "switching": prey_switching.summarise(measured),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_a")
    parser.add_argument("checkpoint_b")
    parser.add_argument("episodes", type=int)
    parser.add_argument("out", nargs="?")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    jobs = [(cell, seed) for cell in cells() for seed in range(args.episodes)]
    played, measured = {cell: [] for cell in cells()}, {cell: [] for cell in cells()}
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init, initargs=(args.checkpoint_a, args.checkpoint_b)) as pool:
        for cell, row, measure in pool.map(_play, jobs):
            played[cell].append(row)
            measured[cell].append(measure)
    results = [summarise_cell(cell, played[cell], measured[cell]) for cell in cells()]
    print("A:", args.checkpoint_a, " B:", args.checkpoint_b)
    print(f"{'H R M':7}{'coexist':>8}{'H out':>6}{'R out':>6}{'M out':>6}{'mam food':>9}{'caught/ep':>10}{'joint':>7}{'chance':>7}{'rate@11-20':>11}{'rate@21-40':>11}{'pursuit@11-20':>14}{'pursuit@21-40':>14}")

    def number(value, width, digits):
        return f"{'-':>{width}}" if value is None else f"{value:>{width}.{digits}f}"

    for row in results:
        bins = {entry["rabbits"]: entry for entry in row["switching"]["bins"]}
        print(f"{' '.join(row['cell'][s] for s in ('predator', 'rabbit', 'mammoth')):7}{row['coexisting']:>8}"
              + "".join(f"{row['episodes'] - row['survived'][s]:>6}" for s in ("predator", "rabbit", "mammoth"))
              + number(row["mammoth_food_share"], 9, 2) + f"{row['rabbits_caught_per_episode']:>10.1f}"
              + number(row["joint_arrival_rate"], 7, 3) + number(row["joint_arrival_rate_chance"], 7, 3)
              + "".join(number(bins[b]["catch_rate_x1e5"], 11, 1) for b in SHOWN_BINS)
              + "".join(number(bins[b]["pursuit"], 14, 3) for b in SHOWN_BINS))
    order = [b for b in range(len(prey_switching.RABBIT_BINS)) if prey_switching.bin_label(b) in SHOWN_BINS]
    contrasts = []
    # One species' source changed from B to A, the others as named.
    for name, position in (("rabbit", 1), ("predator", 0), ("mammoth", 2)):
        for cell in cells():
            if cell[position] != "A":
                continue
            other = tuple("B" if i == position else source for i, source in enumerate(cell))
            for b in order:
                for key, denominator, label in (("catches", "rabbit_hunter_steps", "catch rate"), ("pursuit", "sightings", "pursuit")):
                    ratio, interval = paired_ratio(measured[cell], measured[other], key, denominator, b)
                    contrasts.append({"species": name, "cell": "".join(cell), "against": "".join(other), "rabbits": prey_switching.bin_label(b), "measure": label, "ratio": ratio, "ci": interval})
    print("paired ratios, one species' source changed (A over B):")
    for row in contrasts:
        print(f"  {row['species']:9}{row['cell']}/{row['against']}  {row['rabbits']:6}{row['measure']:11}{row['ratio']:.2f} {row['ci'] and [round(q, 2) for q in row['ci']]}")
    if args.out:
        Path(args.out).write_text(json.dumps({"checkpoint_a": args.checkpoint_a, "checkpoint_b": args.checkpoint_b, "cells": results, "contrasts": contrasts, "episodes": {"".join(cell): played[cell] for cell in cells()}}, indent=1))


if __name__ == "__main__":
    main()
