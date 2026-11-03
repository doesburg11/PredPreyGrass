"""Do hunters turn to mammoths in the lean season? Seeded evaluation episodes
(same seeds and action sampling as evaluate.py, whole episodes), counted by
phase of the rabbits' grass cycle: the first half of a cycle is the grass's
good season, the second half its lean one. Rabbit numbers follow with a lag,
so read the phases in turn, not the two halves. Descriptive: phases differ in
rabbit numbers, hunter numbers and condition together.

The first cycle is left out by default: it starts from the initial
populations, not from a previous season.

--period, --amplitude and --start replace the checkpoint's seasons, to see what a
seasonal world does to policies trained with or without them (observations do
not change). Such policies never trained in that world. --rabbits replaces
the number of rabbits at the start.

    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.seasons \\
        <checkpoint dir> <episodes> [out.json] [--period P] [--amplitude A] [--start S] [--rabbits R] [--skip-cycles N] [--workers N]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous import evaluate
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import PARTY_SIZES, StagHuntContinuous

PHASES = 8
COUNTERS = (
    "physics_steps",
    "grass_steps",
    "rabbit_steps",
    "hunter_steps",
    "mammoth_steps",
    "catches",
    "kills",
    "group_kills",
    "rabbit_energy",
    "mammoth_energy",
)
_worker = {}


def cycle_phase(time_step, period, start=0):
    """(whole cycles passed, phase index) of a physics step; cycles before
    start are negative."""
    time_step -= start
    return time_step // period, (time_step % period) * PHASES // period


def kills(stats, smallest=1):
    return sum(stats[f"kills_party_{size}"] for size in range(smallest, PARTY_SIZES + 1))


def play_episode(env, choose, seed):
    """Plays an episode with choose(obs) -> actions. Populations, edible
    patches and catches count at each physics step, in that step's phase;
    hunts resolve at the end of a decision, so kills and mammoth food count in
    the phase of its last physics step. Returns the counters per cycle and
    phase."""
    grass = env.food["grass"]
    period = grass.season_period
    if period < PHASES:
        raise ValueError(f"seasons needs a world with a grass_season_period of at least {PHASES}")
    cycles = {}

    def row_at(time_step):
        cycle, phase = cycle_phase(time_step, period, grass.season_start)
        return cycles.setdefault(cycle, [dict.fromkeys(COUNTERS, 0.0) for _ in range(PHASES)])[phase]

    physics_step = env._physics_step

    def counting(moves):
        stats = env.hunt_stats
        row = row_at(env.time_step)
        row["physics_steps"] += 1
        row["grass_steps"] += sum(grass.edible(patch) for patch in grass.patches)
        row["rabbit_steps"] += len(env.rabbits)
        row["hunter_steps"] += len(env.predators)
        row["mammoth_steps"] += len(env.mammoths)
        caught, energy = stats["rabbits_caught"], stats["rabbit_energy"]
        result = physics_step(moves)
        row["catches"] += stats["rabbits_caught"] - caught
        row["rabbit_energy"] += stats["rabbit_energy"] - energy
        return result

    env._physics_step = counting
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            stats = env.hunt_stats
            before = kills(stats), kills(stats, 2), stats["mammoth_energy"]
            obs, _, terms, truncs, _ = env.step(choose({a: obs[a] for a in sorted(obs) if a in env.active}))
            row = row_at(env.time_step - 1)
            row["kills"] += kills(stats) - before[0]
            row["group_kills"] += kills(stats, 2) - before[1]
            row["mammoth_energy"] += stats["mammoth_energy"] - before[2]
            if terms["__all__"] or truncs["__all__"]:
                break
    finally:
        del env._physics_step
    return {
        "seed": seed,
        "outcome": evaluate.outcome(env),
        "period": period,
        "amplitude": grass.season_amplitude,
        "start": grass.season_start,
        "rabbit_count": env.count["rabbit"],
        "extinct_at": dict(env.extinct_at),
        "grass_eaten": grass.total_consumed,
        "physics_steps": env.time_step,
        "cycles": {str(cycle): rows for cycle, rows in cycles.items()},
    }


def summarise(rows, skip_cycles=1):
    """Per phase over all episodes' cycles from skip_cycles on: mean
    populations and edible patches over the physics steps played, prey taken
    and food per hunter step x 1000, mammoths' share of the hunters' food.
    None where nothing was played or eaten."""
    pooled = [dict.fromkeys(COUNTERS, 0.0) for _ in range(PHASES)]
    for row in rows:
        for cycle, phases in row["cycles"].items():
            if int(cycle) >= skip_cycles:
                for into, values in zip(pooled, phases):
                    for key in COUNTERS:
                        into[key] += values[key]
    phases = []
    for phase, row in enumerate(pooled):
        steps, hunters = row["physics_steps"], row["hunter_steps"]
        food = row["rabbit_energy"] + row["mammoth_energy"]
        phases.append(
            {
                "phase": phase,
                "physics_steps": steps,
                "grass": row["grass_steps"] / steps if steps else None,
                "rabbits": row["rabbit_steps"] / steps if steps else None,
                "hunters": hunters / steps if steps else None,
                "mammoths": row["mammoth_steps"] / steps if steps else None,
                "catches_per_1000_hunter_steps": 1000 * row["catches"] / hunters if hunters else None,
                "kills_per_1000_hunter_steps": 1000 * row["kills"] / hunters if hunters else None,
                "group_kills": row["group_kills"],
                "food_per_1000_hunter_steps": 1000 * food / hunters if hunters else None,
                "mammoth_share": row["mammoth_energy"] / food if food else None,
            }
        )
    outcomes = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    return {
        "episodes": len(rows),
        "skip_cycles": skip_cycles,
        "period": rows[0]["period"],
        "amplitude": rows[0]["amplitude"],
        "start": rows[0]["start"],
        "outcomes": outcomes,
        "grass_eaten_per_1000_steps": 1000 * sum(row["grass_eaten"] for row in rows) / sum(row["physics_steps"] for row in rows),
        "phases": phases,
    }


def _init(checkpoint, period, amplitude, start, rabbits):
    import torch

    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.checkpoints import checkpoint_env_config, load_modules

    torch.set_num_threads(1)
    config = checkpoint_env_config(checkpoint)
    if period is not None:
        config["grass_season_period"] = period
    if amplitude is not None:
        config["grass_season_amplitude"] = amplitude
    if start is not None:
        config["grass_season_start"] = start
    if rabbits is not None:
        config["rabbit_count"] = rabbits
    config["truncate_on_prey_extinction"] = False
    _worker["modules"] = load_modules(checkpoint)
    _worker["env"] = StagHuntContinuous(config)


def _play(seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    return play_episode(_worker["env"], lambda obs: evaluate.sample_actions(_worker["modules"], obs, generator), seed)


def show(number):
    return "       -" if number is None else f"{number:8.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("episodes", type=int)
    parser.add_argument("out", nargs="?")
    parser.add_argument("--period", type=int)
    parser.add_argument("--amplitude", type=float)
    parser.add_argument("--start", type=int)
    parser.add_argument("--rabbits", type=int)
    parser.add_argument("--skip-cycles", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be at least 1")
    if args.skip_cycles < 0:
        parser.error("skip-cycles must not be negative")
    if args.workers < 1:
        parser.error("workers must be at least 1")
    context = get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=_init, initargs=(args.checkpoint, args.period, args.amplitude, args.start, args.rabbits)) as pool:
        rows = list(pool.map(_play, range(args.episodes)))
    summary = summarise(rows, args.skip_cycles)
    print(f"period {summary['period']} amplitude {summary['amplitude']} start {summary['start']} outcomes {summary['outcomes']} grass eaten per 1000 steps (whole episodes) {summary['grass_eaten_per_1000_steps']:.1f}")
    keys = ("grass", "rabbits", "hunters", "mammoths", "catches_per_1000_hunter_steps", "kills_per_1000_hunter_steps", "food_per_1000_hunter_steps", "mammoth_share")
    print("phase   " + " ".join(f"{key[:8]:>8s}" for key in keys) + "  group_kills")
    for phase in summary["phases"]:
        print(f"{phase['phase']:5d}   " + " ".join(show(phase[key]) for key in keys) + f"  {phase['group_kills']:.0f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"checkpoint": args.checkpoint, "summary": summary, "rows": rows}))


if __name__ == "__main__":
    main()
