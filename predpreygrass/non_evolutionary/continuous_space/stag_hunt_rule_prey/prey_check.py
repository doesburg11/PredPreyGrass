"""What do the prey's rules do on their own? Seeded episodes with one harmless
hunter (it stands still, never joins, pays no costs, has no young, catches
nothing and frightens nothing), so the populations show the world's own limits: rabbits, mammoths, mammoth
energy and edible patches, by stretch of the episode. Hunters that hunt change
all of this; this only checks the prey before training anything.

--set replaces settings, e.g. --set grass_respawn_delay=800.

    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.prey_check \\
        <episodes> [out.json] [--set KEY=VALUE ...] [--workers N]
"""

import argparse
import ast
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import MOVES, StagHuntRulePrey

STRETCH = 1000  # physics steps
STOP = MOVES - 1
IDLE_HUNTER = {
    "predator_count": 1,
    "reproduction_max_predator": 1,
    "energy_predator_resting_metabolic_cost": 0.0,
    "energy_predator_speed_cost": 0.0,
    "truncate_on_prey_extinction": False,
    "rabbit_flee_distance": 0.0,
    "mammoth_flee_distance": 0.0,
}
_worker = {}


def play_episode(env, seed):
    """Per STRETCH physics steps: means of rabbits, mammoths, mammoth energy
    and edible patches of both layers, sampled at every decision."""
    env.reset(seed=seed)
    samples = {}
    while True:
        row = samples.setdefault(env.time_step // STRETCH, [])
        energies = [m.energy for m in env.mammoths]
        row.append(
            (
                len(env.rabbits),
                len(env.mammoths),
                float(np.mean(energies)) if energies else float("nan"),
                sum(env.food["grass"].edible(p) for p in env.food["grass"].patches),
                sum(env.food["steppe"].edible(p) for p in env.food["steppe"].patches),
            )
        )
        _, _, terms, truncs, _ = env.step({a: np.array([STOP, 0]) for a in env.active})
        if terms["__all__"] or truncs["__all__"]:
            break
    return {
        "seed": seed,
        "extinct_at": dict(env.extinct_at),
        "births": dict(env.births),
        "stretches": {str(k): [float(x) for x in np.nanmean(np.array(v, dtype=float), axis=0)] if np.isfinite(np.array(v, dtype=float)).any() else None for k, v in sorted(samples.items())},
    }


def summarise(rows):
    stretches = sorted({int(k) for row in rows for k in row["stretches"]})
    table = {}
    for stretch in stretches:
        values = np.array([row["stretches"][str(stretch)] for row in rows if row["stretches"].get(str(stretch))], dtype=float)
        table[str(stretch)] = [float(x) for x in np.nanmean(values, axis=0)]
    return {
        "episodes": len(rows),
        "rabbits_out": sum("rabbit" in row["extinct_at"] for row in rows),
        "mammoths_out": sum("mammoth" in row["extinct_at"] for row in rows),
        "births": {s: float(np.mean([row["births"][s] for row in rows])) for s in ("rabbit", "mammoth")},
        "by_stretch": table,
    }


def harmless_world(config):
    """The world with its hunter unable to catch a rabbit on contact."""
    env = StagHuntRulePrey(config)
    contacts = env._contacts

    def no_catch(animal):
        return None, contacts(animal)[1]

    env._contacts = no_catch
    return env


def _init(config):
    _worker["env"] = harmless_world(config)


def _play(seed):
    return play_episode(_worker["env"], seed)


def parse_setting(text):
    key, _, value = text.partition("=")
    if not value:
        raise argparse.ArgumentTypeError(f"{text}: expected KEY=VALUE")
    try:
        return key, ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return key, value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes", type=int)
    parser.add_argument("out", nargs="?")
    parser.add_argument("--set", dest="settings", type=parse_setting, action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = dict(IDLE_HUNTER, **dict(args.settings))
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init, initargs=(config,)) as pool:
        rows = list(pool.map(_play, range(args.episodes)))
    summary = summarise(rows)
    print(f"settings {dict(args.settings)}: rabbits out {summary['rabbits_out']}, mammoths out {summary['mammoths_out']} of {summary['episodes']}; births per episode {summary['births']}")
    print("from step  rabbits  mammoths  mammoth E    grass   steppe")
    for stretch, (rabbits, mammoths, energy, grass, steppe) in summary["by_stretch"].items():
        print(f"{int(stretch) * STRETCH:9d} {rabbits:8.1f} {mammoths:9.1f} {energy:10.1f} {grass:8.1f} {steppe:8.1f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"settings": dict(args.settings), "summary": summary, "rows": rows}))


if __name__ == "__main__":
    main()
