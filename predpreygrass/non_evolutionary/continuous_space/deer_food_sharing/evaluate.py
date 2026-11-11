"""Evaluation of trained (or untrained) policies: whether predators and deer
coexist to the end of each episode, and the catch counters.

Plays seeded episodes 0, 1, ... with actions sampled from the policies'
distributions as in training (not a replay of training's samples: one torch
generator per episode, seeded with the episode's seed, agents in sorted
order), in parallel processes.

    python -m predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.evaluate \\
        <checkpoint dir | untrained> <episodes> [out.json] [--workers N] [--set KEY=JSON ...]

With `untrained`, the policies are freshly initialised (seed 0) in the world
of config/config_env.py. --set changes settings of the world the policies
play in (a setting that changes the observations does not fit a checkpoint
trained without it).
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.callbacks import derived_rates
from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.environment import SPECIES, DeerStagHunt, new_stats, species_of

_worker = {}


def sample_actions(modules, obs, generator):
    """Every agent's action, one batched forward pass per species."""
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
        move = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(1)
        actions.update(zip(agents, move.tolist()))
    return actions


def outcome(env):
    """The species still alive at the end."""
    alive = [s for s in SPECIES if env.animals[s]]
    return "coexisting" if len(alive) == len(SPECIES) else "only_" + "_".join(alive) if alive else "none"


def play(env, modules, seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, terms, truncs, _ = env.step(sample_actions(modules, {a: obs[a] for a in sorted(obs) if a in env.active}, generator))
        if terms["__all__"] or truncs["__all__"]:
            break
    return {
        "seed": seed,
        "outcome": outcome(env),
        "physics_steps": env.time_step,
        "extinct_at": dict(env.extinct_at),
        "stats": env.episode_stats,
    }


def _init(checkpoint, overrides=None):
    import torch

    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.checkpoints import checkpoint_env_config, load_modules, untrained_modules

    torch.set_num_threads(1)
    if checkpoint == "untrained":
        from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.config.config_env import config_env

        config = dict(config_env)
        _worker["modules"] = untrained_modules(config, seed=0)
    else:
        config = checkpoint_env_config(checkpoint)
        _worker["modules"] = load_modules(checkpoint)
    config.update(overrides or {})
    _worker["env"] = DeerStagHunt(config)


def _play(seed):
    return play(_worker["env"], _worker["modules"], seed)


def summarise(rows):
    """Totals over episodes: outcomes, coexistence to the last physics step,
    catch counters and rates."""
    outcomes = Counter(row["outcome"] for row in rows)
    totals = Counter()
    for row in rows:
        totals.update(row["stats"] or new_stats())
    return {
        "episodes": len(rows),
        "outcomes": dict(outcomes),
        # Both species alive at the episode's last physics step (an episode
        # ends early only when a species dies out).
        "coexisting": outcomes["coexisting"],
        "totals": dict(totals),
        **derived_rates(totals),
        "extinction_steps": {s: sorted(row["extinct_at"][s] for row in rows if s in row["extinct_at"]) for s in SPECIES},
    }


def parse_overrides(items):
    """{key: value} from KEY=JSON strings (--set)."""
    overrides = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set needs KEY=JSON, not {item!r}")
        overrides[key] = json.loads(value)
    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("episodes", type=int)
    parser.add_argument("out", nargs="?")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    args = parser.parse_args()
    overrides = parse_overrides(args.set)
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init, initargs=(args.checkpoint, overrides)) as pool:
        rows = sorted(pool.map(_play, range(args.episodes)), key=lambda row: row["seed"])
    summary = {"checkpoint": str(args.checkpoint), "overrides": overrides, **summarise(rows)}
    for key, value in summary.items():
        if key != "totals":
            print(f"{key}: {round(value, 3) if isinstance(value, float) else value}")
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))


if __name__ == "__main__":
    main()
