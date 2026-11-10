"""Evaluation of a checkpoint of any arm: whether hunters and deer coexist
to the end of each episode, the catch counters and the hunters' genes.

Every episode starts from the checkpoint's gene pool (pooled over its
environments; founders drawn with the episode's seed), with mutation at
births as in training. Plays seeded episodes 0, 1, ... with actions sampled
from the policies' distributions as in training (one torch generator per
episode, seeded with the episode's seed, agents in sorted order), in
parallel processes.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.evaluate \\
        <checkpoint dir> <episodes> [out.json] [--workers N] [--set KEY=JSON ...]
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.callbacks import derived_rates
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.environment import SPECIES, EvolvingDeerStagHunt, new_stats, species_of

_worker = {}


def sample_actions(actors, obs, generator):
    """Every agent's action, one batched pass per species."""
    import torch

    actions = {}
    for species in SPECIES:
        agents = [agent for agent in obs if species_of(agent) == species]
        if not agents:
            continue
        batch = np.stack([np.asarray(obs[agent], dtype=np.float32) for agent in agents])
        logits = actors[species].logits(batch).to(torch.float64)
        move = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(1)
        actions.update(zip(agents, move.tolist()))
    return actions


def outcome(env):
    """The species still alive at the end."""
    alive = [s for s in SPECIES if env.animals[s]]
    return "coexisting" if len(alive) == len(SPECIES) else "only_" + "_".join(alive) if alive else "none"


def play(env, actors, seed, pool=None):
    """One episode; pool: (genomes, generations) its founders are drawn from."""
    import torch

    if pool is not None:
        env.set_founder_pool(*pool)
    generator = torch.Generator().manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, terms, truncs, _ = env.step(sample_actions(actors, {a: obs[a] for a in sorted(obs) if a in env.active}, generator))
        if terms["__all__"] or truncs["__all__"]:
            break
    return {
        "seed": seed,
        "outcome": outcome(env),
        "physics_steps": env.time_step,
        "extinct_at": dict(env.extinct_at),
        "stats": env.episode_stats,
    }


def _init(checkpoint, overrides=None, deer=None):
    import torch

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import evaluation_env_config, load_actors
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.environment import pooled_gene_pool

    torch.set_num_threads(1)
    config = evaluation_env_config(checkpoint, overrides)
    _worker["env"] = EvolvingDeerStagHunt(config)
    _worker["pool"] = pooled_gene_pool(config["gene_initial_pool"])
    _worker["actors"] = load_actors(checkpoint, _worker["env"], deer)


def _play(seed):
    return play(_worker["env"], _worker["actors"], seed, _worker["pool"])


GENE_KEYS = ("_mean", "_selection")


def summarise(rows):
    """Totals over episodes: outcomes, coexistence to the last physics step,
    catch counters and rates, adults caught per episode, and the episodes'
    mean gene statistics."""
    outcomes = Counter(row["outcome"] for row in rows)
    totals = Counter()
    genes = {}
    for row in rows:
        stats = row["stats"] or new_stats()
        totals.update({k: v for k, v in stats.items() if not k.startswith("gene") and k != "generation_mean"})
        for key, value in stats.items():
            if key.startswith("gene") or key == "generation_mean":
                genes.setdefault(key, []).append(value)
    return {
        "episodes": len(rows),
        "outcomes": dict(outcomes),
        # Both species alive at the episode's last physics step (an episode
        # ends early only when a species dies out).
        "coexisting": outcomes["coexisting"],
        "totals": dict(totals),
        "adults_caught_per_episode": totals["adults_caught"] / len(rows) if rows else None,
        **derived_rates(totals),
        "genes": {key: float(np.mean(values)) for key, values in sorted(genes.items())},
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


def evaluate(checkpoint, episodes, workers=10, overrides=None, deer=None):
    """deer: a deer module directory to act instead of the checkpoint's own
    (checkpoints.checkpoint_deer)."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import checkpoint_deer

    deer_used = str(deer or checkpoint_deer(checkpoint))
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_init, initargs=(str(checkpoint), overrides, deer and str(deer))) as pool:
        rows = sorted(pool.map(_play, range(episodes)), key=lambda row: row["seed"])
    return {"checkpoint": str(checkpoint), "deer": deer_used, "overrides": overrides or {}, **summarise(rows)}, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("episodes", type=int)
    parser.add_argument("out", nargs="?")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    args = parser.parse_args()
    summary, rows = evaluate(args.checkpoint, args.episodes, args.workers, parse_overrides(args.set))
    for key, value in summary.items():
        if key != "totals":
            print(f"{key}: {round(value, 3) if isinstance(value, float) else value}")
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))


if __name__ == "__main__":
    main()
