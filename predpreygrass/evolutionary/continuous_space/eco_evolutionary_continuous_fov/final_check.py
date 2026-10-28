"""Final check of a trained checkpoint, the sustainability guard of
DESIGN_fov_tradeoff.md (test 3: both species coexist in at least 45 of 50
episodes).

Plays the checkpoint's policies (tournament.play_episode: actions sampled as
in training, one torch generator per episode seeded with the episode's seed)
for seeded episodes 0, 1, ..., with founders drawn from the checkpoint's
gene pool, held fixed. Episodes run in parallel processes, one environment
and one set of policies per process.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.final_check \\
        <checkpoint dir> <episodes> [out.json] [--workers N]
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

_worker = {}


def _init(checkpoint):
    import torch

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.checkpoints import (
        checkpoint_env_config,
        load_gene_pool,
        load_modules,
        use_gene_pool,
    )
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.environment import PredPreyGrassContinuous

    torch.set_num_threads(1)
    env = PredPreyGrassContinuous(checkpoint_env_config(checkpoint))
    use_gene_pool(env, load_gene_pool(checkpoint))
    _worker.update(env=env, modules=load_modules(checkpoint))


def _play(seed):
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.tournament import play_episode

    return dict(seed=seed, **play_episode(_worker["env"], _worker["modules"], seed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("episodes", type=int)
    parser.add_argument("out", nargs="?")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init, initargs=(args.checkpoint,)) as pool:
        rows = sorted(pool.map(_play, range(args.episodes)), key=lambda row: row["seed"])
    outcomes = Counter(row["outcome"] for row in rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "episodes": args.episodes,
        "coexisting": outcomes["time_limit"],
        "prey_extinct": outcomes["prey_extinct"],
        "predators_extinct": outcomes["predators_extinct"],
        "both_extinct": outcomes["both_extinct"],
        "catch_risk": 1000 * sum(r["prey_caught"] for r in rows) / sum(r["prey_exposure"] for r in rows),
        "mean_predators": float(np.mean([r["mean_predators"] for r in rows])),
        "mean_prey": float(np.mean([r["mean_prey"] for r in rows])),
        "predator_births": float(np.mean([r["predator_births"] for r in rows])),
        "prey_births": float(np.mean([r["prey_births"] for r in rows])),
    }
    for key, value in summary.items():
        print(f"{key}: {round(value, 3) if isinstance(value, float) else value}")
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))


if __name__ == "__main__":
    main()
