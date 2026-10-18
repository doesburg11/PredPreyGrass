"""Separate learning from body evolution in a run's tournament.

In tournament.py a checkpoint plays with its own policy and its own evolved
gene pool, so "newer predators catch more" mixes the two. Here the focal
species' policy and gene pool are taken from different checkpoints: every
policy iteration crossed with every gene-pool iteration, each against fixed
opponents (the other species' policy and gene pool from one checkpoint).
Episode k uses seed + k in every cell (common random numbers).

Effects, for each focal species, as the change in catch risk that helps the
focal species (predators: higher; prey: lower), newer minus older:
- policy effect: newer policy, same gene pool and opponent;
- body effect: newer gene pool, same policy and opponent;
averaged over all ordered pairs of iterations, the other factor's levels and
the opponents, with paired-seed bootstrap 95% intervals. Cells where policy
and gene pool come from different checkpoints never occurred in training.

    python -m predpreygrass.evolutionary.eco_evolutionary_continuous_traits.analysis.cross_policy_bodies \\
        play <run_dir> --iterations 300,1500,2990 --episodes 100 --seed 10000 --workers 14
    python -m predpreygrass.evolutionary.eco_evolutionary_continuous_traits.analysis.cross_policy_bodies \\
        analyze <run_dir> [<run_dir> ...] --out analysis/results/cross_policy_bodies.json
"""

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

CSV = "cross_policy_bodies.csv"
SPECIES = ("predator", "prey")
FIELDS = ("focal", "policy_iter", "pool_iter", "opponent_iter", "episode", "seed", "outcome", "prey_caught", "prey_exposure")
REPLICATES = 10_000
BOOTSTRAP_SEED = 20261018

_worker = {}


def _init(run_dir, env_config):
    import torch

    torch.set_num_threads(1)
    _worker.update(run_dir=Path(run_dir), env_config=env_config, modules={}, pools={}, env=None)


def _module(species, iteration):
    from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.checkpoints import load_module
    from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.tournament import checkpoint_dir

    key = (species, iteration)
    if key not in _worker["modules"]:
        _worker["modules"][key] = load_module(checkpoint_dir(_worker["run_dir"], iteration), species)
    return _worker["modules"][key]


def _pool(iteration):
    from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.checkpoints import load_gene_pool
    from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.tournament import checkpoint_dir

    if iteration not in _worker["pools"]:
        _worker["pools"][iteration] = load_gene_pool(checkpoint_dir(_worker["run_dir"], iteration))
    return _worker["pools"][iteration]


def _play(task):
    from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.checkpoints import use_gene_pool
    from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.environment import PredPreyGrassContinuous
    from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.tournament import play_episode

    focal, policy_iter, pool_iter, opponent_iter, episode, seed = task
    other = "prey" if focal == "predator" else "predator"
    if _worker["env"] is None:
        _worker["env"] = PredPreyGrassContinuous(dict(_worker["env_config"]))
    env = _worker["env"]
    modules = {focal: _module(focal, policy_iter), other: _module(other, opponent_iter)}
    use_gene_pool(env, {focal: _pool(pool_iter)[focal], other: _pool(opponent_iter)[other]})
    result = play_episode(env, modules, seed)
    return dict(
        focal=focal,
        policy_iter=policy_iter,
        pool_iter=pool_iter,
        opponent_iter=opponent_iter,
        episode=episode,
        seed=seed,
        outcome=result["outcome"],
        prey_caught=result["prey_caught"],
        prey_exposure=result["prey_exposure"],
    )


def play(args):
    from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.tournament import checkpoint_dir, run_env_config

    run_dir = Path(args.run_dir)
    iterations = [int(i) for i in args.iterations.split(",")]
    for iteration in iterations:
        if not checkpoint_dir(run_dir, iteration).is_dir():
            raise SystemExit(f"no checkpoint for iteration {iteration} in {run_dir}")
    env_config = run_env_config(run_dir, iterations[-1])
    tasks = [
        (focal, policy, pool, opponent, episode, args.seed + episode)
        for focal in SPECIES
        for policy, pool, opponent in itertools.product(iterations, repeat=3)
        for episode in range(args.episodes)
    ]
    out = run_dir / CSV
    print(f"{len(tasks)} episodes on {args.workers} workers -> {out}", flush=True)
    start = time.time()
    with (
        ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init, initargs=(run_dir, env_config)) as pool,
        open(out, "w", encoding="utf-8") as csv,
    ):
        csv.write(",".join(FIELDS) + "\n")
        for done, row in enumerate(pool.map(_play, tasks, chunksize=1), 1):
            csv.write(",".join(str(row[field]) for field in FIELDS) + "\n")
            csv.flush()
            if done % max(1, len(tasks) // 20) == 0:
                elapsed = time.time() - start
                print(f"{done}/{len(tasks)}, {elapsed / 60:.1f} min, ~{elapsed / done * (len(tasks) - done) / 60:.1f} min left", flush=True)


def effects(caught, exposure, sign):
    """(policy effect, body effect, share of policy comparisons that help,
    share of body comparisons that help) from (policy, pool, opponent, seed)
    arrays; sign +1 for predators, -1 for prey."""
    risk = sign * 1000 * caught.sum(-1) / exposure.sum(-1)  # (policy, pool, opponent)
    n = risk.shape[0]
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    policy = np.array([risk[b] - risk[a] for a, b in pairs])  # over pool, opponent
    body = np.array([risk[:, b] - risk[:, a] for a, b in pairs])  # over policy, opponent
    return np.array(
        [policy.mean(), body.mean(), np.mean((policy > 0) + 0.5 * (policy == 0)), np.mean((body > 0) + 0.5 * (body == 0))]
    )


def analyze(args):
    names = ["policy_effect", "body_effect", "policy_share_helping", "body_share_helping"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = {"method": __doc__.split("\n\n")[1].strip(), "replicates": REPLICATES, "runs": {}}
    for run_dir in args.run_dirs:
        df = pd.read_csv(Path(run_dir) / CSV)
        iterations = sorted(df.policy_iter.unique())
        seeds = sorted(df.seed.unique())
        run = {"iterations": [int(i) for i in iterations], "episodes": len(seeds)}
        for focal, sign in (("predator", 1), ("prey", -1)):
            part = df[df.focal == focal]
            index = pd.MultiIndex.from_product([iterations] * 3 + [seeds])
            table = part.set_index(["policy_iter", "pool_iter", "opponent_iter", "seed"]).reindex(index)
            assert table.prey_exposure.notna().all() and (table.prey_exposure > 0).all()
            shape = (len(iterations),) * 3 + (len(seeds),)
            caught = table.prey_caught.to_numpy().reshape(shape)
            exposure = table.prey_exposure.to_numpy().reshape(shape)
            estimate = effects(caught, exposure, sign)
            boot = np.empty((REPLICATES, len(names)))
            for b in range(REPLICATES):
                sample = rng.integers(0, len(seeds), len(seeds))
                boot[b] = effects(caught[..., sample], exposure[..., sample], sign)
            cells = 1000 * caught.sum(-1) / exposure.sum(-1)
            run[focal] = {
                **{name: {"estimate": float(estimate[k]), "ci95": np.quantile(boot[:, k], [0.025, 0.975]).tolist()} for k, name in enumerate(names)},
                # Catch risk per (policy, gene pool), averaged over opponents.
                "catch_risk_policy_by_pool": np.round(cells.mean(-1), 4).tolist(),
            }
        output["runs"][Path(run_dir).name] = run
    out = Path(args.out)
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({r: {s: {k: v for k, v in d[s].items() if k != "catch_risk_policy_by_pool"} for s in SPECIES} for r, d in output["runs"].items()}, indent=1))
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("play")
    p.add_argument("run_dir")
    p.add_argument("--iterations", default="300,1500,2990")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--workers", type=int, default=14)
    a = sub.add_parser("analyze")
    a.add_argument("run_dirs", nargs="+")
    a.add_argument("--out", required=True)
    args = parser.parse_args()
    play(args) if args.command == "play" else analyze(args)


if __name__ == "__main__":
    main()
