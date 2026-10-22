"""Evolution with fixed behaviour: both species act with the policies of one
checkpoint, which never change, while the gene pool evolves across episodes
exactly as in training (environment.py). RLlib cannot run with nothing to
train, so this replaces train.py when both policies are frozen.

Each island (process) plays episodes back to back from the checkpoint's
pooled gene pool, like one RLlib simulation worker. Together the islands play
as many environment steps as `iterations` training iterations of
`train_batch_size` steps. An environment step is one decision (one
env.step call, up to action_repeat physics steps), as RLlib counts it. Each
island plays an equal share, as each RLlib worker samples a fixed fragment
of every training batch. Every `checkpoint_every` iterations' worth of steps
the islands' gene pools are written as <out>/checkpoint/iter_<n>/gene_pools.json,
the layout train.py uses, and each episode's trait statistics go to
<out>/islands/island_<k>.jsonl. The last snapshot is also written to
<out>/checkpoint/gene_pools.json, where train.py puts the final one.

    python -m predpreygrass.evolutionary.eco_evolutionary_continuous_traits.evolve_frozen \\
        <checkpoint> <out> --islands 7 --iterations 1500
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path


def island(task):
    import torch

    from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.checkpoints import checkpoint_env_config, load_modules
    from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.environment import PredPreyGrassContinuous
    from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.tournament import sample_actions

    k, checkpoint, out, steps_per_snapshot, snapshots, seed, neutral = task
    torch.set_num_threads(1)
    config = checkpoint_env_config(checkpoint)
    config["trait_initial_gene_pool"] = str(Path(checkpoint) / "gene_pools.json")
    if neutral:
        # Neutral control: genes are inherited and mutate as usual but have no
        # effect on movement, vision or costs. Every animal gets the species
        # base body, so the ecology (and so drift) is not matched to CT-05.
        config["trait_expression"] = False
    env = PredPreyGrassContinuous(config)
    modules = load_modules(checkpoint)
    generator = torch.Generator().manual_seed(seed)
    states, steps, episode = [], 0, 0
    log = open(Path(out) / "islands" / f"island_{k}.jsonl", "w", encoding="utf-8")
    while len(states) < snapshots:
        obs, _ = env.reset(seed=seed + episode)
        while True:
            obs, _, terms, truncs, _ = env.step(sample_actions(modules, obs, generator, env))
            steps += 1  # one decision, as RLlib counts environment steps
            while len(states) < snapshots and steps >= (len(states) + 1) * steps_per_snapshot:
                states.append(env.gene_pool_state())
            if terms["__all__"] or truncs["__all__"] or len(states) >= snapshots:
                break
            obs = {a: o for a, o in obs.items() if not (terms.get(a) or truncs.get(a))}
        log.write(json.dumps({"episode": episode, "steps": steps, "stats": env.trait_stats()}) + "\n")
        log.flush()
        episode += 1
    log.close()
    return k, states


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    parser.add_argument("--islands", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--train-batch-size", type=int, default=4800)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--neutral",
        action="store_true",
        help="genes have no effect (trait_expression=False): genetic neutrality, every animal "
        "with the species base body, so its demography differs from the expressed arms",
    )
    args = parser.parse_args()
    out = Path(args.out)
    (out / "islands").mkdir(parents=True, exist_ok=True)
    snapshots = args.iterations // args.checkpoint_every
    steps_per_snapshot = args.checkpoint_every * args.train_batch_size / args.islands
    (out / "settings.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    tasks = [
        (k, str(Path(args.checkpoint).resolve()), str(out), steps_per_snapshot, snapshots, args.seed + 1_000_000 * k, args.neutral)
        for k in range(args.islands)
    ]
    start = time.time()
    results = {}
    with ProcessPoolExecutor(args.islands, mp_context=get_context("spawn")) as pool:
        for k, states in pool.map(island, tasks):
            results[k] = states
            print(f"island {k} done, {(time.time() - start) / 60:.1f} min", flush=True)
    for n in range(snapshots):
        target = out / "checkpoint" / f"iter_{(n + 1) * args.checkpoint_every:06d}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "gene_pools.json").write_text(json.dumps([results[k][n] for k in range(args.islands)]) + "\n")
    final = json.dumps([results[k][snapshots - 1] for k in range(args.islands)]) + "\n"
    (out / "checkpoint" / "gene_pools.json").write_text(final)
    print(f"{snapshots} gene-pool snapshots written to {out / 'checkpoint'}", flush=True)


if __name__ == "__main__":
    main()
