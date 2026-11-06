"""Evaluate hunters against the rabbits their run evolved: as evaluate.py, but
each episode's founders come from one lineage's gene pool at the checkpoint
(gene_pools.load_pools; episode seed s from lineage s % lineages). Also
records the founders' and final rabbits' mean genes. Run from the
repository root:

    python predpreygrass/non_evolutionary/continuous_space/stag_hunt_rule_prey/launchers/gene_eval.py \\
        CHECKPOINT POOL_DIR EPISODES WORKERS OUT
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey import evaluate
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.launchers.gene_pools import load_pools, use_pool

_w = {}


def init(checkpoint, pools):
    import torch

    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.checkpoints import checkpoint_env_config, load_modules

    torch.set_num_threads(1)
    config = checkpoint_env_config(checkpoint)
    if not config["rabbit_genes"]:
        raise SystemExit("this checkpoint's run had no rabbit genes")
    config["truncate_on_prey_extinction"] = False
    config["rabbit_gene_log"] = ""
    evaluate._worker["modules"] = load_modules(checkpoint)
    evaluate._worker["env"] = evaluate.StagHuntRulePrey(config)
    _w["pools"] = pools


def play(seed):
    env = evaluate._worker["env"]
    use_pool(env, _w["pools"], seed)
    row = evaluate._play(seed)
    row["lineage"] = seed % len(_w["pools"])
    row["founders"] = [list(g) for g in env.episode_founders]
    row["final_genes"] = env.episode_gene_stats
    return row


def main():
    checkpoint, pool_dir, episodes, workers, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    pools = load_pools(pool_dir, checkpoint)
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=init, initargs=(checkpoint, pools)) as pool:
        rows = sorted(pool.map(play, range(episodes)), key=lambda row: row["seed"])
    summary = {"checkpoint": checkpoint, "pool_dir": pool_dir, "lineages": len(pools), **evaluate.summarise(rows)}
    with open(out, "w") as handle:
        json.dump({"summary": summary, "pools": pools, "episodes": rows}, handle, indent=1)
    print(checkpoint, "lineages", len(pools), summary["outcomes"], "rabbits_mean %.1f predators_mean %.1f" % (summary["rabbits_mean"], summary["predators_mean"]))


if __name__ == "__main__":
    main()
