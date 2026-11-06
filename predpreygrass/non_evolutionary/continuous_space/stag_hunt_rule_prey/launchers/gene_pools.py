"""Gene pools logged by a run with rabbit_genes (environment.py,
rabbit_gene_log): one pool_*.jsonl file per environment instance (one lineage
of rabbits), one line per episode.

    load_pools(pool_dir, checkpoint=None) -> [genomes per lineage]

Each lineage's last nonempty pool logged no later than the checkpoint was
saved (its algorithm_state.pkl), or its last pool without a checkpoint.
Lineages are sorted by file name. Pools are matched to the checkpoint by
wall-clock time on one machine: the pool is that of the last episode the
lineage finished before the checkpoint was saved.
"""

import json
from pathlib import Path


def load_pools(pool_dir, checkpoint=None):
    until = (Path(checkpoint) / "algorithm_state.pkl").stat().st_mtime if checkpoint else float("inf")
    pools = []
    for path in sorted(Path(pool_dir).glob("pool_*.jsonl")):
        last = None
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["time"] <= until and row["genomes"]:
                    last = row["genomes"]
        if last is not None:
            pools.append(last)
        else:
            print(f"warning: no pool in {path.name} up to the checkpoint; lineage left out", flush=True)
    if not pools:
        raise SystemExit(f"no gene pools in {pool_dir} up to {checkpoint}")
    return pools


def use_pool(env, pools, seed):
    """Evaluation: the founders of the episode with this seed come from
    lineage seed % len(pools), fixed (no updating, no logging)."""
    env.fixed_founders = [env._clipped(g) for g in pools[seed % len(pools)]]
    env.gene_log = ""
