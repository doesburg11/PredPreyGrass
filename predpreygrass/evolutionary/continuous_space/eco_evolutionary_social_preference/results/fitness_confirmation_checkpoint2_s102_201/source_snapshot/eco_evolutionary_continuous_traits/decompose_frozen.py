"""Where does the advantage of predator speed go in the evolving frozen arm?
(RESULTS.md section 7)

In the both-frozen replication runs the offspring-based selection
differential on predator speed summed to +0.5 to +0.9 per island, yet mean
speed hardly changed. This re-runs that arm (both policies frozen at the
branch point, gene pools evolving from the branch point's as in
evolve_frozen.py) with every predator lineage tracked, and splits each
episode's change in mean predator speed exactly (Price equation over the
episode's founders):

    survivors' mean - founders' mean = cov(n, z) / mean(n)          (selection)
                                     + sum n (d - z) / sum n         (transmission)

with z a founder's speed gene, n its lineage's members alive at the end
(the next founders are drawn from them) and d their mean speed gene
(the within-lineage term: mutation along the lineage and selection among
its descendants' mutations). Selection is further split
into fecundity, cov(m, z) / mean(m) with m = 1 + the lineage's births, and
survival, the rest. Founder sampling is the change from one episode's
survivors to the next episode's founders. The offspring-based differential
the training logs use is recorded next to it.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.decompose_frozen --islands 4 --episodes 250
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.mechanism_speed import BRANCH_POINT, RUNS

OUT = RUNS / "decompose_frozen"


def island(task):
    import torch

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import load_modules
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.mechanism_speed import make_env
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.tournament import sample_actions

    k, episodes = task
    torch.set_num_threads(1)
    env = make_env(BRANCH_POINT, BRANCH_POINT, fixed_speeds=False)
    modules = load_modules(BRANCH_POINT)
    records = []
    for episode in range(episodes):
        seed = 1_000_000 * k + episode
        generator = torch.Generator().manual_seed(seed)
        obs, _ = env.reset(seed=seed)
        z = np.array([env.family[f"predator_{i}"][1][0] for i in range(env.count["predator"])])
        while True:
            obs, _, terms, truncs, _ = env.step(sample_actions(modules, obs, generator, env))
            if terms["__all__"] or truncs["__all__"]:
                break
            obs = {a: o for a, o in obs.items() if not (terms.get(a) or truncs.get(a))}
        n = np.zeros(len(z))
        d_sum = np.zeros(len(z))
        births = np.zeros(len(z))
        for predator in env.predators:
            i = env.lineage[predator.id]
            n[i] += 1
            d_sum[i] += predator.genome[0]
        for agent, (species, _, offspring) in env.family.items():
            if species == "predator":
                births[env.lineage[agent]] += offspring
        record = {
            "episode": episode,
            "founders_mean": float(z.mean()),
            "founders": z.tolist(),
            "alive": n.tolist(),
            "births": births.tolist(),
            "survivors_mean": float(d_sum.sum() / n.sum()) if n.sum() else None,
            "logged_selection": env.selection_differentials("predator")[0] if env.selection_differentials("predator") else None,
        }
        if n.sum():
            m = 1 + births
            record["selection"] = float(np.mean((n / n.mean() - 1) * (z - z.mean())))
            record["fecundity"] = float(np.mean((m / m.mean() - 1) * (z - z.mean())))
            record["transmission"] = float((d_sum - n * z).sum() / n.sum())
        records.append(record)
    return k, records


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=250)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with ProcessPoolExecutor(args.islands, mp_context=get_context("spawn")) as pool:
        for k, records in pool.map(island, [(k, args.episodes) for k in range(args.islands)]):
            (OUT / f"island_{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            print(f"island {k} done, {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
