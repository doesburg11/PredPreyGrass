"""Mechanism test for the predator-speed effect (RESULTS.md section 5, README
TODO): does extra speed pay only once the predator policy has adapted to it?

Each episode starts from the branch point's gene pools (CT-05 at iteration
1,500), but the 6 predator founders get fixed speed genes from SPEEDS, mixed
within the episode: every value once, two values (drawn per episode) twice,
in a random order. Their other genes, and the prey, are as in the
branch point. The predators then act with either

- the branch-point predator policy, which is what frozen predators kept
  ("frozen"), or
- the predator policy at the end of a branch in which predators learned
  (both learning or prey frozen, seeds 1-3: "learned").

The prey act with the branch-point prey policy and gene pool ("same prey"),
or, as a secondary condition, with the policy and gene pool at the end of
the run whose predators are tested ("own prey"; the frozen predators are
then scored against each run's prey too).

Every predator descends from one founder (lineages are tracked through
births), and descendants inherit the founder's speed up to the usual
mutation. Per founder this records the lineage's members alive at the end of
the episode (its share of the next episode's founders, which are drawn from
the survivors), its births and its catches. Episode seeds are common to all
conditions, so founders, speed assignments and spawn positions match.

The learned policies differ from the frozen one in everything they learned,
not only in how they use speed: a steeper fitness slope on speed under them
shows that the later policies make speed pay more, not that they adapted to
the speed gene itself.

Pre-specified primary outcome (fixed before the run, 2026-10-24): per
condition, the slope of relative lineage fitness (members alive at the end,
divided by the episode's mean over founders) on founder speed, least squares
over all founders of all episodes with a mean above zero. Primary contrast,
same prey: the mean slope over the 6 learned runs minus the frozen slope,
with a 95% interval from a bootstrap over episode seeds (paired across
conditions); also reported per learned run. Secondary: the same with
lineage births (the fecundity measure logged in training) and catches, and
the own-prey conditions (each learned run against frozen predators facing
the same prey).

Writes one JSON line per episode to runs/mechanism_speed/<condition>.jsonl;
analysis/mechanism_speed.py analyses them.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.mechanism_speed --episodes 400
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parent
RUNS = MODULE / "runs"
BRANCH_POINT = RUNS / "eco_evolutionary_continuous_traits_2026-10-16_20-53-30_161103" / "checkpoint" / "iter_001500"
LEARNED = {f"{arm}_s{seed}": RUNS / f"blrep_{arm}_s{seed}" / "checkpoint" for arm in ("both_learn", "prey_frozen") for seed in (1, 2, 3)}
SPEEDS = (-0.2, 0.0, 0.2, 0.4)
OUT = RUNS / "mechanism_speed"
CHUNK = 25  # episodes per task


def conditions():
    """{name: (predator checkpoint, prey checkpoint)}."""
    out = {"frozen__same_prey": (BRANCH_POINT, BRANCH_POINT)}
    for run, checkpoint in LEARNED.items():
        out[f"{run}__same_prey"] = (checkpoint, BRANCH_POINT)
        out[f"{run}__own_prey"] = (checkpoint, checkpoint)
        out[f"frozen__prey_of_{run}"] = (BRANCH_POINT, checkpoint)
    return out


def make_env(predator_checkpoint, prey_checkpoint):
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import (
        checkpoint_env_config,
        load_gene_pool,
        use_gene_pool,
    )
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.environment import PredPreyGrassContinuous

    class FixedSpeedEnv(PredPreyGrassContinuous):
        """Predator founders with fixed speed genes; lineages, their births
        and catches tracked per founder."""

        def reset(self, *, seed=None, options=None):
            self.speed_rng = np.random.default_rng(None if seed is None else [seed, 7])
            self.lineage, self.catches_by = {}, {}
            obs, infos = super().reset(seed=seed, options=options)
            for i in range(self.count["predator"]):
                self.lineage[f"predator_{i}"] = i
            return obs, infos

        def _founder_genomes(self, species, count):
            genomes = super()._founder_genomes(species, count)
            if species == "predator":
                extra = self.speed_rng.choice(SPEEDS, size=count - len(SPEEDS), replace=count - len(SPEEDS) > len(SPEEDS))
                speeds = np.concatenate([SPEEDS, extra])[self.speed_rng.permutation(count)]
                genomes = [np.concatenate([[s], g[1:]]) for s, g in zip(speeds, genomes)]
            return genomes

        def _reproduce(self, rewards, terminateds, truncateds, infos):
            born = super()._reproduce(rewards, terminateds, truncateds, infos)
            for child, parent in born:
                if parent in self.lineage:
                    self.lineage[child] = self.lineage[parent]
            return born

        def _energy(self, catches, eaten, terminateds, truncateds, infos):
            for predator, _ in catches:
                if predator.alive:
                    founder = self.lineage[predator.id]
                    self.catches_by[founder] = self.catches_by.get(founder, 0) + 1
            return super()._energy(catches, eaten, terminateds, truncateds, infos)

    config = checkpoint_env_config(BRANCH_POINT)
    config["trait_initial_gene_pool"] = None
    env = FixedSpeedEnv(config)
    predator_pool, prey_pool = load_gene_pool(BRANCH_POINT), load_gene_pool(prey_checkpoint)
    use_gene_pool(env, {"predator": predator_pool["predator"], "prey": prey_pool["prey"]})
    return env


def run_chunk(task):
    import torch

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import load_module
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.tournament import sample_actions

    name, predator_checkpoint, prey_checkpoint, seeds = task
    torch.set_num_threads(1)
    env = make_env(predator_checkpoint, prey_checkpoint)
    modules = {"predator": load_module(predator_checkpoint, "predator"), "prey": load_module(prey_checkpoint, "prey")}
    records = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        obs, _ = env.reset(seed=seed)
        founders = [env.family[f"predator_{i}"][1][0] for i in range(env.count["predator"])]
        steps = 0
        while True:
            obs, _, terms, truncs, _ = env.step(sample_actions(modules, obs, generator, env))
            steps += 1
            if terms["__all__"] or truncs["__all__"]:
                break
            obs = {a: o for a, o in obs.items() if not (terms.get(a) or truncs.get(a))}
        n = len(founders)
        alive, births = [0] * n, [0] * n
        for predator in env.predators:
            alive[env.lineage[predator.id]] += 1
        for agent, (species, _, offspring) in env.family.items():
            if species == "predator":
                births[env.lineage[agent]] += offspring
        records.append(
            {
                "seed": seed,
                "steps": steps,
                "speed": [float(s) for s in founders],
                "alive": alive,
                "births": births,
                "catches": [env.catches_by.get(i, 0) for i in range(n)],
                "prey_alive": len(env.prey),
            }
        )
    return name, records


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--only", nargs="*", help="condition names (default: all)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = {k: v for k, v in conditions().items() if not args.only or k in args.only}
    tasks = [
        (name, str(pred), str(prey), list(range(start, min(start + CHUNK, args.episodes))))
        for name, (pred, prey) in chosen.items()
        for start in range(0, args.episodes, CHUNK)
    ]
    (OUT / "settings.json").write_text(json.dumps({"episodes": args.episodes, "speeds": SPEEDS, "conditions": {k: [str(p), str(q)] for k, (p, q) in chosen.items()}}, indent=2) + "\n")
    results = {name: [] for name in chosen}
    start, done = time.time(), 0
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        for name, records in pool.map(run_chunk, tasks):
            results[name] += records
            done += 1
            print(f"{done}/{len(tasks)} chunks, {(time.time() - start) / 60:.1f} min", flush=True)
    for name, records in results.items():
        records.sort(key=lambda r: r["seed"])
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"written {len(results)} conditions to {OUT}", flush=True)


if __name__ == "__main__":
    main()
