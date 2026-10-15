"""Final check of a trained checkpoint: outcomes, catch risk, populations,
and behaviour (speed choices, time near grass, catches near grass).

Plays a checkpoint's policies (actions sampled as in training, each
species' agents in sorted order, one torch generator per episode seeded with
the episode's seed) for a number of seeded episodes in this environment.
Checkpoints of add_walls runs in the milestone world work too: this
environment reproduces add_walls exactly (SPEC.md, section 10), so all runs
are measured with the same code and seeds.

"Near grass" means within 64 units (two body diameters) of an available
patch, as in add_walls's RESULTS.md. Catch risk is prey caught per prey per
1,000 physics steps.

Usage, from the repository root:

    .conda/bin/python base_environment_continuous/analysis/final_check.py \\
        <checkpoint dir> <episodes> [out.json]

<checkpoint dir> is a run's checkpoint/ or checkpoint/iter_<n> directory.
Episodes use seeds 0, 1, ...
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.checkpoints import (
    checkpoint_env_config,
    load_gene_pool,
    load_modules,
    use_gene_pool,
)
from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.environment import PredPreyGrassContinuous
from predpreygrass.evolutionary.eco_evolutionary_continuous_traits.tournament import classify, sample_actions

NEAR_GRASS = 64.0


def near_grass_distance(env, position):
    patches = [p.position for p in env.grass.patches if env.grass.available(p)]
    return min((env.torus.distance(position, p) for p in patches), default=math.inf)


def play(env, modules, seed, stats):
    generator = torch.Generator().manual_seed(seed)
    catches = []
    move = env._move

    def recording_move(animal, action, catcher, aways, caught):
        before = len(caught)
        move(animal, action, catcher, aways, caught)
        if len(caught) > before:
            catcher = caught[-1][0]
            catches.append((catcher.velocity.mag(), near_grass_distance(env, catcher.position)))

    env._move = recording_move
    try:
        obs, _ = env.reset(seed=seed)
        exposure = {"predator": 0.0, "prey": 0.0}
        caught = 0
        while True:
            ordered = {agent: obs[agent] for agent in sorted(obs)}
            actions = sample_actions(modules, ordered, generator)
            for animal in env.predators + env.prey:
                if animal.id not in env.active:
                    continue
                key = animal.species
                stats[f"{key}_decisions"] += 1
                stats[f"{key}_stopped"] += actions[animal.id] // 16 == 2
                stats[f"{key}_near_grass"] += near_grass_distance(env, animal.position) <= NEAR_GRASS
            before = (env.time_step, len(env.predators), len(env.prey))
            obs, _, terms, truncs, infos = env.step(actions)
            steps = env.time_step - before[0]
            exposure["predator"] += steps * (before[1] + len(env.predators)) / 2
            exposure["prey"] += steps * (before[2] + len(env.prey)) / 2
            for agent, info in infos.items():
                if agent.startswith("prey") and terms.get(agent) and not info.get("starved"):
                    caught += 1
            if terms["__all__"] or truncs["__all__"]:
                break
    finally:
        env._move = move
    stats["catches"] += len(catches)
    stats["catches_near_grass"] += sum(d <= NEAR_GRASS for _, d in catches)
    stats["catches_standing"] += sum(speed < 0.5 for speed, _ in catches)
    return {
        "seed": seed,
        "outcome": classify(len(env.predators), len(env.prey)),
        "physics_steps": env.time_step,
        "prey_caught": caught,
        "prey_exposure": exposure["prey"],
        "mean_predators": exposure["predator"] / max(env.time_step, 1),
        "mean_prey": exposure["prey"] / max(env.time_step, 1),
    }


def main():
    checkpoint, episodes = sys.argv[1], int(sys.argv[2])
    torch.set_num_threads(1)
    env = PredPreyGrassContinuous(checkpoint_env_config(checkpoint))
    # Founders from the gene pool evolved by the checkpoint, kept fixed.
    use_gene_pool(env, load_gene_pool(checkpoint))
    modules = load_modules(checkpoint)
    stats = Counter()
    rows = [play(env, modules, seed, stats) for seed in range(episodes)]
    outcomes = Counter(row["outcome"] for row in rows)
    summary = {
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "coexisting": outcomes["time_limit"],
        "prey_extinct": outcomes["prey_extinct"],
        "predators_extinct": outcomes["predators_extinct"],
        "both_extinct": outcomes["both_extinct"],
        "catch_risk": 1000 * sum(r["prey_caught"] for r in rows) / sum(r["prey_exposure"] for r in rows),
        "mean_predators": float(np.mean([r["mean_predators"] for r in rows])),
        "mean_prey": float(np.mean([r["mean_prey"] for r in rows])),
        "predators_stopped": stats["predator_stopped"] / stats["predator_decisions"],
        "prey_stopped": stats["prey_stopped"] / stats["prey_decisions"],
        "predators_near_grass": stats["predator_near_grass"] / stats["predator_decisions"],
        "prey_near_grass": stats["prey_near_grass"] / stats["prey_decisions"],
        "catches_near_grass": stats["catches_near_grass"] / max(stats["catches"], 1),
        "catches_by_standing_predator": stats["catches_standing"] / max(stats["catches"], 1),
    }
    for key, value in summary.items():
        print(f"{key}: {round(value, 3) if isinstance(value, float) else value}")
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))


if __name__ == "__main__":
    main()
