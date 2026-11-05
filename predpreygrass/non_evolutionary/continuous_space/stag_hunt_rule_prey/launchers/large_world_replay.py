"""Replay trained rule-prey hunters, without training, in larger worlds at equal
densities. usage: large_world_replay.py CHECKPOINT AREA_FACTOR EPISODES WORKERS OUT"""

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey import evaluate

COUNTS = (
    "predator_count", "rabbit_count", "mammoth_count",
    "reproduction_max_predator", "reproduction_max_rabbit", "reproduction_max_mammoth",
    "reproduction_predator_pool", "reproduction_rabbit_pool", "reproduction_mammoth_pool",
    "grass_count", "grass_cluster_count", "steppe_count", "steppe_cluster_count",
)


def scaled(config, factor):
    out = dict(config)
    side = math.sqrt(factor)
    out["width"] = round(config["width"] * side)
    out["height"] = round(config["height"] * side)
    for key in COUNTS:
        out[key] = config[key] * factor
    return out


def init(checkpoint, factor):
    import torch

    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.checkpoints import (
        checkpoint_env_config,
        load_modules,
    )

    torch.set_num_threads(1)
    config = scaled(checkpoint_env_config(checkpoint), factor)
    config["truncate_on_prey_extinction"] = False
    evaluate._worker["modules"] = load_modules(checkpoint)
    evaluate._worker["env"] = evaluate.StagHuntRulePrey(config)


def main():
    checkpoint, factor, episodes, workers, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=init, initargs=(checkpoint, factor)) as pool:
        rows = sorted(pool.map(evaluate._play, range(episodes)), key=lambda row: row["seed"])
    summary = {"checkpoint": checkpoint, "area_factor": factor, **evaluate.summarise(rows)}
    with open(out, "w") as handle:
        json.dump({"summary": summary, "episodes": rows}, handle, indent=1)
    print(factor, summary["outcomes"], {k: (len(v), sorted(v)[len(v) // 2] if v else None) for k, v in summary["extinction_steps"].items()})


if __name__ == "__main__":
    main()
