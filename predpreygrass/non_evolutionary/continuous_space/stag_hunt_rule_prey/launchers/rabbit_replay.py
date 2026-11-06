"""Replay trained hunters, no training, against rabbits with other settings.
usage: rabbit_replay.py CHECKPOINT EPISODES WORKERS OUT KEY=VALUE..."""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey import evaluate


def init(checkpoint, settings):
    import torch

    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.checkpoints import (
        checkpoint_env_config,
        load_modules,
    )

    torch.set_num_threads(1)
    config = checkpoint_env_config(checkpoint)
    for key in settings:
        assert key in config, key
    config.update(settings)
    config["truncate_on_prey_extinction"] = False
    if config.get("rabbit_genes") and config.get("rabbit_gene_founders") is None:
        # The gene pool would carry over between this worker's episodes.
        raise SystemExit("rabbit genes are on: use launchers/gene_eval.py, or set rabbit_genes false")
    evaluate._worker["modules"] = load_modules(checkpoint)
    evaluate._worker["env"] = evaluate.StagHuntRulePrey(config)


def main():
    checkpoint, episodes, workers, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    settings = {k: json.loads(v) for k, v in (item.split("=") for item in sys.argv[5:])}
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=init, initargs=(checkpoint, settings)) as pool:
        rows = sorted(pool.map(evaluate._play, range(episodes)), key=lambda row: row["seed"])
    summary = {"checkpoint": checkpoint, "settings": settings, **evaluate.summarise(rows)}
    with open(out, "w") as handle:
        json.dump({"summary": summary, "episodes": rows}, handle, indent=1)
    t = summary["totals"]
    total = t["mammoth_energy"] + t["rabbit_energy"]
    median = lambda v: (len(v), sorted(v)[len(v) // 2] if v else None)
    print(settings, summary["outcomes"], {k: median(v) for k, v in summary["extinction_steps"].items()},
          "caught", t["rabbits_caught"], "rabbit births", t["births_rabbit"], "hunter births", t["births_predator"],
          "steps", t["physics_steps"], "share %.2f" % (t["mammoth_energy"] / total),
          "joint %.2fx" % (summary["joint_arrival_rate"] / summary["joint_arrival_rate_chance"]))


if __name__ == "__main__":
    main()
