"""Run the ERL predator-prey-grass simulation (no RLlib/PPO/Ray -- pure
Python/NumPy, per Ackley & Littman 1991's actual compute profile).

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.run_erl_simulation \\
        --steps 200000 --seed 41 --log-every 500 --constraint-window 5000
"""

import argparse
import time
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.metrics import CsvLogger
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import ErlWorld

FIELDNAMES = [
    "step",
    "predator_count",
    "prey_count",
    "predator_eval_weight_absmean",
    "predator_action_weight_absmean",
    "prey_eval_weight_absmean",
    "prey_action_weight_absmean",
    "predator_eval_site_change_rate",
    "predator_action_site_change_rate",
    "prey_eval_site_change_rate",
    "prey_action_site_change_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500, help="Steps between metric log rows.")
    parser.add_argument(
        "--constraint-window", type=int, default=5000,
        help="Steps between resetting the functional-constraint tracker's window "
             "(rates() reflects change since the last reset, not the whole run).",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = dict(config_erl)
    if args.seed is not None:
        cfg["seed"] = args.seed
    rng = np.random.default_rng(cfg["seed"])
    world = ErlWorld(cfg, rng)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path.home() / "erl_results" / f"ERL_BALDWIN_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = CsvLogger(out_dir / "progress.csv", FIELDNAMES)

    start = time.time()
    last_window_reset = 0
    extinction_step = None

    for step in range(1, args.steps + 1):
        world.step()
        counts = world.population_counts()

        if counts["predator"] == 0 or counts["prey"] == 0:
            extinction_step = step
            break

        if step % args.log_every == 0:
            row = {"step": step, "predator_count": counts["predator"], "prey_count": counts["prey"]}
            row.update(world.genome_stats())
            for species in ("predator", "prey"):
                rates = world.constraint_trackers[species].rates()
                row[f"{species}_eval_site_change_rate"] = rates["eval_site_change_rate"]
                row[f"{species}_action_site_change_rate"] = rates["action_site_change_rate"]
            logger.log(row)
            logger.flush()

        if step - last_window_reset >= args.constraint_window:
            for tracker in world.constraint_trackers.values():
                tracker.reset_window()
            last_window_reset = step

    elapsed = time.time() - start
    logger.close()

    print(f"Finished at step {world.current_step} in {elapsed:.1f}s ({world.current_step / max(elapsed, 1e-9):.0f} steps/sec).")
    if extinction_step is not None:
        print(f"Population extinction at step {extinction_step} (one species reached 0).")
    else:
        print(f"Reached step limit ({args.steps}) without extinction.")
    print(f"Final population: {world.population_counts()}")
    print(f"Log written to: {out_dir / 'progress.csv'}")


if __name__ == "__main__":
    main()
