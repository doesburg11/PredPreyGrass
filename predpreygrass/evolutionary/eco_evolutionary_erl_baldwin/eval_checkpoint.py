"""Load a checkpoint (see checkpoint.py) and inspect/visualize it standalone --
no training loop, no re-running from scratch. Two things this is for:

  1. Sanity-checking an evolved population's behavior before committing to a
     full-scale run (does it still forage/evade sensibly? did a strategy go
     extinct in a weird way?).
  2. A quick, real read on the proximate-vs-ultimate-reward question (see
     analyze_proximate_reward.py) for the checkpoint's current population,
     without needing a completed lineage_fitness.csv.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.eval_checkpoint \\
        ~/simulation_results/erl_results/ERL_BALDWIN_.../checkpoints/checkpoint_step_50000.pkl \\
        --steps 2000 --render snapshots
"""

import argparse
import time
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.checkpoint import load_checkpoint
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.metrics import lineage_fieldnames, lineage_record


def _print_summary(world, label: str):
    counts = world.population_counts()
    stats = world.genome_stats()
    print(f"--- {label} (step {world.current_step}) ---")
    print(f"Population: {counts}")
    print(
        f"eval_weight_absmean={stats['eval_weight_absmean']:.3f}  "
        f"action_weight_absmean={stats['action_weight_absmean']:.3f}"
    )
    living = [a for a in world.agents if a.alive]
    if living:
        offspring = np.array([a.offspring_count for a in living])
        print(f"offspring_count over {len(living)} living agents: mean={offspring.mean():.2f}, max={offspring.max()}")
        top = sorted(living, key=lambda a: -a.offspring_count)[:5]
        names = ["visual_N", "visual_S", "visual_E", "visual_W", "in_tree", "health_norm", "energy_norm", "alarm_signal"]
        print("Top 5 by offspring_count (eval_weights -- the evolved reward each carries):")
        for a in top:
            weights = ", ".join(f"{names[i]}={w:.2f}" for i, w in enumerate(a.genome.eval_weights))
            print(f"  agent {a.agent_id}: offspring={a.offspring_count}, gen={a.generation}, [{weights}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Path to a checkpoint_step_*.pkl file.")
    parser.add_argument(
        "--steps", type=int, default=2000,
        help="Steps to run forward from the checkpoint for the eval/visualization pass "
             "(0 = just inspect/render the saved state as-is, no further evolution or learning).",
    )
    parser.add_argument(
        "--render", type=str, default="snapshots", choices=["none", "live", "snapshots"],
        help="'snapshots' (default) saves PNGs -- works headless over SSH, unlike 'live'.",
    )
    parser.add_argument("--render-every", type=int, default=100, help="Steps between rendered frames.")
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory for frames/lineage CSV.")
    parser.add_argument(
        "--log-lineage", action="store_true",
        help="Also write a lineage_fitness.csv for this eval window (deaths during --steps "
             "only) -- off by default since eval runs are usually too short to be meaningful "
             "for analyze_proximate_reward.py.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    world = load_checkpoint(checkpoint_path)
    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parent.parent / "eval" / checkpoint_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_summary(world, "Loaded checkpoint")

    lineage_logger = None
    if args.log_lineage:
        from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.metrics import CsvLogger

        lineage_logger = CsvLogger(out_dir / "eval_lineage.csv", lineage_fieldnames(world.obs_dim))
        world.on_agent_death = lambda agent, death_step: lineage_logger.log(
            lineage_record(agent, death_step, censored=False)
        )

    renderer = None
    if args.render != "none":
        from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.visualize import WorldRenderer

        if args.render == "live":
            import matplotlib.pyplot as plt

            plt.ion()
        renderer = WorldRenderer(world.grid_size)
        renderer.show(world, world.current_step) if args.render == "live" else renderer.save(
            world, world.current_step, out_dir / "frames"
        )

    start = time.time()
    for _ in range(args.steps):
        world.step()
        if world.population_counts()["agent"] == 0:
            break
        if renderer is not None and world.current_step % args.render_every == 0:
            renderer.show(world, world.current_step) if args.render == "live" else renderer.save(
                world, world.current_step, out_dir / "frames"
            )
    elapsed = time.time() - start

    if lineage_logger is not None:
        for agent in world.agents:
            if agent.alive:
                lineage_logger.log(lineage_record(agent, world.current_step, censored=True))
        lineage_logger.close()

    if renderer is not None:
        renderer.close()

    if args.steps > 0:
        print(f"\nRan {args.steps} eval steps in {elapsed:.1f}s.")
        _print_summary(world, "After eval steps")
    if args.render == "snapshots":
        print(f"Frames written to: {out_dir / 'frames'}")
    if lineage_logger is not None:
        print(f"Eval lineage log written to: {out_dir / 'eval_lineage.csv'}")


if __name__ == "__main__":
    main()
