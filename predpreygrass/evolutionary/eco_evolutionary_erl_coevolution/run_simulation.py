"""Run one eco_evolutionary_erl_coevolution simulation (pure Python/NumPy, no RLlib).

For batches of seeds/strategies use study.py instead; this is for a single,
inspectable, checkpointed run (long runs, rendering, resuming).

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.run_simulation \\
        --preset step1 --steps 200000 --seed 1 --log-every 500

    # resume an interrupted run (--steps is how many MORE steps to run, not a total):
    python -m ...run_simulation --resume-from .../checkpoints --steps 100000
"""

import argparse
import time
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.config import PRESETS
from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.metrics import (
    CsvLogger,
    lineage_fieldnames,
    lineage_record,
    truncate_csv_after_step,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.study import parse_overrides
from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.world import STRATEGIES, ErlWorld
from predpreygrass.global_config import ERL_RESULTS_DIR

DEATH_FIELDS = [
    "agent_death_carnivore", "agent_death_agent_attack", "agent_death_starvation",
    "agent_death_wounds", "agent_death_tree_fall",
    "carnivore_death_agent_attack", "carnivore_death_starvation", "carnivore_death_wounds",
]

FIELDNAMES = [
    "step",
    "agent_count",
    "carnivore_count",
    "carnivore_births",
    "carnivore_immigrants",
    *DEATH_FIELDS,
    "eval_weight_absmean",
    "action_weight_absmean",
    "eval_site_change_rate",
    "action_site_change_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="step1")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Config override on top of the preset; repeatable.")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None, choices=list(STRATEGIES))
    parser.add_argument("--log-every", type=int, default=500, help="Steps between metric log rows.")
    parser.add_argument(
        "--constraint-window", type=int, default=5000,
        help="Steps between resetting the functional-constraint tracker's window.",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument(
        "--render", type=str, default="none", choices=["none", "live", "snapshots"],
        help="'live' pops up a matplotlib window, 'snapshots' saves PNG frames to out_dir/frames/.",
    )
    parser.add_argument("--render-every", type=int, default=2000, help="Steps between rendered frames.")
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="A checkpoint_step_*.pkl file or a checkpoints/ directory (uses the latest). "
             "--steps then means steps to run THIS invocation. config comes from the checkpoint.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=50_000,
                        help="Steps between periodic checkpoints (0 disables; a final one is always written).")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint directory (default: out_dir/checkpoints).")
    return parser.parse_args()


def main():
    args = parse_args()

    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.is_dir():
            found = latest_checkpoint(resume_path)
            if found is None:
                raise FileNotFoundError(f"No checkpoint_step_*.pkl found in {resume_path}")
            resume_path = found

    if resume_path is not None:
        world = load_checkpoint(resume_path)
        out_dir = Path(args.out_dir) if args.out_dir else resume_path.parent.parent
        print(f"Resumed from {resume_path} at step {world.current_step} (strategy={world.strategy}).")
    else:
        overrides = parse_overrides(args.set)
        unknown = set(overrides) - set(PRESETS[args.preset])
        if unknown:
            raise KeyError(f"--set: not config keys: {sorted(unknown)}")
        cfg = {**PRESETS[args.preset], **overrides}
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.strategy is not None:
            cfg["strategy"] = args.strategy
        world = ErlWorld(cfg, np.random.default_rng(cfg["seed"]))
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path(args.out_dir) if args.out_dir else (
            Path(ERL_RESULTS_DIR) / f"ERL_COEVO_{args.preset}_{cfg['strategy']}_seed{cfg['seed']}_{timestamp}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / "checkpoints"
    resuming = resume_path is not None

    if resuming:
        # Log rows can be flushed past the last checkpoint before a crash; drop
        # them, since resuming re-simulates those steps.
        truncate_csv_after_step(out_dir / "progress.csv", "step", world.current_step)
        truncate_csv_after_step(out_dir / "lineage_fitness.csv", "death_step", world.current_step)
        world.constraint_tracker.reset_window()

    logger = CsvLogger(out_dir / "progress.csv", FIELDNAMES, append=resuming)
    lineage_logger = CsvLogger(
        out_dir / "lineage_fitness.csv", lineage_fieldnames(world.obs_dim), append=resuming
    )
    world.on_agent_death = lambda agent, death_step: lineage_logger.log(
        lineage_record(agent, death_step, censored=False)
    )

    renderer = None
    if args.render != "none":
        from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.visualize import WorldRenderer

        if args.render == "live":
            import matplotlib.pyplot as plt

            plt.ion()
        renderer = WorldRenderer(world.grid_size)

    def render(step):
        if renderer is None:
            return
        if args.render == "live":
            renderer.show(world, step)
        else:
            renderer.save(world, step, out_dir / "frames")

    render(world.current_step)

    end_on_carnivore = world.cfg.get("end_on_carnivore_extinction", False)
    start = time.time()
    start_step = world.current_step
    last_window_reset = world.current_step
    last_checkpoint_step = world.current_step
    end_reason = "budget"

    for _ in range(args.steps):
        world.step()
        step = world.current_step
        counts = world.population_counts()

        if counts["agent"] == 0:
            end_reason = "agent_extinct"
        elif counts["carnivore"] == 0 and end_on_carnivore and not world.immigration_active():
            end_reason = "carnivore_extinct"

        if step % args.log_every == 0 or end_reason != "budget":
            row = {
                "step": step,
                "agent_count": counts["agent"],
                "carnivore_count": counts["carnivore"],
                "carnivore_births": world.carnivore_births,
                "carnivore_immigrants": world.carnivore_immigrants,
                **{field: 0 for field in DEATH_FIELDS},
                **world.death_counts(),
            }
            row.update(world.genome_stats())
            row.update(world.constraint_tracker.rates())
            logger.log(row)
            logger.flush()
            lineage_logger.flush()

        if end_reason != "budget":
            render(step)
            break

        if step % args.render_every == 0:
            render(step)

        if step - last_window_reset >= args.constraint_window:
            world.constraint_tracker.reset_window()
            last_window_reset = step

        if args.checkpoint_every > 0 and step - last_checkpoint_step >= args.checkpoint_every:
            logger.flush()
            lineage_logger.flush()
            save_checkpoint(world, checkpoint_dir / f"checkpoint_step_{step}.pkl")
            last_checkpoint_step = step

    elapsed = time.time() - start
    logger.close()
    lineage_logger.close()
    if renderer is not None:
        renderer.close()

    final_checkpoint = checkpoint_dir / f"checkpoint_step_{world.current_step}.pkl"
    save_checkpoint(world, final_checkpoint)

    ran = world.current_step - start_step
    print(f"Finished at step {world.current_step} in {elapsed:.1f}s ({ran / max(elapsed, 1e-9):.0f} steps/sec).")
    if end_reason == "agent_extinct":
        print(f"Agent population extinction at step {world.current_step}.")
    elif end_reason == "carnivore_extinct":
        print(f"Carnivore population extinction at step {world.current_step}.")
    else:
        print(f"Reached step limit ({args.steps}) without extinction.")
    print(f"Final population: {world.population_counts()}")
    print(f"Deaths: {world.death_counts()}  carnivore births: {world.carnivore_births}, "
          f"immigrants: {world.carnivore_immigrants}")
    print(f"Log written to: {out_dir / 'progress.csv'}")
    print(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
