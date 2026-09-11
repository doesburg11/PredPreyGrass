"""Run the ERL predator-prey-grass simulation (no RLlib/PPO/Ray -- pure
Python/NumPy, per Ackley & Littman 1991's actual compute profile).

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.run_erl_simulation \\
        --steps 200000 --seed 41 --log-every 500 --constraint-window 5000

    # resume an interrupted run (--steps is how many MORE steps to run, not a total):
    python -m ...run_erl_simulation --resume-from .../checkpoints --steps 100000
"""

import argparse
import time
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.config import config_erl
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.metrics import (
    CsvLogger,
    lineage_fieldnames,
    lineage_record,
    truncate_csv_after_step,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import ErlWorld
from predpreygrass.global_config import ERL_RESULTS_DIR

FIELDNAMES = [
    "step",
    "agent_count",
    "carnivore_count",
    "eval_weight_absmean",
    "action_weight_absmean",
    "eval_site_change_rate",
    "action_site_change_rate",
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
    parser.add_argument(
        "--strategy", type=str, default=None,
        choices=["ERL", "E", "L", "F", "B", "C", "ERLC", "K", "ERLK", "S", "ERLS"],
        help="Override config_erl['strategy'] -- ERL/E/L/F/B are Ackley & Littman's 5 "
             "comparative conditions; C/ERLC add the cooperation breeding-bonus, K/ERLK "
             "add the kin-selection damage discount, S/ERLS add the alarm-call "
             "communication mechanism (see world.py module docstring for all three).",
    )
    parser.add_argument(
        "--render", type=str, default="none", choices=["none", "live", "snapshots"],
        help="Visualize the grid: 'live' pops up a live-updating matplotlib window, "
             "'snapshots' saves PNG frames to out_dir/frames/ every --render-every steps. "
             "Adds overhead -- use a short --steps run, not a full comparative-study run.",
    )
    parser.add_argument("--render-every", type=int, default=2000, help="Steps between rendered frames.")
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume from a checkpoint: either a checkpoint_step_*.pkl file, or a "
             "checkpoints/ directory (uses the highest-step checkpoint in it). "
             "--steps then means steps to run THIS invocation, not a total across resumes. "
             "config/seed/strategy come from the checkpoint, not from --seed/--strategy.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=50_000,
        help="Steps between periodic checkpoints (0 disables periodic saves; a final "
             "checkpoint is always written when the run ends, regardless of this value).",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Override checkpoint directory (default: out_dir/checkpoints).",
    )
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
        # checkpoints/ lives directly under the original run's out_dir.
        out_dir = Path(args.out_dir) if args.out_dir else resume_path.parent.parent
        print(f"Resumed from {resume_path} at step {world.current_step} (strategy={world.strategy}).")
    else:
        cfg = dict(config_erl)
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.strategy is not None:
            cfg["strategy"] = args.strategy
        rng = np.random.default_rng(cfg["seed"])
        world = ErlWorld(cfg, rng)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path(args.out_dir) if args.out_dir else Path(ERL_RESULTS_DIR) / f"ERL_BALDWIN_{timestamp}"

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / "checkpoints"
    resuming = resume_path is not None

    if resuming:
        # progress.csv/lineage_fitness.csv are flushed more often than checkpoints
        # are saved (--log-every vs --checkpoint-every), so a crash between the
        # last checkpoint and the next log flush can leave rows describing steps
        # PAST world.current_step -- those steps will be re-simulated from the
        # checkpoint, so the stale rows must be dropped first or they'd end up
        # duplicated (and out of order) once we append the replayed ones.
        truncate_csv_after_step(out_dir / "progress.csv", "step", world.current_step)
        truncate_csv_after_step(out_dir / "lineage_fitness.csv", "death_step", world.current_step)
        # The functional-constraint window mid-accumulated at checkpoint time is
        # discarded rather than resumed -- it already reflects an interrupted,
        # partial window, and resuming it would make the first post-resume window
        # longer than --constraint-window (see world.py's FunctionalConstraintTracker).
        world.constraint_tracker.reset_window()

    logger = CsvLogger(out_dir / "progress.csv", FIELDNAMES, append=resuming)

    # Per-agent lineage log for the "proximate vs. ultimate reward" analysis
    # (analyze_proximate_reward.py): pairs each agent's evolved eval_weights with
    # its realized offspring_count at death. Fired from World, not written by it.
    # Deliberately records ONLY real deaths, never survivors -- a run that will
    # be resumed later has no way to know which of its survivors are "final";
    # analyze_proximate_reward.py's --checkpoint option adds a checkpoint's live
    # agents as censored rows at analysis time instead, so this file never needs
    # a second, contradicting row for the same agent after a resume.
    lineage_logger = CsvLogger(
        out_dir / "lineage_fitness.csv", lineage_fieldnames(world.obs_dim), append=resuming
    )
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
        if args.render == "live":
            renderer.show(world, world.current_step)
        else:
            renderer.save(world, world.current_step, out_dir / "frames")

    start = time.time()
    last_window_reset = world.current_step
    last_checkpoint_step = world.current_step
    extinction_step = None

    for _ in range(args.steps):
        world.step()
        step = world.current_step
        counts = world.population_counts()

        # Extinction of the adaptive agent population ends the run, matching
        # the paper ("simulation ends after 1 million steps or extinction").
        # Carnivores respawn every carnivore_spawn_interval steps regardless
        # and are not part of the survival-time measurement.
        if counts["agent"] == 0:
            extinction_step = step
            if renderer is not None:
                renderer.show(world, step) if args.render == "live" else renderer.save(world, step, out_dir / "frames")
            break

        if step % args.log_every == 0:
            row = {"step": step, "agent_count": counts["agent"], "carnivore_count": counts["carnivore"]}
            row.update(world.genome_stats())
            row.update(world.constraint_tracker.rates())
            logger.log(row)
            logger.flush()
            lineage_logger.flush()

        if renderer is not None and step % args.render_every == 0:
            renderer.show(world, step) if args.render == "live" else renderer.save(world, step, out_dir / "frames")

        if step - last_window_reset >= args.constraint_window:
            world.constraint_tracker.reset_window()
            last_window_reset = step

        if args.checkpoint_every > 0 and step - last_checkpoint_step >= args.checkpoint_every:
            # Flush logs first so a checkpoint always corresponds to a fully-written
            # prefix of progress.csv/lineage_fitness.csv, not a torn one.
            logger.flush()
            lineage_logger.flush()
            save_checkpoint(world, checkpoint_dir / f"checkpoint_step_{step}.pkl")
            last_checkpoint_step = step

    # Deliberately no survivor/censored-row logging here -- see the lineage_logger
    # construction above for why (this run may be resumed again later). Use
    # analyze_proximate_reward.py --checkpoint <this run's final checkpoint> to
    # include current survivors in the analysis without persisting them here.

    elapsed = time.time() - start
    logger.close()
    lineage_logger.close()
    if renderer is not None:
        renderer.close()

    # Always write a final checkpoint, independent of --checkpoint-every, so
    # every run leaves at least one resumable/evaluable state (see checkpoint.py
    # and eval_checkpoint.py) -- and so a run that ends in extinction still
    # records that final (agent-empty) state rather than nothing.
    final_checkpoint = checkpoint_dir / f"checkpoint_step_{world.current_step}.pkl"
    save_checkpoint(world, final_checkpoint)

    print(f"Finished at step {world.current_step} in {elapsed:.1f}s ({world.current_step / max(elapsed, 1e-9):.0f} steps/sec).")
    if extinction_step is not None:
        print(f"Agent population extinction at step {extinction_step}.")
    else:
        print(f"Reached step limit ({args.steps}) without extinction.")
    print(f"Final population: {world.population_counts()}")
    print(f"Log written to: {out_dir / 'progress.csv'}")
    print(f"Lineage log written to: {out_dir / 'lineage_fitness.csv'}")
    print(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
