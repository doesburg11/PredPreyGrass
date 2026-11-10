"""Evaluate one run of any arm at its marks (RESULTS.md sections 1 and 2):
evaluate.py (50 episodes, seeds 0-49) and coop_fair.py (20 episodes) at
each checkpoint, against the deer the checkpoint was saved with (section 2:
D(t), checkpoints.checkpoint_deer), written to
results/<run>_iter<mark>_{eval,coop}.json, with the checkpoint's budget
(physics steps, episodes, births, generations) and its deer.

--panel (section 2, also reported): the final checkpoint against each of
deer_stag_hunt touch_1-3's final deer, evaluate.py 50 episodes, to
results/<run>_final_vs_<touch_j>_eval.json.

Run from the repository root:
    python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/evaluate_run.py \\
        RUN [MARK ...] [--workers 16] [--panel]
"""

import argparse
import json
from pathlib import Path

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import evaluate
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import budget, checkpoint_at
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import DEER_RUNS_DIR, RUNS_DIR
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.coop_fair import coop

RESULTS = Path(__file__).resolve().parents[1] / "results"
MARKS = (250, 500, 750, 1000)
EPISODES = {"eval": 50, "coop": 20}
PANEL = ("touch_1", "touch_2", "touch_3")


def panel_deer(source):
    """The final deer module of a deer_stag_hunt run, its world checked."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import deer_schedule
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_env import config_env

    deer_schedule.check_source(DEER_RUNS_DIR / source, config_env)
    return deer_schedule.module_dir(deer_schedule.source_checkpoint(DEER_RUNS_DIR / source, deer_schedule.LAST))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("marks", nargs="*", type=int)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--panel", action="store_true")
    args = parser.parse_args()
    run = RUNS_DIR / args.run
    RESULTS.mkdir(exist_ok=True)
    for mark in args.marks or MARKS:
        checkpoint = checkpoint_at(run, mark)
        stem = RESULTS / f"{args.run}_iter{mark}"
        summary, rows = evaluate.evaluate(checkpoint, EPISODES["eval"], args.workers)
        summary["budget"] = budget(checkpoint)
        Path(f"{stem}_eval.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
        coop_summary, coop_rows = coop(checkpoint, EPISODES["coop"], args.workers)
        Path(f"{stem}_coop.json").write_text(json.dumps({"checkpoint": str(checkpoint), "deer": summary["deer"], "summary": coop_summary, "episodes": coop_rows}, indent=1))
        print(
            f"{args.run} iter {mark}: coexisting {summary['coexisting']}/50, adults/episode {summary['adults_caught_per_episode']:.2f}, "
            f"attempt success {summary.get('adult_attempt_success')}, adult/chance {coop_summary['adult_over_chance']}, budget {summary['budget']}",
            flush=True,
        )
    if args.panel:
        final = run / "checkpoint"
        for source in PANEL:
            summary, rows = evaluate.evaluate(final, EPISODES["eval"], args.workers, deer=panel_deer(source))
            Path(RESULTS / f"{args.run}_final_vs_{source}_eval.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
            print(f"{args.run} final vs {source}'s final deer: coexisting {summary['coexisting']}/50, adults/episode {summary['adults_caught_per_episode']:.2f}", flush=True)


if __name__ == "__main__":
    main()
