"""RESULTS.md section 3's cross-play panel: at a checkpoint mark, every
run's hunters (policy plus the checkpoint's founder gene pool) against
every other run's deer at the same mark, evaluate.py 50 episodes, seeds
0-49, to results/<hunters>_iter<mark>_vs_<deer>_eval.json. Own-run cells
are the own-world evaluations (launchers/evaluate_run.py) and are not
repeated here. Existing result files whose recorded checkpoint, deer and
hashes match are kept, so the panel can be resumed.

Each pair is refused unless both runs learned their deer (section 3) in
the same world, with the same deer observation and action spaces.
Evaluations run one at a time, each with its own worker pool.

Run from the repository root:
    python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/cross_play.py \\
        MARK RUN [RUN ...] [--workers 16]
"""

import argparse
import json
from pathlib import Path

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import evaluate
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import checkpoint_at, checkpoint_deer, checkpoint_env_config, run_settings
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import RUNS_DIR
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.deer_schedule import sha256

RESULTS = Path(__file__).resolve().parents[1] / "results"
EPISODES = 50


def gene_free(config):
    return {k: v for k, v in config.items() if not k.startswith("gene_")}


def check_pair(hunters, deer):
    """Refuse a hunter checkpoint and a deer checkpoint that are not both
    section 3 checkpoints of one world (gene settings aside: the hunters'
    world is used)."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.environment import EvolvingDeerStagHunt

    for checkpoint in (hunters, deer):
        if not run_settings(checkpoint)["config_ppo"].get("deer_learn"):
            raise ValueError(f"{checkpoint}: not a learning-deer (section 3) run")
    a, b = checkpoint_env_config(hunters), checkpoint_env_config(deer)
    if gene_free(a) != gene_free(b):
        differ = sorted(k for k in set(gene_free(a)) | set(gene_free(b)) if a.get(k) != b.get(k))
        raise ValueError(f"{hunters} and {deer}: different worlds in {differ}")
    ea, eb = EvolvingDeerStagHunt(a), EvolvingDeerStagHunt(b)
    if ea.get_observation_space("deer_0") != eb.get_observation_space("deer_0") or ea.get_action_space("deer_0") != eb.get_action_space("deer_0"):
        raise ValueError(f"{hunters} and {deer}: deer observation or action spaces differ")


def cell_path(hunter_run, deer_run, mark):
    return RESULTS / f"{hunter_run}_iter{mark}_vs_{deer_run}_eval.json"


def provenance(hunters, deer_dir):
    return {
        "checkpoint": str(hunters),
        "deer": str(deer_dir),
        "predator_sha256": sha256(Path(hunters) / "learner_group" / "learner" / "rl_module" / "predator_policy" / "module_state.pkl"),
        "deer_sha256": sha256(Path(deer_dir) / "module_state.pkl"),
    }


def done(path, expected):
    if not path.is_file():
        return False
    data = json.loads(path.read_text())
    return data.get("provenance") == expected and data["summary"]["episodes"] == EPISODES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mark", type=int)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs):
        raise SystemExit("runs repeated")
    RESULTS.mkdir(exist_ok=True)
    checkpoints = {run: checkpoint_at(RUNS_DIR / run, args.mark) for run in args.runs}
    for hunter_run in args.runs:
        for deer_run in args.runs:
            if hunter_run == deer_run:
                continue
            hunters, deer_dir = checkpoints[hunter_run], checkpoint_deer(checkpoints[deer_run])
            check_pair(hunters, checkpoints[deer_run])
            expected = provenance(hunters, deer_dir)
            path = cell_path(hunter_run, deer_run, args.mark)
            if done(path, expected):
                continue
            summary, rows = evaluate.evaluate(hunters, EPISODES, args.workers, deer=deer_dir)
            if summary["deer"] != str(deer_dir) or summary["checkpoint"] != str(hunters):
                raise RuntimeError(f"{path}: evaluated {summary['checkpoint']} with {summary['deer']}")
            path.write_text(json.dumps({"provenance": expected, "hunter_run": hunter_run, "deer_run": deer_run, "mark": args.mark, "summary": summary, "episodes": rows}, indent=1))
            print(f"iter {args.mark}: {hunter_run} hunters vs {deer_run} deer: coexisting {summary['coexisting']}/50, adults/episode {summary['adults_caught_per_episode']:.2f}", flush=True)


if __name__ == "__main__":
    main()
