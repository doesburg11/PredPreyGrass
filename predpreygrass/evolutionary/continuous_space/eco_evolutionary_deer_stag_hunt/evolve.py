"""The E arm (DESIGN.md, DESIGN_2.md): evolution only. Hunters move by their
genes (genes.e_arm_logits: the innate bias with stop moves masked, the
cruise move when nothing is in view); the deer are replayed from a
deer_stag_hunt run, as in arms L and B (RESULTS.md section 2). No learning.

Run k uses seed k and the deer of deer_stag_hunt touch_((k - 1) mod 3) + 1,
as run k of L and B (launchers/replicate.py), copied into the run with their
manifest (deer_schedule.py). Deer switch per environment: an environment
moves to D(t) when its own physics steps reach its share (worker_targets) of
the median, over the matched L and B runs, of their total physics steps at
the end of iteration t - 1 (their iterations.jsonl), that is, when they
switched; the thresholds are saved in the run (deer_thresholds.json) and
every switch is logged in its episodes file. Each of WORKERS processes runs one
environment (one population, as each of RLlib's env runners does in L and
B; with --match, WORKERS must equal their number of environments) through
consecutive episodes until its share of the physics-step budget is used.
Every episode is seeded from the run's own generator, so an E run is
reproducible (L and B runs are not: RLlib seeds only their first episodes).
A share is reached after a whole decision, so an environment may overrun it
by up to action_repeat - 1 physics steps. The budget at marks 250, 500, 750
and 1000 is given directly (--targets) or matched (--match) to the median
total physics steps of L and B runs' checkpoints at those iterations
(DESIGN.md, runs and budget); section 2 always matches. At each mark, every environment's gene pool and
lifetime counters are saved in a checkpoint of train.py's layout
(runs/<NAME>/checkpoint/iter_<n>/, the last one runs/<NAME>/checkpoint/),
and every finished episode's statistics go to
runs/<NAME>/episodes_<environment>.jsonl.

Run from the repository root, e.g. under systemd-run --user:

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.evolve \\
        NAME K --match RUN ... [--marks 250 500 ...] [--workers 14]
"""

import argparse
import json
import os
import re
import time
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import deer_schedule, genes
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import BUDGET, GENE_POOLS, budget, checkpoint_at
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_env import config_env
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import RUNS_DIR
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.environment import EvolvingDeerStagHunt, species_of

MARKS = (250, 500, 750, 1000)


def matched_targets(runs, marks=MARKS):
    """{mark: median total physics steps of the runs' checkpoints there}
    and the runs' number of environments (one value, or an error)."""
    budgets = [budget(checkpoint_at(run, mark)) for run in runs for mark in marks]
    environments = {b["environments"] for b in budgets}
    if len(environments) != 1:
        raise SystemExit(f"the matched runs differ in their number of environments: {sorted(environments)}")
    targets = {mark: int(np.median([budget(checkpoint_at(run, mark))["physics_steps"] for run in runs])) for mark in marks}
    return targets, environments.pop()


def physics_steps_by_iteration(run):
    """{iteration: total physics steps at its end} from a run's
    iterations.jsonl (train.py)."""
    rows = [json.loads(line) for line in (Path(run) / "iterations.jsonl").read_text().splitlines() if line.strip()]
    steps = {row["iteration"]: row["physics_steps"] for row in rows}
    if sorted(steps) != list(range(1, len(steps) + 1)):
        raise SystemExit(f"{run}: iterations.jsonl is not one row per iteration from 1")
    return steps


def switch_thresholds(runs, last):
    """{t: median over the runs of their total physics steps at the end of
    iteration t - 1} for every switch iteration t up to last: when the L
    and B runs moved to D(t)."""
    per_run = [physics_steps_by_iteration(run) for run in runs]
    return {t: int(np.median([steps[t - 1] for steps in per_run])) for t in deer_schedule.switch_iterations(last)}


def deer_at(switches, physics_steps):
    """The source iteration of the deer an environment acts with after
    physics_steps of its own, given its {t: threshold} switches (thresholds
    must not decrease with t); several thresholds passed at once take the
    last."""
    order = sorted(switches)
    if any(switches[a] > switches[b] for a, b in zip(order, order[1:])):
        raise ValueError("deer switch thresholds must not decrease")
    passed = [t for t in order if physics_steps >= switches[t]]
    return deer_schedule.source_iteration(passed[-1]) if passed else deer_schedule.source_iteration(1)


class DeerClock:
    """deer_at for one environment, advanced step by step."""

    def __init__(self, switches):
        deer_at(switches, 0)  # checks the thresholds
        self.order = sorted(switches)
        self.switches = switches
        self.next = 0

    def active(self, physics_steps):
        while self.next < len(self.order) and physics_steps >= self.switches[self.order[self.next]]:
            self.next += 1
        return deer_schedule.source_iteration(self.order[self.next - 1]) if self.next else deer_schedule.source_iteration(1)


def worker_targets(targets, workers):
    """[{mark: share}] per environment: the budget split exactly (the first
    total % workers environments take one step more)."""
    return [{mark: total // workers + (index < total % workers) for mark, total in targets.items()} for index in range(workers)]


def run_worker(args):
    """One environment's evolution; returns {mark: (gene pool state,
    lifetime counters)} and writes its finished episodes to a JSONL file."""
    index, env_config, run, seed, targets, switches, beta = args
    import torch

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import deer_module_at

    torch.set_num_threads(1)
    env = EvolvingDeerStagHunt(env_config)
    clock = DeerClock(switches)
    active = clock.active(0)
    deer = deer_module_at(deer_schedule.deer_path(run, active), env)
    lay = genes.layout(env)
    rng = np.random.default_rng([seed, index])
    generator = torch.Generator().manual_seed(seed * 1000 + index)
    marks = sorted(targets)
    snapshots = {}
    log = open(Path(run) / f"episodes_{index:02d}.jsonl", "w")
    seeds = np.random.default_rng([seed, index, 1])
    started = time.time()
    try:
        while len(snapshots) < len(marks):
            obs, _ = env.reset(seed=int(seeds.integers(2**31 - 100)))
            while True:
                actions = act(env, obs, deer, lay, beta, rng, generator)
                obs, _, terms, truncs, _ = env.step(actions)
                now = clock.active(env.lifetime["physics_steps"])
                if now != active:
                    active = now
                    deer = deer_module_at(deer_schedule.deer_path(run, active), env)
                    log.write(json.dumps({"deer_switch": active, "physics_steps": env.lifetime["physics_steps"]}) + "\n")
                for mark in marks:
                    if mark not in snapshots and env.lifetime["physics_steps"] >= targets[mark]:
                        if active != deer_schedule.source_iteration(mark):
                            raise RuntimeError(f"worker {index}: at mark {mark} the deer are of iteration {active}, not {deer_schedule.source_iteration(mark)}")
                        snapshots[mark] = (env.gene_pool_state(), dict(env.lifetime))
                        print(f"worker {index}: mark {mark} at {env.lifetime['physics_steps']} physics steps, {env.lifetime['episodes']} episodes, {time.time() - started:.0f}s", flush=True)
                if terms["__all__"] or truncs["__all__"]:
                    log.write(json.dumps(env.episode_stats) + "\n")
                    log.flush()
                    break
                if len(snapshots) == len(marks):
                    break
    finally:
        log.close()
    return snapshots


def act(env, obs, deer, lay, beta, rng, generator):
    """Predators by genes.e_arm_logits (numpy rng), deer by their policy
    (torch generator), agents in sorted order."""
    import torch
    from ray.rllib.core.columns import Columns

    agents = sorted(a for a in obs if a in env.active)
    actions = {}
    hunters = [a for a in agents if species_of(a) == "predator"]
    if hunters:
        moves = genes.sample(genes.e_arm_logits(np.stack([obs[a] for a in hunters]), lay, beta), rng)
        actions.update(zip(hunters, (int(m) for m in moves)))
    herd = [a for a in agents if species_of(a) == "deer"]
    if herd:
        batch = torch.from_numpy(np.stack([np.asarray(obs[a], dtype=np.float32) for a in herd]))
        with torch.inference_mode():
            logits = deer.forward_inference({Columns.OBS: batch})[Columns.ACTION_DIST_INPUTS]
        moves = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(1)
        actions.update(zip(herd, moves.tolist()))
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("k", type=int, help="run number: seed k, deer of deer_stag_hunt run ((k - 1) mod 3) + 1")
    parser.add_argument("--match", nargs="+", required=True, help="L and B run directories (same deer as run k) to match the budget and deer switches to")
    parser.add_argument("--marks", type=int, nargs="+", default=list(MARKS), help="iterations to match with --match")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        raise SystemExit("NAME: letters, digits, _ and - only")
    run = RUNS_DIR / args.name
    if run.exists():
        raise SystemExit(f"refusing: {run} exists")
    if args.k < 1:
        raise SystemExit("K starts at 1")
    targets, environments = matched_targets(args.match, args.marks)
    if environments != args.workers:
        raise SystemExit(f"--workers {args.workers} differs from the matched runs' {environments} environments")
    marks = sorted(targets)
    if any(targets[a] > targets[b] for a, b in zip(marks, marks[1:])):
        raise SystemExit(f"budgets must not decrease: {targets}")

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import train
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.replicate import deer_source

    source = deer_source(args.k)
    for matched in args.match:
        their = deer_schedule.manifest(matched)
        if their is None or their["source_run"] != str(Path(source).resolve()):
            raise SystemExit(f"{matched} does not replay the deer of {source}")
    if config_env["deer_catch_rule"] != "touch":
        raise SystemExit("section 2 runs in the touch world (config/config_env.py)")
    thresholds = switch_thresholds(args.match, max(marks))

    settings = {
        "arm": "E",
        "beta": genes.BETA0,
        "deer_schedule": os.path.abspath(source),
        "deer_checkpoint": None,
        "seed": args.k,
        "iterations": max(marks),
        "num_env_runners": args.workers,
        "physics_step_targets": targets,
        "matched_to": [os.path.abspath(r) for r in args.match] if args.match else None,
        "vf_share_layers": False,
    }
    run.mkdir(parents=True)
    train.save_source_code(run, dict(config_env), SimpleNamespace(**settings))
    deer_schedule.copy_schedule(source, run, max(marks), dict(config_env))
    (run / "deer_thresholds.json").write_text(json.dumps({str(t): v for t, v in thresholds.items()}) + "\n")
    print(f"E run {args.name}: budget {targets}, {args.workers} environments", flush=True)
    shares = worker_targets(targets, args.workers)
    switch_shares = worker_targets(thresholds, args.workers)
    tasks = [(i, dict(config_env), str(run), args.k, shares[i], switch_shares[i], genes.BETA0) for i in range(args.workers)]
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(run_worker, tasks)
    for mark in marks:
        checkpoint = checkpoint_at(run, mark)
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / GENE_POOLS).write_text(json.dumps([r[mark][0] for r in results]) + "\n")
        (checkpoint / BUDGET).write_text(json.dumps([r[mark][1] for r in results]) + "\n")
        deer_schedule.write_active(checkpoint, deer_schedule.source_iteration(mark), run)
        print(f"checkpoint {checkpoint}: {budget(checkpoint)}", flush=True)


if __name__ == "__main__":
    main()
