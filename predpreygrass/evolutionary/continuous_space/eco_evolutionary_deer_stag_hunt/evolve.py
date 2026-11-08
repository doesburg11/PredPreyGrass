"""The E arm (DESIGN.md): evolution only. Hunters move by their genes
(genes.e_arm_logits: the innate bias with stop moves masked, the cruise move
when nothing is in view); the deer use a frozen deer_stag_hunt policy, as in
arms L and B. No learning.

Run k uses seed k and the deer of deer_stag_hunt run ((k - 1) mod 3) + 1, as
run k of L and B (launchers/replicate.py). Each of WORKERS processes runs one
environment (one population, as each of RLlib's env runners does in L and
B; with --match, WORKERS must equal their number of environments) through
consecutive episodes until its share of the physics-step budget is used.
Every episode is seeded from the run's own generator, so an E run is
reproducible (L and B runs are not: RLlib seeds only their first episodes).
A share is reached after a whole decision, so an environment may overrun it
by up to action_repeat - 1 physics steps. The budget at marks 250, 500, 750
and 1000 is given directly (--targets) or matched (--match) to the median
total physics steps of L and B runs' checkpoints at those iterations
(DESIGN.md, runs and budget). At each mark, every environment's gene pool and
lifetime counters are saved in a checkpoint of train.py's layout
(runs/<NAME>/checkpoint/iter_<n>/, the last one runs/<NAME>/checkpoint/),
and every finished episode's statistics go to
runs/<NAME>/episodes_<environment>.jsonl.

Run from the repository root, e.g. under systemd-run --user:

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.evolve \\
        NAME K (--match RUN ... [--marks 250 500 ...] | --targets 250=N,500=N,...) [--workers 14]
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

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import genes
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


def parse_targets(text):
    targets = {}
    for item in text.split(","):
        mark, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--targets needs MARK=STEPS items, not {item!r}")
        targets[int(mark)] = int(value)
    return targets


def worker_targets(targets, workers):
    """[{mark: share}] per environment: the budget split exactly (the first
    total % workers environments take one step more)."""
    return [{mark: total // workers + (index < total % workers) for mark, total in targets.items()} for index in range(workers)]


def run_worker(args):
    """One environment's evolution; returns {mark: (gene pool state,
    lifetime counters)} and writes its finished episodes to a JSONL file."""
    index, env_config, run, seed, targets, beta = args
    import torch

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import deer_module

    torch.set_num_threads(1)
    env = EvolvingDeerStagHunt(env_config)
    deer = deer_module(run, env)
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
                for mark in marks:
                    if mark not in snapshots and env.lifetime["physics_steps"] >= targets[mark]:
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--match", nargs="+", help="L and B run directories to match the budget to")
    group.add_argument("--targets", help="MARK=PHYSICS_STEPS,... total over all environments")
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
    if args.match:
        targets, environments = matched_targets(args.match, args.marks)
        if environments != args.workers:
            raise SystemExit(f"--workers {args.workers} differs from the matched runs' {environments} environments")
    else:
        targets = parse_targets(args.targets)
    marks = sorted(targets)
    if any(targets[a] > targets[b] for a, b in zip(marks, marks[1:])):
        raise SystemExit(f"budgets must not decrease: {targets}")

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import train
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.replicate import deer_checkpoint

    deer = deer_checkpoint(args.k)

    settings = {
        "arm": "E",
        "beta": genes.BETA0,
        "deer_checkpoint": os.path.abspath(deer),
        "seed": args.k,
        "iterations": max(marks),
        "num_env_runners": args.workers,
        "physics_step_targets": targets,
        "matched_to": [os.path.abspath(r) for r in args.match] if args.match else None,
        "vf_share_layers": False,
    }
    run.mkdir(parents=True)
    train.save_source_code(run, dict(config_env), SimpleNamespace(**settings))
    train.copy_deer_policy(deer, run)
    print(f"E run {args.name}: budget {targets}, {args.workers} environments", flush=True)
    shares = worker_targets(targets, args.workers)
    tasks = [(i, dict(config_env), str(run), args.k, shares[i], genes.BETA0) for i in range(args.workers)]
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(run_worker, tasks)
    for mark in marks:
        checkpoint = checkpoint_at(run, mark)
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / GENE_POOLS).write_text(json.dumps([r[mark][0] for r in results]) + "\n")
        (checkpoint / BUDGET).write_text(json.dumps([r[mark][1] for r in results]) + "\n")
        print(f"checkpoint {checkpoint}: {budget(checkpoint)}", flush=True)


if __name__ == "__main__":
    main()
