"""Does a second predator causally help catch a prey? A counterfactual test
of group hunting (README, "Group hunting: counterfactual test").

measure_predators.py found that when another predator heads for the same
prey (joint pursuit), the prey is caught more often, also by the focal
predator. That is an association: both predators may simply pick the same
easy prey. This script removes the selection: at a joint encounter it
copies the whole world and replays the same decision from the same state
in three branches,

- factual: every agent's chosen action;
- helper stops: the helper(s) heading for the prey stand still instead;
- helper turns away: the helper(s) run straight away from the prey;

and records whether the prey is caught, by anyone and by the focal
predator, and whether the focal predator is fed from it (the same as
catching it, unless the environment shares a catch's energy:
energy_catch_share_radius), within 1 and within HORIZON decisions (helpers keep the
replacement action throughout; everyone else samples from the policy with
the same seed in every branch). If the focal predator catches the prey
less often without its helper, the helper causally helps it: group
hunting. If it catches it more often, the helper competes.

Placebo: at a solo encounter, the same replay with a predator that is
nearby (within 200) but not heading for the prey stopped instead. Its
effect should be near zero; a clear effect would mean the method itself is
biased. Determinism check: the factual branch must reproduce what actually
happened in the episode (reported as mismatches).

Usage, from the repository root:

    python -m predpreygrass.non_evolutionary.continuous_space.foraging_swarms.counterfactual_hunting \\
        <checkpoint> <label> [--episodes 50] [--seed 0] [--workers 25]
"""

import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import torch

from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.checkpoints import (
    checkpoint_env_config,
    load_modules,
)
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.environment import (
    DIRECTIONS,
    SPEEDS,
    PredPreyGrassContinuous,
)
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.geometry import Vector
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.measure_predators import (
    APPROACHING,
    ENCOUNTER,
    JOIN_RADIUS,
    _UNITS,
    intended_velocity,
    toward,
)
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.measure_swarms import (
    MIN_AGE,
    layout,
    sample_actions,
)

HERE = Path(__file__).resolve().parent
HORIZON = 3  # decisions followed in each branch
PLACEBO_RADIUS = 200.0
CAP = 40  # joint encounters, and placebo encounters, replayed per episode
REPLAY_SHARE = 0.25  # chance that a qualifying encounter is replayed (until CAP)
STOP_ACTION = SPEEDS.index(0.0) * DIRECTIONS
BOOTSTRAP = 2000
BRANCHES = ("factual", "stop", "away")


def away_action(env, helper, prey_position):
    """Full speed in the direction most directly away from the prey."""
    dx, dy = env.torus.offset(helper.position, prey_position)
    return int(np.argmax(_UNITS @ np.array([-dx, -dy])))  # speed index 0: full speed


def rollout(env, modules, actions, overrides, target, seed, horizon=HORIZON):
    """Replay from a copy of env: actions for the first decision, with the
    overridden agents fixed for every decision. Returns (catcher id or
    None, decision index of the catch or None, ids fed from the catch)."""
    sim = copy.deepcopy(env)
    sim.__dict__.pop("_move", None)  # drop any recording wrapper of the original
    move = sim._move
    caught = {}

    def recording(animal, action, catcher, aways, catches):
        if catcher is not None and animal.id not in caught:
            caught[animal.id] = (catcher.id, k)
        move(animal, action, catcher, aways, catches)

    sim._move = recording
    generator = torch.Generator().manual_seed(seed)
    step_actions = dict(actions)
    for k in range(horizon):
        if k > 0:
            step_actions = sample_actions(modules, obs, generator)
        step_actions.update({a: v for a, v in overrides.items() if a in sim.active})
        step_actions = {a: step_actions[a] for a in sim.active}
        obs, _, terms, truncs, _ = sim.step(step_actions)
        if target in caught:
            fed = next((c["fed"] for c in sim.catch_log if c["prey"] == target), [])
            return (*caught[target], fed)
        if terms["__all__"] or truncs["__all__"]:
            break
    return None, None, []


def find_encounters(env, obs, actions, born_at, decision, lay):
    """[(focal, target prey id, helper ids, bystander ids)] for predators
    old enough with a prey within ENCOUNTER: helpers are other old
    predators within JOIN_RADIUS whose chosen action heads for that prey;
    bystanders are other old predators within PLACEBO_RADIUS that do not."""
    out = []
    start = lay["prey"]
    for p in env.predators:
        o = obs.get(p.id)
        if o is None or decision - born_at[p.id] < MIN_AGE or o[start] <= 0:
            continue
        reach = p.view_distance
        dx, dy, d = o[start + 1] * reach, o[start + 2] * reach, o[start + 3] * reach
        if d > ENCOUNTER:
            continue
        spot = Vector((p.position.x + dx) % env.width, (p.position.y + dy) % env.height)
        target = min(env.prey, key=lambda q: env.torus.distance(spot, q.position))
        if env.torus.distance(spot, target.position) > 1.0:
            continue
        helpers, bystanders = [], []
        for q in env.predators:
            if q is p or decision - born_at[q.id] < MIN_AGE or q.id not in actions:
                continue
            dist = env.torus.distance(p.position, q.position)
            tx, ty = env.torus.offset(q.position, target.position)
            heading = toward(*intended_velocity(int(actions[q.id])), tx, ty) >= APPROACHING
            if heading and dist <= JOIN_RADIUS:
                helpers.append(q.id)
            elif not heading and dist <= PLACEBO_RADIUS:
                bystanders.append(q.id)
        out.append((p.id, target.id, helpers, bystanders))
    return out


def play(env, modules, seed):
    rng = np.random.default_rng(100_000 + seed)
    generator = torch.Generator().manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    lay = layout(env)
    born_at = {}
    decision = 0
    records = []  # (kind, focal, target, {branch: (catcher, k)}), resolved after the real step
    sums, counts = defaultdict(float), defaultdict(int)
    n_joint = n_placebo = 0
    real_catches = {}
    move = env._move

    def recording(animal, action, catcher, aways, catches):
        if catcher is not None and animal.id not in real_catches:
            real_catches[animal.id] = catcher.id
        move(animal, action, catcher, aways, catches)

    env._move = recording
    try:
        while True:
            for p in env.predators:
                born_at.setdefault(p.id, decision)
            actions = sample_actions(modules, obs, generator)
            pending = []
            for focal, target, helpers, bystanders in find_encounters(env, obs, actions, born_at, decision, lay):
                if helpers and n_joint < CAP and rng.random() < REPLAY_SHARE:
                    kind, changed = "joint", helpers
                    n_joint += 1
                elif not helpers and bystanders and n_placebo < CAP and rng.random() < REPLAY_SHARE:
                    kind, changed = "placebo", bystanders[:1]
                    n_placebo += 1
                else:
                    continue
                target_position = next(q.position for q in env.prey if q.id == target)
                branch_seed = int(rng.integers(2**31))
                overrides = {
                    "factual": {},
                    "stop": {h: STOP_ACTION for h in changed},
                    "away": {h: away_action(env, next(q for q in env.predators if q.id == h), target_position) for h in changed},
                }
                outcome = {b: rollout(env, modules, actions, overrides[b], target, branch_seed) for b in BRANCHES}
                pending.append((kind, focal, target, outcome))
            real_catches.clear()
            obs, _, terms, truncs, _ = env.step(actions)
            decision += 1
            for kind, focal, target, outcome in pending:
                fact_catcher, fact_k, _ = outcome["factual"]
                real = real_catches.get(target)
                counts["determinism_checked"] += 1
                if (fact_catcher if fact_k == 0 else None) != real:
                    counts["determinism_mismatch"] += 1
                for b, (catcher, k, fed) in outcome.items():
                    for horizon in (1, HORIZON):
                        within = k is not None and k < horizon
                        key = f"{kind}|{b}|h{horizon}"
                        sums[f"{key}|anyone"] += within
                        sums[f"{key}|self"] += within and catcher == focal
                        sums[f"{key}|other"] += within and catcher != focal
                        sums[f"{key}|fed"] += within and focal in fed
                        counts[key] += 1
            if terms["__all__"] or truncs["__all__"]:
                break
    finally:
        env._move = move
    return {"seed": seed, "sums": dict(sums), "counts": dict(counts)}


# --- Summary --------------------------------------------------------------------


def paired_differences(episodes, rng, resamples=BOOTSTRAP):
    """factual minus counterfactual, per kind, horizon, outcome and branch,
    with 95% intervals from resampling whole episodes. Positive: the
    changed predator(s) raised that outcome."""
    out = {}
    for kind in ("joint", "placebo"):
        for horizon in (1, HORIZON):
            for outcome in ("anyone", "self", "other", "fed"):
                for branch in ("stop", "away"):
                    fk, ck = f"{kind}|factual|h{horizon}", f"{kind}|{branch}|h{horizon}"
                    fs = np.array([e["sums"].get(f"{fk}|{outcome}", 0.0) for e in episodes])
                    cs = np.array([e["sums"].get(f"{ck}|{outcome}", 0.0) for e in episodes])
                    n = np.array([e["counts"].get(fk, 0) for e in episodes])
                    if n.sum() == 0:
                        continue

                    def stat(index):
                        return (fs[index].sum() - cs[index].sum()) / max(n[index].sum(), 1)

                    draws = [stat(rng.integers(len(episodes), size=len(episodes))) for _ in range(resamples)]
                    out[f"{kind}|h{horizon}|{outcome}|factual_minus_{branch}"] = {
                        "factual": float(fs.sum() / n.sum()),
                        "counterfactual": float(cs.sum() / n.sum()),
                        "difference": float(stat(np.arange(len(episodes)))),
                        "low": float(np.percentile(draws, 2.5)),
                        "high": float(np.percentile(draws, 97.5)),
                        "n": int(n.sum()),
                    }
    return out


def format_summary(label, s):
    lines = [f"== {label}: {s['episodes']} episodes; determinism mismatches {s['determinism_mismatch']} of {s['determinism_checked']}"]
    lines.append("   kind     horizon outcome  branch  factual  counterf.  difference (95% interval)  n")
    for key, d in s["differences"].items():
        kind, h, outcome, branch = key.split("|")
        lines.append(f"   {kind:8s} {h:7s} {outcome:8s} {branch.replace('factual_minus_', ''):6s}  {d['factual']:.3f}    {d['counterfactual']:.3f}"
                     f"     {d['difference']:+.3f} ({d['low']:+.3f} to {d['high']:+.3f})  {d['n']}")
    return "\n".join(lines)


_worker: dict = {}


def _init(checkpoint, env_config):
    torch.set_num_threads(1)
    _worker["env"] = PredPreyGrassContinuous(env_config)
    _worker["modules"] = load_modules(checkpoint)


def _episode(seed):
    return play(_worker["env"], _worker["modules"], seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("label")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(25, os.cpu_count() or 1))
    parser.add_argument("--out-dir", default=str(HERE / "analysis" / "results"))
    parser.add_argument("--override", type=json.loads, default={},
                        help="environment settings changed at test time, e.g. a shared carcass for policies trained without one")
    args = parser.parse_args()
    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    env_config = checkpoint_env_config(checkpoint)
    env_config.update(args.override)
    start = time.time()
    seeds = list(range(args.seed, args.seed + args.episodes))
    with get_context("spawn").Pool(min(args.workers, len(seeds)), initializer=_init, initargs=(checkpoint, env_config)) as pool:
        episodes = pool.map(_episode, seeds, chunksize=1)
    summary = {
        "label": args.label,
        "checkpoint": checkpoint,
        "override": args.override,
        "episodes": len(episodes),
        "seeds": seeds,
        "determinism_checked": sum(e["counts"].get("determinism_checked", 0) for e in episodes),
        "determinism_mismatch": sum(e["counts"].get("determinism_mismatch", 0) for e in episodes),
        "differences": paired_differences(episodes, np.random.default_rng(0)),
        "settings": dict(HORIZON=HORIZON, PLACEBO_RADIUS=PLACEBO_RADIUS, CAP=CAP, REPLAY_SHARE=REPLAY_SHARE,
                         ENCOUNTER=ENCOUNTER, JOIN_RADIUS=JOIN_RADIUS, APPROACHING=APPROACHING, MIN_AGE=MIN_AGE),
        "wall_seconds": round(time.time() - start, 1),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(summary, indent=1) + "\n")
    text = format_summary(args.label, summary)
    (out / f"{args.label}.txt").write_text(text + "\n")
    print(text)
    print(f"({summary['wall_seconds']} s; written to {out / args.label}.json)")


if __name__ == "__main__":
    sys.exit(main())
