"""
Counterfactual test of what the mate-proximity association is made of (RESULTS.md, Iterations 14, 16, 17).

Structural fact this rests on: a predator's observation (`_get_observation`) has five channels -- border, ONE
predator layer shared by both sexes (it holds only an energy value), prey, grass, fruit. A predator therefore
cannot tell whether a neighbor is male or female, nor whether it is its recorded mate. Whatever the female policy
does differently "with her mate nearby" can only be a response to the predator layer as such (their presence and
energy), or to something correlated with it. This script asks how sensitive the policy is to edits of that layer.

Method (policy-only, no training): sample on-policy states from rollouts in the env each run was trained in
(analysis_env.py), for one sex's policy, restricted to agents that have a recorded mate history (near/far/dead/
abandoned; virgins excluded) at states where a target (fruit or prey) is visible. So the estimand is conditional on
having reproduced, being alive, and seeing a target -- not the policy's whole state distribution. For each state the
policy is queried on edited copies of the SAME observation:
  base      the observation as it was
  alone     every other predator removed from the window (only the agent's own cell in the predator layer kept)
  +d{1,2,3} `alone` plus ONE inserted predator (energy --insert-energy, default 8) at Chebyshev distance d
  +d2 e3 / e12   the same at d=2 with a low-/high-energy neighbor
Every insertion effect is AVERAGED OVER ALL legal placements at that distance (any in-grid cell not already holding
a predator; food cells are allowed), so there is no placement sampling noise, and the energy variants share the
placements of +d2 exactly. The approach bias is the same metric as the other scripts (expected distance-after under a
uniform mover minus under the policy). It is computed two ways: "total" uses, for EACH edited observation, the
environment's blocking rule (a move onto a cell holding another predator leaves the agent in place; a frozen-neighbor
snapshot -- the real env moves agents sequentially), so an inserted predator changes both the policy's probabilities
and the mechanics; "policy" ignores blocking entirely (unblocked distances everywhere, as in the other scripts), so
it isolates the change in the policy's action probabilities. Total minus policy is the mechanical part. Effects are paired per state on a common complete-case set of states (the same states for
every condition) and reported with a 95% episode-cluster percentile bootstrap (whole episodes resampled; the
observational near-minus-away difference is bootstrapped the same way). Output gives states and episodes used.

  crowding effect  = bias(+dK) - bias(alone): what adding ONE anonymous predator does to this policy
  isolation effect = bias(alone) - bias(base): what removing whoever was actually there does
  observational near-away = mean unblocked bias with the recorded mate within --near-radius minus mean otherwise, same
                            states (comparable to the near-away numbers of the other scripts); needs >= 5 episodes in
                            each arm

What it can and cannot show: the "policy" crowding effect measures the policy's sensitivity to anonymous predator-layer
edits; the "total" effect adds collision mechanics and can be nonzero even for a policy that ignores its observation.
A policy crowding effect of similar size and sign to the observational near-away difference makes generic-neighbor
responsiveness a plausible route for the association; it does NOT show the association is explained by it (the observed groups also differ in
density, number and energy of neighbors, location, history and survival, and an inserted predator on a random legal
cell is not how real neighbors are distributed). It says nothing about training-time coordination.

Example:
  python -m predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_counterfactual_crowding \
      --run CONTROL_s42=~/simulation_results/ray_results/PPO_FIXED_PREY_DENSITY_CONTROL_SEED42 --episodes 20
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from predpreygrass.non_evolutionary.predator_sexual_reproduction.analysis_env import make_env
from predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_mate_contingency import mate_bucket
from predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_prey_approach_from_checkpoint import (
    POLICY_IDS,
    action_geometry,
    load_modules,
    module_probs,
    policy_of,
)

PRED_CH, PREY_CH, GRASS_CH, FRUIT_CH = 1, 2, 3, 4
TARGETS = ("prey", "fruit")


def remove_neighbors(obs, offset):
    """Copy of obs with every predator except the agent's own (center cell) removed."""
    out = obs.copy()
    own = out[PRED_CH, offset, offset]
    out[PRED_CH] = 0.0
    out[PRED_CH, offset, offset] = own
    return out


def placements(alone, offset, dist):
    """All in-grid cells at Chebyshev distance `dist` from the center holding no predator (food cells allowed)."""
    W = alone.shape[1]
    return [
        (x, y)
        for x in range(W)
        for y in range(W)
        if max(abs(x - offset), abs(y - offset)) == dist and alone[0, x, y] == 0 and alone[PRED_CH, x, y] == 0
    ]


def with_predator(alone, cell, energy):
    out = alone.copy()
    out[PRED_CH, cell[0], cell[1]] = energy
    return out


def blocked_actions(obs, offset, moves):
    """True for actions whose destination cell holds another predator (the env leaves the agent in place)."""
    return np.array(
        [(dx, dy) != (0, 0) and obs[PRED_CH, offset + dx, offset + dy] > 0 for dx, dy in moves], dtype=bool
    )


def edited_d_after(d_free, d_stay, obs, offset, moves, blocking=True):
    if not blocking:
        return d_free
    return np.where(blocked_actions(obs, offset, moves), d_stay, d_free)


def approach_bias(probs, d_after):
    return d_after.mean() - probs @ d_after


def build_bank(env_config, modules, policy_id, n_episodes, seed0, every, near_radius, max_states):
    env = make_env(env_config)
    moves = np.array([env.action_to_move_tuple[a] for a in range(env.num_actions)])
    offset = (env.predator_obs_range - 1) // 2
    rng = np.random.default_rng(seed0)
    bank = []  # dicts: obs, episode, near(bool), geo per target
    for ep in range(n_episodes):
        observations, _ = env.reset(seed=seed0 + ep)
        active = list(observations.keys())
        step = 0
        while True:
            targets = {
                "prey": np.array(list(env.prey_positions.values()), dtype=int).reshape(-1, 2),
                "fruit": np.array(
                    [p for f, p in env.fruit_positions.items() if env.fruit_energies[f] > 0], dtype=int
                ).reshape(-1, 2),
            }
            actions = {}
            for pid in POLICY_IDS:
                agents = [a for a in active if policy_of(a) == pid]
                if not agents:
                    continue
                rows = module_probs(modules[pid], [observations[a] for a in agents])
                for i, agent in enumerate(agents):
                    if pid == policy_id and step % every == 0:
                        bucket = mate_bucket(env, agent, near_radius)
                        if bucket in ("near", "far", "dead", "abandoned"):  # has a recorded mate history
                            pos = np.array(env.agent_positions[agent], dtype=int)
                            geo = {t: action_geometry(pos, targets[t], moves, env.grid_size, offset) for t in TARGETS}
                            stay = {
                                t: (np.sqrt(((targets[t] - pos) ** 2).sum(1)).min() if len(targets[t]) else None)
                                for t in TARGETS
                            }
                            if any(g is not None for g in geo.values()):
                                bank.append(
                                    {
                                        "obs": np.array(observations[agent], dtype=np.float64),
                                        "episode": ep,
                                        "near": bucket == "near",
                                        "geo": {t: ((g[0], stay[t]) if g is not None else None) for t, g in geo.items()},
                                    }
                                )
                    actions[agent] = int(rng.choice(env.num_actions, p=rows[i]))
            observations, _, terminations, truncations, _ = env.step(actions)
            step += 1
            active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
            if terminations.get("__all__") or truncations.get("__all__"):
                break
    if len(bank) > max_states:
        keep = np.sort(rng.choice(len(bank), size=max_states, replace=False))
        bank = [bank[i] for i in keep]
    return bank, offset, moves


def conditions(insert_energy):
    return [("base", None, None), ("alone", None, None)] + [
        (f"+d{d}", d, insert_energy) for d in (1, 2, 3)
    ] + [("+d2 e3", 2, 3.0), ("+d2 e12", 2, 12.0)]


def evaluate(bank, offset, moves, module, insert_energy, blocking=True):
    """bias[cond][target] arrays (N,) with NaN where the condition or the target is not available. Insertion
    conditions average the bias over ALL legal placements."""
    conds = conditions(insert_energy)
    N = len(bank)
    bias = {name: {t: np.full(N, np.nan) for t in TARGETS} for name, _, _ in conds}
    # single-observation conditions
    for name in ("base", "alone"):
        obs_list = [b["obs"] if name == "base" else remove_neighbors(b["obs"], offset) for b in bank]
        probs = module_probs(module, obs_list)
        for i, b in enumerate(bank):
            for t in TARGETS:
                if b["geo"][t] is not None:
                    d = edited_d_after(*b["geo"][t], obs_list[i], offset, moves, blocking)
                    bias[name][t][i] = approach_bias(probs[i], d)
    # insertion conditions: every legal placement, then average per state
    for name, dist, energy in conds:
        if dist is None:
            continue
        flat, owner = [], []
        for i, b in enumerate(bank):
            alone = remove_neighbors(b["obs"], offset)
            for cell in placements(alone, offset, dist):
                flat.append(with_predator(alone, cell, energy))
                owner.append(i)
        if not flat:
            continue
        probs = module_probs(module, flat)
        acc = {t: np.zeros(N) for t in TARGETS}
        cnt = {t: np.zeros(N) for t in TARGETS}
        for k, i in enumerate(owner):
            for t in TARGETS:
                g = bank[i]["geo"][t]
                if g is not None:
                    acc[t][i] += approach_bias(probs[k], edited_d_after(*g, flat[k], offset, moves, blocking))
                    cnt[t][i] += 1
        for t in TARGETS:
            ok = cnt[t] > 0
            bias[name][t][ok] = acc[t][ok] / cnt[t][ok]
    return bias


def common_mask(bias, target):
    ok = np.ones(len(next(iter(bias.values()))[target]), dtype=bool)
    for name in bias:
        ok &= ~np.isnan(bias[name][target])
    return ok


MIN_EPISODES = 5


def _cluster_stat(fn, episodes, rng, n_boot=2000):
    """fn(state_weights) -> statistic; episode-cluster percentile bootstrap (state weight = its episode's multiplicity)."""
    eps, inv = np.unique(episodes, return_inverse=True)
    if len(eps) < MIN_EPISODES:
        return None
    point = fn(np.ones(len(episodes)))
    mult = rng.multinomial(len(eps), np.ones(len(eps)) / len(eps), size=n_boot)
    boots = np.array([fn(m[inv].astype(float)) for m in mult])
    boots = boots[~np.isnan(boots)]
    if len(boots) < 100:
        return None
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, len(eps)


def wmean(x, w):
    w = np.asarray(w, dtype=float)
    return np.nan if w.sum() == 0 else float((x * w).sum() / w.sum())


def summarize(label, sex, bank, bias_total, bias_policy, rng):
    lines = []
    episodes = np.array([b["episode"] for b in bank])
    near = np.array([b["near"] for b in bank])
    n_neighbors = np.array([(b["obs"][PRED_CH] > 0).sum() - 1 for b in bank])  # others in window (own cell excluded)
    lines.append(
        f"{label} {sex}: {len(bank)} sampled states from {len(np.unique(episodes))} episodes, "
        f"{int((n_neighbors > 0).sum())} with >=1 other predator in view, "
        f"{int(near.sum())} with the recorded mate within the near radius"
    )
    fmt = lambda r: "n/a" if r is None or not np.isfinite(r[0]) else f"{r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}]"
    for t in TARGETS:
        ok = common_mask(bias_total, t) & common_mask(bias_policy, t)  # same states for every condition and version
        ep, nr = episodes[ok], near[ok]
        n_used, n_ep = int(ok.sum()), len(np.unique(ep))
        if n_used < 30:
            lines.append(f"    {t:5}: only {n_used} common states, skipped")
            continue
        tot = {name: bias_total[name][t][ok] for name in bias_total}
        pol = {name: bias_policy[name][t][ok] for name in bias_policy}
        n_near_ep, n_away_ep = len(np.unique(ep[nr])), len(np.unique(ep[~nr]))
        parts = [f"common states={n_used}, episodes={n_ep} (near {n_near_ep}, away {n_away_ep})"]
        if min(n_near_ep, n_away_ep) >= MIN_EPISODES:
            obs = _cluster_stat(lambda w: wmean(pol["base"], w * nr) - wmean(pol["base"], w * ~nr), ep, rng)
        else:
            obs = None
        parts.append("observational near-away=" + fmt(obs))
        parts.append("isolation policy=" + fmt(_cluster_stat(lambda w: wmean(pol["alone"] - pol["base"], w), ep, rng)))
        parts.append("isolation total=" + fmt(_cluster_stat(lambda w: wmean(tot["alone"] - tot["base"], w), ep, rng)))
        lines.append(f"    {t:5}: " + " | ".join(parts))
        for name in ("+d1", "+d2", "+d3", "+d2 e3", "+d2 e12"):
            rp = _cluster_stat(lambda w, name=name: wmean(pol[name] - pol["alone"], w), ep, rng)
            rt = _cluster_stat(lambda w, name=name: wmean(tot[name] - tot["alone"], w), ep, rng)
            lines.append(f"           crowding {name:8}: policy={fmt(rp)} | total={fmt(rt)}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=path to a ray_results experiment dir")
    parser.add_argument("--checkpoint", type=int, default=29)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--every", type=int, default=5, help="sample every k-th step")
    parser.add_argument("--max-states", type=int, default=6000)
    parser.add_argument("--near-radius", type=int, default=3)
    parser.add_argument("--insert-energy", type=float, default=8.0)
    parser.add_argument("--sexes", nargs="+", default=["female", "male"], choices=["female", "male"])
    parser.add_argument("--seed", type=int, default=5000)
    args = parser.parse_args()
    if args.every < 1 or args.near_radius < 0 or args.insert_energy <= 0:
        raise SystemExit("--every must be >= 1, --near-radius >= 0, --insert-energy > 0")
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    for spec in args.run:
        label, path = spec.split("=", 1)
        path = os.path.expanduser(path)
        config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
        trial = sorted(glob.glob(os.path.join(path, "PPO_*/")))[0]
        modules = load_modules(os.path.join(trial, f"checkpoint_{args.checkpoint:06d}"))
        for sex in args.sexes:
            pid = f"predator_{sex}_policy"
            bank, offset, moves = build_bank(config, modules, pid, args.episodes, args.seed, args.every, args.near_radius, args.max_states)
            if args.near_radius > offset:
                raise SystemExit(f"--near-radius {args.near_radius} exceeds the observation offset {offset}")
            if len(bank) < 100:
                print(f"{label} {sex}: only {len(bank)} sampled states, skipped", flush=True)
                continue
            bias_total = evaluate(bank, offset, moves, modules[pid], args.insert_energy, blocking=True)
            bias_policy = evaluate(bank, offset, moves, modules[pid], args.insert_energy, blocking=False)
            for line in summarize(f"{label} ckpt{args.checkpoint}", sex, bank, bias_total, bias_policy, rng):
                print(line, flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
