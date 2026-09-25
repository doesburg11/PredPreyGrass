"""
Counterfactual test of what the mate-proximity association is made of (RESULTS.md, Iterations 14 and 16).

Structural fact this rests on: a predator's observation (`_get_observation`) has five channels -- border, ONE
predator layer shared by both sexes (it holds only an energy value), prey, grass, fruit. A predator therefore
cannot tell whether a neighbor is male or female, nor whether it is its recorded mate. Whatever the female policy
does differently "with her mate nearby" can only be a response to nearby predators in general (their presence and
energy), which mates are among. The observational near-vs-away tests cannot separate that from mate-specific
responsiveness; a counterfactual edit of the observation can.

Method (policy-only, no training): sample on-policy states from rollouts in the env each run was trained in
(analysis_env.py), for one sex's policy, at states where a target (fruit or prey) is visible. For each state, query
the policy on edited copies of the SAME observation and compute the approach bias (same metric as the other
scripts: expected distance-after under a uniform mover minus under the policy):
  base      the observation as it was
  alone     every other predator removed from the window (only the agent's own cell in the predator layer kept)
  +d{1,2,3} `alone` plus ONE inserted predator (energy --insert-energy, default 8) at Chebyshev distance d,
            on a cell with no prey/grass/fruit, inside the grid
  +d2 with energy 3 / 12   the same insertion at d=2 with a low- / high-energy neighbor
Effects are paired per state and reported with an episode-cluster bootstrap:
  crowding effect = bias(+dK) - bias(alone): what adding ONE generic predator does to this policy
  isolation effect = bias(alone) - bias(base): what removing whoever was actually there does
Alongside, the observational near-minus-away difference on the same sampled states (mate within --near-radius vs
not) so the size of the counterfactual effect can be compared with the size of the association it is meant to explain.

What it can and cannot show: if inserting an anonymous predator moves the policy about as much as the
observational mate-proximity association, the association is explained by a response to crowding, not to the partner
as such. It says nothing about training-time coordination and only covers edits of the predator layer; inserted
neighbors are an approximation (a real neighbor comes with a history and correlated surroundings).

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


def insert_predator(alone, offset, dist, energy, rng):
    """`alone` plus one predator at Chebyshev distance `dist`, on a free in-grid cell with no prey/grass/fruit.
    Returns None if no such cell exists in this window."""
    W = alone.shape[1]
    cells = []
    for x in range(W):
        for y in range(W):
            if max(abs(x - offset), abs(y - offset)) != dist:
                continue
            if alone[0, x, y] != 0 or alone[PRED_CH, x, y] != 0:
                continue  # border (outside the grid) or occupied
            if alone[PREY_CH, x, y] != 0 or alone[GRASS_CH, x, y] != 0 or alone[FRUIT_CH, x, y] != 0:
                continue
            cells.append((x, y))
    if not cells:
        return None
    x, y = cells[int(rng.integers(len(cells)))]
    out = alone.copy()
    out[PRED_CH, x, y] = energy
    return out


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
                            if any(g is not None for g in geo.values()):
                                bank.append(
                                    {
                                        "obs": np.array(observations[agent], dtype=np.float64),
                                        "episode": ep,
                                        "near": bucket == "near",
                                        "geo": {t: (g[0] if g is not None else None) for t, g in geo.items()},
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
    return bank, offset


def conditions(insert_energy):
    return [("base", None, None), ("alone", None, None)] + [
        (f"+d{d}", d, insert_energy) for d in (1, 2, 3)
    ] + [("+d2 e3", 2, 3.0), ("+d2 e12", 2, 12.0)]


def evaluate(bank, offset, module, insert_energy, seed):
    """bias[cond][target] arrays (N,) with NaN where the condition or the target is not available."""
    rng = np.random.default_rng(seed)
    conds = conditions(insert_energy)
    N = len(bank)
    obs_by_cond = {name: [None] * N for name, _, _ in conds}
    for i, b in enumerate(bank):
        alone = remove_neighbors(b["obs"], offset)
        for name, dist, energy in conds:
            if name == "base":
                obs_by_cond[name][i] = b["obs"]
            elif name == "alone":
                obs_by_cond[name][i] = alone
            else:
                obs_by_cond[name][i] = insert_predator(alone, offset, dist, energy, rng)
    bias = {name: {t: np.full(N, np.nan) for t in TARGETS} for name, _, _ in conds}
    for name, _, _ in conds:
        idx = [i for i in range(N) if obs_by_cond[name][i] is not None]
        if not idx:
            continue
        probs = module_probs(module, [obs_by_cond[name][i] for i in idx])
        for k, i in enumerate(idx):
            for t in TARGETS:
                d_after = bank[i]["geo"][t]
                if d_after is not None:
                    bias[name][t][i] = approach_bias(probs[k], d_after)
    return bias


def cluster_mean_diff(x, y, episodes, rng, n_boot=2000):
    """Mean of (x-y) over states where both exist, with an episode-cluster percentile bootstrap."""
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 30:
        return None
    d, ep = (x - y)[ok], episodes[ok]
    eps = np.unique(ep)
    s = np.array([d[ep == e].sum() for e in eps])
    c = np.array([(ep == e).sum() for e in eps], dtype=float)
    point = d.sum() / len(d)
    idx = rng.integers(0, len(eps), size=(n_boot, len(eps)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, int(ok.sum())


def summarize(label, sex, bank, bias, rng):
    lines = []
    episodes = np.array([b["episode"] for b in bank])
    near = np.array([b["near"] for b in bank])
    n_neighbors = np.array([(b["obs"][PRED_CH] > 0).sum() - 1 for b in bank])  # others in window (own cell excluded)
    lines.append(
        f"{label} {sex}: {len(bank)} sampled states, {int((n_neighbors > 0).sum())} with >=1 other predator in view, "
        f"{int(near.sum())} with the recorded mate within the near radius"
    )
    fmt = lambda r: "n/a" if r is None else f"{r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}] (n={r[3]})"
    for t in TARGETS:
        base = bias["base"][t]
        parts = []
        # observational association on the same sampled states (near vs not near)
        both = ~np.isnan(base)
        if both[near].sum() >= 30 and both[~near].sum() >= 30:
            obs_diff = base[near & both].mean() - base[~near & both].mean()
            parts.append(f"observational near-away={obs_diff:+.3f}")
        parts.append("isolation (alone-base)=" + fmt(cluster_mean_diff(bias["alone"][t], base, episodes, rng)))
        for name in ("+d1", "+d2", "+d3", "+d2 e3", "+d2 e12"):
            parts.append(f"crowding {name}-alone=" + fmt(cluster_mean_diff(bias[name][t], bias["alone"][t], episodes, rng)))
        lines.append(f"    {t:5}: " + " | ".join(parts))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=path to a ray_results experiment dir")
    parser.add_argument("--checkpoint", type=int, default=29)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--every", type=int, default=5, help="sample every k-th step")
    parser.add_argument("--max-states", type=int, default=20000)
    parser.add_argument("--near-radius", type=int, default=3)
    parser.add_argument("--insert-energy", type=float, default=8.0)
    parser.add_argument("--sexes", nargs="+", default=["female", "male"], choices=["female", "male"])
    parser.add_argument("--seed", type=int, default=5000)
    args = parser.parse_args()
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
            bank, offset = build_bank(config, modules, pid, args.episodes, args.seed, args.every, args.near_radius, args.max_states)
            if len(bank) < 100:
                print(f"{label} {sex}: only {len(bank)} sampled states, skipped", flush=True)
                continue
            bias = evaluate(bank, offset, modules[pid], args.insert_energy, args.seed)
            for line in summarize(f"{label} ckpt{args.checkpoint}", sex, bank, bias, rng):
                print(line, flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
