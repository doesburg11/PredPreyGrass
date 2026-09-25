"""
Who is scattered, and why? For trained band policies (no training), sample every --every steps and, for each living predator,
measure how far it is from the rest of its band, then group predators by how they came to be in that band:

  founder      started in the band and never changed band
  born         born into the band and never changed band
  moved        changed band at least once (a marriage moved it, or its mother, into another band), split by the time since its
               last move: < 25, 25-100, > 100 steps

Measures (per group): the mean Chebyshev distance to the band centroid computed over the OTHER members; the share of observations
with no band-mate within band_share_range ("out of sharing range", which includes a band of one); and the share more than
--far cells from the centroid ("scattered"). 95% episode-cluster percentile bootstrap (whole episodes resampled).

Reading it: if `moved` members are far more often out of range right after a move and drift back over time, the marriage rule
(membership changes without moving) is a main source of scatter; if `founder` and `born` members scatter about as much, the
policies themselves do not keep bands together. Descriptive; one checkpoint per run; the centroid of a band spread over
several clusters is not a meaningful centre, so the nearest-band-mate share is the more robust measure.

Example:
  python -m predpreygrass.non_evolutionary.predator_bands.analyze_band_scatter \
      --run MEAT060_s42=~/simulation_results/ray_results/PPO_PREDATOR_BANDS_CALIB_F008_MEAT060_SEED42 --checkpoint 19 --episodes 20
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

from predpreygrass.non_evolutionary.predator_bands.analyze_band_behavior import (
    InstrumentedBandsEnv,
    founder_roles,
    load_modules,
    module_probs,
    policy_of,
    POLICY_IDS,
    ratio_ci,
    fmt,
)

GROUPS = ("founder", "born", "moved <25", "moved 25-100", "moved >100")


def group_of(agent, role, moved_at, step):
    if agent in moved_at:
        since = step - moved_at[agent]
        return "moved <25" if since < 25 else ("moved 25-100" if since <= 100 else "moved >100")
    return "founder" if role != "born" else "born"


def rollout(config, modules, n_episodes, seed0, every, far):
    env = InstrumentedBandsEnv(config)
    rng = np.random.default_rng(seed0)
    episodes = []
    for ep in range(n_episodes):
        observations, _ = env.reset(seed=seed0 + ep)
        roles = founder_roles(env)
        prev_band = dict(env.agent_band)
        moved_at = {}
        active = list(observations.keys())
        acc = {g: np.zeros(4) for g in GROUPS}  # n, sum centroid distance (over obs with >=1 band-mate), n with band-mate, n out of range
        far_count = {g: 0 for g in GROUPS}
        while True:
            actions = {}
            for pid in POLICY_IDS:
                agents = [a for a in active if policy_of(a) == pid]
                if not agents:
                    continue
                rows = module_probs(modules[pid], [observations[a] for a in agents])
                for i, agent in enumerate(agents):
                    actions[agent] = int(rng.choice(env.num_actions, p=rows[i]))
            observations, _, terminations, truncations, _ = env.step(actions)
            step = env.current_step
            for a, b in env.agent_band.items():
                if a in prev_band and prev_band[a] != b:
                    moved_at[a] = step
                prev_band[a] = b
            if step % every == 0:
                bands = defaultdict(list)
                for a, p in env.predator_positions.items():
                    if a in env.agent_band:
                        bands[env.agent_band[a]].append((a, np.array(p)))
                for members in bands.values():
                    for a, p in members:
                        others = [q for o, q in members if o != a]
                        g = group_of(a, roles.get(a, "born"), moved_at, step)
                        acc[g][0] += 1
                        if others:
                            others = np.array(others)
                            acc[g][1] += float(np.abs(others.mean(axis=0) - p).max())
                            acc[g][2] += 1
                            in_range = (np.abs(others - p).max(axis=1) <= env.band_share_range).any()
                            acc[g][3] += 0 if in_range else 1
                            far_count[g] += float(np.abs(others.mean(axis=0) - p).max()) > far
                        else:
                            acc[g][3] += 1  # alone in its band: out of sharing range by definition
            active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
            if terminations.get("__all__") or truncations.get("__all__"):
                break
        episodes.append({"acc": acc, "far": far_count})
    return episodes


def summarize(label, episodes, far, rng):
    out = [f"== {label}: {len(episodes)} episodes; distances are Chebyshev cells; 'far' = more than {far} cells from the centroid of the others"]
    for g in GROUPS:
        n = [e["acc"][g][0] for e in episodes]
        if sum(n) == 0:
            out.append(f"   {g:13} no observations")
            continue
        dist = ratio_ci([e["acc"][g][1] for e in episodes], [e["acc"][g][2] for e in episodes], rng)
        oor = ratio_ci([e["acc"][g][3] for e in episodes], n, rng)
        fr = ratio_ci([e["far"][g] for e in episodes], [e["acc"][g][2] for e in episodes], rng)
        out.append(
            f"   {g:13} n={int(sum(n)):6d} | centroid distance {fmt(dist, 2)} | out of sharing range {fmt(oor, 3)} | far {fmt(fr, 3)}"
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=path to a ray_results experiment dir")
    parser.add_argument("--checkpoint", type=int, default=19)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--far", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=6000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    for spec in args.run:
        label, path = spec.split("=", 1)
        path = os.path.expanduser(path)
        config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
        trial = sorted(glob.glob(os.path.join(path, "PPO_*/")))[0]
        modules = load_modules(os.path.join(trial, f"checkpoint_{args.checkpoint:06d}"))
        episodes = rollout(config, modules, args.episodes, args.seed, args.every, args.far)
        for line in summarize(f"{label} ckpt{args.checkpoint}", episodes, args.far, rng):
            print(line, flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
