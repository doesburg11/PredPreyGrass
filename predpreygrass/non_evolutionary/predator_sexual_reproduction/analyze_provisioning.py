"""
Provisioning test for the female mate-proximity association (RESULTS.md, Iteration 14).

Hypothesis under test: a female approaches food less when her recorded mate is nearby because he can donate part
of a successful hunt to her (`_apply_male_gift`, only to his recorded mate within predator_gift_range), so a
nearby, recently-successful mate means she is being fed and forages less urgently. This is a first, sharper
test than a bare correlation of per-life gifts with approach bias (which would also pick up survival duration,
location, the mate's own success, and so on). Two contrasts, females only, in the env each run was trained in:

 (1) Gift timing. Females with a living mate within --near-radius are split by whether they RECEIVED a gift
     within the last --gift-window steps ("gift_recent") or not ("near_nogift"). If provisioning drives the
     association, near_nogift should look like "away" (far + dead + abandoned) and gift_recent should carry
     the drop. Reported: near_nogift-away, gift_recent-away, gift_recent-near_nogift.
 (2) Energy adjustment. The female's own energy (visible to the policy) is a natural mediator: a fed female is
     richer. The near-minus-away approach-bias difference is recomputed WITHIN bands of her own energy and
     averaged with weights equal to each band's pooled decision count (direct standardization). If energy carries
     the association, the adjusted difference shrinks toward zero relative to the crude one.

All intervals are 95% episode-cluster bootstrap percentiles (whole episodes resampled; every statistic is
recomputed inside each replicate). What this can and cannot show: it is still observational. A shrunken adjusted
difference or a near_nogift group that resembles "away" would be evidence CONSISTENT with an energy/provisioning
explanation; it would not exclude other explanations that also track energy or gift timing, and a surviving
association after adjustment would show the explanation is incomplete, not that it is wrong. Gift receipt is
measured from the change in every predator's energy around the gift call (InstrumentedMixin), so it is exact
whatever the rules; energy bands are fixed edges, not fitted.

Example:
  python -m predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_provisioning \
      --run CONTROL_s42=~/simulation_results/ray_results/PPO_FIXED_PREY_DENSITY_CONTROL_SEED42 --episodes 30
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_energy_sources import make_instrumented_env
from predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_mate_contingency import mate_bucket
from predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_prey_approach_from_checkpoint import (
    POLICY_IDS,
    action_geometry,
    load_modules,
    module_probs,
    new_stats,
    policy_of,
    record,
)

TARGETS = ("prey", "fruit")
GROUPS = ("gift_recent", "near_nogift", "far", "dead", "abandoned")  # virgin females are excluded
ENERGY_EDGES = (4.0, 8.0, 12.0)  # bands: <4, 4-8, 8-12, >=12 (predator_creation_energy_threshold is 12)
N_BANDS = len(ENERGY_EDGES) + 1


def energy_band(e):
    return int(np.searchsorted(ENERGY_EDGES, e, side="right"))


def run_episodes(env_config, modules, n_episodes, seed0, near_radius, gift_window):
    env = make_instrumented_env(env_config)
    moves = np.array([env.action_to_move_tuple[a] for a in range(env.num_actions)])
    offset = (env.predator_obs_range - 1) // 2
    sample_rng = np.random.default_rng(seed0)
    per_episode = []
    for ep in range(n_episodes):
        observations, _ = env.reset(seed=seed0 + ep)
        active = list(observations.keys())
        stats = {(t, g, b): new_stats() for t in TARGETS for g in GROUPS for b in range(N_BANDS)}
        energy_sum = {g: [0.0, 0] for g in GROUPS}
        last_gift_step = {}
        seen_gifts = {}
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
                    probs = rows[i]
                    if pid == "predator_female_policy":
                        bucket = mate_bucket(env, agent, near_radius)
                        if bucket == "near":
                            since = step - last_gift_step.get(agent, -10**9)
                            group = "gift_recent" if since <= gift_window else "near_nogift"
                        elif bucket in ("far", "dead", "abandoned"):
                            group = bucket
                        else:
                            group = None  # virgin
                        if group is not None:
                            energy = env.agent_energies[agent]
                            band = energy_band(energy)
                            energy_sum[group][0] += energy
                            energy_sum[group][1] += 1
                            pos = np.array(env.agent_positions[agent], dtype=int)
                            for t in TARGETS:
                                geo = action_geometry(pos, targets[t], moves, env.grid_size, offset)
                                if geo is not None:
                                    record(stats[(t, group, band)], geo, probs)
                    actions[agent] = int(sample_rng.choice(env.num_actions, p=probs))
            observations, _, terminations, truncations, _ = env.step(actions)
            step += 1
            # Gift receipts observed during THIS step become visible to the NEXT decision.
            for a, total in list(env.gift_received.items()):
                if total > seen_gifts.get(a, 0.0) + 1e-12:
                    last_gift_step[a] = step
                    seen_gifts[a] = total
            active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
            if terminations.get("__all__") or truncations.get("__all__"):
                break
        per_episode.append((stats, energy_sum))
    return per_episode


def _arrays(per_episode):
    """num/den arrays of shape (episodes, groups, bands) per target."""
    E = len(per_episode)
    out = {}
    for t in TARGETS:
        num = np.zeros((E, len(GROUPS), N_BANDS))
        den = np.zeros((E, len(GROUPS), N_BANDS))
        for e, (stats, _) in enumerate(per_episode):
            for gi, g in enumerate(GROUPS):
                for b in range(N_BANDS):
                    s = stats[(t, g, b)]
                    num[e, gi, b], den[e, gi, b] = s["bias"], s["n"]
        out[t] = (num, den)
    return out


GI = {g: i for i, g in enumerate(GROUPS)}
NEAR = [GI["gift_recent"], GI["near_nogift"]]
AWAY = [GI["far"], GI["dead"], GI["abandoned"]]


def _ratio(num, den, groups):
    n = num[..., groups, :].sum(axis=-2).sum(axis=-1)
    d = den[..., groups, :].sum(axis=-2).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, n / d, np.nan)


def _adjusted(num, den):
    """Energy-band-standardized near-minus-away. Works on (..., groups, bands) totals."""
    nn, nd = num[..., NEAR, :].sum(axis=-2), den[..., NEAR, :].sum(axis=-2)
    an, ad = num[..., AWAY, :].sum(axis=-2), den[..., AWAY, :].sum(axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = nn / nd - an / ad
    valid = (nd > 0) & (ad > 0)
    w = np.where(valid, nd + ad, 0.0)
    diff = np.where(valid, diff, 0.0)
    tot = w.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, (w * diff).sum(axis=-1) / tot, np.nan)


def _stats_from_totals(num, den):
    """num/den shape (..., groups, bands) -> dict of named statistics, each shape (...)."""
    g = lambda name: [GI[name]]
    r = lambda groups: _ratio(num, den, groups)
    return {
        "crude near-away": r(NEAR) - r(AWAY),
        "energy-adjusted near-away": _adjusted(num, den),
        "near_nogift-away": r(g("near_nogift")) - r(AWAY),
        "gift_recent-away": r(g("gift_recent")) - r(AWAY),
        "gift_recent-near_nogift": r(g("gift_recent")) - r(g("near_nogift")),
    }


def summarize(label, per_episode, rng, n_boot=2000):
    lines = []
    E = len(per_episode)
    en = {g: (sum(ep[1][g][0] for ep in per_episode), sum(ep[1][g][1] for ep in per_episode)) for g in GROUPS}
    lines.append(
        f"{label}: female decisions by group (n, mean own energy): "
        + ", ".join(f"{g} n={n} e={(s / n if n else float('nan')):.1f}" for g, (s, n) in en.items())
    )
    arrs = _arrays(per_episode)
    for t in TARGETS:
        num, den = arrs[t]
        point = _stats_from_totals(num.sum(axis=0), den.sum(axis=0))
        idx = rng.integers(0, E, size=(n_boot, E))
        boot = _stats_from_totals(num[idx].sum(axis=1), den[idx].sum(axis=1))
        parts = []
        for name, val in point.items():
            b = boot[name][~np.isnan(boot[name])]
            if len(b) >= 100 and not np.isnan(val):
                lo, hi = np.percentile(b, [2.5, 97.5])
                parts.append(f"{name}={val:+.3f} [{lo:+.3f},{hi:+.3f}]")
            else:
                parts.append(f"{name}=n/a")
        lines.append(f"    {t:5}: " + " | ".join(parts))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=path to a ray_results experiment dir")
    parser.add_argument("--checkpoint", type=int, default=29)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--near-radius", type=int, default=3, help="also the gift range in this module's defaults")
    parser.add_argument("--gift-window", type=int, default=10, help="steps after a received gift counted as 'recent'")
    args = parser.parse_args()
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    for spec in args.run:
        label, path = spec.split("=", 1)
        path = os.path.expanduser(path)
        config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
        trial = sorted(glob.glob(os.path.join(path, "PPO_*/")))[0]
        modules = load_modules(os.path.join(trial, f"checkpoint_{args.checkpoint:06d}"))
        per_episode = run_episodes(config, modules, args.episodes, args.seed, args.near_radius, args.gift_window)
        for line in summarize(f"{label} ckpt{args.checkpoint}", per_episode, rng):
            print(line, flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
