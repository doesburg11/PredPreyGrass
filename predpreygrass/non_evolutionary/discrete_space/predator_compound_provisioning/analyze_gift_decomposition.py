"""Decompose the gift-bit eligibility gap into range and mate-need parts (RESULTS.md, Iteration 24 analysis).

Each predator decision state (read before the step, as in analyze_compound_gift.py) goes into one bucket:

- no_mate:      no recorded mate, or the mate is dead
- mate_dying:   mate energy - homeostatic cost <= 0 (a gift can't rescue it)
- no_afford:    giver energy - homeostatic cost - gift amount <= 0
- otherwise a 2x2 cell of range (mate within predator_gift_range, Chebyshev) x need (mate energy below the
  reproduction threshold): in_need (= eligible), in_full, out_need, out_full

Per episode and sex, the policy probability of the gift bit (sum over actions 9..17) is averaged per bucket. Contrasts:

- need_in  = P(in_need) - P(in_full):  does the gift bit track the mate's need when the mate is in range?
- range_need = P(in_need) - P(out_need): does it track range when the mate needs energy? The gift range equals the
  observation radius (3), so an in-range mate is always tagged and visible and an out-of-range mate never is: this
  contrast can't separate range from seeing the mate.

The mate's energy is observable: the predator layer holds energy values and the mate channel marks the mate's cell
when it is inside the observation window.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.analyze_compound_gift import (
    SEXES,
    sex_of,
    summarize,
)
from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.compound_gift_env import (
    CompoundGiftPredPreyGrass,
)
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.analyze_prey_approach_from_checkpoint import (
    POLICY_IDS,
    load_modules,
    module_probs,
    policy_of,
)

BUCKETS = ("no_mate", "mate_dying", "no_afford", "in_need", "in_full", "out_need", "out_full")


def bucket_of(env, agent):
    mate = env.agent_mate.get(agent)
    if mate is None or mate not in env.predator_positions:
        return "no_mate"
    cost = env.homeostatic_energy_cost_per_step_predator
    if env.agent_energies[mate] - cost <= 0:
        return "mate_dying"
    if env.agent_energies[agent] - cost - env.learned_gift_amount <= 0:
        return "no_afford"
    a, b = env.predator_positions[agent], env.predator_positions[mate]
    in_range = max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= env.predator_gift_range
    need = env.agent_energies[mate] < env.predator_creation_energy_threshold
    return ("in_" if in_range else "out_") + ("need" if need else "full")


def rollout(config, modules, episodes, seed0):
    if config.get("compound_gift") not in ("on", "inert"):
        raise SystemExit("not a compound-gift run (config_env.compound_gift must be 'on' or 'inert')")
    env = CompoundGiftPredPreyGrass(config)
    rng, rows = np.random.default_rng(seed0), []
    for episode in range(episodes):
        observations, _ = env.reset(seed=seed0 + episode)
        active = list(observations)
        stats = {sex: {b: [0.0, 0] for b in BUCKETS} for sex in SEXES}
        while True:
            actions = {}
            for policy_id in POLICY_IDS:
                agents = [a for a in active if policy_of(a) == policy_id]
                if not agents:
                    continue
                probs = module_probs(modules[policy_id], [observations[a] for a in agents])
                for i, agent in enumerate(agents):
                    if "predator" in agent:
                        cell = stats[sex_of(agent)][bucket_of(env, agent)]
                        cell[0] += float(probs[i, env.num_actions :].sum())
                        cell[1] += 1
                    actions[agent] = int(rng.choice(len(probs[i]), p=probs[i]))
            observations, _, terminations, truncations, _ = env.step(actions)
            active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
            if terminations.get("__all__") or truncations.get("__all__"):
                break
        rows.append(episode_row(stats))
    return rows


def episode_row(stats):
    row = {}
    for sex in SEXES:
        total = sum(n for _, n in stats[sex].values())
        p = {}
        for b in BUCKETS:
            s, n = stats[sex][b]
            p[b] = s / n if n else np.nan
            row[f"{sex}_p_{b}"] = p[b]
            row[f"{sex}_share_{b}"] = n / total if total else np.nan
        row[f"{sex}_need_in"] = p["in_need"] - p["in_full"]
        row[f"{sex}_range_need"] = p["in_need"] - p["out_need"]
    return row


MIN_EPISODES = 10  # a contrast needs both of its cells in at least this many of the episodes
MARGIN = 0.02


def contrast_values(rows, key):
    return np.array([r[key] for r in rows if not np.isnan(r[key])], dtype=float)


def pair_test(give_rows, inert_rows, key, rng, n_boot=2000):
    """Pre-registered per-pair rule: GIVE contrast >= MARGIN with its 95% interval excluding zero, GIVE minus INERT
    >= MARGIN with the 95% interval of that difference excluding zero (independent episode bootstraps), and both
    runs with the contrast defined in >= MIN_EPISODES episodes. Returns (passed, details)."""
    g, i = contrast_values(give_rows, key), contrast_values(inert_rows, key)
    details = {"give_n": len(g), "inert_n": len(i)}
    if len(g) < MIN_EPISODES or len(i) < MIN_EPISODES:
        return False, {**details, "reason": "too few episodes"}
    gb = g[rng.integers(0, len(g), size=(n_boot, len(g)))].mean(1)
    ib = i[rng.integers(0, len(i), size=(n_boot, len(i)))].mean(1)
    g_lo, g_hi = np.percentile(gb, [2.5, 97.5])
    d_lo, d_hi = np.percentile(gb - ib, [2.5, 97.5])
    details.update(give=float(g.mean()), give_ci=[float(g_lo), float(g_hi)], inert=float(i.mean()),
                   diff=float(g.mean() - i.mean()), diff_ci=[float(d_lo), float(d_hi)])
    passed = g.mean() >= MARGIN and g_lo > 0 and g.mean() - i.mean() >= MARGIN and d_lo > 0
    return bool(passed), details


def score(paths, pairs, need=5):
    """paths: per-run JSON outputs of this script; pairs: [(give_label, inert_label)]. Prints D1/D2 per pair."""
    runs = {}
    for path in paths:
        runs.update(json.load(open(path)))
    rng, verdict = np.random.default_rng(0), {}
    for test, key in (("D1_need_in", "predator_male_need_in"), ("D2_range_need", "predator_male_range_need")):
        results = [pair_test(runs[g]["rows"], runs[i]["rows"], key, rng) for g, i in pairs]
        passes = sum(p for p, _ in results)
        verdict[test] = {"passes": passes, "met": passes >= need,
                         "pairs": {f"{g}|{i}": d | {"passed": p} for (g, i), (p, d) in zip(pairs, results)}}
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="LABEL=run-dir (one run per process)")
    parser.add_argument("--checkpoint", type=int, default=29)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    label, path = args.run.split("=", 1)
    config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
    trials = sorted(glob.glob(os.path.join(path, "*PPO_*/")))
    if len(trials) != 1:
        raise SystemExit(f"expected one trial under {path}, found {len(trials)}")
    modules = load_modules(os.path.join(trials[0], f"checkpoint_{args.checkpoint:06d}"))
    rows = rollout(config, modules, args.episodes, args.seed)
    rng = np.random.default_rng(0)
    out = {key: summarize([r[key] for r in rows], rng) for key in rows[0]}
    with open(args.out, "w") as f:
        json.dump({label: {"summary": out, "rows": rows}}, f, indent=1)
    for key in sorted(out):
        if out[key] is not None and ("need_in" in key or "range_need" in key):
            m, lo, hi, n = out[key]
            print(f"{label} {key:32}: {m:+.4f} [{lo:+.4f},{hi:+.4f}] (episodes {n})", flush=True)


if __name__ == "__main__":
    main()
