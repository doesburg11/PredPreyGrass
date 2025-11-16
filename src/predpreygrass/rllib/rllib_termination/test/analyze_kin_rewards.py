"""Analyze kin reward effects from agent_fitness_stats.json.

This script is meant to be run manually after an evaluation produced
`agent_fitness_stats.json` (see `evaluate_ppo_from_checkpoint_debug.py`).

It computes, per group (e.g. type_1_prey, type_1_predator):
- mean/median/max kin_kickbacks
- correlations between kin_kickbacks and offspring_count / cumulative_reward
- comparisons between agents with kin_kickbacks > 0 and == 0
"""

import json
import os
from collections import defaultdict
import numpy as np

# Adjust this to point to a specific eval dir
EVAL_DIR = "/home/doesburg/Dropbox/02_marl_results/predpreygrass_results/ray_results/kin_kick_back=1.0/PPO_PredPreyGrass_e9a5c_00000_0_2025-11-16_16-07-05/checkpoint_000001/eval_checkpoint_000001_2025-11-16_22-15-00"  # e.g. ".../checkpoint_000001/eval_checkpoint_000001_2025-11-16_12-34-56"
STATS_FILE = os.path.join(EVAL_DIR, "agent_fitness_stats.json")

with open(STATS_FILE, "r") as f:
    stats = json.load(f)

# Group by high-level type, using your parse_uid convention:
def parse_uid(uid):
    # Simplified version; you can import the one from evaluate_ppo_from_checkpoint_debug.py instead
    import re
    match = re.match(r"(type_\d+_(?:predator|prey))_(\d+)(?:#(\d+))?", uid)
    if match:
        group, idx, lifetime = match.groups()
        return group, int(idx), int(lifetime) if lifetime is not None else 0
    else:
        return uid, None, None

group_data = defaultdict(list)
for uid, s in stats.items():
    group, _, _ = parse_uid(uid)
    reward = s.get("cumulative_reward", 0.0)
    off = s.get("offspring_count", 0)
    life = s.get("lifetime", 0)
    kin = s.get("kin_kickbacks", 0)
    group_data[group].append({
        "uid": uid,
        "reward": reward,
        "offspring": off,
        "lifetime": life,
        "kin": kin,
        "off_per_step": off / life if life > 0 else 0.0,
    })

def summarize_group(group, entries):
    if not entries:
        print(f"\n## {group} (no data)")
        return
    rewards = np.array([e["reward"] for e in entries], dtype=float)
    offs = np.array([e["offspring"] for e in entries], dtype=float)
    kin = np.array([e["kin"] for e in entries], dtype=float)

    print(f"\n## {group} ##")
    print(f"n = {len(entries)}")
    print(f"mean reward         = {rewards.mean():.2f}")
    print(f"mean offspring      = {offs.mean():.2f}")
    print(f"mean kin_kickbacks  = {kin.mean():.2f}")
    print(f"median kin_kickbacks= {np.median(kin):.2f}")
    print(f"max kin_kickbacks   = {kin.max():.2f}")

    # simple correlations, guard div-zero
    def corr(a, b, name):
        if np.all(a == a[0]) or np.all(b == b[0]) or len(a) < 2:
            print(f"corr(kin, {name})   = undefined (no variance)")
            return
        c = np.corrcoef(a, b)[0, 1]
        print(f"corr(kin, {name})   = {c:.3f}")

    corr(kin, offs, "offspring")
    corr(kin, rewards, "reward")

    # Compare kin>0 vs kin==0
    has_kin = kin > 0
    if has_kin.any():
        print("\n  Subset comparison (kin_kickbacks > 0 vs == 0):")
        def sub_mean(mask, arr):
            return arr[mask].mean() if mask.any() else float("nan")

        print(f"    n(kin>0)              = {has_kin.sum()}")
        print(f"    mean offspring (kin>0)= {sub_mean(has_kin, offs):.2f}")
        print(f"    mean reward    (kin>0)= {sub_mean(has_kin, rewards):.2f}")
        print(f"    mean offspring (kin=0)= {sub_mean(~has_kin, offs):.2f}")
        print(f"    mean reward    (kin=0)= {sub_mean(~has_kin, rewards):.2f}")

for group, entries in sorted(group_data.items()):
    summarize_group(group, entries)