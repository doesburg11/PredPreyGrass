"""
Why did base_environment_step_energy seed 45 fail to establish a predator population? (RESULTS.md section 20)

Reads the runs' Ray result logs under ~/simulation_results/ray_results (training-time policy entropy,
predator births per episode, predator training samples per iteration) plus the saved evaluation data in
clustering_density_data/ (greedy) and sampled_action_data/ (actions sampled from the policy), and prints:
  - predator and prey policy entropy by training window (ln 9 = 2.197 is a uniform-random policy)
  - predator births per completed episode by training window
  - predator training samples per iteration by window, and cumulative samples at iteration 300
  - greedy vs sampled-action evaluation of every iteration-300 policy (predator extinction, mean predators)

Usage: python analyze_seed45.py
"""
import glob
import json
import math
import os

import numpy as np

RR = os.path.expanduser("~/simulation_results/ray_results")
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 43, 44, 45, 46, 47]
WINDOWS = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300)]


def result_path(kind, seed):
    if seed == 42:
        if kind == "base":
            return (f"{RR}/base_v_drive_2026-09-06/PPO_BASE_ENVIRONMENT_SEED42_2026-09-05_18-55-45/"
                    "PPO_PredPreyGrass_a2fe1_00000_0_2026-09-05_18-55-45/result.json")
        return glob.glob(f"{RR}/PPO_STEP_ENERGY_ADDITIVE_PREDEASE_CONFIRM500_SEED42/PPO_*/result.json")[0]
    name = (f"PPO_BASE_ENVIRONMENT_SEED{seed}_MULTISEED" if kind == "base"
            else f"PPO_STEP_ENERGY_RUNB_SEED{seed}_MULTISEED")
    return glob.glob(f"{RR}/{name}/PPO_*/result.json")[0]


LOGS = {(k, s): [json.loads(line) for line in open(result_path(k, s))] for k in ("step", "base") for s in SEEDS}


def series(kind, seed, getter):
    out = []
    for lo, hi in WINDOWS:
        vals = [getter(r) for r in LOGS[(kind, seed)] if lo < r["training_iteration"] <= hi]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out


entropy = lambda pid: (lambda r: r.get("learners", {}).get(pid, {}).get("entropy"))
births = lambda r: r.get("env_runners", {}).get("ecology/births_predator")
pred_steps = lambda r: r["env_runners"]["num_module_steps_sampled"]["predator_policy"] / 1000

hdr = "  ".join(f"{lo + 1}-{hi}".rjust(7) for lo, hi in WINDOWS)
print("PREDATOR policy entropy by training window (ln 9 = 2.197 is uniform-random)")
print(f"config seed | {hdr}")
for kind in ("step", "base"):
    for s in SEEDS:
        print(f"{kind:4s}  {s}  | " + "  ".join(f"{x:7.2f}" for x in series(kind, s, entropy("predator_policy"))))
print("\nPREY policy entropy, base_environment_step_energy")
for s in SEEDS:
    print(f"step  {s}  | " + "  ".join(f"{x:7.2f}" for x in series("step", s, entropy("prey_policy"))))
print("\nPredator births per completed episode, base_environment_step_energy")
for s in SEEDS:
    print(f"step  {s}  | " + "  ".join(f"{x:7.1f}" for x in series("step", s, births)))
print("\nPredator training samples per iteration (thousands); cumulative samples at iteration 300 (millions)")
for kind in ("step", "base"):
    for s in SEEDS:
        life = [r for r in LOGS[(kind, s)] if r["training_iteration"] == 300][0]["env_runners"][
            "num_module_steps_sampled_lifetime"]["predator_policy"] / 1e6
        print(f"{kind:4s}  {s}  | " + "  ".join(f"{x:7.1f}" for x in series(kind, s, pred_steps)) + f"  | {life:5.2f}")

print("\nGREEDY vs SAMPLED-ACTION evaluation of the iteration-300 policies (30 episodes each)")
print("config seed | greedy: early-ended, mean predators | sampled: predator extinction, mean predators")
for kind in ("step", "base"):
    for s in SEEDS:
        g = json.load(open(os.path.join(HERE, "clustering_density_data", f"dose_{kind}_{s}.json")))
        m = json.load(open(os.path.join(HERE, "sampled_action_data", f"sampled_{kind}_{s}.json")))
        print(f"{kind:4s}  {s}  | {100 * np.mean([x['steps'] < 1000 for x in g]):4.0f}%  {np.mean([x['n_pred'] for x in g]):6.2f}"
              f"      | {100 * np.mean([x['pred_extinct'] for x in m]):4.0f}%  {np.mean([x['mean_pred'] for x in m]):6.2f}")
