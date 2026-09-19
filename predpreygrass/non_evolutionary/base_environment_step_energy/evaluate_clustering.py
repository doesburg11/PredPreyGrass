"""
Predator spatial-clustering evaluation for one training seed (see RESULTS.md sections 13-16).

Usage (from the repo root, with the project's Python):
    python predpreygrass/non_evolutionary/base_environment_step_energy/evaluate_clustering.py <seed>

For the given seed it loads the iteration-300 checkpoint (checkpoint_000029) of the seed's
base_environment policy and its base_environment_step_energy (run B) policy, runs 30
deterministic (greedy) evaluation episodes each (environment reset seeds 100-129), and computes
one Clark-Evans index R per episode (predators; R < 1 clustered, R = 1 random, R > 1 dispersed;
expected nearest-neighbour distance under CSR = 0.5 * sqrt(area / N), area = 25*25).
Writes clustering_seed<seed>.json next to this file. Checkpoint paths are those used in this
investigation (ray_results under ~/simulation_results); seed 42 uses the pre-existing runs.
Consolidated raw results for all six seeds: clustering_results.json.
"""
import sys, json, math, glob, os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
torch.set_num_threads(1)
from ray.rllib.core.rl_module.rl_module import RLModule

RR = os.path.expanduser("~/simulation_results/ray_results")
N_EPISODES = 30
EVAL_SEEDS = list(range(100, 100 + N_EPISODES))

def ckpt_dir(kind, seed, ckpt="checkpoint_000029"):
    if seed == 42:
        if kind == "base":
            root = f"{RR}/base_v_drive_2026-09-06/PPO_BASE_ENVIRONMENT_SEED42_2026-09-05_18-55-45/PPO_PredPreyGrass_a2fe1_00000_0_2026-09-05_18-55-45"
        else:
            root = f"{RR}/PPO_STEP_ENERGY_ADDITIVE_PREDEASE_CONFIRM500_SEED42/PPO_PredPreyGrass_eed47_00000_0_2026-09-17_07-30-45"
    else:
        name = f"PPO_BASE_ENVIRONMENT_SEED{seed}_MULTISEED" if kind == "base" else f"PPO_STEP_ENERGY_RUNB_SEED{seed}_MULTISEED"
        root = glob.glob(f"{RR}/{name}/PPO_*")[0]
    return f"{root}/{ckpt}"

def load_modules(path):
    return {
        "predator_policy": RLModule.from_checkpoint(f"{path}/learner_group/learner/rl_module/predator_policy"),
        "prey_policy": RLModule.from_checkpoint(f"{path}/learner_group/learner/rl_module/prey_policy"),
    }

def act(obs, module):
    with torch.no_grad():
        out = module._forward_inference({"obs": torch.tensor(obs).float().unsqueeze(0)})
    return torch.argmax(out["action_dist_inputs"], dim=-1).item()

def clark_evans(per_step_positions, area=625.0):
    o = e = 0.0; n_s = 0
    for pos in per_step_positions:
        n = len(pos)
        if n < 2: continue
        pts = np.array(pos, dtype=float)
        nn = []
        for i in range(n):
            d = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1)); d[i] = np.inf; nn.append(d.min())
        o += np.mean(nn); e += 0.5 * math.sqrt(area / n); n_s += 1
    return None if n_s == 0 else o / e

def run_episode(Env, cfg, mods, seed):
    env = Env(cfg)
    obs, _ = env.reset(seed=seed)
    active = list(obs.keys()); traj = []; step = 0
    while step < 1000 and active:
        acts = {a: act(obs[a], mods["predator_policy" if "predator" in a else "prey_policy"]) for a in active}
        obs, rew, term, trunc, _ = env.step(acts)
        active = [a for a in obs if not term.get(a, False) and not trunc.get(a, False)]
        traj.append([p for a, p in env.agent_positions.items() if "predator" in a])
        step += 1
        if trunc.get("__all__", False) or term.get("__all__", False): break
    return traj

def evaluate(kind, seed, ckpt):
    if kind == "base":
        from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass as Env
        from predpreygrass.non_evolutionary.base_environment.config_env import config_env as cfg
    else:
        from predpreygrass.non_evolutionary.base_environment_step_energy.predpreygrass_rllib_env import PredPreyGrass as Env
        from predpreygrass.non_evolutionary.base_environment_step_energy.config_env import config_env as cfg
    mods = load_modules(ckpt_dir(kind, seed, ckpt))
    rs = []
    for s in EVAL_SEEDS:
        r = clark_evans(run_episode(Env, cfg, mods, s))
        if r is not None: rs.append(r)
    return rs

if __name__ == "__main__":
    seed = int(sys.argv[1])
    out = {"seed": seed, "ckpt": "checkpoint_000029"}
    out["base"] = evaluate("base", seed, "checkpoint_000029")
    out["step"] = evaluate("step", seed, "checkpoint_000029")
    if seed == 42:
        out["base_final"] = evaluate("base", seed, "checkpoint_000099")
        out["step_final"] = evaluate("step", seed, "checkpoint_000049")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"clustering_seed{seed}.json")
    json.dump(out, open(path, "w"))
    print(f"seed {seed} done: base mean R={np.mean(out['base']):.4f} (n={len(out['base'])}), step mean R={np.mean(out['step']):.4f} (n={len(out['step'])})")
