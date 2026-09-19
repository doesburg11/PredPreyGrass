"""
Clustering evaluation that also records predator density (see RESULTS.md section 17).

Usage (repo root, project Python), one task per invocation, kind in {base, step, runA, freerest}:
    python predpreygrass/non_evolutionary/base_environment_step_energy/evaluate_clustering_density.py <kind> <seed>

kind base/step: the seed's base_environment / base_environment_step_energy policy (iteration-300 checkpoint);
runA: run A (rest 0.10, move +0.10) seed-42 policy; freerest: an earlier run in which resting was free
(rest 0, move 0.15/0.05; a design since removed from the code) seed-42 policy, evaluated with the
equivalent additive-cost keys.
30 deterministic episodes (reset seeds 100-129), CPU-only. Per episode records the Clark-Evans R
and the mean predator count. Writes clustering_density_data/dose_<kind>_<seed>.json.
Analysis (random-placement null, density-adjusted test): analyze_clustering_density.py.
"""
import sys, json, math, glob, os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
torch.set_num_threads(1)
from ray.rllib.core.rl_module.rl_module import RLModule

RR = os.path.expanduser("~/simulation_results/ray_results")
EVAL_SEEDS = list(range(100, 130))
CK = "checkpoint_000029"

def root_for(task):
    kind, seed = task
    if kind == "runA": return glob.glob(f"{RR}/PPO_STEP_ENERGY_ADDITIVE_CONFIRM500_SEED42/PPO_*")[0]
    if kind == "freerest":  return glob.glob(f"{RR}/PPO_STEP_ENERGY_CONFIRM500_SEED42/PPO_*")[0]
    if seed == 42:
        return (f"{RR}/base_v_drive_2026-09-06/PPO_BASE_ENVIRONMENT_SEED42_2026-09-05_18-55-45/PPO_PredPreyGrass_a2fe1_00000_0_2026-09-05_18-55-45"
                if kind == "base" else f"{RR}/PPO_STEP_ENERGY_ADDITIVE_PREDEASE_CONFIRM500_SEED42/PPO_PredPreyGrass_eed47_00000_0_2026-09-17_07-30-45")
    name = f"PPO_BASE_ENVIRONMENT_SEED{seed}_MULTISEED" if kind == "base" else f"PPO_STEP_ENERGY_RUNB_SEED{seed}_MULTISEED"
    return glob.glob(f"{RR}/{name}/PPO_*")[0]

OVERRIDES = {"runA": {"move_energy_cost_per_step_predator": 0.10},
             "freerest": {"homeostatic_energy_cost_per_step_predator": 0.0, "homeostatic_energy_cost_per_step_prey": 0.0,
                     "move_energy_cost_per_step_predator": 0.15, "move_energy_cost_per_step_prey": 0.05}}

def act(obs, m):
    with torch.no_grad():
        out = m._forward_inference({"obs": torch.tensor(obs).float().unsqueeze(0)})
    return torch.argmax(out["action_dist_inputs"], dim=-1).item()

def clark_evans(traj, area=625.0):
    o = e = 0.0; n_s = 0
    for pos in traj:
        n = len(pos)
        if n < 2: continue
        pts = np.array(pos, dtype=float); nn = []
        for i in range(n):
            d = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1)); d[i] = np.inf; nn.append(d.min())
        o += np.mean(nn); e += 0.5 * math.sqrt(area / n); n_s += 1
    return None if n_s == 0 else o / e

def run_episode(Env, cfg, mods, seed):
    env = Env(cfg); obs, _ = env.reset(seed=seed)
    active = list(obs.keys()); traj = []; step = 0
    while step < 1000 and active:
        acts = {a: act(obs[a], mods["predator_policy" if "predator" in a else "prey_policy"]) for a in active}
        obs, rew, term, trunc, _ = env.step(acts)
        active = [a for a in obs if not term.get(a, False) and not trunc.get(a, False)]
        traj.append([p for a, p in env.agent_positions.items() if "predator" in a]); step += 1
        if trunc.get("__all__", False) or term.get("__all__", False): break
    return traj

if __name__ == "__main__":
    kind, seed = sys.argv[1], int(sys.argv[2])
    task = (kind, seed)
    if kind == "base":
        from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass as Env
        from predpreygrass.non_evolutionary.base_environment.config_env import config_env as cfg
    else:
        from predpreygrass.non_evolutionary.base_environment_step_energy.predpreygrass_rllib_env import PredPreyGrass as Env
        from predpreygrass.non_evolutionary.base_environment_step_energy.config_env import config_env as cfg
        cfg = {**cfg, **OVERRIDES.get(kind, {})}
    r = root_for(task)
    mods = {p: RLModule.from_checkpoint(f"{r}/{CK}/learner_group/learner/rl_module/{p}") for p in ("predator_policy", "prey_policy")}
    rows = []
    for s in EVAL_SEEDS:
        traj = run_episode(Env, cfg, mods, s)
        R = clark_evans(traj)
        if R is not None:
            rows.append({"R": R, "n_pred": float(np.mean([len(p) for p in traj])), "steps": len(traj)})
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clustering_density_data", f"dose_{kind}_{seed}.json")
    json.dump(rows, open(out, "w"))
    print(f"{kind} seed {seed} done: meanR={np.mean([x['R'] for x in rows]):.4f} meanN={np.mean([x['n_pred'] for x in rows]):.2f} n={len(rows)}")
