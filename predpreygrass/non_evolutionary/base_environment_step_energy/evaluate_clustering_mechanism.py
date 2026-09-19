"""
Birth-and-dispersal mechanism measurements (see RESULTS.md section 18).

Usage (repo root, project Python), one task per invocation, kind in {base, step, runA, freerest}:
    python predpreygrass/non_evolutionary/base_environment_step_energy/evaluate_clustering_mechanism.py <kind> <seed>

Same policies/episodes as evaluate_clustering_density.py (iteration-300 checkpoints, 30 deterministic
episodes, reset seeds 100-129, CPU-only), instrumented to record, per episode: the Clark-Evans R;
the fraction of predator actions that are noop and the fraction of predator-steps in which the
predator actually changed cell, and mean displacement per step; the number of predator births
(from which per-predator birth rate follows); the mean parent-offspring distance at offspring ages
1,2,3,5,10,20,30,50,75,100 steps (the spawn-position call is wrapped to capture each newborn's
parent); and the fraction of predators with another predator within 1.5 cells.
Writes clustering_mechanism_data/mech_<kind>_<seed>.json.
Analysis: analyze_clustering_mechanism.py.
"""
import sys, json, math, glob, os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
torch.set_num_threads(1)
from ray.rllib.core.rl_module.rl_module import RLModule

RR = os.path.expanduser("~/simulation_results/ray_results")
EVAL_SEEDS = list(range(100, 130)); CK = "checkpoint_000029"
AGES = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]

def root_for(kind, seed):
    if kind == "runA": return glob.glob(f"{RR}/PPO_STEP_ENERGY_ADDITIVE_CONFIRM500_SEED42/PPO_*")[0]
    if kind == "freerest": return glob.glob(f"{RR}/PPO_STEP_ENERGY_CONFIRM500_SEED42/PPO_*")[0]
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

def eu(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

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
    state = {"p": env._next_predator_idx, "q": env._next_prey_idx}; pending = []
    orig = env._find_available_spawn_position
    def patched(ref, occ):
        res = orig(ref, occ)
        if env._next_predator_idx != state["p"]:
            pending.append((f"predator_{env._next_predator_idx - 1}", tuple(ref))); state["p"] = env._next_predator_idx
        elif env._next_prey_idx != state["q"]:
            state["q"] = env._next_prey_idx
        return res
    env._find_available_spawn_position = patched

    active = list(obs.keys()); t = 0
    prev = {a: tuple(p) for a, p in env.agent_positions.items() if "predator" in a}
    traj = []; noop = pact = moved = disp_n = 0; disp_sum = 0.0; pred_steps = 0; adj_sum = 0.0; adj_n = 0
    births = []                                   # [newborn, parent, t_birth]
    age_sum = {a: 0.0 for a in AGES}; age_cnt = {a: 0 for a in AGES}
    while t < 1000 and active:
        acts = {a: act(obs[a], mods["predator_policy" if "predator" in a else "prey_policy"]) for a in active}
        for a, ac in acts.items():
            if "predator" in a: pact += 1; noop += (ac == 4)
        obs, rew, term, trunc, _ = env.step(acts)
        active = [a for a in obs if not term.get(a, False) and not trunc.get(a, False)]
        cur = {a: tuple(p) for a, p in env.agent_positions.items() if "predator" in a}
        for a, p in cur.items():
            if a in prev:
                d = eu(p, prev[a]); disp_sum += d; disp_n += 1; moved += (d > 0)
        for nid, ref in pending:
            par = [a for a, p in cur.items() if p == ref and a != nid]
            if par and nid in cur: births.append([nid, par[0], t])
        pending.clear()
        for nid, pid, tb in births:
            age = t - tb
            if age in age_sum and nid in cur and pid in cur:
                age_sum[age] += eu(cur[nid], cur[pid]); age_cnt[age] += 1
        pos = list(cur.values()); traj.append(pos); pred_steps += len(pos)
        if len(pos) >= 2:
            pts = np.array(pos, float); c = 0
            for i in range(len(pos)):
                d = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1)); d[i] = np.inf; c += (d.min() <= 1.5)
            adj_sum += c / len(pos); adj_n += 1
        prev = cur; t += 1
        if trunc.get("__all__", False) or term.get("__all__", False): break
    return {"R": clark_evans(traj), "noop_frac": noop / max(pact, 1), "moved_frac": moved / max(disp_n, 1),
            "mean_disp": disp_sum / max(disp_n, 1), "births": len(births), "pred_steps": pred_steps,
            "adj_frac": adj_sum / max(adj_n, 1), "age_sum": age_sum, "age_cnt": age_cnt}

if __name__ == "__main__":
    kind, seed = sys.argv[1], int(sys.argv[2])
    if kind == "base":
        from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass as Env
        from predpreygrass.non_evolutionary.base_environment.config_env import config_env as cfg
    else:
        from predpreygrass.non_evolutionary.base_environment_step_energy.predpreygrass_rllib_env import PredPreyGrass as Env
        from predpreygrass.non_evolutionary.base_environment_step_energy.config_env import config_env as cfg
        cfg = {**cfg, **OVERRIDES.get(kind, {})}
    r = root_for(kind, seed)
    mods = {p: RLModule.from_checkpoint(f"{r}/{CK}/learner_group/learner/rl_module/{p}") for p in ("predator_policy", "prey_policy")}
    eps = []
    for s in EVAL_SEEDS:
        e = run_episode(Env, cfg, mods, s)
        if e["R"] is not None: eps.append(e)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clustering_mechanism_data", f"mech_{kind}_{seed}.json")
    json.dump(eps, open(out, "w"))
    print(f"{kind} s{seed} done: R={np.mean([e['R'] for e in eps]):.4f} noop={np.mean([e['noop_frac'] for e in eps]):.3f} "
          f"disp={np.mean([e['mean_disp'] for e in eps]):.3f} births={np.mean([e['births'] for e in eps]):.1f} n={len(eps)}")
