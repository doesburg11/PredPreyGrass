"""
Evaluation with actions SAMPLED from each policy (as in training) rather than taken greedily
(see RESULTS.md section 20).

Usage (repo root, project Python), kind in {base, step}:
    python predpreygrass/non_evolutionary/base_environment_step_energy/evaluate_sampled_actions.py <kind> <seed>

Loads the seed's iteration-300 checkpoint (checkpoint_000029), runs 30 episodes (reset seeds 100-129,
torch seeded per episode), CPU-only, and records per episode the number of steps, whether the
predator population went extinct, the final predator count and the mean predator count.
Writes sampled_action_data/sampled_<kind>_<seed>.json. Greedy counterparts: clustering_density_data/.
"""
import sys, json, glob, os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np, torch
torch.set_num_threads(1)
from ray.rllib.core.rl_module.rl_module import RLModule
RR=os.path.expanduser("~/simulation_results/ray_results"); CK="checkpoint_000029"; SEEDS=list(range(100,130))
def root(kind,seed):
    if seed==42:
        return (f"{RR}/base_v_drive_2026-09-06/PPO_BASE_ENVIRONMENT_SEED42_2026-09-05_18-55-45/PPO_PredPreyGrass_a2fe1_00000_0_2026-09-05_18-55-45" if kind=="base"
                else f"{RR}/PPO_STEP_ENERGY_ADDITIVE_PREDEASE_CONFIRM500_SEED42/PPO_PredPreyGrass_eed47_00000_0_2026-09-17_07-30-45")
    n=f"PPO_BASE_ENVIRONMENT_SEED{seed}_MULTISEED" if kind=="base" else f"PPO_STEP_ENERGY_RUNB_SEED{seed}_MULTISEED"
    return glob.glob(f"{RR}/{n}/PPO_*")[0]
kind,seed=sys.argv[1],int(sys.argv[2])
if kind=="base":
    from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass as Env
    from predpreygrass.non_evolutionary.base_environment.config_env import config_env as cfg
else:
    from predpreygrass.non_evolutionary.base_environment_step_energy.predpreygrass_rllib_env import PredPreyGrass as Env
    from predpreygrass.non_evolutionary.base_environment_step_energy.config_env import config_env as cfg
r=root(kind,seed)
mods={p: RLModule.from_checkpoint(f"{r}/{CK}/learner_group/learner/rl_module/{p}") for p in ("predator_policy","prey_policy")}
def act_sampled(obs,m):
    with torch.no_grad(): out=m._forward_inference({"obs":torch.tensor(obs).float().unsqueeze(0)})
    return torch.distributions.Categorical(logits=out["action_dist_inputs"]).sample().item()
rows=[]
for s in SEEDS:
    torch.manual_seed(s)
    env=Env(cfg); obs,_=env.reset(seed=s); active=list(obs.keys()); t=0; npred=[]
    while t<1000 and active:
        acts={a: act_sampled(obs[a], mods["predator_policy" if "predator" in a else "prey_policy"]) for a in active}
        obs,rew,term,trunc,_=env.step(acts); active=[a for a in obs if not term.get(a,False) and not trunc.get(a,False)]
        npred.append(env.current_num_predators); t+=1
        if trunc.get("__all__",False) or term.get("__all__",False): break
    rows.append({"steps":t,"pred_extinct":env.current_num_predators<=0,"final_pred":env.current_num_predators,"mean_pred":float(np.mean(npred))})
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"sampled_action_data",f"sampled_{kind}_{seed}.json")
json.dump(rows,open(out,"w"))
print(f"{kind} s{seed} sampled: pred_extinct={np.mean([x['pred_extinct'] for x in rows])*100:.0f}% mean_pred={np.mean([x['mean_pred'] for x in rows]):.2f}")
