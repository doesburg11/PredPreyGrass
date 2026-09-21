"""
Compare what the male and the female policy of a checkpoint would do on EXACTLY THE SAME observations.

The rollout analysis (analyze_prey_approach_from_checkpoint.py) lets every policy create its own states, so its
approach / step-onto ratios mix action preference with the states each policy visits. Here the states come from a
fixed bank generated once by a uniform-random policy (independent of any trained policy, and of sex), and the
environment is NOT stepped while querying the policies. Each policy's action distribution is read off the same bank.

Metrics, per policy (male / female), over the bank states that have a visible target:
  approach_bias = E_uniform[d_after] - E_policy[d_after]   (cells; positive = moves toward the nearest target
                                                            more than a random mover would)
  P(step onto target) when the target is adjacent, next to the uniform-random value on the same states
  P(noop) over all states
and the paired male-minus-female difference with an episode-level bootstrap interval (states of one episode are
resampled together). Intervals cover the states of the bank for ONE checkpoint, not training-seed variation.

Example:
  python -m predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_shared_states \
      --run K05_s42=~/simulation_results/ray_results/PPO_PREDATOR_SEXUAL_REPRODUCTION_PROP_K05_MB1024_SEED42 \
      --run REF=~/simulation_results/ray_results/PPO_PREDATOR_SEXUAL_REPRODUCTION_REF_PEN02_MB1024_SEED42 \
      --checkpoints 19 24 29 --episodes 20
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_prey_approach_from_checkpoint import (
    action_geometry,
    load_modules,
    module_probs,
)
from predpreygrass.non_evolutionary.predator_sexual_reproduction.predpreygrass_rllib_env import PredPreyGrass

TARGETS = ("prey", "fruit")
POLICIES = ("predator_male_policy", "predator_female_policy")


def build_bank(env_config, n_episodes, seed0, every=5):
    """Sample predator observations (both sexes) from uniform-random-policy episodes; store target geometry."""
    env = PredPreyGrass(env_config)
    moves = np.array([env.action_to_move_tuple[a] for a in range(env.num_actions)])
    offset = (env.predator_obs_range - 1) // 2
    rng = np.random.default_rng(seed0)
    obs_list, ep_ids, cheb = [], [], {t: [] for t in TARGETS}
    d_after = {t: [] for t in TARGETS}
    lands = {t: [] for t in TARGETS}
    for ep in range(n_episodes):
        observations, _ = env.reset(seed=seed0 + ep)
        active = list(observations.keys())
        while True:
            if env.current_step % every == 0:
                targets = {
                    "prey": np.array(list(env.prey_positions.values()), dtype=int).reshape(-1, 2),
                    "fruit": np.array(
                        [p for f, p in env.fruit_positions.items() if env.fruit_energies[f] > 0], dtype=int
                    ).reshape(-1, 2),
                }
                for agent in active:
                    if "predator" not in agent or agent not in env.agent_positions:
                        continue
                    pos = np.array(env.agent_positions[agent], dtype=int)
                    obs_list.append(np.asarray(observations[agent], dtype=np.float32))
                    ep_ids.append(ep)
                    for t in TARGETS:
                        geo = action_geometry(pos, targets[t], moves, env.grid_size, offset)
                        if geo is None:
                            d_after[t].append(np.full(env.num_actions, np.nan))
                            lands[t].append(np.zeros(env.num_actions, dtype=bool))
                            cheb[t].append(0)
                        else:
                            d_after[t].append(geo[0])
                            lands[t].append(geo[1])
                            cheb[t].append(geo[2])
            actions = {a: int(rng.integers(env.num_actions)) for a in active}
            observations, _, terminations, truncations, _ = env.step(actions)
            active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
            if terminations.get("__all__") or truncations.get("__all__"):
                break
    return {
        "obs": np.stack(obs_list),
        "episode": np.array(ep_ids),
        "cheb": {t: np.array(cheb[t]) for t in TARGETS},
        "d_after": {t: np.stack(d_after[t]) for t in TARGETS},
        "lands": {t: np.stack(lands[t]) for t in TARGETS},
        "n_actions": env.num_actions,
        "noop": env.noop_action_id,
    }


def policy_probs(module, obs, batch=4096):
    out = []
    for i in range(0, len(obs), batch):
        out.append(module_probs(module, list(obs[i : i + batch])))
    return np.concatenate(out)


def per_state_metrics(bank, probs, target):
    """Per-state quantities (NaN where not applicable): approach bias, adjacent step-onto prob, and the uniform reference."""
    d = bank["d_after"][target]
    visible = bank["cheb"][target] > 0
    bias = np.full(len(probs), np.nan)
    e_uniform = np.full(len(probs), np.nan)
    if visible.any():
        e_uniform[visible] = d[visible].mean(1)
    e_policy = np.nansum(np.where(visible[:, None], d, 0.0) * probs, axis=1)
    bias[visible] = e_uniform[visible] - e_policy[visible]
    adjacent = bank["cheb"][target] == 1
    p_land = np.full(len(probs), np.nan)
    u_land = np.full(len(probs), np.nan)
    lands = bank["lands"][target].astype(float)
    p_land[adjacent] = (lands * probs).sum(1)[adjacent]
    u_land[adjacent] = lands.mean(1)[adjacent]
    return bias, p_land, u_land


def boot_mean_diff(x, y, episodes, rng, n_boot=500):
    """Mean of x - y over states with finite values, resampling whole episodes."""
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return float("nan"), float("nan"), float("nan")
    diff = (x - y)[ok]
    ep = episodes[ok]
    point = diff.mean()
    uniq = np.unique(ep)
    sums = np.array([diff[ep == e].sum() for e in uniq])
    cnts = np.array([(ep == e).sum() for e in uniq], float)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    return point, *np.percentile(boots, [2.5, 97.5])


def summarize(label, bank, probs_m, probs_f, rng):
    lines = []
    ep = bank["episode"]
    for t in TARGETS:
        bm, pm, um = per_state_metrics(bank, probs_m, t)
        bf, pf, uf = per_state_metrics(bank, probs_f, t)
        n_vis = int(np.isfinite(bm).sum())
        n_adj = int(np.isfinite(pm).sum())
        d_pt, d_lo, d_hi = boot_mean_diff(bm, bf, ep, rng)
        s_pt, s_lo, s_hi = boot_mean_diff(pm, pf, ep, rng)
        lines.append(
            f"{label:26} {t:5} n_visible={n_vis:6d} approach_bias M={np.nanmean(bm):+.3f} F={np.nanmean(bf):+.3f} "
            f"diff(M-F)={d_pt:+.3f} [{d_lo:+.3f},{d_hi:+.3f}] | adjacent n={n_adj:5d}: "
            f"P(step onto {t}) M={np.nanmean(pm):.3f} F={np.nanmean(pf):.3f} random={np.nanmean(um):.3f} "
            f"diff(M-F)={s_pt:+.3f} [{s_lo:+.3f},{s_hi:+.3f}]"
        )
    noop = bank["noop"]
    lines.append(f"{label:26} noop  P(noop) M={probs_m[:, noop].mean():.3f} F={probs_f[:, noop].mean():.3f} (uniform {1 / bank['n_actions']:.3f})")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=path to a ray_results experiment dir")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[29])
    parser.add_argument("--episodes", type=int, default=20, help="random-policy episodes for the state bank")
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--bank-config", type=str, default=None,
                        help="run_config.json of the run whose env config builds the bank (default: first --run)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    torch.set_num_threads(2)
    runs = [(r.split("=", 1)[0], os.path.expanduser(r.split("=", 1)[1])) for r in args.run]
    cfg_path = os.path.expanduser(args.bank_config) if args.bank_config else os.path.join(runs[0][1], "run_config.json")
    env_config = json.load(open(cfg_path))["config_env"]
    print(f"building state bank from {args.episodes} random-policy episodes ...", flush=True)
    bank = build_bank(env_config, args.episodes, args.seed)
    print(f"bank: {len(bank['obs'])} predator states; visible prey in {(bank['cheb']['prey'] > 0).sum()}, "
          f"adjacent prey in {(bank['cheb']['prey'] == 1).sum()}, visible fruit in {(bank['cheb']['fruit'] > 0).sum()}", flush=True)
    rng = np.random.default_rng(0)
    output = []

    def emit(line):
        print(line, flush=True)
        output.append(line)

    for label, path in runs:
        trial = sorted(glob.glob(os.path.join(path, "PPO_PredPreyGrass_*/")))[0]
        for k in args.checkpoints:
            modules = load_modules(os.path.join(trial, f"checkpoint_{k:06d}"))
            pm = policy_probs(modules["predator_male_policy"], bank["obs"])
            pf = policy_probs(modules["predator_female_policy"], bank["obs"])
            for line in summarize(f"{label} ckpt{k}", bank, pm, pf, rng):
                emit(line)
            emit("")
    if args.out:
        with open(os.path.expanduser(args.out), "w") as f:
            json.dump(output, f, indent=1)


if __name__ == "__main__":
    main()
