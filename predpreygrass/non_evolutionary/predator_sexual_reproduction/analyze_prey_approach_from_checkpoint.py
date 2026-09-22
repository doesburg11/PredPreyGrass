"""
Measure directly whether trained predator policies approach or avoid prey (and, as a
control, fruit), by rolling out saved checkpoints.

For every predator decision where a target (prey or fruit) is visible (within the
predator's observation window) and not co-located, the policy's action distribution is
compared against a uniform-random mover on the SAME state:

  approach_bias = E_uniform[d_after] - E_policy[d_after]
      d_after = Euclidean distance from the destination cell to the nearest target.
      Positive = the policy moves closer than a random mover would; negative = avoids.
  P(step onto target) vs random, in states where a target is adjacent (Chebyshev 1):
      for prey this is the probability of choosing the action that triggers a hunt.

Comparing predator_female with predator_male (and prey with fruit) separates "does not
approach prey at all" from "does not approach anything". Confidence intervals come from
bootstrapping over episodes. --selftest verifies the metric's sign with synthetic
always-approach / always-avoid policies.

Example:
  python -m predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_prey_approach_from_checkpoint \
      --run POSCONTROL=~/simulation_results/ray_results/PPO_PREDATOR_SEXUAL_REPRODUCTION_POSITIVE_CONTROL_SEED42 \
      --run REALISTIC=~/simulation_results/ray_results/PPO_PREDATOR_SEXUAL_REPRODUCTION_REALISTIC_SEED42 \
      --checkpoints 0 4 9 --episodes 40
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from predpreygrass.non_evolutionary.predator_sexual_reproduction.predpreygrass_rllib_env import PredPreyGrass

POLICY_IDS = ("predator_male_policy", "predator_female_policy", "prey_policy")
SEXES = ("predator_male", "predator_female")
TARGETS = ("prey", "fruit")


def policy_of(agent_id):
    if "predator_male" in agent_id:
        return "predator_male_policy"
    if "predator_female" in agent_id:
        return "predator_female_policy"
    return "prey_policy"


def load_modules(checkpoint_dir):
    from ray.rllib.core.rl_module.rl_module import RLModule

    root = os.path.join(checkpoint_dir, "learner_group", "learner", "rl_module")
    return {pid: RLModule.from_checkpoint(os.path.join(root, pid)) for pid in POLICY_IDS}


def module_probs(module, observations):
    x = torch.as_tensor(np.stack(observations), dtype=torch.float32)
    with torch.no_grad():
        logits = module._forward_inference({"obs": x})["action_dist_inputs"]
    probs = torch.softmax(logits, dim=-1).double().numpy()
    return probs / probs.sum(axis=1, keepdims=True)


def action_geometry(pos, targets, moves, grid_size, offset):
    """None unless the nearest target is visible (Chebyshev <= offset) and none is co-located."""
    if len(targets) == 0:
        return None
    cheb = np.abs(targets - pos).max(axis=1)
    if (cheb == 0).any() or cheb.min() > offset:
        return None
    new = np.clip(pos + moves, 0, grid_size - 1)
    delta = new[:, None, :] - targets[None, :, :]
    d_after = np.sqrt((delta ** 2).sum(-1)).min(axis=1)
    lands = (delta == 0).all(-1).any(axis=1)
    return d_after, lands, int(cheb.min())


def synthetic_probs(mode, geometry, n_actions):
    uniform = np.full(n_actions, 1.0 / n_actions)
    if mode == "random" or geometry is None:
        return uniform
    d_after = geometry[0]
    best = np.isclose(d_after, d_after.min() if mode == "greedy_approach" else d_after.max())
    return best / best.sum()


def new_stats():
    return {"n": 0, "bias": 0.0, "n_adj": 0, "p_land": 0.0, "u_land": 0.0}


def record(stats, geometry, probs):
    d_after, lands, cheb_min = geometry
    stats["n"] += 1
    stats["bias"] += d_after.mean() - float(probs @ d_after)
    if cheb_min == 1:
        stats["n_adj"] += 1
        stats["p_land"] += float(probs @ lands)
        stats["u_land"] += float(lands.mean())


def run_episodes(env_config, modules, mode, n_episodes, seed0):
    env = PredPreyGrass(env_config)
    moves = np.array([env.action_to_move_tuple[a] for a in range(env.num_actions)])
    offset = (env.predator_obs_range - 1) // 2
    sample_rng = np.random.default_rng(seed0)
    per_episode = []
    for ep in range(n_episodes):
        observations, _ = env.reset(seed=seed0 + ep)
        active = list(observations.keys())
        stats = {(sex, tgt): new_stats() for sex in SEXES for tgt in TARGETS}
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
                if mode == "checkpoint":
                    rows = module_probs(modules[pid], [observations[a] for a in agents])
                for i, agent in enumerate(agents):
                    is_predator = pid != "prey_policy"
                    if is_predator:
                        pos = np.array(env.agent_positions[agent], dtype=int)
                        geo = {t: action_geometry(pos, targets[t], moves, env.grid_size, offset) for t in TARGETS}
                        probs = rows[i] if mode == "checkpoint" else synthetic_probs(mode, geo["prey"], env.num_actions)
                        sex = "predator_male" if pid == "predator_male_policy" else "predator_female"
                        for t in TARGETS:
                            if geo[t] is not None:
                                record(stats[(sex, t)], geo[t], probs)
                    else:
                        probs = rows[i] if mode == "checkpoint" else np.full(env.num_actions, 1.0 / env.num_actions)
                    actions[agent] = int(sample_rng.choice(env.num_actions, p=probs))
            observations, _, terminations, truncations, _ = env.step(actions)
            active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
            if terminations.get("__all__") or truncations.get("__all__"):
                break
        per_episode.append(stats)
    return per_episode


def bootstrap_ratio(numerators, denominators, rng, n_boot=2000):
    num, den = np.asarray(numerators, float), np.asarray(denominators, float)
    point = num.sum() / den.sum() if den.sum() > 0 else float("nan")
    if den.sum() == 0:
        return point, float("nan"), float("nan")
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    d = den[idx].sum(1)
    ok = d > 0
    boots = num[idx].sum(1)[ok] / d[ok]
    return point, *np.percentile(boots, [2.5, 97.5])


def summarize(label, per_episode, rng):
    lines = []
    for sex in SEXES:
        for tgt in TARGETS:
            rows = [ep[(sex, tgt)] for ep in per_episode]
            n = sum(r["n"] for r in rows)
            if n == 0:
                lines.append(f"{label:22} {sex[9:]:6} {tgt:5} no visible-target decisions")
                continue
            bias, lo, hi = bootstrap_ratio([r["bias"] for r in rows], [r["n"] for r in rows], rng)
            n_adj = sum(r["n_adj"] for r in rows)
            adj = ""
            if n_adj:
                p = sum(r["p_land"] for r in rows) / n_adj
                u = sum(r["u_land"] for r in rows) / n_adj
                adj = f" | adjacent n={n_adj:5d}: P(step onto {tgt})={p:.3f} vs random {u:.3f} (x{p / u:.2f})"
            lines.append(
                f"{label:22} {sex[9:]:6} {tgt:5} n={n:6d} approach_bias={bias:+.3f} cells [{lo:+.3f},{hi:+.3f}]{adj}"
            )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=path to a ray_results experiment dir")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 4, 9], help="checkpoint indices")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--selftest", action="store_true", help="synthetic approach/avoid policies on the first run's config")
    parser.add_argument("--out", type=str, default=None, help="optional path for a JSON dump of the summary lines")
    args = parser.parse_args()
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    runs = [(r.split("=", 1)[0], os.path.expanduser(r.split("=", 1)[1])) for r in args.run]
    output = []

    def emit(line):
        print(line, flush=True)
        output.append(line)

    if args.selftest:
        label, path = runs[0]
        config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
        for mode in ("random", "greedy_approach", "greedy_avoid"):
            for line in summarize(f"selftest:{mode}", run_episodes(config, None, mode, min(args.episodes, 8), args.seed), rng):
                emit(line)
        return

    for label, path in runs:
        config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
        trial = sorted(glob.glob(os.path.join(path, "PPO_*/")))[0]
        for k in args.checkpoints:
            checkpoint = os.path.join(trial, f"checkpoint_{k:06d}")
            modules = load_modules(checkpoint)
            per_episode = run_episodes(config, modules, "checkpoint", args.episodes, args.seed)
            for line in summarize(f"{label} ckpt{k}", per_episode, rng):
                emit(line)
            emit("")
    if args.out:
        with open(os.path.expanduser(args.out), "w") as f:
            json.dump(output, f, indent=1)


if __name__ == "__main__":
    main()
