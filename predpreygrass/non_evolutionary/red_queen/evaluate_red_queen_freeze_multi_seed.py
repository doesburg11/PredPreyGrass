"""
Multi-seed, multi-checkpoint-pair Red Queen freeze/unfreeze evaluation.

This is a methodologically stronger replacement for
`evaluate_red_queen_freeze_type_1_only.py`, which ran each of its 4 conditions
exactly once (seed=42, single episode) against a single hardcoded checkpoint
pair. That's not enough evidence to tell a real Red Queen signal (relative
fitness eroding for whichever side is "behind" in training) from one run's
noise. This script fixes both gaps:

  1. Every condition is run across multiple seeds; results are reported as
     mean +/- std, not a single number.
  2. The "early" vs "late" checkpoint split is not hardcoded to one pair --
     you can pass several checkpoint iterations and every consecutive
     (early, late) pair is evaluated, so you can see whether the pattern
     holds consistently across training rather than being an artifact of
     one arbitrarily chosen pair.

For each (early, late) checkpoint pair, four conditions are run (same
structure as the original script):
  - frozen_prey:      predator=late,  prey=early   (mismatched)
  - frozen_predator:  predator=early, prey=late    (mismatched)
  - static_early:     predator=early, prey=early   (matched control)
  - static_late:      predator=late,  prey=late    (matched control)

Per condition/seed, the same fitness proxies as the original are collected
(total reward, offspring counts by type, avg prey offspring, avg prey
lifespan). Results are printed as a summary table and saved to a JSON file
for later analysis/plotting.

Usage:
    python -m predpreygrass.non_evolutionary.red_queen.evaluate_red_queen_freeze_multi_seed \\
        --base-path /path/to/ray_results/PPO_RUN_NAME \\
        --checkpoint-iters 300 600 1000 \\
        --seeds 0 1 2 3 4 \\
        --max-steps 1000 \\
        --out red_queen_results.json

Checkpoint directory naming: by default this expects
`<base_path>/checkpoint_iter_<N>/...` (matching the original script's
example run). If your run uses RLlib Tune's standard zero-padded naming
instead, pass e.g. --checkpoint-dir-template "checkpoint_{iter:06d}".
"""
import argparse
import json
import os
import statistics
from collections import defaultdict

import torch
from ray.rllib.core.rl_module.rl_module import RLModule

from predpreygrass.non_evolutionary.red_queen.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.red_queen.config.config_env_eval import config_env

# Metrics we track per episode; all are "higher = fitter for prey" except
# total_reward, which is a joint (predator+prey) signal kept for reference.
METRIC_KEYS = ["total_reward", "avg_prey_offspring", "avg_prey_lifespan", "prey_survivor_count"]


def policy_mapping_fn(agent_id, *_args, **_kwargs):
    return "_".join(agent_id.split("_")[:3])  # 'type_1_predator' or 'type_1_prey'


def policy_pi(observation, policy_module, deterministic=True):
    obs_tensor = torch.tensor(observation).float().unsqueeze(0)
    with torch.no_grad():
        action_output = policy_module._forward_inference({"obs": obs_tensor})
    logits = action_output.get("action_dist_inputs")
    if logits is None:
        raise KeyError("policy_pi: action_dist_inputs not found in action_output.")
    return torch.argmax(logits, dim=-1).item() if deterministic else torch.distributions.Categorical(logits=logits).sample().item()


class RLModuleCache:
    """Avoids reloading the same checkpoint's weights repeatedly across seeds/pairs."""

    def __init__(self):
        self._cache = {}

    def load(self, ckpt_path, policy_id):
        key = (ckpt_path, policy_id)
        if key not in self._cache:
            self._cache[key] = RLModule.from_checkpoint(
                os.path.join(ckpt_path, "learner_group", "learner", "rl_module", policy_id)
            )
        return self._cache[key]


def run_one_episode(pred_ckpt_path, prey_ckpt_path, module_cache, max_steps, seed):
    rl_modules = {
        "type_1_predator": module_cache.load(pred_ckpt_path, "type_1_predator"),
        "type_1_prey": module_cache.load(prey_ckpt_path, "type_1_prey"),
    }

    env = PredPreyGrass(config=config_env)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0

    for _ in range(max_steps):
        action_dict = {}
        for agent_id in env.agents:
            policy_id = policy_mapping_fn(agent_id)
            action_dict[agent_id] = policy_pi(obs[agent_id], rl_modules[policy_id])

        obs, rewards, terminations, truncations, _ = env.step(action_dict)
        total_reward += sum(rewards.values())

        if terminations.get("__all__") or truncations.get("__all__"):
            break

    prey_stats = [s for s in env.unique_agent_stats.values() if "prey" in s["policy_group"]]
    if prey_stats:
        avg_offspring = sum(a["offspring_count"] for a in prey_stats) / len(prey_stats)
        finished = [a for a in prey_stats if a["death_step"]]
        avg_lifetime = (
            sum(a["death_step"] - a["birth_step"] for a in finished) / len(finished) if finished else 0.0
        )
    else:
        avg_offspring = 0.0
        avg_lifetime = 0.0

    return {
        "total_reward": total_reward,
        "avg_prey_offspring": avg_offspring,
        "avg_prey_lifespan": avg_lifetime,
        "prey_survivor_count": env.active_num_prey,
        "steps_run": env.current_step,
    }


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def run_condition(pred_ckpt_path, prey_ckpt_path, module_cache, max_steps, seeds):
    per_seed = [run_one_episode(pred_ckpt_path, prey_ckpt_path, module_cache, max_steps, seed) for seed in seeds]
    summary = {"n_seeds": len(seeds), "per_seed": per_seed}
    for key in METRIC_KEYS:
        m, s = mean_std([r[key] for r in per_seed])
        summary[key] = {"mean": m, "std": s}
    return summary


def checkpoint_dir(base_path, iteration, template):
    return os.path.join(base_path, template.format(iter=iteration))


def print_pair_summary(early_iter, late_iter, results):
    print(f"\n{'=' * 70}\nCheckpoint pair: early=iter_{early_iter}  late=iter_{late_iter}\n{'=' * 70}")
    header = f"{'condition':<16}" + "".join(f"{k:>22}" for k in METRIC_KEYS)
    print(header)
    for cond_name, summary in results.items():
        row = f"{cond_name:<16}"
        for key in METRIC_KEYS:
            m, s = summary[key]["mean"], summary[key]["std"]
            row += f"{m:>15.2f} +/-{s:>5.2f}"
        print(row)

    # Simple monotonicity signal (not a formal significance test -- just a
    # readable diagnostic): does the mismatched condition's prey-fitness mean
    # fall on the "expected" side of both matched controls?
    fp = results["frozen_prey"]["avg_prey_offspring"]["mean"]  # predator advantaged
    fpred = results["frozen_predator"]["avg_prey_offspring"]["mean"]  # prey advantaged
    se = results["static_early"]["avg_prey_offspring"]["mean"]
    sl = results["static_late"]["avg_prey_offspring"]["mean"]
    lo, hi = min(se, sl), max(se, sl)
    print(f"\nDiagnostic (avg_prey_offspring): matched controls span [{lo:.2f}, {hi:.2f}]")
    print(f"  frozen_prey (predator-advantaged)   = {fp:.2f}  {'BELOW range (as expected)' if fp < lo else 'inside/above range (no clear signal)'}")
    print(f"  frozen_predator (prey-advantaged)   = {fpred:.2f}  {'ABOVE range (as expected)' if fpred > hi else 'inside/below range (no clear signal)'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-path", required=True, help="Ray results run directory containing checkpoint_* subfolders")
    ap.add_argument("--checkpoint-iters", type=int, nargs="+", required=True, help="Sorted list of checkpoint iterations to compare, e.g. 300 600 1000")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Seeds to average each condition over")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--checkpoint-dir-template", default="checkpoint_iter_{iter}", help="Format string for checkpoint subfolder names")
    ap.add_argument("--out", default="red_queen_results.json")
    args = ap.parse_args()

    iters = sorted(args.checkpoint_iters)
    if len(iters) < 2:
        raise ValueError("Need at least 2 checkpoint iterations to form an (early, late) pair.")

    module_cache = RLModuleCache()
    all_results = {}

    for early_iter, late_iter in zip(iters[:-1], iters[1:]):
        early_ckpt = checkpoint_dir(args.base_path, early_iter, args.checkpoint_dir_template)
        late_ckpt = checkpoint_dir(args.base_path, late_iter, args.checkpoint_dir_template)

        conditions = {
            "frozen_prey": (late_ckpt, early_ckpt),  # predator=late, prey=early
            "frozen_predator": (early_ckpt, late_ckpt),  # predator=early, prey=late
            "static_early": (early_ckpt, early_ckpt),
            "static_late": (late_ckpt, late_ckpt),
        }

        pair_results = {}
        for cond_name, (pred_ckpt, prey_ckpt) in conditions.items():
            print(f"Running condition '{cond_name}' (early={early_iter}, late={late_iter}) over {len(args.seeds)} seeds...")
            pair_results[cond_name] = run_condition(pred_ckpt, prey_ckpt, module_cache, args.max_steps, args.seeds)

        print_pair_summary(early_iter, late_iter, pair_results)
        all_results[f"{early_iter}_vs_{late_iter}"] = pair_results

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results (per-seed and aggregated) written to {args.out}")


if __name__ == "__main__":
    main()
