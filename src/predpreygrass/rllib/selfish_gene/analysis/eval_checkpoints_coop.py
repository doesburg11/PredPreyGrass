import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch

from ray.rllib.core.rl_module.rl_module import RLModule

from predpreygrass.rllib.selfish_gene.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.selfish_gene.config.config_env_selfish_gene import config_env as base_config
from predpreygrass.rllib.selfish_gene.analysis.coop_metrics import (
    load_episode,
    compute_assortment_index,
    compute_assortment_index_by_policy,
    compute_kin_proximity_advantage,
    compute_kin_proximity_advantage_by_policy,
)


def policy_mapping_fn(agent_id: str) -> str:
    parts = agent_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    raise ValueError(f"Unrecognized agent_id format: {agent_id}")


def policy_pi(observation, policy_module: RLModule, deterministic=True):
    obs_tensor = torch.tensor(observation).float().unsqueeze(0)
    with torch.no_grad():
        action_output = policy_module._forward_inference({"obs": obs_tensor})
    logits = action_output.get("action_dist_inputs")
    if logits is None:
        raise KeyError("policy_pi: action_dist_inputs not found in action_output.")
    if deterministic:
        return torch.argmax(logits, dim=-1).item()
    else:
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item()


def _load_training_env_config_from_run(checkpoint_path: str, base_cfg: dict) -> dict:
    candidates = [
        os.path.join(os.path.dirname(checkpoint_path), "run_config.json"),
        os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "run_config.json"),
    ]
    training_env_cfg = None
    for cand in candidates:
        if os.path.isfile(cand):
            try:
                with open(cand, "r") as f:
                    rc = json.load(f)
                if isinstance(rc, dict):
                    # Prefer explicit env_config if present; else, intersect known keys
                    if isinstance(rc.get("env_config"), dict):
                        training_env_cfg = rc["env_config"]
                    else:
                        training_env_cfg = {k: rc[k] for k in base_cfg.keys() if k in rc}
                break
            except Exception:
                pass

    if not isinstance(training_env_cfg, dict):
        return base_cfg

    obs_keys = {
        "grid_size",
        "num_obs_channels",
        "predator_obs_range",
        "prey_obs_range",
        "include_visibility_channel",
        "mask_observation_with_visibility",
        "respect_los_for_movement",
        "include_kin_density_channel",
        "kin_density_radius",
        "kin_density_norm_cap",
        "kin_density_los_aware",
        "type_1_action_range",
        "type_2_action_range",
    }
    merged = dict(base_cfg)
    for k in obs_keys:
        if k in training_env_cfg:
            merged[k] = training_env_cfg[k]
    return merged


def find_checkpoints(run_path: str):
    p = Path(run_path)
    if p.name.startswith("checkpoint_") and p.is_dir():
        return [p]
    # search one level deep for checkpoint_* folders
    cps = sorted([d for d in p.rglob("checkpoint_*") if d.is_dir()], key=lambda d: int(re.findall(r"(\d+)$", d.name)[0]))
    return cps


def run_eval_for_checkpoint(checkpoint_dir: Path, episodes: int, max_steps: int | None, log_root: Path, seed: int):
    # Load RLModules
    rl_module_dir = checkpoint_dir / "learner_group" / "learner" / "rl_module"
    if not rl_module_dir.is_dir():
        raise FileNotFoundError(f"RLModule directory not found: {rl_module_dir}")
    rl_modules = {}
    for pid_dir in rl_module_dir.iterdir():
        if pid_dir.is_dir():
            rl_modules[pid_dir.name] = RLModule.from_checkpoint(str(pid_dir))

    # Build env config aligned with training
    cfg = dict(base_config)
    cfg = _load_training_env_config_from_run(str(checkpoint_dir), cfg)
    cfg["enable_coop_logging"] = True
    ckpt_id = checkpoint_dir.name
    ckpt_log_dir = log_root / ckpt_id
    ckpt_log_dir.mkdir(parents=True, exist_ok=True)
    cfg["coop_log_dir"] = str(ckpt_log_dir)
    if max_steps is not None:
        cfg["max_steps"] = int(max_steps)

    env = PredPreyGrass(cfg)
    for ep in range(episodes):
        obs, _ = env.reset(seed=int(seed + ep))
        done = False
        while not done:
            actions = {}
            for agent_id in env.agents:
                pid = policy_mapping_fn(agent_id)
                module = rl_modules.get(pid)
                if module is None:
                    # Default to no-op action 0 if module missing (shouldn't happen in multi-policy runs)
                    actions[agent_id] = 0
                else:
                    actions[agent_id] = policy_pi(obs[agent_id], module, deterministic=True)
            obs, rew, term, trunc, _ = env.step(actions)
            done = term.get("__all__", False) or trunc.get("__all__", False)
    env.close()

    # Analyze
    episode_files = sorted(ckpt_log_dir.glob("episode_*.json"))
    episodes_data = [load_episode(fp) for fp in episode_files]
    return {
        "ai": compute_assortment_index(episodes_data, los_aware=True, n_bootstrap=0, seed=seed),
        "kpa": compute_kin_proximity_advantage(episodes_data, los_aware=True, n_bootstrap=0, seed=seed),
        "ai_by_policy": compute_assortment_index_by_policy(episodes_data, los_aware=True, n_bootstrap=0, seed=seed),
        "kpa_by_policy": compute_kin_proximity_advantage_by_policy(episodes_data, los_aware=True, n_bootstrap=0, seed=seed),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints with coop logging and summarize AI/KPA.")
    parser.add_argument("--run", required=True, help="Path to Ray run dir or a specific checkpoint_* dir")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per checkpoint")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode (override config)")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for eval")
    parser.add_argument("--limit", type=int, default=None, help="Process only the last N checkpoints")
    parser.add_argument("--out", default="output/coop_eval_summary.csv", help="CSV path for summary output")
    parser.add_argument("--log-root", default="output/coop_eval_logs", help="Root directory for per-checkpoint episode logs")

    args = parser.parse_args()
    run_path = args.run
    cps = find_checkpoints(run_path)
    if not cps:
        raise FileNotFoundError(f"No checkpoints found under: {run_path}")
    if args.limit is not None and args.limit > 0:
        cps = cps[-args.limit:]

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for cp in cps:
        m = re.search(r"(\d+)$", cp.name)
        iteration = int(m.group(1)) if m else -1
        print(f"[EVAL] Checkpoint {cp} (iter={iteration}) → generating {args.episodes} episodes")
        metrics = run_eval_for_checkpoint(cp, episodes=args.episodes, max_steps=args.max_steps, log_root=log_root, seed=args.seed)
        # Flatten key metrics for CSV
        row = {
            "checkpoint": cp.name,
            "iteration": iteration,
            "ai": metrics["ai"].get("ai", 0.0),
            "ai_n": metrics["ai"].get("n", 0),
            "kpa": metrics["kpa"].get("kpa", 0.0),
            "kpa_n_with": metrics["kpa"].get("n_with", 0),
            "kpa_n_without": metrics["kpa"].get("n_without", 0),
        }
        rows.append(row)

        # Also persist full JSON per-checkpoint
        json_out = log_root / f"metrics_{cp.name}.json"
        with open(json_out, "w") as f:
            json.dump(metrics, f, indent=2)

    # Write summary CSV
    fieldnames = ["checkpoint", "iteration", "ai", "ai_n", "kpa", "kpa_n_with", "kpa_n_without"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[DONE] Wrote summary to {out_csv} and per-checkpoint metrics JSON to {log_root}")


if __name__ == "__main__":
    main()
