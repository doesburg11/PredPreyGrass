import argparse
import json
from pathlib import Path
import time
from typing import Dict, Any, List

import numpy as np

try:
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
except Exception:
    ray = None
    Algorithm = object  # type: ignore

from predpreygrass.rllib.kin_selection.predpreygrass_rllib_env import PredPreyGrass


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate helping metrics across checkpoints.")
    p.add_argument("--results-dir", type=str, required=True, help="Ray results directory containing experiment folder(s)")
    p.add_argument("--env-config", type=str, required=False, help="Path to JSON file with env config overrides")
    p.add_argument("--n-episodes", type=int, default=10, help="Episodes per checkpoint")
    p.add_argument("--bootstrap", type=int, default=200, help="Bootstrap samples for CI")
    p.add_argument("--confidence", type=float, default=0.95, help="Confidence level for CI")
    p.add_argument("--stochastic", action="store_true", help="Enable exploration during evaluation (stochastic policy)")
    p.add_argument("--out", type=str, default="helping_eval.csv", help="Output CSV path")
    return p.parse_args()


def load_env_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        return json.load(f)


def list_checkpoints(results_dir: Path) -> List[Path]:
    cks = []
    for exp in results_dir.iterdir():
        if not exp.is_dir():
            continue
        for ck in exp.rglob("checkpoint_*"):
            if (ck / "algorithm_state.pkl").exists() or (ck / "checkpoint.pkl").exists():
                cks.append(ck)
    cks.sort(key=lambda p: p.stat().st_mtime)
    return cks


def bootstrap_ci(values: np.ndarray, n_boot: int, conf: float):
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(123)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        samples.append(values[idx].mean())
    lo = np.percentile(samples, (1 - conf) / 2 * 100)
    hi = np.percentile(samples, (1 + conf) / 2 * 100)
    return (float(lo), float(hi))


def run_eval_on_checkpoint(ckpt: Path, env_config: Dict[str, Any], n_episodes: int, stochastic: bool):
    # Lazy import to avoid Ray requirement for dry runs
    from ray.tune.registry import register_env
    from ray.rllib.algorithms.ppo import PPOConfig

    def env_creator(cfg):
        return PredPreyGrass(cfg)

    register_env("PredPreyGrass", env_creator)

    algo_cfg = PPOConfig().environment(env="PredPreyGrass", env_config=env_config).framework("torch")
    algo = algo_cfg.build()
    algo.restore(str(ckpt))

    helping_rates = []
    received_means = []
    type_counts = []  # per-episode policy group counts

    for ep in range(n_episodes):
        env = env_creator(env_config)
        obs, info = env.reset()
        done = {"__all__": False}
        steps = 0
        shares = 0
        recv_total = 0.0
        while not done["__all__"]:
            actions = {}
            for aid in env.agents:
                pol_id = f"type_{aid.split('_')[1]}_{aid.split('_')[2]}"
                policy = algo.get_policy(pol_id)
                # Extract obs for policy (Dict or Box)
                aobs = obs[aid]
                if isinstance(aobs, dict):
                    tobs = aobs["obs"]
                    extra = {"action_mask": aobs.get("action_mask")}
                else:
                    tobs = aobs
                    extra = {}
                act = policy.compute_single_action(tobs, explore=stochastic, state=None, prev_action=None, prev_reward=None, extra_input_dict=extra)[0]
                actions[aid] = int(act)
            obs, rew, term, trunc, inf = env.step(actions)
            steps += 1
            for ainfo in inf.values():
                if not ainfo:
                    continue
                shares += int(ainfo.get("shared", 0))
                recv_total += float(ainfo.get("received_share", 0.0))
            done = {"__all__": term.get("__all__", False) or trunc.get("__all__", False)}
        helping_rates.append(shares / max(steps, 1))
        received_means.append(recv_total / max(steps, 1))
        # type frequencies snapshot at end
        counts = {k: 0 for k in ("type_1_prey", "type_2_prey", "type_1_predator", "type_2_predator")}
        for aid in env.agents:
            group = "_".join(aid.split("_")[:3])
            if group in counts:
                counts[group] += 1
        type_counts.append(counts)

    # Aggregate
    hr = np.array(helping_rates)
    rc = np.array(received_means)
    hr_lo, hr_hi = bootstrap_ci(hr, n_boot=env_config.get("bootstrap", 200), conf=env_config.get("confidence", 0.95))
    rc_lo, rc_hi = bootstrap_ci(rc, n_boot=env_config.get("bootstrap", 200), conf=env_config.get("confidence", 0.95))

    mean_counts = {k: float(np.mean([c[k] for c in type_counts])) for k in type_counts[0].keys()}

    return {
        "helping_rate": float(hr.mean()),
        "helping_rate_lo": hr_lo,
        "helping_rate_hi": hr_hi,
        "received_share_mean": float(rc.mean()),
        "received_share_mean_lo": rc_lo,
        "received_share_mean_hi": rc_hi,
        **{f"count_{k}": v for k, v in mean_counts.items()},
    }


def main():
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser()
    env_overrides = load_env_config(args.env_config) if args.env_config else {}

    # Force deterministic eval unless stochastic requested
    env_overrides.setdefault("action_mask_enabled", True)
    env_overrides.setdefault("share_enabled", True)

    checkpoints = list_checkpoints(results_dir)
    if not checkpoints:
        print("No checkpoints found under", results_dir)
        return

    out_path = Path(args.out).expanduser()
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    header_written = out_path.exists()
    with open(out_path, "a") as f:
        if not header_written:
            f.write(
                "iteration,helping_rate,helping_rate_lo,helping_rate_hi,received_share_mean,received_share_mean_lo,received_share_mean_hi," \
                "count_type_1_prey,count_type_2_prey,count_type_1_predator,count_type_2_predator\n"
            )
        for ck in checkpoints:
            # Extract iteration from path name pattern: .../checkpoint_000050
            try:
                iter_num = int(str(ck.name).split("_")[-1])
            except Exception:
                iter_num = int(time.time())
            metrics = run_eval_on_checkpoint(ck, env_overrides, n_episodes=args.n_episodes, stochastic=args.stochastic)
            line = (
                f"{iter_num},{metrics['helping_rate']:.6f},{metrics['helping_rate_lo']:.6f},{metrics['helping_rate_hi']:.6f},"
                f"{metrics['received_share_mean']:.6f},{metrics['received_share_mean_lo']:.6f},{metrics['received_share_mean_hi']:.6f},"
                f"{metrics['count_type_1_prey']:.2f},{metrics['count_type_2_prey']:.2f},{metrics['count_type_1_predator']:.2f},{metrics['count_type_2_predator']:.2f}\n"
            )
            print(line.strip())
            f.write(line)

if __name__ == "__main__":
    main()
