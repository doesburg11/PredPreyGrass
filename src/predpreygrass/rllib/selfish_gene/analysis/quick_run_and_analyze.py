import json
from pathlib import Path

from predpreygrass.rllib.selfish_gene.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.selfish_gene.config.config_env_selfish_gene import config_env as base_config
from predpreygrass.rllib.selfish_gene.analysis.coop_metrics import (
    compute_assortment_index,
    compute_assortment_index_by_policy,
    compute_kin_proximity_advantage,
    compute_kin_proximity_advantage_by_policy,
)


def run_episodes(n_episodes=10, max_steps=None, log_dir=None, seed=0, config_overrides=None):
    cfg = dict(base_config)
    cfg["enable_coop_logging"] = True
    if max_steps is not None:
        cfg["max_steps"] = int(max_steps)
    if log_dir is not None:
        cfg["coop_log_dir"] = str(log_dir)
    if config_overrides:
        cfg.update(config_overrides)

    env = PredPreyGrass(cfg)
    # Create a deterministic sequence of episode seeds
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(seed + ep))
        done = False
        while not done:
            actions = {a: 0 for a in env.agents}
            obs, rew, term, trunc, info = env.step(actions)
            done = term.get("__all__", False) or trunc.get("__all__", False)
    env.close()


def analyze(log_dir, los_aware=False, bootstrap=0, seed=0, by_policy=True):
    from predpreygrass.rllib.selfish_gene.analysis.coop_metrics import load_episode

    p = Path(log_dir)
    files = sorted(p.glob("episode_*.json"))
    episodes = [load_episode(fp) for fp in files]
    ai = compute_assortment_index(episodes, los_aware=los_aware, n_bootstrap=bootstrap, seed=seed)
    kpa = compute_kin_proximity_advantage(episodes, los_aware=los_aware, n_bootstrap=bootstrap, seed=seed)
    out = {"ai": ai, "kpa": kpa}
    if by_policy:
        out["ai_by_policy"] = compute_assortment_index_by_policy(episodes, los_aware=los_aware, n_bootstrap=bootstrap, seed=seed)
        out["kpa_by_policy"] = compute_kin_proximity_advantage_by_policy(episodes, los_aware=los_aware, n_bootstrap=bootstrap, seed=seed)
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quick harness to run episodes with coop logging and analyze metrics.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode (override config)")
    parser.add_argument("--log-dir", default=None, help="Directory for episode logs (default: from config)")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for episodes and analysis")
    parser.add_argument("--los-aware", action="store_true", help="Use LOS for neighbor checks in analysis")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap iterations for CI")
    parser.add_argument("--by-policy", action="store_true", help="Include per-policy breakdowns")
    parser.add_argument("--config-override", action="append", default=[], help="Override env config as key=value (can be repeated)")

    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    for kv in args.config_override:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        # Try to parse numbers, booleans
        v_strip = v.strip()
        if v_strip.lower() in ("true", "false"):
            val = v_strip.lower() == "true"
        else:
            try:
                if "." in v_strip:
                    val = float(v_strip)
                else:
                    val = int(v_strip)
            except ValueError:
                val = v_strip
        overrides[k.strip()] = val

    log_dir = args.log_dir or base_config.get("coop_log_dir", "output/coop_logs")

    print(f"[RUN] episodes={args.episodes} max_steps={args.max_steps} log_dir={log_dir} overrides={overrides}")
    run_episodes(n_episodes=args.episodes, max_steps=args.max_steps, log_dir=log_dir, seed=args.seed, config_overrides=overrides)

    print("[ANALYZE] computing metrics...")
    out = analyze(log_dir=log_dir, los_aware=args.los_aware, bootstrap=args.bootstrap, seed=args.seed, by_policy=args.by_policy)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
