import time
import argparse
import numpy as np

from predpreygrass.rllib.v3_1.predpreygrass_rllib_env import PredPreyGrass as PredPreyGrassOriginal
from predpreygrass.rllib.jax.predpreygrass_rllib_env import PredPreyGrass as PredPreyGrassJax


def run_benchmark(env_cls, config, steps, warmup=100, label="env"):
    env = env_cls(config)
    obs, _ = env.reset(seed=config.get("seed", 123))

    # Warmup
    for _ in range(warmup):
        actions = {a: env.action_spaces[a].sample() for a in list(env.agents)}
        env.step(actions)

    start = time.perf_counter()
    step_count = 0
    while step_count < steps:
        if not env.agents:  # all agents dead; reset to keep step loop running
            env.reset(seed=config.get("seed", 123))
        actions = {a: env.action_spaces[a].sample() for a in list(env.agents)}
        env.step(actions)
        step_count += 1
    duration = time.perf_counter() - start
    sps = steps / duration if duration > 0 else float('inf')
    return {"label": label, "steps": steps, "seconds": duration, "steps_per_sec": sps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000, help="Number of timed steps (after warmup)")
    parser.add_argument("--warmup", type=int, default=250, help="Number of warmup steps")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    base_config = {
        "debug_mode": False,
        "max_steps": 10_000_000,
        "seed": args.seed,
        # Keep defaults for population sizes etc.
    }

    results = []

    # 1. Original environment (non-JAX file)
    results.append(run_benchmark(PredPreyGrassOriginal, base_config, args.steps, args.warmup, label="original_v3_1"))

    # 2. JAX file with JAX disabled (measures overhead of refactor)
    cfg_no_jax = dict(base_config)
    cfg_no_jax["use_jax_core"] = False
    results.append(run_benchmark(PredPreyGrassJax, cfg_no_jax, args.steps, args.warmup, label="jax_file_no_jax"))

    # 3. JAX file with JAX enabled
    cfg_jax = dict(base_config)
    cfg_jax["use_jax_core"] = True
    try:
        results.append(run_benchmark(PredPreyGrassJax, cfg_jax, args.steps, args.warmup, label="jax_enabled"))
    except Exception as e:
        results.append({"label": "jax_enabled", "error": str(e), "steps": 0, "seconds": 0, "steps_per_sec": 0})

    # Reporting
    print("\nEnvironment Speed Benchmark (steps_per_second)\n" + "-" * 55)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<18} ERROR: {r['error']}")
        else:
            print(f"{r['label']:<18} {r['steps_per_sec']:>10.1f} sps  (steps={r['steps']}, seconds={r['seconds']:.3f})")

    # Relative improvements
    base = next((r for r in results if r['label'] == 'original_v3_1'), None)
    jax = next((r for r in results if r['label'] == 'jax_enabled'), None)
    if base and jax and 'error' not in base and 'error' not in jax and base['steps_per_sec'] > 0:
        improvement = (jax['steps_per_sec'] / base['steps_per_sec'] - 1.0) * 100.0
        print(f"\nRelative improvement (jax_enabled vs original_v3_1): {improvement:.2f}%")

    no_jax = next((r for r in results if r['label'] == 'jax_file_no_jax'), None)
    if base and no_jax and 'error' not in base and 'error' not in no_jax and base['steps_per_sec'] > 0:
        overhead = (no_jax['steps_per_sec'] / base['steps_per_sec'] - 1.0) * 100.0
        print(f"Overhead of jax file (no_jax vs original): {overhead:.2f}%")


if __name__ == "__main__":
    main()
