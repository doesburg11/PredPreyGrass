"""Does a genuinely LARGER effective population size -- not more founders alone,
but more carrying capacity (grid area + grass scaled together, so density and
every other parameter stay identical) -- weaken genetic drift enough to let
evolution more reliably find something like `avoider`'s extreme region?

Motivation: classical population genetics says drift's strength scales
roughly as 1/N_e. `centralized_prey_emergence_check.py` (the calibrated,
pooled, genome-conditioned learner) found real but RARE (10%, n=30-confirmed)
sustained large-genome-divergence events, with a pattern (a small, consistent
minority succeeding while the majority don't, under identical setup) that
looks more like founder-effect/stochastic luck than a systematic learning-
capacity ceiling. If that diagnosis is right, a larger population -- which
directly weakens drift's relative strength without touching the learner at
all -- should raise the sustained-divergence rate. If it doesn't, the
bottleneck is elsewhere (the fitness landscape's own shape, or something
else), not drift-vs-selection balance as such.

Scaling: `SCALE` multiplies grid AREA (grid_size scales by sqrt(SCALE)) and
n_initial_active_prey/predator/initial_num_grass all by SCALE together, so
grass density, prey density, and predator density are UNCHANGED from the
baseline config -- this raises carrying capacity (confirmed directly: at
SCALE=4, population reaches ~300 within 200 steps, vs. ~70-90 at baseline),
not just the founder count, which would otherwise just decay back to the
same grass-limited plateau. Everything else (mutation rate/std, lr_multiplier,
the calibrated pooled prey policy) is held identical to
centralized_prey_emergence_check.py for a clean, single-variable comparison.

Throughput at SCALE=4 is much worse than linear (~7 steps/sec vs. baseline's
~40-65, confirmed by direct measurement) -- likely O(N^2)-ish costs in
agent-agent interactions, not something to fix here, just budget around
(shorter step count than the baseline emergence check's 20,000).

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.population_scale_emergence_check \\
        --steps 10000 --seeds 10 --checkpoint-every 2000 --scale 4
"""

import argparse
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.centralized_predator import CentralizedPredatorPolicy
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.centralized_prey import CentralizedPreyPolicy
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.config import (
    N_ACTIONS,
    OBS_DIM,
    PREDATOR_OBS_DIM,
    config_env_flagship,
    config_erl_flagship,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.driver import Trial13Driver
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import FEATURE_NAMES
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.polymorphism_check import (
    DELTA_BIC_THRESHOLD,
    MIN_LIVE_AGENTS,
    bic_bimodality,
)
from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass

PRED_PROX_IDX = FEATURE_NAMES.index("predator_proximity")


def scaled_config_env(scale: float) -> dict:
    config_env = dict(config_env_flagship)
    base_grid = config_env_flagship["grid_size"]
    config_env["grid_size"] = round(base_grid * math.sqrt(scale))
    config_env["n_initial_active_prey"] = round(config_env_flagship["n_initial_active_prey"] * scale)
    config_env["n_initial_active_predator"] = round(config_env_flagship["n_initial_active_predator"] * scale)
    config_env["initial_num_grass"] = round(config_env_flagship["initial_num_grass"] * scale)
    return config_env


def run_one(seed: int, steps: int, checkpoint_every: int, lr_multiplier: float, scale: float) -> dict:
    try:
        cfg = dict(config_erl_flagship)
        cfg["seed"] = seed
        cfg["lr_positive"] = cfg["lr_positive"] * lr_multiplier
        cfg["lr_negative"] = cfg["lr_negative"] * lr_multiplier

        config_env = scaled_config_env(scale)
        rng = np.random.default_rng(seed)
        env = PredPreyGrass(config_env)
        predator_policy = CentralizedPredatorPolicy(
            PREDATOR_OBS_DIM, N_ACTIONS, rng,
            init_std=cfg["predator_founder_weight_std"],
            lr_positive=cfg["predator_lr_positive"],
            lr_negative=cfg["predator_lr_negative"],
        )
        prey_policy = CentralizedPreyPolicy(
            OBS_DIM, OBS_DIM, N_ACTIONS, rng,
            init_std=cfg["founder_weight_std"],
            lr_positive=cfg["lr_positive"],
            lr_negative=cfg["lr_negative"],
        )
        driver = Trial13Driver(env, predator_policy, cfg, rng, prey_policy=prey_policy)
        driver.reset()

        rng_null = np.random.default_rng(seed)
        snapshots = []
        for step in range(1, steps + 1):
            driver.step()
            if driver.population_counts()["prey"] == 0:
                break
            if step % checkpoint_every == 0:
                registry = driver.registry
                if len(registry) < MIN_LIVE_AGENTS:
                    continue
                weights = np.array([s.genome.eval_weights[PRED_PROX_IDX] for s in registry.values()])
                bic_result = bic_bimodality(weights, seed, rng_null)
                bic_result["step"] = step
                bic_result["mean_generation"] = float(np.mean([s.generation for s in registry.values()]))
                bic_result["n_live"] = len(registry)
                snapshots.append(bic_result)

        sustained = False
        if len(snapshots) >= 3:
            sustained = all(c["delta_bic"] > DELTA_BIC_THRESHOLD for c in snapshots[-3:])

        return {"seed": seed, "ok": True, "snapshots": snapshots, "sustained_bimodal_last3": sustained}
    except Exception:
        return {"seed": seed, "ok": False, "stderr_tail": traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--lr-multiplier", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=4.0, help="Population/grass/grid-area multiplier.")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.seeds < 1 or args.steps < 1 or args.checkpoint_every < 1:
        raise ValueError("--seeds, --steps, and --checkpoint-every must all be >= 1.")
    if not (math.isfinite(args.lr_multiplier) and args.lr_multiplier >= 0):
        raise ValueError(f"--lr-multiplier must be finite and >= 0, got {args.lr_multiplier}.")
    if not (math.isfinite(args.scale) and args.scale >= 1.0):
        # A Codex review caught that scale < 1 (or very small values) could round one or
        # more founder counts to zero, silently returning a degenerate "successful" run --
        # this script is specifically about testing LARGER populations, so reject shrinking.
        raise ValueError(f"--scale must be finite and >= 1.0 (this experiment scales UP), got {args.scale}.")
    if args.workers is not None and args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}.")

    import os
    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)

    scaled = scaled_config_env(args.scale)
    print(f"Launching {args.seeds} seeds, {args.steps} steps each, scale={args.scale}x "
          f"(grid_size={scaled['grid_size']}, n_prey={scaled['n_initial_active_prey']}, "
          f"n_predator={scaled['n_initial_active_predator']}, n_grass={scaled['initial_num_grass']}), "
          f"lr_multiplier={args.lr_multiplier}, checkpoint every {args.checkpoint_every} steps, "
          f"{workers} parallel workers.")

    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, seed, args.steps, args.checkpoint_every, args.lr_multiplier, args.scale): seed
            for seed in range(1, args.seeds + 1)
        }
        done = 0
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                row = fut.result()
            except Exception:
                row = {"seed": seed, "ok": False, "stderr_tail": traceback.format_exc()}
            results.append(row)
            done += 1
            print(f"[{done}/{args.seeds}] seed={seed}: {'OK' if row['ok'] else 'FAILED'}", flush=True)

    elapsed = time.time() - start
    print(f"\nAll runs finished in {elapsed:.0f}s.\n")

    ok_results = [r for r in results if r["ok"]]
    all_steps = sorted({c["step"] for r in ok_results for c in r["snapshots"]})

    print(f"{'step':<10}{'mean_gen':<12}{'mean_n_live':<14}{'n_eligible':<12}{'frac_bimodal':<14}"
          f"{'null_fp_rate':<14}{'mean_delta_bic'}")
    for step in all_steps:
        rows_at_step = [c for r in ok_results for c in r["snapshots"] if c["step"] == step]
        if not rows_at_step:
            continue
        n = len(rows_at_step)
        bimodal = [c for c in rows_at_step if c["delta_bic"] > DELTA_BIC_THRESHOLD]
        null_bimodal = [c for c in rows_at_step if c["null_delta_bic"] > DELTA_BIC_THRESHOLD]
        frac_bimodal = len(bimodal) / n
        null_fp_rate = len(null_bimodal) / n
        mean_dbic = sum(c["delta_bic"] for c in rows_at_step) / n
        mean_gen = sum(c["mean_generation"] for c in rows_at_step) / n
        mean_n_live = sum(c["n_live"] for c in rows_at_step) / n
        print(f"{step:<10}{mean_gen:<12.1f}{mean_n_live:<14.0f}{n:<12}{frac_bimodal:<14.2f}"
              f"{null_fp_rate:<14.2f}{mean_dbic:.1f}")

    n_sustained = sum(1 for r in ok_results if r.get("sustained_bimodal_last3"))
    print(f"\nSustained polymorphism (delta_BIC > {DELTA_BIC_THRESHOLD:.0f} at ALL of a seed's last 3 "
          f"checkpoints): {n_sustained}/{len(ok_results)} seeds.")

    print("\nPer-seed mode-gap trajectories (gap = mode_means[1] - mode_means[0], only where bimodal):")
    for r in ok_results:
        gaps = [(c["step"], c["mean_generation"], c["mode_means"][1] - c["mode_means"][0])
                for c in r["snapshots"] if c["delta_bic"] > DELTA_BIC_THRESHOLD]
        if not gaps:
            print(f"  seed {r['seed']}: never bimodal")
            continue
        gap_str = "; ".join(f"gen{g:.0f}:{gap:+.2f}" for _, g, gap in gaps)
        print(f"  seed {r['seed']}: {gap_str}")

    print(
        "\nCompare against centralized_prey_emergence_check.py's baseline (scale=1x, n=30): 3/30 (10.0%) "
        "seeds showed SUSTAINED large separation (>=0.5 held across >=2 consecutive checkpoints). If a "
        "larger effective population raises that rate, drift-vs-selection balance (not a learning-capacity "
        "ceiling) was the dominant remaining constraint; if it doesn't, the bottleneck is elsewhere."
    )

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\n{len(failed)} run(s) failed. First failure's stderr tail:")
        print(failed[0]["stderr_tail"])


if __name__ == "__main__":
    main()
