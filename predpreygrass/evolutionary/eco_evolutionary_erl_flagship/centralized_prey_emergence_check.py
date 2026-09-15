"""Does POOLING experience across the whole live prey population -- rather than
each individual learning alone within its own short lifetime -- let evolution
discover something like `avoider`'s extreme, fitness-consequential region of
`eval_weights` space, where the default per-individual-learner architecture did
not (polymorphism_check.py: no growing structure over ~260 generations, even at
the 20x learning rate that makes reward-to-behavior coupling strong)?

Motivated directly by the existing Wang2019 replication's own diagnosed root
cause for its null result (each genotype trial got only ~20 episodes of RL
training before being scored -- too few to separate real genotype quality from
noise). See centralized_prey.py's docstring for the full reasoning and
centralized_predator.py for the pooling pattern this reuses.

Same emergence-test methodology as polymorphism_check.py (real evolution,
mutation on, standard neutral random founder `eval_weights` init, no fixed/
mixed founders), for direct comparability -- only the prey policy architecture
differs: `CentralizedPreyPolicy` (one shared, genome-conditioned action
network, pooling every living prey's experience) instead of each prey's own
`action_weights`/`action_bias`.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.centralized_prey_emergence_check \\
        --steps 20000 --seeds 10 --checkpoint-every 2000
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


def run_one(seed: int, steps: int, checkpoint_every: int, lr_multiplier: float) -> dict:
    try:
        cfg = dict(config_erl_flagship)
        cfg["seed"] = seed
        cfg["lr_positive"] = cfg["lr_positive"] * lr_multiplier
        cfg["lr_negative"] = cfg["lr_negative"] * lr_multiplier
        # No fixed/mixed founders -- standard neutral random init, matching
        # polymorphism_check.py exactly, for direct comparability.

        config_env = dict(config_env_flagship)
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
                snapshots.append(bic_result)

        sustained = False
        if len(snapshots) >= 3:
            sustained = all(c["delta_bic"] > DELTA_BIC_THRESHOLD for c in snapshots[-3:])

        return {"seed": seed, "ok": True, "snapshots": snapshots, "sustained_bimodal_last3": sustained}
    except Exception:
        return {"seed": seed, "ok": False, "stderr_tail": traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument(
        "--lr-multiplier", type=float, default=0.05,
        help="Default is the CALIBRATED value for the pooled/genome-conditioned architecture "
             "(see centralized_prey.py's module docstring, point 3) -- NOT the 20x that was "
             "calibrated for the default per-individual architecture. Every currently-living "
             "prey's update now lands on the SAME shared weights each step (even batched, "
             "the effective step size scales with how many examples are averaged in), so the "
             "per-individual-optimal learning rate is far too high here. Validated by a direct "
             "sweep: 0.02-0.05 gives robust, healthy populations (5/5 seeds, ~70-90 prey, "
             "matching the default architecture's own plateau); 0.1 already shows real fragility "
             "(1/5 seeds went extinct); 1.0+ collapses reliably.",
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.seeds < 1 or args.steps < 1 or args.checkpoint_every < 1:
        raise ValueError("--seeds, --steps, and --checkpoint-every must all be >= 1.")
    if not (math.isfinite(args.lr_multiplier) and args.lr_multiplier >= 0):
        raise ValueError(f"--lr-multiplier must be finite and >= 0, got {args.lr_multiplier}.")
    if args.workers is not None and args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}.")

    import os
    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)

    print(f"Launching {args.seeds} seeds, {args.steps} steps each, lr_multiplier={args.lr_multiplier}, "
          f"centralized/genome-conditioned prey policy, checkpoint every {args.checkpoint_every} steps, "
          f"{workers} parallel workers.")

    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, seed, args.steps, args.checkpoint_every, args.lr_multiplier): seed
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
            print(f"[{done}/{args.seeds}] seed={seed}: {'OK' if row['ok'] else 'FAILED'}")

    elapsed = time.time() - start
    print(f"\nAll runs finished in {elapsed:.0f}s.\n")

    ok_results = [r for r in results if r["ok"]]
    all_steps = sorted({c["step"] for r in ok_results for c in r["snapshots"]})

    print(f"{'step':<10}{'mean_gen':<12}{'n_eligible':<12}{'frac_bimodal':<14}{'null_fp_rate':<14}"
          f"{'mean_delta_bic':<16}{'mode_means (when bimodal)'}")
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
        mode_str = ""
        if bimodal:
            mode_pairs = [f"({m[0]:+.2f}, {m[1]:+.2f})" for m in (c["mode_means"] for c in bimodal)]
            mode_str = "; ".join(mode_pairs[:5]) + (" ..." if len(mode_pairs) > 5 else "")
        print(f"{step:<10}{mean_gen:<12.1f}{n:<12}{frac_bimodal:<14.2f}{null_fp_rate:<14.2f}"
              f"{mean_dbic:<16.1f}{mode_str}")

    n_sustained = sum(1 for r in ok_results if r.get("sustained_bimodal_last3"))
    print(f"\nSustained polymorphism (delta_BIC > {DELTA_BIC_THRESHOLD:.0f} at ALL of a seed's last 3 "
          f"checkpoints): {n_sustained}/{len(ok_results)} seeds.")

    # Per-seed trajectories, not just the aggregate table's first-5-bimodal-seeds-per-
    # checkpoint view (which doesn't track seed identity across checkpoints and so can't
    # show whether a given seed's gap is actually GROWING over time, the decisive
    # question -- see polymorphism_check.py's own gap-growth check for why this matters:
    # an earlier result there looked exciting from the aggregate table alone and turned
    # out to be flat/non-growing once checked per-seed).
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
        "\nCompare directly against polymorphism_check.py's per-individual-learner result: there, mode "
        "gaps stayed ~0.1-0.2 and did not grow over ~260 generations, far short of avoider's -2.0. If "
        "pooling lets evolution reach substantially larger, growing mode separations here, that "
        "confirms training-volume-per-individual (not the REINFORCE-vs-PPO distinction) was the real "
        "bottleneck; if it doesn't, the bottleneck is elsewhere (e.g. the fitness landscape itself, not "
        "the learner)."
    )

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\n{len(failed)} run(s) failed. First failure's stderr tail:")
        print(failed[0]["stderr_tail"])


if __name__ == "__main__":
    main()
