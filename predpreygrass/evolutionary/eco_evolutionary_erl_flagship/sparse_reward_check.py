"""Does the literal, fitness-identical sparse reward ("+10 if reproduction occurs,
nothing else") beat a hand-shaped, evolved reward genome (`avoider`) once they
actually compete for the same food/space -- completing the algorithm x reward-density
2x2 this project has now measured three of four cells of:

    sparse reward   + PPO (gamma+GAE)        : WINS   (project's reward-density initiative)
    dense/shaped    + PPO                    : loses  (same initiative)
    dense/shaped    + this module's 1-step
                      local reinforcement    : WINS   (avoider beats anti_adaptive, positive_control.py)
    sparse reward   + this module's 1-step
                      local reinforcement    : ??? -- this script

The prediction (see README.md's status section): sparse loses here, not because it's
less TRUE (it IS the fitness signal itself, no proxy involved) but because this
module's reinforcement rule (`e_now - prev_eval`, a documented one-step simplification
of Ackley & Littman 1991 -- NOT full-trajectory REINFORCE, which handles sparse
reward fine via Monte Carlo returns) has no multi-step credit assignment at all. A
sparse signal gives it almost nothing to learn from between the rare reproduction
events. This mirrors an earlier, already-tested finding for the PREDATOR's own reward
design (sparse reproduction-only was tried there too and lost, for the same reason) --
this script is the first direct test for PREY.

Faithfulness note: `eval_weights` is a linear function of 8 observed features, and
"did I just reproduce" isn't one of them -- a genome literally cannot represent this
reward. The only faithful way to test it is to bypass eval_weights entirely and use
the environment's own reproduction signal directly as the reinforcement (see
driver.py's `PreyGenomeState.sparse_mode` / `_select_prey_action`), which is what this
script does -- not a proxy genome (e.g. "maximize energy_norm"), the actual thing.

Methodology: mirrors polymorphism_maintenance_check.py, not positive_control.py's
segregated comparison -- this investigation already found segregated, isolated-
population comparisons can be actively misleading (a strategy that looks strong alone
can lose decisively once it has to compete). Founders are seeded 50/50 directly from
`avoider` (genome-driven) and sparse-reproduction (genome-bypassing) prey, mutation
stays on for the avoider side, real competition for the same food/space, at the 20x
learning rate already confirmed to make reward-genome content matter for fitness.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.sparse_reward_check \\
        --steps 20000 --seeds 10 --checkpoint-every 2000
"""

import argparse
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.centralized_predator import CentralizedPredatorPolicy
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.config import (
    N_ACTIONS,
    PREDATOR_OBS_DIM,
    config_env_flagship,
    config_erl_flagship,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.driver import Trial13Driver
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.positive_control import GENOMES
from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass

MIN_LIVE_AGENTS = 10


def run_one(seed: int, steps: int, checkpoint_every: int, lr_multiplier: float) -> dict:
    try:
        cfg = dict(config_erl_flagship)
        cfg["seed"] = seed
        cfg["lr_positive"] = cfg["lr_positive"] * lr_multiplier
        cfg["lr_negative"] = cfg["lr_negative"] * lr_multiplier
        # Cluster A = avoider (genome-driven, eval_weights matters); cluster B's
        # eval_weights vector is irrelevant (sparse_reproduction_cluster="b" makes
        # _select_prey_action bypass it entirely) -- zeros here purely as a placeholder.
        cfg["mixed_founder_weights"] = (GENOMES["avoider"], [0.0] * len(GENOMES["avoider"]))
        cfg["sparse_reproduction_cluster"] = "b"

        config_env = dict(config_env_flagship)
        rng = np.random.default_rng(seed)
        env = PredPreyGrass(config_env)
        predator_policy = CentralizedPredatorPolicy(
            PREDATOR_OBS_DIM, N_ACTIONS, rng,
            init_std=cfg["predator_founder_weight_std"],
            lr_positive=cfg["predator_lr_positive"],
            lr_negative=cfg["predator_lr_negative"],
        )
        driver = Trial13Driver(env, predator_policy, cfg, rng)
        driver.reset()
        n_avoider_founders = sum(1 for s in driver.registry.values() if not s.sparse_mode)
        n_sparse_founders = sum(1 for s in driver.registry.values() if s.sparse_mode)

        deaths_avoider, deaths_sparse = 0, 0

        def on_death(state, death_step):
            nonlocal deaths_avoider, deaths_sparse
            if state.sparse_mode:
                deaths_sparse += 1
            else:
                deaths_avoider += 1

        driver.on_agent_death = on_death

        snapshots = []
        for step in range(1, steps + 1):
            driver.step()
            if driver.population_counts()["prey"] == 0:
                break
            if step % checkpoint_every == 0:
                registry = driver.registry
                if len(registry) < MIN_LIVE_AGENTS:
                    continue
                avoider_states = [s for s in registry.values() if not s.sparse_mode]
                sparse_states = [s for s in registry.values() if s.sparse_mode]
                snapshots.append({
                    "step": step,
                    "mean_generation": float(np.mean([s.generation for s in registry.values()])),
                    "n_avoider": len(avoider_states),
                    "n_sparse": len(sparse_states),
                    "offspring_avoider_living": sum(s.offspring_count for s in avoider_states),
                    "offspring_sparse_living": sum(s.offspring_count for s in sparse_states),
                })

        # Total offspring by condition = total individuals of that condition EVER present
        # (still-living + dead) minus that condition's own founder count -- NOT
        # living-offspring-count plus death-count, which a Codex review caught as wrong
        # (a death means one individual of that condition died, not that it reproduced
        # once; conflating the two both double-counts children who later died and
        # undercounts parents who died after producing several). sparse_mode is inherited
        # exactly, so every individual's condition is well-defined throughout.
        final_avoider = [s for s in driver.registry.values() if not s.sparse_mode]
        final_sparse = [s for s in driver.registry.values() if s.sparse_mode]
        total_offspring_avoider = len(final_avoider) + deaths_avoider - n_avoider_founders
        total_offspring_sparse = len(final_sparse) + deaths_sparse - n_sparse_founders

        return {
            "seed": seed, "ok": True, "snapshots": snapshots,
            "final_step": driver.current_step,
            "final_n_avoider": len(final_avoider), "final_n_sparse": len(final_sparse),
            "n_avoider_founders": n_avoider_founders, "n_sparse_founders": n_sparse_founders,
            "deaths_avoider": deaths_avoider, "deaths_sparse": deaths_sparse,
            "total_offspring_avoider": total_offspring_avoider,
            "total_offspring_sparse": total_offspring_sparse,
        }
    except Exception:
        return {"seed": seed, "ok": False, "stderr_tail": traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--lr-multiplier", type=float, default=20.0)
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

    print(f"Launching {args.seeds} seeds (50/50 avoider/sparse-reproduction founders), "
          f"{args.steps} steps each, lr_multiplier={args.lr_multiplier}, "
          f"checkpoint every {args.checkpoint_every} steps, {workers} parallel workers.")

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

    print(f"{'step':<10}{'mean_gen':<12}{'mean_share_avoider':<20}{'share_range'}")
    all_steps = sorted({s["step"] for r in ok_results for s in r["snapshots"]})
    for step in all_steps:
        rows_at_step = [s for r in ok_results for s in r["snapshots"] if s["step"] == step]
        if not rows_at_step:
            continue
        shares = [s["n_avoider"] / (s["n_avoider"] + s["n_sparse"]) for s in rows_at_step]
        mean_gen = np.mean([s["mean_generation"] for s in rows_at_step])
        print(f"{step:<10}{mean_gen:<12.1f}{np.mean(shares):<20.2f}[{np.min(shares):.2f}, {np.max(shares):.2f}]")

    n = len(ok_results)
    if n:
        final_shares = [r["final_n_avoider"] / max(r["final_n_avoider"] + r["final_n_sparse"], 1) for r in ok_results]
        total_offspring_avoider = [r["total_offspring_avoider"] for r in ok_results]
        total_offspring_sparse = [r["total_offspring_sparse"] for r in ok_results]
        print(f"\nFinal (n={n} seeds):")
        print(f"  mean final population share (avoider): {np.mean(final_shares):.2f} "
              f"[{np.min(final_shares):.2f}, {np.max(final_shares):.2f}]")
        print(f"  mean total offspring -- avoider: {np.mean(total_offspring_avoider):.0f}, "
              f"sparse-reproduction: {np.mean(total_offspring_sparse):.0f}")
        n_avoider_wins = sum(1 for s in final_shares if s > 0.9)
        n_sparse_wins = sum(1 for s in final_shares if s < 0.1)
        print(f"  seeds resolved to >90% avoider: {n_avoider_wins}/{n}; "
              f"seeds resolved to >90% sparse-reproduction: {n_sparse_wins}/{n}")

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\n{len(failed)} run(s) failed. First failure's stderr tail:")
        print(failed[0]["stderr_tail"])


if __name__ == "__main__":
    main()
