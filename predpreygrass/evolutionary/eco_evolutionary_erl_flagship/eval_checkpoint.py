"""Load a Trial 13 checkpoint (see checkpoint.py) and inspect it standalone -- no
training loop, no re-running from scratch. For sanity-checking an evolved prey
population's genomes before committing to a longer run, and for a quick, real read
on the proximate-vs-ultimate-reward question (see analyze_proximate_reward.py) for
the checkpoint's current population, without needing a completed lineage_fitness.csv.

No rendering support -- flagship's own PyGame renderer
(non_evolutionary/base_environment/utils/pygame_grid_renderer_rllib.py) is a
separate, heavier interactive system out of scope for this lightweight module; see
this module's README.md.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.eval_checkpoint \\
        ~/simulation_results/erl_results/ERL_FLAGSHIP_.../checkpoints/checkpoint_step_5000.pkl \\
        --steps 1000
"""

import argparse
import time
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.checkpoint import load_checkpoint
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.driver import Trial13Driver
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import FEATURE_NAMES
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.metrics import CsvLogger, lineage_fieldnames, lineage_record
from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass


def _print_summary(driver: Trial13Driver, label: str):
    counts = driver.population_counts()
    stats = driver.genome_stats()
    print(f"--- {label} (step {driver.current_step}) ---")
    print(f"Population: {counts}")
    print(
        f"eval_weight_absmean={stats['eval_weight_absmean']:.3f}  "
        f"action_weight_absmean={stats['action_weight_absmean']:.3f}  "
        f"predator_action_weight_absmean={stats['predator_action_weight_absmean']:.3f}"
    )
    living = list(driver.registry.values())
    if living:
        offspring = np.array([s.offspring_count for s in living])
        print(f"offspring_count over {len(living)} living prey: mean={offspring.mean():.2f}, max={offspring.max()}")
        top = sorted(living, key=lambda s: -s.offspring_count)[:5]
        print("Top 5 by offspring_count (eval_weights -- the evolved reward each carries):")
        for state in top:
            weights = ", ".join(f"{FEATURE_NAMES[i]}={w:.2f}" for i, w in enumerate(state.genome.eval_weights))
            print(f"  agent {state.agent_id}: offspring={state.offspring_count}, gen={state.generation}, [{weights}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Path to a checkpoint_step_*.pkl file.")
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Steps to run forward from the checkpoint for the eval pass "
             "(0 = just inspect the saved state as-is, no further evolution or learning).",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory for the eval lineage CSV.")
    parser.add_argument(
        "--log-lineage", action="store_true",
        help="Also write an eval_lineage.csv for this eval window (deaths during --steps only).",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    payload = load_checkpoint(checkpoint_path)
    cfg = payload["cfg"]
    config_env = payload["config_env"]
    rng = np.random.default_rng()
    rng.bit_generator.state = payload["rng_state"]

    env = PredPreyGrass(config_env)
    env.reset()
    env.restore_state_snapshot(payload["env_snapshot"])

    predator_policy = payload["predator_policy"]  # LEARNED weights, not reconstructed fresh
    driver = Trial13Driver(env, predator_policy, cfg, rng)
    driver.registry = payload["registry"]
    driver.predator_registry = payload["predator_registry"]
    driver.current_step = payload["current_step"]

    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parent.parent / "eval" / checkpoint_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_summary(driver, "Loaded checkpoint")

    lineage_logger = None
    if args.log_lineage:
        lineage_logger = CsvLogger(out_dir / "eval_lineage.csv", lineage_fieldnames(len(FEATURE_NAMES)))
        driver.on_agent_death = lambda state, death_step: lineage_logger.log(
            lineage_record(state, death_step, censored=False)
        )

    start = time.time()
    for _ in range(args.steps):
        driver.step()
        # Only prey extinction stops an eval pass -- see
        # run_trial13_simulation.py's main loop for why predator extinction no
        # longer does (generational depth, not early stopping, is the goal).
        if driver.population_counts()["prey"] == 0:
            break
    elapsed = time.time() - start

    if lineage_logger is not None:
        for state in driver.registry.values():
            lineage_logger.log(lineage_record(state, driver.current_step, censored=True))
        lineage_logger.close()

    if args.steps > 0:
        print(f"\nRan {args.steps} eval steps in {elapsed:.1f}s.")
        _print_summary(driver, "After eval steps")
    if lineage_logger is not None:
        print(f"Eval lineage log written to: {out_dir / 'eval_lineage.csv'}")


if __name__ == "__main__":
    main()
