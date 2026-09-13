"""Run Trial 13: an evolved prey reward genome inside flagship's PredPreyGrass
ecology, against a frozen PPO predator. No RLlib training loop for Trial 13 itself
(no ray.init()/PPOConfig/Tuner) -- flagship's env is reused directly as a plain
Python simulator, per this module's README.md ("RLlib or not").

Usage (staged validation -- see README.md for the full rationale):
    # Stage 0: smoke test, mechanics only
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.run_trial13_simulation \\
        --steps 5000 --seed 1 --log-every 200

    # Stage 1: single-seed pilot
    python -m ...run_trial13_simulation --steps 100000 --seed 1 --log-every 1000

    # resume an interrupted run (--steps is how many MORE steps to run, not a total):
    python -m ...run_trial13_simulation --resume-from .../checkpoints --steps 50000
"""

import argparse
import time
from pathlib import Path

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.config import config_env_flagship, config_erl_flagship, OBS_DIM
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.driver import Trial13Driver
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.metrics import (
    CsvLogger,
    lineage_fieldnames,
    lineage_record,
    truncate_csv_after_step,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.predator_policy import FrozenPredatorPolicy
from predpreygrass.global_config import ERL_RESULTS_DIR
from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass

FIELDNAMES = [
    "step",
    "prey_count",
    "predator_count",
    "eval_weight_absmean",
    "action_weight_absmean",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000, help="Default matches Stage 0's smoke-test budget; raise for later stages, see this module's docstring.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=200, help="Steps between metric log rows.")
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument(
        "--fixed-eval-weights", type=str, default=None,
        help="Comma-separated floats (length 8, see features.FEATURE_NAMES) to use as every "
             "founder's eval_weights instead of a random init.",
    )
    parser.add_argument(
        "--predator-checkpoint-dir", type=str, default=None,
        help="Override the frozen predator_policy checkpoint directory "
             "(default: config.py's DEFAULT_PREDATOR_CHECKPOINT_DIR).",
    )
    parser.add_argument(
        "--predator-deterministic", action="store_true",
        help="Argmax predator actions instead of sampling from its action distribution.",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume from a checkpoint: either a checkpoint_step_*.pkl file, or a "
             "checkpoints/ directory (uses the highest-step checkpoint in it). "
             "--steps then means steps to run THIS invocation, not a total across resumes. "
             "cfg/config_env/seed come from the checkpoint, not from --seed/--fixed-eval-weights.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=5000,
        help="Steps between periodic checkpoints (0 disables periodic saves; a final "
             "checkpoint is always written when the run ends, regardless of this value).",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Override checkpoint directory (default: out_dir/checkpoints).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.is_dir():
            found = latest_checkpoint(resume_path)
            if found is None:
                raise FileNotFoundError(f"No checkpoint_step_*.pkl found in {resume_path}")
            resume_path = found

    if resume_path is not None:
        if args.fixed_eval_weights is not None:
            raise ValueError("--fixed-eval-weights has no effect with --resume-from (no new founders are spawned).")
        payload = load_checkpoint(resume_path)
        cfg = payload["cfg"]
        config_env = payload["config_env"]
        rng = np.random.default_rng()
        rng.bit_generator.state = payload["rng_state"]

        env = PredPreyGrass(config_env)
        env.reset()
        env.restore_state_snapshot(payload["env_snapshot"])

        predator_policy = FrozenPredatorPolicy(
            cfg["predator_checkpoint_dir"], deterministic=cfg["predator_deterministic"], seed=cfg["seed"]
        )
        driver = Trial13Driver(env, predator_policy, cfg, rng)
        driver.registry = payload["registry"]
        driver.current_step = payload["current_step"]

        out_dir = Path(args.out_dir) if args.out_dir else resume_path.parent.parent
        print(f"Resumed from {resume_path} at step {driver.current_step}.")
    else:
        cfg = dict(config_erl_flagship)
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.fixed_eval_weights is not None:
            weights = [float(x) for x in args.fixed_eval_weights.split(",")]
            if len(weights) != OBS_DIM:
                raise ValueError(f"--fixed-eval-weights must have exactly {OBS_DIM} values, got {len(weights)}.")
            cfg["fixed_eval_weights"] = weights
        if args.predator_checkpoint_dir is not None:
            cfg["predator_checkpoint_dir"] = args.predator_checkpoint_dir
        if args.predator_deterministic:
            cfg["predator_deterministic"] = True

        config_env = dict(config_env_flagship)
        rng = np.random.default_rng(cfg["seed"])

        env = PredPreyGrass(config_env)
        predator_policy = FrozenPredatorPolicy(
            cfg["predator_checkpoint_dir"], deterministic=cfg["predator_deterministic"], seed=cfg["seed"]
        )
        driver = Trial13Driver(env, predator_policy, cfg, rng)
        driver.reset()

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path(args.out_dir) if args.out_dir else Path(ERL_RESULTS_DIR) / f"ERL_FLAGSHIP_{timestamp}"

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / "checkpoints"
    resuming = resume_path is not None

    if resuming:
        # Same reasoning as erl_baldwin's run_erl_simulation.py: logs are flushed
        # more often than checkpoints are saved, so a crash between the last
        # checkpoint and the next log flush can leave rows past driver.current_step
        # that resuming will re-simulate -- drop them first or they'd be duplicated.
        truncate_csv_after_step(out_dir / "progress.csv", "step", driver.current_step)
        truncate_csv_after_step(out_dir / "lineage_fitness.csv", "death_step", driver.current_step)

    logger = CsvLogger(out_dir / "progress.csv", FIELDNAMES, append=resuming)

    # Per-agent lineage log for the proximate-vs-ultimate-reward analysis
    # (analyze_proximate_reward.py). Deliberately records only real deaths, never
    # survivors -- see erl_baldwin's run_erl_simulation.py for why (a resumable
    # run has no way to know which survivors are "final").
    lineage_logger = CsvLogger(
        out_dir / "lineage_fitness.csv", lineage_fieldnames(OBS_DIM), append=resuming
    )
    driver.on_agent_death = lambda state, death_step: lineage_logger.log(
        lineage_record(state, death_step, censored=False)
    )

    start = time.time()
    last_checkpoint_step = driver.current_step
    extinction_step = None
    predator_extinction_step = None

    for _ in range(args.steps):
        driver.step()
        step = driver.current_step
        counts = driver.population_counts()

        # Extinction of the evolved prey population ends the run -- matching
        # Trial 12's "simulation ends after N steps or extinction" convention.
        if counts["prey"] == 0:
            extinction_step = step
            break

        # Predator extinction also ends the run: once the fixed threat is gone,
        # the rest of the budget just watches prey grow unchecked against no
        # selection pressure from predation -- not what this trial is testing,
        # and it wastes the remaining step budget on uninformative dynamics.
        if counts["predator"] == 0:
            predator_extinction_step = step
            break

        if step % args.log_every == 0:
            row = {"step": step, "prey_count": counts["prey"], "predator_count": counts["predator"]}
            row.update(driver.genome_stats())
            logger.log(row)
            logger.flush()
            lineage_logger.flush()

        if args.checkpoint_every > 0 and step - last_checkpoint_step >= args.checkpoint_every:
            logger.flush()
            lineage_logger.flush()
            save_checkpoint(driver, cfg, config_env, checkpoint_dir / f"checkpoint_step_{step}.pkl")
            last_checkpoint_step = step

    elapsed = time.time() - start
    logger.close()
    lineage_logger.close()

    final_checkpoint = checkpoint_dir / f"checkpoint_step_{driver.current_step}.pkl"
    save_checkpoint(driver, cfg, config_env, final_checkpoint)

    print(
        f"Finished at step {driver.current_step} in {elapsed:.1f}s "
        f"({driver.current_step / max(elapsed, 1e-9):.1f} steps/sec)."
    )
    if extinction_step is not None:
        print(f"Prey population extinction at step {extinction_step}.")
    elif predator_extinction_step is not None:
        print(f"Predator population extinction at step {predator_extinction_step} -- run stopped early.")
    else:
        print(f"Reached step limit ({args.steps}) without extinction.")
    print(f"Final population: {driver.population_counts()}")
    print(f"Log written to: {out_dir / 'progress.csv'}")
    print(f"Lineage log written to: {out_dir / 'lineage_fitness.csv'}")
    print(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
