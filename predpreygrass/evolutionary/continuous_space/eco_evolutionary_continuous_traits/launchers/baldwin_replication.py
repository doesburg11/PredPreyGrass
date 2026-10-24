"""Replication of the Baldwin-style branches (RESULTS.md section 4).

Every arm starts from CT-05 at iteration 1,500 (its policies and pooled gene
pool) and runs 1,500 iterations' worth of environment steps on 7 workers or
islands, with CT-05's environment settings. Seeds are new (the exploratory
BL-1..4 runs were unseeded RLlib runs and BL-2 used seed 0).

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.launchers.baldwin_replication ARM SEED [ITERATIONS]

ARM is one of
  both_learn      (BL-1)  RLlib, both policies trained
  pred_frozen     (BL-3)  RLlib, predator policy frozen
  prey_frozen     (BL-4)  RLlib, prey policy frozen
  both_frozen     (BL-2)  evolve_frozen.py, both policies frozen
  neutral_frozen  genetic-neutrality reference: evolve_frozen.py with trait_expression
                  off (base bodies, so demography differs from the other arms)
Output: runs/blrep_<ARM>_s<SEED>/ (or <SEED>_smoke with ITERATIONS given).
"""

import json
import subprocess
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
RUNS = MODULE / "runs"
CT05 = RUNS / "eco_evolutionary_continuous_traits_2026-10-16_20-53-30_161103"
START = CT05 / "checkpoint" / "iter_001500"
WORKERS = 7
RLLIB_ARMS = {"both_learn": [], "pred_frozen": ["predator_policy"], "prey_frozen": ["prey_policy"]}
FROZEN_ARMS = {"both_frozen": False, "neutral_frozen": True}


def main():
    arm, seed = sys.argv[1], int(sys.argv[2])
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    if arm not in RLLIB_ARMS and arm not in FROZEN_ARMS:
        raise SystemExit(f"unknown arm {arm!r}")
    name = f"blrep_{arm}_s{seed}" + ("" if len(sys.argv) <= 3 else "_smoke")
    out = RUNS / name
    if (out / "checkpoint").exists():
        raise SystemExit(f"{out} already has a checkpoint folder; refusing to overwrite")

    if arm in FROZEN_ARMS:
        command = [
            sys.executable, "-m", "predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.evolve_frozen",
            str(START), str(out), "--islands", str(WORKERS), "--iterations", str(iterations), "--seed", str(seed),
        ]
        if FROZEN_ARMS[arm]:
            command.append("--neutral")
        raise SystemExit(subprocess.call(command))

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits import train

    env = json.loads((CT05 / "source_code" / "settings.json").read_text())["config_env"]
    env["trait_initial_gene_pool"] = str(START / "gene_pools.json")
    train.config_env.clear()
    train.config_env.update(env)
    train.config_ppo.update(
        iterations=iterations,
        init_checkpoint=str(START),
        frozen_policies=RLLIB_ARMS[arm],
        num_env_runners=WORKERS,
        seed=seed,
        checkpoint_dir=str(out / "checkpoint"),
        tensorboard_dir=str(out / "tensorboard"),
    )
    train.main()


if __name__ == "__main__":
    main()
