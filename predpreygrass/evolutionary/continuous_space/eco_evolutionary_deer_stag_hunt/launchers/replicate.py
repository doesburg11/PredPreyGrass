"""One training run of arm L or B from scratch (RESULTS.md section 1).

Run k of every arm uses seed k and the deer of deer_stag_hunt run
((k - 1) mod 3) + 1 (DESIGN.md, deer). Run from the repository root, e.g.
under systemd-run --user:

    python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/replicate.py \\
        NAME ARM K [ITERATIONS] [ENV_RUNNERS]

Output: runs/<NAME>/ (refuses to overwrite). The E arm is evolve.py, matched
to the L and B runs (launchers/run_main.sh).
"""

import re
import sys

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import train
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import DEER_RUNS_DIR, RUNS_DIR


def deer_checkpoint(k):
    """deer_stag_hunt's final checkpoint whose deer run k uses."""
    return DEER_RUNS_DIR / f"run_{(k - 1) % 3 + 1}" / "checkpoint"


def main():
    if not 4 <= len(sys.argv) <= 6 or not re.fullmatch(r"[A-Za-z0-9_-]+", sys.argv[1]) or sys.argv[2] not in train.ARMS:
        raise SystemExit("usage: replicate.py NAME ARM(L|B) K [ITERATIONS] [ENV_RUNNERS]")
    k = int(sys.argv[3])
    if k < 1:
        raise SystemExit("K starts at 1")
    run = RUNS_DIR / sys.argv[1]
    if run.exists():
        raise SystemExit(f"refusing: {run} exists")
    train.config_ppo.update(
        arm=sys.argv[2],
        seed=k,
        deer_checkpoint=str(deer_checkpoint(k)),
        tensorboard_dir=str(run / "tensorboard"),
        checkpoint_dir=str(run / "checkpoint"),
    )
    if len(sys.argv) > 4:
        train.config_ppo["iterations"] = int(sys.argv[4])
    if len(sys.argv) > 5:
        train.config_ppo["num_env_runners"] = int(sys.argv[5])
    train.main()


if __name__ == "__main__":
    main()
