"""One training run of arm L or B from scratch, deer learning alongside
(RESULTS.md section 3, DESIGN_3.md).

Run k of each arm uses seed k; a fresh deer policy is trained in every run,
in the touch world (config/config_env.py). Run from the repository root,
e.g. under systemd-run --user:

    python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/replicate_3.py \\
        NAME ARM K [ITERATIONS] [ENV_RUNNERS]

Output: runs/<NAME>/ (refuses to overwrite).
"""

import re
import sys

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import train
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import RUNS_DIR


def main():
    if not 4 <= len(sys.argv) <= 6 or not re.fullmatch(r"[A-Za-z0-9_-]+", sys.argv[1]) or sys.argv[2] not in train.ARMS:
        raise SystemExit("usage: replicate_3.py NAME ARM(L|B) K [ITERATIONS] [ENV_RUNNERS]")
    k = int(sys.argv[3])
    if k < 1:
        raise SystemExit("K starts at 1")
    run = RUNS_DIR / sys.argv[1]
    if run.exists():
        raise SystemExit(f"refusing: {run} exists")
    if train.config_env["deer_catch_rule"] != "touch":
        raise SystemExit("section 3 runs in the touch world (config/config_env.py)")
    train.config_ppo.update(
        arm=sys.argv[2],
        seed=k,
        deer_learn=True,
        deer_schedule=None,
        deer_checkpoint=None,
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
