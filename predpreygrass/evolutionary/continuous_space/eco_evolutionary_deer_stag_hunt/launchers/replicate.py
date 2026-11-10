"""One training run of arm L or B from scratch (RESULTS.md section 2).

Run k of every arm uses seed k and the deer of deer_stag_hunt touch_((k - 1)
mod 3) + 1, replayed in training order (deer_schedule.py), in the touch
world (config/config_env.py). Run from the repository root, e.g. under
systemd-run --user:

    python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/replicate.py \\
        NAME ARM K [ITERATIONS] [ENV_RUNNERS]

Output: runs/<NAME>/ (refuses to overwrite). The E arm is evolve.py, matched
to the L and B runs of the same deer (launchers/run_pilot_2.sh,
launchers/run_main_2.sh). Section 1's runs (one frozen final deer of
deer_stag_hunt's party-rule run_1-3) were started by an earlier version of
this script; their settings are in each run's source_code/.
"""

import re
import sys

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import train
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import DEER_RUNS_DIR, RUNS_DIR


def deer_source(k):
    """The deer_stag_hunt run whose deer run k replays."""
    return DEER_RUNS_DIR / f"touch_{(k - 1) % 3 + 1}"


def main():
    if not 4 <= len(sys.argv) <= 6 or not re.fullmatch(r"[A-Za-z0-9_-]+", sys.argv[1]) or sys.argv[2] not in train.ARMS:
        raise SystemExit("usage: replicate.py NAME ARM(L|B) K [ITERATIONS] [ENV_RUNNERS]")
    k = int(sys.argv[3])
    if k < 1:
        raise SystemExit("K starts at 1")
    run = RUNS_DIR / sys.argv[1]
    if run.exists():
        raise SystemExit(f"refusing: {run} exists")
    if train.config_env["deer_catch_rule"] != "touch":
        raise SystemExit("section 2 runs in the touch world (config/config_env.py)")
    train.config_ppo.update(
        arm=sys.argv[2],
        seed=k,
        deer_schedule=str(deer_source(k)),
        deer_checkpoint=None,
        deer_learn=False,
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
