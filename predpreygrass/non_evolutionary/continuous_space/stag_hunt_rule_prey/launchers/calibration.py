"""Calibration runs, trained from scratch with the settings of
config/config_env.py and config/config_ppo.py except for ARMS. Run from the
repository root, e.g. under systemd-run --user:

    python predpreygrass/non_evolutionary/continuous_space/stag_hunt_rule_prey/launchers/calibration.py \\
        ARM [ITERATIONS] [ENV_RUNNERS]

Output: runs/calibration_<arm>/ (refuses to overwrite).
"""

import sys

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey import train
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.config.config_ppo import RUNS_DIR

ARMS = {
    # The world of stag_hunt_continuous's round 19c with the prey on rules.
    "base": {},
    # Scarcer food, chosen with prey_check: about 46 rabbits, about 15
    # mammoths at energy about 10, so a typical pair can win at ratio 0.6.
    "lean": {"grass_overgrazing_delay": 800, "steppe_overgrazing_delay": 800},
    # Rabbits nearer to real ones, chosen after the replay in RESULTS.md, section
    # 5: wider view, earlier flight. One run, no further rabbit changes after it.
    "harder_rabbits": {"rabbit_fov": 340, "rabbit_flee_distance": 200.0},
}


def main():
    if not 2 <= len(sys.argv) <= 4 or sys.argv[1] not in ARMS:
        raise SystemExit(f"usage: calibration.py {{{','.join(ARMS)}}} [ITERATIONS] [ENV_RUNNERS]")
    arm = sys.argv[1]
    run = RUNS_DIR / f"calibration_{arm}"
    if run.exists():
        raise SystemExit(f"refusing: {run} exists")
    train.config_env.update(ARMS[arm])
    train.config_ppo["tensorboard_dir"] = str(run / "tensorboard")
    train.config_ppo["checkpoint_dir"] = str(run / "checkpoint")
    if len(sys.argv) > 2:
        train.config_ppo["iterations"] = int(sys.argv[2])
    if len(sys.argv) > 3:
        train.config_ppo["num_env_runners"] = int(sys.argv[3])
    train.main()


if __name__ == "__main__":
    main()
