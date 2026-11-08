"""Independent training runs of the configured world from scratch (README.md;
RESULTS.md, section 1). Each run starts from fresh network weights and
draws its episodes from the operating system's entropy, so runs differ by
chance only.

Run from the repository root, e.g. under systemd-run --user:

    python predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/launchers/replicate.py \\
        NAME [ITERATIONS] [ENV_RUNNERS] [--catch-rule party|touch]

--catch-rule sets the world's deer_catch_rule (default: config/config_env.py,
which is "party"; "touch" is RESULTS.md section 2's world).

Output: runs/<NAME>/ (refuses to overwrite).
"""

import re
import sys

from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt import train
from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.config.config_ppo import RUNS_DIR

USAGE = "usage: replicate.py NAME [ITERATIONS] [ENV_RUNNERS] [--catch-rule party|touch]"


def main():
    args = sys.argv[1:]
    if "--catch-rule" in args:
        i = args.index("--catch-rule")
        if i + 1 >= len(args) or args[i + 1] not in ("party", "touch"):
            raise SystemExit(USAGE)
        train.config_env["deer_catch_rule"] = args[i + 1]
        del args[i : i + 2]
    sys.argv = sys.argv[:1] + args
    if not 2 <= len(sys.argv) <= 4 or not re.fullmatch(r"[A-Za-z0-9_-]+", sys.argv[1]):
        raise SystemExit(USAGE)
    run = RUNS_DIR / sys.argv[1]
    if run.exists():
        raise SystemExit(f"refusing: {run} exists")
    train.config_ppo["tensorboard_dir"] = str(run / "tensorboard")
    train.config_ppo["checkpoint_dir"] = str(run / "checkpoint")
    if len(sys.argv) > 2:
        train.config_ppo["iterations"] = int(sys.argv[2])
    if len(sys.argv) > 3:
        train.config_ppo["num_env_runners"] = int(sys.argv[3])
    train.main()


if __name__ == "__main__":
    main()
