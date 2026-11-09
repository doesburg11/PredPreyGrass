"""RESULTS.md section 3: check that a run's world and training settings are
section 2's (touch_1's saved settings) except for the stated differences:
grass_regrowth "logistic" (instead of "instant") and 500 iterations (instead
of 1000); output folders may differ too.

    check_logistic_world.py            # the current config, before training
    check_logistic_world.py CHECKPOINT # a trained run's saved settings
"""

import sys
from pathlib import Path

from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt import checkpoints

RUNS = Path(__file__).resolve().parents[1] / "runs"
REFERENCE = RUNS / "touch_1" / "checkpoint"
ENV_CHANGES = {"grass_regrowth": ("instant", "logistic")}
PPO_CHANGES = {"iterations": (1000, 500)}
PPO_FREE = {"tensorboard_dir", "checkpoint_dir"}


def differences(reference, candidate, changes, free=()):
    problems = []
    for key in sorted(set(reference) | set(candidate)):
        if key in free:
            continue
        old, new = reference.get(key), candidate.get(key)
        expected = changes.get(key, (old, old))
        if (old, new) != expected:
            problems.append(f"{key}: section 2 {old!r}, this run {new!r}, expected {expected[1]!r}")
    return problems


def main():
    reference = checkpoints.run_settings(REFERENCE)
    ref_env = checkpoints.checkpoint_env_config(REFERENCE)
    if len(sys.argv) > 1:
        candidate = checkpoints.run_settings(sys.argv[1])
        env = checkpoints.checkpoint_env_config(sys.argv[1])
        ppo = candidate["config_ppo"]
    else:
        from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt import train
        env, ppo = dict(train.config_env), dict(train.config_ppo)
        env["deer_catch_rule"] = "touch"  # set by replicate.py --catch-rule touch
        ppo["iterations"] = 500  # set by replicate.py NAME 500
    problems = differences(ref_env, env, ENV_CHANGES) + differences(reference["config_ppo"], ppo, PPO_CHANGES, PPO_FREE)
    if problems:
        raise SystemExit("not section 3's world:\n  " + "\n  ".join(problems))
    print("section 3's world: as section 2 except grass_regrowth logistic and 500 iterations")


if __name__ == "__main__":
    main()
