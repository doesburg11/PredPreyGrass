"""RESULTS.md section 5: check that a run's world and training settings are
section 2's (touch_1's saved settings) except for the stated differences:
deer_young_solo "certain" (instead of "graded") and 500 iterations (instead
of 1000); output folders may differ too.

    check_control_world.py            # the current config, before training
    check_control_world.py CHECKPOINT # a trained run's saved settings
"""

import sys

from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt import checkpoints
from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.launchers.check_logistic_world import PPO_FREE, REFERENCE, differences

ENV_CHANGES = {"deer_young_solo": ("graded", "certain")}
PPO_CHANGES = {"iterations": (1000, 500)}


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
        env["deer_young_solo"] = "certain"  # set by replicate.py --set
        ppo["iterations"] = 500  # set by replicate.py NAME 500
    problems = differences(ref_env, env, ENV_CHANGES) + differences(reference["config_ppo"], ppo, PPO_CHANGES, PPO_FREE)
    if problems:
        raise SystemExit("not section 5's world:\n  " + "\n  ".join(problems))
    print("section 5's world: as section 2 except deer_young_solo certain and 500 iterations")


if __name__ == "__main__":
    main()
