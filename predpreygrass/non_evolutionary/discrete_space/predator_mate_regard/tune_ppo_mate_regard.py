"""Train Iteration 21 (mate-regard reward). Reuses predator_sexual_reproduction's PPO script with the
MateRegardPredPreyGrass env; runs land in this module's runs/ directory."""
import argparse
import sys

from predpreygrass.non_evolutionary.discrete_space.predator_mate_regard.config_env import config_env
from predpreygrass.non_evolutionary.discrete_space.predator_mate_regard.mate_regard_env import (
    MateRegardPredPreyGrass,
    mate_regard_env_creator,
)
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction import (
    tune_ppo_predator_sexual_reproduction as core,
)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mate-regard-weight", type=float, required=True)
    own, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    core.PredPreyGrass = MateRegardPredPreyGrass
    core.env_creator = mate_regard_env_creator  # importable creator for remote env runners
    core.config_env = {**config_env, "compound_gift": "on", "mate_regard_weight": own.mate_regard_weight}
    core.__file__ = __file__  # core.main picks its runs/ directory from __file__
    core.main()


if __name__ == "__main__":
    main()
