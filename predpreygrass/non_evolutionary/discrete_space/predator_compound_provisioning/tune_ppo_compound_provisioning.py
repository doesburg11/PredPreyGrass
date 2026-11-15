"""Train the Iteration 20 compound movement-plus-provisioning environment.

This reuses the validated PPO configuration of predator_sexual_reproduction,
but substitutes the 18-action CompoundGiftPredPreyGrass environment and stores
runs under this module's own runs/ directory.
"""
import argparse
import sys

from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.compound_gift_env import (
    CompoundGiftPredPreyGrass,
    compound_env_creator,
)
from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.config_env import config_env
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction import (
    tune_ppo_predator_sexual_reproduction as core,
)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--compound-gift", choices=("on", "inert"), default="on")
    own, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    core.PredPreyGrass = CompoundGiftPredPreyGrass
    # The env runners need an importable creator; see compound_env_creator.
    core.env_creator = compound_env_creator
    core.config_env = {**config_env, "compound_gift": own.compound_gift}
    # core.main uses its module global __file__ to choose its runs directory.
    # Point that lookup to this wrapper so provenance stays in this module.
    core.__file__ = __file__
    core.main()


if __name__ == "__main__":
    main()
