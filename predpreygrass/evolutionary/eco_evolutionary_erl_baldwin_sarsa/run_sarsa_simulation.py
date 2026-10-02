"""Run the SARSA(lambda) Baldwin variant using the baseline simulation CLI.

All baseline flags remain available.  ``--out-dir`` is recommended so SARSA
results are kept separate from existing REINFORCE runs.
"""

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin import run_erl_simulation as runner
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_sarsa.config import config_sarsa
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_sarsa.world import SarsaWorld


def main():
    runner.config_erl = config_sarsa
    runner.ErlWorld = SarsaWorld
    runner.EXPECTED_WORLD_CLASS = SarsaWorld
    runner.RUN_NAME_PREFIX = "ERL_BALDWIN_SARSA"
    runner.main()


if __name__ == "__main__":
    main()
