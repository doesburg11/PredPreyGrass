"""Run the actor-critic Baldwin variant using the baseline CLI."""

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin import run_erl_simulation as runner
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_actor_critic.config import config_actor_critic
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_actor_critic.world import ActorCriticWorld


def main():
    runner.config_erl = config_actor_critic
    runner.ErlWorld = ActorCriticWorld
    runner.EXPECTED_WORLD_CLASS = ActorCriticWorld
    runner.RUN_NAME_PREFIX = "ERL_BALDWIN_ACTOR_CRITIC"
    runner.main()


if __name__ == "__main__":
    main()
