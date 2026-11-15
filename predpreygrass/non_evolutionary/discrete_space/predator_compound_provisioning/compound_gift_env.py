"""Compound-action version of the sexed predator environment.

Predator action ``a`` has movement ``a % 9`` and a transfer bit ``a >= 9``.
Both treatment arms have all 18 actions.  The `on` arm transfers energy after
all agents move; the `inert` arm makes the same request but transfers nothing.
This removes Iteration 19's extra-stay-action confound.
"""
import gymnasium

from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.predpreygrass_rllib_env import (
    PredPreyGrass,
)


class CompoundGiftPredPreyGrass(PredPreyGrass):
    """Sexed predator environment with 18 compound predator actions and 9 prey actions."""

    def __init__(self, config=None):
        supplied = dict(config or {})
        mode = supplied.get("compound_gift", "on")
        if mode is True:
            mode = "on"
        if mode is False:
            mode = "off"
        if mode not in ("on", "inert"):
            raise ValueError(f"compound_gift must be 'on' or 'inert' (got {mode!r})")

        # Disable the base's dedicated 10th action, then reuse its gift
        # transfer implementation and metrics through _gift_givers below.
        base_config = {**supplied, "learned_gift": False}
        super().__init__(base_config)
        self.compound_gift_mode = mode
        self.learned_gift_mode = mode
        self.give_action_id = 0  # non-None sentinel enabling base Step 2b; _is_give is overridden.
        self.compound_action_count = 2 * self.num_actions
        predator_space = gymnasium.spaces.Discrete(self.compound_action_count)
        prey_space = gymnasium.spaces.Discrete(self.num_actions)
        self.action_spaces = {
            agent: predator_space if "predator" in agent else prey_space for agent in self.possible_agents
        }
        self.num_predator_actions = self.compound_action_count

    def _is_give(self, agent, action) -> bool:
        # The action bit is decoded separately. This prevents action 0 (the
        # sentinel above) from being treated as an old dedicated give action.
        return False

    def _gift_givers(self, action_dict):
        return [
            agent
            for agent, action in action_dict.items()
            if agent in self.agent_positions and "predator" in agent and action >= self.num_actions
        ]

    def _movement_action(self, agent, action):
        return action % self.num_actions if "predator" in agent else action

    def _get_move(self, agent, action):
        return super()._get_move(agent, self._movement_action(agent, action))

    def _get_movement_energy_cost(self, agent, action):
        return super()._get_movement_energy_cost(agent, self._movement_action(agent, action))


def compound_env_creator(config):
    """RLlib env creator. Module-level in an importable module, so remote env runners unpickle it by reference
    and build the compound environment (patching the core tune script's PredPreyGrass global only affects the
    driver process: its workers re-import the core module and got the 9-action base env)."""
    from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.config_env import config_env

    return CompoundGiftPredPreyGrass({**config_env, **(config or {})})
