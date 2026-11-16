"""Mate-regard reward on top of the 18-action compound-gift environment (RESULTS.md Iteration 21).

Each predator's reward gains ``mate_regard_weight * reward_predator_per_energy * intake(mate)``, where
``intake(mate)`` is the energy its recorded mate took in this step: food eaten plus gifts received. With weight w,
a predator values its mate's energy intake at w times its own. w = 0 reproduces the compound-gift environment
exactly. The mate is read from the bond at the start of the step, i.e. the bond a gift in this step goes to.
Rewards only: dynamics, actions and observations are those of CompoundGiftPredPreyGrass.
"""
import math

from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.compound_gift_env import (
    CompoundGiftPredPreyGrass,
)


class MateRegardPredPreyGrass(CompoundGiftPredPreyGrass):
    def __init__(self, config=None):
        super().__init__(config)
        w = float((config or {}).get("mate_regard_weight", 0.0))
        if not (0.0 <= w < math.inf):
            raise ValueError(f"mate_regard_weight must be finite and >= 0 (got {w})")
        self.mate_regard_weight = w
        self.mate_regard_reward_total = 0.0

    def reset(self, *, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self.mate_regard_reward_total = 0.0
        return out

    def step(self, action_dict):
        mates = dict(self.agent_mate)
        observations, rewards, terminations, truncations, infos = super().step(action_dict)
        if self.mate_regard_weight > 0.0:
            scale = self.mate_regard_weight * self.reward_predator_per_energy
            for agent in rewards:
                if "predator" not in agent:
                    continue
                mate = mates.get(agent)
                intake = self.predator_energy_intake.get(mate, 0.0) if mate is not None else 0.0
                if intake > 0.0:
                    bonus = scale * intake
                    rewards[agent] += bonus
                    self.cumulative_rewards[agent] = self.cumulative_rewards.get(agent, 0.0) + bonus
                    self.mate_regard_reward_total += bonus
        return observations, rewards, terminations, truncations, infos

    def _build_episode_training_metrics(self):
        metrics = super()._build_episode_training_metrics()
        metrics["mate_regard_weight"] = float(self.mate_regard_weight)
        metrics["mate_regard_reward_total"] = float(self.mate_regard_reward_total)
        return metrics


def mate_regard_env_creator(config):
    """RLlib env creator, module-level so remote env runners import it by reference (see compound_env_creator)."""
    from predpreygrass.non_evolutionary.discrete_space.predator_mate_regard.config_env import config_env

    return MateRegardPredPreyGrass({**config_env, **(config or {})})
