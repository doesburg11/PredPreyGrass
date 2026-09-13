"""Trial 13 driver: wraps flagship's `PredPreyGrass.step()` with genome-driven
prey action selection (evolved eval_weights reward + REINFORCE-learned
action_weights) and a frozen PPO predator, plus the newborn/parent-detection and
offspring-count bookkeeping flagship's env has no concept of.

Structural analog of eco_evolutionary_erl_baldwin/world.py's `_step_agents`/
`_handle_agent_reproduction`, but wrapping flagship's actual env rather than
reimplementing grid mechanics -- see this module's README.md.
"""

from dataclasses import dataclass

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.config import N_ACTIONS, OBS_DIM
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import extract_prey_features
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.genome import Genome, founder_genome, mutate
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.networks import (
    action_probs,
    evaluate,
    reinforce_update,
    sample_action,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.predator_policy import FrozenPredatorPolicy
from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass


@dataclass
class PreyGenomeState:
    """Per-prey bookkeeping for one flagship `prey_<i>` agent_id -- structurally
    analogous to erl_baldwin's `Agent`, and exposing exactly the attributes
    metrics.py's `lineage_record`/`lineage_fieldnames` expect (agent_id,
    generation, born_step, offspring_count, genome)."""

    agent_id: str
    genome: Genome
    action_weights: np.ndarray  # LIVE, learned copy -- diverges from genome.action_weights over life
    action_bias: np.ndarray
    generation: int
    born_step: int
    prev_obs: np.ndarray | None = None
    prev_action: int | None = None
    prev_eval: float | None = None
    offspring_count: int = 0


class Trial13Driver:
    """Owns the flagship env, the frozen predator, and the prey genome registry;
    `step()` is the whole per-step loop. `on_agent_death(state, death_step)`, if
    set, is called for each prey death (mirrors erl_baldwin's World.on_agent_death
    hook, used for lineage-fitness CSV logging -- see run_trial13_simulation.py)."""

    def __init__(self, env: PredPreyGrass, predator_policy: FrozenPredatorPolicy, cfg: dict, rng: np.random.Generator):
        self.env = env
        self.predator_policy = predator_policy
        self.cfg = cfg
        self.rng = rng
        self.current_step = 0
        self.registry: dict[str, PreyGenomeState] = {}
        self.on_agent_death = None

    def reset(self):
        self.env.reset()
        self.current_step = 0
        self.registry = {}
        for agent_id in list(self.env.agents):
            if "prey" in agent_id:
                self.registry[agent_id] = self._new_founder(agent_id)

    def _new_founder(self, agent_id: str) -> PreyGenomeState:
        fixed_eval_weights = self.cfg.get("fixed_eval_weights")
        genome = founder_genome(
            OBS_DIM, N_ACTIONS, self.rng, self.cfg["founder_weight_std"], fixed_eval_weights
        )
        return PreyGenomeState(
            agent_id=agent_id,
            genome=genome,
            action_weights=genome.action_weights.copy(),
            action_bias=genome.action_bias.copy(),
            generation=0,
            born_step=self.current_step,
        )

    def step(self):
        """One environment step: genome-driven prey actions + frozen-predator
        actions -> env.step() -> newborn/parent detection -> death detection."""
        action_dict = {}
        for agent_id in self.env.agents:
            if agent_id not in self.env.agent_positions:
                # Terminated in a PRIOR step but not yet purged from self.env.agents
                # -- flagship defers that removal to the top of its OWN next
                # step() call (predpreygrass_rllib_env.py:238-241), so it's still
                # in this list; skip it exactly as the env's own Step 3 does
                # ("if agent not in self.agent_positions: continue").
                continue
            if "prey" in agent_id:
                action_dict[agent_id] = self._select_prey_action(agent_id)
            elif "predator" in agent_id:
                obs = self.env._get_observation(agent_id)
                action_dict[agent_id] = self.predator_policy.act(obs)

        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)
        self.current_step = self.env.current_step

        self._handle_reproduction(rewards)
        self._handle_deaths(terminations)

        return observations, rewards, terminations, truncations, infos

    def _select_prey_action(self, agent_id: str) -> int:
        state = self.registry[agent_id]
        feat = extract_prey_features(self.env, agent_id)
        e_now = evaluate(feat, state.genome.eval_weights, state.genome.eval_bias)

        if state.prev_obs is not None:
            reinforcement = e_now - state.prev_eval
            reinforce_update(
                state.action_weights, state.action_bias,
                state.prev_obs, state.prev_action, reinforcement,
                self.cfg["lr_positive"], self.cfg["lr_negative"],
            )

        probs = action_probs(feat, state.action_weights, state.action_bias)
        action = sample_action(probs, self.rng)

        state.prev_obs = feat
        state.prev_action = action
        state.prev_eval = e_now
        return action

    def _handle_reproduction(self, rewards: dict):
        """A prey agent whose reward this step equals reproduction_reward_prey just
        reproduced (predpreygrass_rllib_env.py:441-469); flagship places the
        newborn adjacent to it, with a fresh id already present in self.env.agents
        by the time env.step() returns. Detected here from the outside -- no
        modification to flagship's env -- by pairing each such parent to the
        nearest not-yet-registered prey id (the env always spawns the child
        adjacent to its parent, see _find_available_spawn_position)."""
        if not rewards:
            return
        threshold = self.env.reproduction_reward_prey
        parents = [
            agent_id for agent_id, r in rewards.items()
            if "prey" in agent_id and r == threshold and agent_id in self.registry
        ]
        if not parents:
            return
        unmatched_newborns = [
            agent_id for agent_id in self.env.agents
            if "prey" in agent_id and agent_id not in self.registry
        ]
        for parent_id in parents:
            if not unmatched_newborns:
                break  # more reproduction rewards than detected newborns; shouldn't happen
            parent_pos = self.env.agent_positions[parent_id]
            best_id, best_dist = None, None
            for newborn_id in unmatched_newborns:
                pos = self.env.agent_positions[newborn_id]
                dist = abs(pos[0] - parent_pos[0]) + abs(pos[1] - parent_pos[1])
                if best_dist is None or dist < best_dist:
                    best_id, best_dist = newborn_id, dist
            unmatched_newborns.remove(best_id)

            parent_state = self.registry[parent_id]
            parent_state.offspring_count += 1
            child_genome = mutate(
                parent_state.genome, self.rng, self.cfg["mutation_rate"], self.cfg["mutation_std"]
            )
            self.registry[best_id] = PreyGenomeState(
                agent_id=best_id,
                genome=child_genome,
                action_weights=child_genome.action_weights.copy(),
                action_bias=child_genome.action_bias.copy(),
                generation=parent_state.generation + 1,
                born_step=self.current_step,
            )

    def _handle_deaths(self, terminations: dict):
        for agent_id, terminated in terminations.items():
            if terminated and "prey" in agent_id and agent_id in self.registry:
                state = self.registry.pop(agent_id)
                if self.on_agent_death is not None:
                    self.on_agent_death(state, self.current_step)

    def population_counts(self) -> dict[str, int]:
        return {
            "prey": sum(1 for a in self.env.agents if "prey" in a and a in self.env.agent_positions),
            "predator": sum(1 for a in self.env.agents if "predator" in a and a in self.env.agent_positions),
        }

    def genome_stats(self) -> dict[str, float]:
        if not self.registry:
            return {"eval_weight_absmean": float("nan"), "action_weight_absmean": float("nan")}
        eval_abs = np.concatenate([np.abs(s.genome.eval_weights) for s in self.registry.values()])
        action_abs = np.concatenate([np.abs(s.action_weights).ravel() for s in self.registry.values()])
        return {
            "eval_weight_absmean": float(eval_abs.mean()),
            "action_weight_absmean": float(action_abs.mean()),
        }
