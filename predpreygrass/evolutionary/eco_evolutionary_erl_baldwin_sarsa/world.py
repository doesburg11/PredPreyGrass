"""World AL variant replacing lifetime REINFORCE with linear SARSA(lambda).

All ecology, genome, mutation, crossover, and comparative-strategy mechanics
come from ``eco_evolutionary_erl_baldwin``.  Genomic action parameters are
interpreted as inherited initial Q parameters.  Each agent receives private
live copies and zero eligibility traces; neither learned Q changes nor traces
are inherited.
"""

from dataclasses import dataclass

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.genome import (
    crossover,
    founder_genome,
    mutate,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.networks import evaluate
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import (
    Agent,
    ErlWorld as BaselineWorld,
    N_ACTIONS,
    OBS_DIM,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_sarsa.networks import (
    q_values,
    sample_action,
    sarsa_lambda_update,
    softmax_probs,
)


LEARNING_STRATEGIES = frozenset(("ERL", "L", "ERLC", "ERLK", "ERLS"))


@dataclass
class SarsaAgent(Agent):
    """Baseline agent plus non-heritable accumulating eligibility traces."""

    eligibility_weights: np.ndarray | None = None
    eligibility_bias: np.ndarray | None = None
    last_td_error: float = 0.0


class SarsaWorld(BaselineWorld):
    """Baseline ecology with on-policy linear SARSA(lambda) within life."""

    def __init__(self, config: dict, rng: np.random.Generator):
        self._validate_sarsa_config(config)
        super().__init__(config, rng)

    @staticmethod
    def _validate_sarsa_config(config: dict) -> None:
        alpha = config["sarsa_alpha"]
        gamma = config["sarsa_gamma"]
        trace_lambda = config["sarsa_lambda"]
        temperature = config["sarsa_temperature"]
        terminal_bonus = config["sarsa_terminal_bonus"]
        if not (np.isfinite(alpha) and alpha > 0.0):
            raise ValueError("sarsa_alpha must be finite and > 0")
        if not (np.isfinite(gamma) and 0.0 <= gamma <= 1.0):
            raise ValueError("sarsa_gamma must be finite and in [0, 1]")
        if not (np.isfinite(trace_lambda) and 0.0 <= trace_lambda <= 1.0):
            raise ValueError("sarsa_lambda must be finite and in [0, 1]")
        if not (np.isfinite(temperature) and temperature > 0.0):
            raise ValueError("sarsa_temperature must be finite and > 0")
        if not np.isfinite(terminal_bonus):
            raise ValueError("sarsa_terminal_bonus must be finite")

    def _make_agent(self, genome, row: int, col: int, generation: int) -> SarsaAgent:
        return SarsaAgent(
            agent_id=self._next_agent_id,
            row=row,
            col=col,
            energy=self.cfg["initial_energy_agent"],
            health=self.cfg["initial_health_agent"],
            in_tree=False,
            genome=genome,
            action_weights=genome.action_weights.copy(),
            action_bias=genome.action_bias.copy(),
            generation=generation,
            born_step=self.current_step,
            eligibility_weights=np.zeros_like(genome.action_weights),
            eligibility_bias=np.zeros_like(genome.action_bias),
        )

    def _spawn_founder_agent(self):
        cell = self._random_empty_cell()
        if cell is None:
            return
        row, col = cell
        genome = founder_genome(
            self.obs_dim,
            N_ACTIONS,
            self.rng,
            self.cfg["founder_weight_std"],
            fixed_eval_weights=self.cfg.get("fixed_eval_weights"),
        )
        agent = self._make_agent(genome, row, col, generation=0)
        self._next_agent_id += 1
        self.agents.append(agent)
        self.occupant[(row, col)] = agent

    def _choose_action(self, agent: SarsaAgent, obs: np.ndarray) -> int:
        probabilities = softmax_probs(
            q_values(obs, agent.action_weights, agent.action_bias),
            self.cfg["sarsa_temperature"],
        )
        return sample_action(probabilities, self.rng)

    def _update_agent(
        self,
        agent: SarsaAgent,
        reward: float,
        next_obs: np.ndarray | None,
        next_action: int | None,
        *,
        terminal: bool,
    ) -> None:
        agent.last_td_error = sarsa_lambda_update(
            agent.action_weights,
            agent.action_bias,
            agent.eligibility_weights,
            agent.eligibility_bias,
            agent.prev_obs,
            agent.prev_action,
            reward,
            next_obs,
            next_action,
            alpha=self.cfg["sarsa_alpha"],
            gamma=self.cfg["sarsa_gamma"],
            trace_lambda=self.cfg["sarsa_lambda"],
            terminal=terminal,
        )

    def _step_agents(self):
        order = list(self.agents)
        self.rng.shuffle(order)
        learning_enabled = self.strategy in LEARNING_STRATEGIES
        coop = self.strategy in ("C", "ERLC")
        for agent in order:
            if not agent.alive:
                continue
            obs = self._observe_agent(agent)
            evaluation = evaluate(obs, agent.genome.eval_weights, agent.genome.eval_bias)

            if self.strategy == "B":
                action = int(self.rng.integers(0, N_ACTIONS))
            else:
                # Select exactly once before the update.  This cached action is
                # both the SARSA bootstrap target and the behavior actually
                # executed below, even though the update changes Q in between.
                action = self._choose_action(agent, obs)

            if learning_enabled and agent.prev_obs is not None:
                self._update_agent(
                    agent,
                    evaluation - agent.prev_eval,
                    obs,
                    action,
                    terminal=False,
                )

            # Commit the transition before resolving it so a death caused by
            # this action receives exactly one terminal update in _kill_agent.
            agent.prev_obs = obs
            agent.prev_action = action
            agent.prev_eval = evaluation
            self._resolve_agent_action(agent, action, track_forage=coop)

            if agent.alive:
                agent.energy -= self.cfg["basal_energy_cost_agent"]
                if agent.energy <= 0 or agent.health <= 0:
                    self._kill_agent(agent)

    def _kill_agent(self, agent: Agent):
        if not agent.alive:
            return
        if self.strategy in LEARNING_STRATEGIES and agent.prev_obs is not None:
            final_obs = self._observe_agent(agent)
            final_evaluation = evaluate(
                final_obs, agent.genome.eval_weights, agent.genome.eval_bias
            )
            self._update_agent(
                agent,
                final_evaluation
                - agent.prev_eval
                + self.cfg["sarsa_terminal_bonus"],
                None,
                None,
                terminal=True,
            )
        super()._kill_agent(agent)

    def _handle_agent_reproduction(self):
        coop = self.strategy in ("C", "ERLC")
        newborns = []
        for agent in self.agents:
            if not agent.alive:
                continue
            threshold = self.cfg["reproduction_energy_threshold_agent"]
            if coop and self._agent_group_is_cooperative_fit(agent):
                threshold *= 1.0 - self.cfg["coop_threshold_discount_frac"]
            if agent.energy < threshold:
                continue
            if len(self.agents) + len(newborns) >= self.cfg["max_population_cap"]:
                continue
            cell = self._nearest_empty_adjacent(agent.row, agent.col)
            if cell is None:
                continue

            mate = None
            if self.strategy in ("L", "F"):
                child_genome = agent.genome.copy()
            else:
                mate = self._nearest_mate(agent)
                child_genome = agent.genome.copy()
                if mate is not None:
                    child_genome = crossover(agent.genome, mate.genome, self.rng)
                child_genome = mutate(
                    child_genome,
                    self.rng,
                    self.cfg["mutation_rate"],
                    self.cfg["mutation_std"],
                )
            self.constraint_tracker.record(agent.genome.flatten(), child_genome.flatten())

            agent.energy -= self.cfg["reproduction_energy_cost_agent"]
            agent.offspring_count += 1
            if mate is not None:
                mate.offspring_count += 1
            if coop:
                agent.last_reproduce_step = self.current_step

            row, col = cell
            child = self._make_agent(child_genome, row, col, agent.generation + 1)
            self._next_agent_id += 1
            newborns.append(child)
            self.occupant[(row, col)] = child
        self.agents.extend(newborns)


# Familiar name for callers shared with the baseline package.
ErlWorld = SarsaWorld
