"""Baseline ecology with lifetime-only linear TD(0) actor-critic learning."""

from dataclasses import dataclass

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.genome import crossover, founder_genome, mutate
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.networks import action_probs, evaluate, sample_action
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.world import Agent, ErlWorld as BaselineWorld, N_ACTIONS
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_actor_critic.networks import actor_critic_update


LEARNING_STRATEGIES = frozenset(("ERL", "L", "ERLC", "ERLK", "ERLS"))


@dataclass
class ActorCriticAgent(Agent):
    critic_weights: np.ndarray | None = None
    critic_bias: np.ndarray | None = None
    prev_probs: np.ndarray | None = None
    last_td_error: float = 0.0
    last_actor_update_norm: float = 0.0
    last_critic_update_norm: float = 0.0


class ActorCriticWorld(BaselineWorld):
    def __init__(self, config: dict, rng: np.random.Generator):
        self._validate_actor_critic_config(config)
        super().__init__(config, rng)

    @staticmethod
    def _validate_actor_critic_config(config: dict) -> None:
        positive = ("actor_alpha", "critic_beta", "actor_max_update_norm", "critic_max_update_norm")
        for key in positive:
            if not (np.isfinite(config[key]) and config[key] > 0.0):
                raise ValueError(f"{key} must be finite and > 0")
        gamma = config["actor_critic_gamma"]
        if not (np.isfinite(gamma) and 0.0 <= gamma <= 1.0):
            raise ValueError("actor_critic_gamma must be finite and in [0, 1]")
        if not np.isfinite(config["actor_critic_terminal_bonus"]):
            raise ValueError("actor_critic_terminal_bonus must be finite")

    def _make_agent(self, genome, row: int, col: int, generation: int) -> ActorCriticAgent:
        return ActorCriticAgent(
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
            critic_weights=np.zeros(self.obs_dim),
            critic_bias=np.zeros(1),
        )

    def _spawn_founder_agent(self):
        cell = self._random_empty_cell()
        if cell is None:
            return
        row, col = cell
        genome = founder_genome(
            self.obs_dim, N_ACTIONS, self.rng, self.cfg["founder_weight_std"],
            fixed_eval_weights=self.cfg.get("fixed_eval_weights"),
        )
        agent = self._make_agent(genome, row, col, 0)
        self._next_agent_id += 1
        self.agents.append(agent)
        self.occupant[(row, col)] = agent

    def _learn(self, agent: ActorCriticAgent, reward: float, next_obs, *, terminal: bool) -> None:
        td, actor_norm, critic_norm = actor_critic_update(
            agent.action_weights, agent.action_bias, agent.critic_weights, agent.critic_bias,
            agent.prev_obs, agent.prev_probs, agent.prev_action, reward, next_obs,
            actor_alpha=self.cfg["actor_alpha"], critic_beta=self.cfg["critic_beta"],
            gamma=self.cfg["actor_critic_gamma"],
            actor_max_update_norm=self.cfg["actor_max_update_norm"],
            critic_max_update_norm=self.cfg["critic_max_update_norm"], terminal=terminal,
        )
        agent.last_td_error = td
        agent.last_actor_update_norm = actor_norm
        agent.last_critic_update_norm = critic_norm

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
            if learning_enabled and agent.prev_obs is not None:
                self._learn(agent, evaluation - agent.prev_eval, obs, terminal=False)

            if self.strategy == "B":
                probs = None
                action = int(self.rng.integers(0, N_ACTIONS))
            else:
                probs = action_probs(obs, agent.action_weights, agent.action_bias)
                action = sample_action(probs, self.rng)

            agent.prev_obs = obs
            agent.prev_probs = probs
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
            final_evaluation = evaluate(final_obs, agent.genome.eval_weights, agent.genome.eval_bias)
            reward = final_evaluation - agent.prev_eval + self.cfg["actor_critic_terminal_bonus"]
            self._learn(agent, reward, None, terminal=True)
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
            if agent.energy < threshold or len(self.agents) + len(newborns) >= self.cfg["max_population_cap"]:
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
                child_genome = mutate(child_genome, self.rng, self.cfg["mutation_rate"], self.cfg["mutation_std"])
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


ErlWorld = ActorCriticWorld
