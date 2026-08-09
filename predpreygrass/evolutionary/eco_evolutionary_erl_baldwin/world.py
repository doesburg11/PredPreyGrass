"""Self-contained predator-prey-grass world for ERL agents.

Deliberately not built on RLlib/PPO/Ray -- there is no shared,
centrally-trained policy here. Each agent carries its own genome
(genome.py) and its own live, individually-learning action network
(networks.py), updated by a plain per-step Python/NumPy loop. This
reuses the predator-prey-grass ecology already validated elsewhere in
this project (energy-based survival/reproduction, satiation on
predator catches, grass regrowth) rather than replicating Ackley &
Littman's exact world (which additionally has carnivores, trees, and
walls) -- see README.md for what's reused vs. adapted.

Observation (per agent, 9 features, matching Ackley & Littman's
nearest-object-per-compass-direction scheme rather than PredPreyGrass's
dense image-style observation -- the latter is sized for a CNN, not the
single-layer network used here):
  [food_N, food_S, food_E, food_W, danger_N, danger_S, danger_E,
   danger_W, own_energy_normalized]
"food" = grass for prey, prey for predators. "danger" = predators, for
prey only (predators have no in-world danger, so those four channels
are always 0 for them -- kept for a uniform obs_dim across species).
Each directional channel is 0 if nothing relevant is within
`sense_range` cells in that direction, else in (0, 1], closer = larger.

Actions: 0=stay, 1=north, 2=south, 3=east, 4=west (single-cell move,
clipped at the grid boundary).
"""

from dataclasses import dataclass, field

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.genome import (
    Genome,
    crossover,
    founder_genome,
    mutate,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.networks import (
    action_probs,
    evaluate,
    reinforce_update,
    sample_action,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.metrics import FunctionalConstraintTracker

OBS_DIM = 9
N_ACTIONS = 5
_DELTAS = {1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}  # N, S, E, W


@dataclass
class Agent:
    agent_id: int
    species: str  # "predator" or "prey"
    row: int
    col: int
    energy: float
    genome: Genome
    action_weights: np.ndarray  # LIVE, learned copy -- diverges from genome.action_weights over life
    action_bias: np.ndarray
    generation: int
    prev_obs: np.ndarray | None = None
    prev_action: int | None = None
    prev_eval: float | None = None
    alive: bool = True


class ErlWorld:
    def __init__(self, config: dict, rng: np.random.Generator):
        self.cfg = config
        self.rng = rng
        self.grid_size = config["grid_size"]
        self.sense_range = config["sense_range"]
        self.current_step = 0
        self._next_agent_id = 0
        self.agents: list[Agent] = []
        self.grass_energy = np.zeros((self.grid_size, self.grid_size))
        self.constraint_trackers = {
            "predator": FunctionalConstraintTracker(OBS_DIM, N_ACTIONS),
            "prey": FunctionalConstraintTracker(OBS_DIM, N_ACTIONS),
        }
        self.reset()

    # ---- setup ----

    def reset(self):
        self.current_step = 0
        self.agents = []
        self._next_agent_id = 0
        self.grass_energy = self.rng.uniform(
            0.0, self.cfg["max_energy_grass"], size=(self.grid_size, self.grid_size)
        )
        for _ in range(self.cfg["n_initial_prey"]):
            self._spawn_founder("prey")
        for _ in range(self.cfg["n_initial_predators"]):
            self._spawn_founder("predator")

    def _spawn_founder(self, species: str):
        genome = founder_genome(OBS_DIM, N_ACTIONS, self.rng, self.cfg["founder_weight_std"])
        row = int(self.rng.integers(0, self.grid_size))
        col = int(self.rng.integers(0, self.grid_size))
        init_energy = self.cfg[f"initial_energy_{species}"]
        agent = Agent(
            agent_id=self._next_agent_id,
            species=species,
            row=row,
            col=col,
            energy=init_energy,
            genome=genome,
            action_weights=genome.action_weights.copy(),
            action_bias=genome.action_bias.copy(),
            generation=0,
        )
        self._next_agent_id += 1
        self.agents.append(agent)

    # ---- observation ----

    def _nearest_in_direction(self, row: int, col: int, drow: int, dcol: int, targets: set[tuple[int, int]]) -> float:
        for dist in range(1, self.sense_range + 1):
            r, c = row + drow * dist, col + dcol * dist
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                break
            if (r, c) in targets:
                return 1.0 - 0.5 * (dist - 1) / max(self.sense_range - 1, 1)
        return 0.0

    def _observe(self, agent: Agent) -> np.ndarray:
        prey_cells = {(a.row, a.col) for a in self.agents if a.alive and a.species == "prey"}
        predator_cells = {(a.row, a.col) for a in self.agents if a.alive and a.species == "predator"}
        obs = np.zeros(OBS_DIM)
        dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # N, S, E, W
        if agent.species == "prey":
            for i, (dr, dc) in enumerate(dirs):
                obs[i] = self._grass_signal(agent.row, agent.col, dr, dc)
                obs[4 + i] = self._nearest_in_direction(agent.row, agent.col, dr, dc, predator_cells)
        else:  # predator
            for i, (dr, dc) in enumerate(dirs):
                obs[i] = self._nearest_in_direction(agent.row, agent.col, dr, dc, prey_cells)
            # danger channels (4:8) stay 0 -- predators have no in-world threat
        max_e = self.cfg[f"max_energy_{agent.species}_for_norm"]
        obs[8] = min(agent.energy / max_e, 1.0)
        return obs

    def _grass_signal(self, row: int, col: int, drow: int, dcol: int) -> float:
        for dist in range(1, self.sense_range + 1):
            r, c = row + drow * dist, col + dcol * dist
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                break
            if self.grass_energy[r, c] > self.cfg["grass_visibility_threshold"]:
                return 1.0 - 0.5 * (dist - 1) / max(self.sense_range - 1, 1)
        return 0.0

    # ---- step ----

    def step(self):
        self.current_step += 1
        order = list(self.agents)
        self.rng.shuffle(order)

        for agent in order:
            if not agent.alive:
                continue
            obs = self._observe(agent)
            e_now = evaluate(obs, agent.genome.eval_weights, agent.genome.eval_bias)

            # Reinforce the PREVIOUS action using how the evaluation changed
            # since it was taken (Ackley & Littman's temporal-offset CRBP).
            if agent.prev_obs is not None:
                reinforcement = e_now - agent.prev_eval
                reinforce_update(
                    agent.action_weights,
                    agent.action_bias,
                    agent.prev_obs,
                    agent.prev_action,
                    reinforcement,
                    self.cfg["lr_positive"],
                    self.cfg["lr_negative"],
                )

            probs = action_probs(obs, agent.action_weights, agent.action_bias)
            action = sample_action(probs, self.rng)
            self._apply_action(agent, action)

            agent.prev_obs = obs
            agent.prev_action = action
            agent.prev_eval = e_now

            agent.energy -= self.cfg[f"basal_energy_cost_{agent.species}"]
            if agent.energy <= 0:
                agent.alive = False

        self._handle_reproduction()
        self._regrow_grass()
        self.agents = [a for a in self.agents if a.alive]

    def _apply_action(self, agent: Agent, action: int):
        if action == 0:
            self._try_eat(agent)
            return
        dr, dc = _DELTAS[action]
        new_row = min(max(agent.row + dr, 0), self.grid_size - 1)
        new_col = min(max(agent.col + dc, 0), self.grid_size - 1)
        agent.row, agent.col = new_row, new_col
        agent.energy -= self.cfg["movement_energy_cost"]
        self._try_eat(agent)

    def _try_eat(self, agent: Agent):
        if agent.species == "prey":
            g = self.grass_energy[agent.row, agent.col]
            if g > self.cfg["grass_visibility_threshold"]:
                gain = min(g, self.cfg["max_energy_gain_per_bite"])
                agent.energy += gain
                self.grass_energy[agent.row, agent.col] -= gain
        else:
            for other in self.agents:
                if other.alive and other.species == "prey" and other.row == agent.row and other.col == agent.col:
                    agent.energy += self.cfg["max_energy_gain_per_prey"]
                    other.alive = False
                    break

    def _handle_reproduction(self):
        newborns = []
        counts = self.population_counts()
        for agent in self.agents:
            if not agent.alive:
                continue
            threshold = self.cfg[f"reproduction_energy_threshold_{agent.species}"]
            if agent.energy < threshold:
                continue
            if counts[agent.species] >= self.cfg["max_population_per_species"]:
                continue
            mate = self._nearest_mate(agent)
            child_genome = agent.genome.copy()
            if mate is not None:
                child_genome = crossover(agent.genome, mate.genome, self.rng)
            child_genome = mutate(child_genome, self.rng, self.cfg["mutation_rate"], self.cfg["mutation_std"])
            self.constraint_trackers[agent.species].record(agent.genome.flatten(), child_genome.flatten())

            cost = self.cfg[f"reproduction_energy_cost_{agent.species}"]
            agent.energy -= cost
            child = Agent(
                agent_id=self._next_agent_id,
                species=agent.species,
                row=agent.row,
                col=agent.col,
                energy=self.cfg[f"initial_energy_{agent.species}"],
                genome=child_genome,
                action_weights=child_genome.action_weights.copy(),  # LIVE net starts from genome, not parent's learned state
                action_bias=child_genome.action_bias.copy(),
                generation=agent.generation + 1,
            )
            self._next_agent_id += 1
            newborns.append(child)
        self.agents.extend(newborns)

    def _nearest_mate(self, agent: Agent) -> Agent | None:
        best, best_dist = None, self.cfg["mate_search_radius"] + 1
        for other in self.agents:
            if other is agent or not other.alive or other.species != agent.species:
                continue
            dist = abs(other.row - agent.row) + abs(other.col - agent.col)
            if dist <= self.cfg["mate_search_radius"] and dist < best_dist:
                best, best_dist = other, dist
        return best

    def _regrow_grass(self):
        self.grass_energy = np.minimum(
            self.grass_energy + self.cfg["grass_regrow_rate"], self.cfg["max_energy_grass"]
        )

    # ---- summary ----

    def population_counts(self) -> dict[str, int]:
        return {
            "predator": sum(1 for a in self.agents if a.alive and a.species == "predator"),
            "prey": sum(1 for a in self.agents if a.alive and a.species == "prey"),
        }

    def genome_stats(self) -> dict[str, float]:
        """Population-mean genome weight magnitude per species (parallel to the
        `{species}_metabolic_rate_mean`-style metrics used elsewhere in this
        project). Mean |weight| rather than mean weight, since these weights
        have no inherent sign-preference the way a scalar trait like
        metabolic_rate does -- what matters is whether genome values are
        moving away from the founder distribution at all, in either direction.
        """
        out: dict[str, float] = {}
        for species in ("predator", "prey"):
            genomes = [a.genome for a in self.agents if a.alive and a.species == species]
            if not genomes:
                out[f"{species}_eval_weight_absmean"] = float("nan")
                out[f"{species}_action_weight_absmean"] = float("nan")
                continue
            eval_vals = np.concatenate([g.eval_weights for g in genomes])
            action_vals = np.concatenate([g.action_weights.ravel() for g in genomes])
            out[f"{species}_eval_weight_absmean"] = float(np.mean(np.abs(eval_vals)))
            out[f"{species}_action_weight_absmean"] = float(np.mean(np.abs(action_vals)))
        return out
