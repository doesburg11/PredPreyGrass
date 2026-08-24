"""World AL (Ackley & Littman 1991) -- rebuilt to match their described
mechanics, not the predator-prey-grass ecology used elsewhere in this
project. See config.py's module docstring for exactly which numbers are
from the paper vs. chosen by me because the paper never publishes them.

Entities:
  - Agent: the single ADAPTIVE species. Genome-initialized action + eval
    networks, local reinforcement learning during life (subject to
    `strategy`). Omnivorous: eats plants, dead agents, dead carnivores.
  - Carnivore: NON-adaptive. No genome, no network, no learning, ever --
    controlled by a fixed hand-coded rule (seek nearest visible agent) --
    regardless of `strategy`. Only agents' adaptation is toggled by
    strategy; carnivores are a constant hazard in every condition, exactly
    as in the paper's comparative study.
  - Plants: food for agents. Geometric growth up to a crowding limit,
    reseeded if the count falls below `min_plants`.
  - Trees: shelter. One occupant (an agent) per tree; occupant dies if the
    tree dies. Carnivores cannot climb or attack a sheltered agent.
  - Walls: permanent, placed at reset (border + scattered interior).
    Damage whoever walks into one; agents/carnivores "as programmed" never
    choose to walk into a wall, but the RL agents here can and do (their
    behavior isn't hard-coded to avoid it the way the paper's own
    interpretation of "carnivores as programmed" implies for carnivores
    specifically -- carnivores' hard-coded FSA below simply never targets
    a wall cell).

Observation (per agent, matching Figure 4's "INPUT TO AGENT" panel):
  [visual_N, visual_S, visual_E, visual_W, in_tree, health_norm, energy_norm]
  Each visual_* is 0 if the nearest object in that direction (within
  agent_sense_range cells) is empty/nothing, else in [0.5, 1.0] proportional
  to closeness -- matching "value zero if only empty cells visible,
  otherwise 0.5 to 1.0 proportional to the closeness of the cell." An
  explicit constant "bias" input shown in Figure 4 is instead handled as a
  standard network bias term (see networks.py) rather than an extra input
  unit -- a documented, behaviorally-equivalent simplification.

Action (matching the paper exactly): 4 discrete choices, one per compass
direction (their 2-bit encoding == a 4-way choice, implemented here as a
plain categorical/softmax network output rather than literal 2 output
bits -- same information content, simpler and more standard to train).
Effect of choosing a direction is determined by the target cell's contents,
per Figure 5's table -- see `_resolve_action` below.

--- Cooperation (C / ERLC) -- NEW, not in Ackley & Littman 1991 ---

Two additional comparative conditions, layered on top of the five above
without changing them (see `ErlWorld`'s docstring for the full mechanism,
its Houghton (2024) motivation, and known caveats/next steps). In one
sentence: a local group of agents that has collectively demonstrated
foraging, carnivore evasion, and reproduction within a recent window gets
a reproduction-energy-threshold discount, blind to which member supplied
which competency -- a group-fitness credit-sharing mechanism analogous to
Houghton's group-of-four Baldwin-Effect commentary, adapted to this
world's continuous, spatially-local, energy-gated reproduction instead of
his synchronous single-generation toy model.

--- Kin selection (K / ERLK) -- NEW, also not in Ackley & Littman ---

A second, independent cooperation mechanism, added alongside C/ERLC rather
than combined with it (keeping each mechanism testable in isolation, the
same incremental style Ackley & Littman themselves used for E/L/F/B/ERL).
Models Hamilton's rule (kin selection reduces aggression toward relatives
in proportion to relatedness and the discount's own evolved strength) using
machinery already in this world:

  - Relatedness proxy: `genome.genome_similarity` -- an RBF-kernel distance
    over each agent's BEHAVIORAL genes (eval + action weights/biases). This
    is NOT literal genealogical tracking (no parent/lineage bookkeeping
    added) -- it's a proxy that holds because agents mate locally
    (`mate_search_radius`) and reproduce via crossover+mutation, so kin
    really do tend to share more similar weights than unrelated agents.
    Simpler than adding lineage IDs, and avoids inventing a second,
    unvalidated kinship-tracking mechanism when the genome itself already
    encodes the relevant history.
  - Evolvable nepotism trait: `genome.kinship_sensitivity`, a new heritable
    scalar (sigmoid-transformed before use, so any real value is valid).
    Lets the population itself evolve toward or away from kin-biased
    leniency, rather than hard-coding a fixed discount -- the more
    interesting scientific question is whether this trait's value (and,
    per functional-constraint tracking, its own genetic stability) changes
    over generations, not just whether kin-biased damage exists.
  - Mechanism: reuses the EXISTING agent-on-agent aggression branch in
    `_resolve_agent_action` (an agent already deals `agent_attack_damage`
    to another agent it moves onto) -- under "K"/"ERLK" only, that damage
    is discounted by `sigmoid(attacker.genome.kinship_sensitivity) *
    kinship_discount_cap * genome_similarity(attacker, victim)`. No new
    action, no new event type -- same "extend an existing event" style as
    the C/ERLC competency tracking.
  - "K": like "E" (evolution alone, no learning) plus the kinship discount.
    "ERLK": like "ERL" (learning + evolution) plus the kinship discount.
    Neither combines with C/ERLC in this first pass -- see above.

KNOWN NOT YET DONE for kin selection, mirroring the C/ERLC caveats:
  - `kinship_similarity_scale` and `kinship_discount_cap` are first-guess
    values (see config.py), not tuned against any real run.
  - `kinship_sensitivity`'s own drift/constraint over generations isn't fed
    into the validated FunctionalConstraintTracker (deliberately -- see
    `Genome.flatten()`'s docstring) -- only a coarse population-mean stat
    (`genome_stats()`'s `kinship_sensitivity_mean`) exists so far. A
    dedicated tracker would be needed to ask the more interesting question
    (does nepotism itself get genetically assimilated over time).
  - The relatedness-as-genome-similarity proxy will degrade in a large,
    well-mixed population where genome similarity no longer tracks true
    recent common ancestry -- worth checking against actual lineage data
    before trusting a result, not assumed to hold indefinitely.
"""

from dataclasses import dataclass
import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.genome import (
    Genome,
    crossover,
    founder_genome,
    genome_similarity,
    mutate,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.networks import (
    action_probs,
    evaluate,
    reinforce_update,
    sample_action,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.metrics import FunctionalConstraintTracker

OBS_DIM = 7  # visual_N, visual_S, visual_E, visual_W, in_tree, health_norm, energy_norm
N_ACTIONS = 4  # N, S, E, W -- no "stay"; every step targets one adjacent cell

TERRAIN_EMPTY = 0
TERRAIN_WALL = 1
TERRAIN_TREE = 2

_DIRS = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # N, S, E, W -- index matches action id

_NEVER = -10**9  # sentinel: "this competency has never been demonstrated"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Agent:
    agent_id: int
    row: int
    col: int
    energy: float
    health: float
    in_tree: bool
    genome: Genome
    action_weights: np.ndarray  # LIVE, learned copy -- diverges from genome.action_weights over life
    action_bias: np.ndarray
    generation: int
    prev_obs: np.ndarray | None = None
    prev_action: int | None = None
    prev_eval: float | None = None
    alive: bool = True
    # --- cooperation bookkeeping (unused unless strategy is "C"/"ERLC") ---
    last_forage_step: int = _NEVER
    last_evade_step: int = _NEVER
    last_reproduce_step: int = _NEVER


@dataclass
class Carnivore:
    carnivore_id: int
    row: int
    col: int
    energy: float
    health: float
    alive: bool = True


@dataclass
class Corpse:
    kind: str  # "agent" or "carnivore"
    energy: float


class ErlWorld:
    """`config["strategy"]` selects one of Ackley & Littman's five comparative
    conditions, applied ONLY to the adaptive Agent population (Carnivores are
    always the same fixed hard-coded hazard, in every condition -- matching
    the paper, where only the "agents" are the experimental subject):
      - "ERL": full model -- genome inherited/mutated/crossed-over at
        reproduction, live action network learns during life. Default.
      - "E" (evolution alone): genome inherited/mutated as normal, but the
        live action network never learns.
      - "L" (learning alone): learns normally, but genome is cloned exactly
        (no mutation, no crossover) -- inheritance still happens, only
        genetic improvement is switched off. Matches the paper's own
        description: "L can never move beyond the randomly generated
        evaluation functions found in the initial populations."
      - "F" (neither): no learning, genome cloned exactly (no mutation/crossover).
      - "B" (Brownian/luck alone): action selection ignores the network
        (and hence the genome) entirely, choosing uniformly at random
        every step.

    Four additional conditions, NOT from Ackley & Littman -- see this
    module's docstring for both mechanisms:
      - "C" (cooperation alone): like "E" (evolution, no learning), plus a
        group-fitness reproduction-threshold discount.
      - "ERLC": like "ERL", plus the same group-fitness discount. Tests
        whether cooperation adds anything on top of learning+evolution.
      - "K" (kin selection alone): like "E", plus an evolved kinship-based
        discount on agent-on-agent attack damage.
      - "ERLK": like "ERL", plus the same kinship discount.
    For "ERL"/"E"/"L"/"F"/"B" neither the cooperation nor kin-selection code
    paths are ever entered (see `_handle_agent_reproduction`'s `coop =
    self.strategy in ("C", "ERLC")` guard and `_resolve_agent_action`'s
    `self.strategy in ("K", "ERLK")` guard) -- those five behave
    byte-identically to before either was added. C/ERLC and K/ERLK are
    independent of each other too -- neither strategy pair triggers the
    other's mechanism.
    """

    def __init__(self, config: dict, rng: np.random.Generator):
        self.cfg = config
        self.rng = rng
        self.strategy = config.get("strategy", "ERL")
        assert self.strategy in (
            "ERL", "E", "L", "F", "B", "C", "ERLC", "K", "ERLK",
        ), self.strategy
        self.grid_size = config["grid_size"]
        self.current_step = 0
        self._next_agent_id = 0
        self._next_carnivore_id = 0
        self.agents: list[Agent] = []
        self.carnivores: list[Carnivore] = []
        self.constraint_tracker = FunctionalConstraintTracker(OBS_DIM, N_ACTIONS)
        self.reset()

    # ---- setup ----

    def reset(self):
        self.current_step = 0
        self._next_agent_id = 0
        self._next_carnivore_id = 0
        self.agents = []
        self.carnivores = []
        self.occupant: dict[tuple[int, int], object] = {}  # (row,col) -> Agent | Carnivore, alive only
        self.corpses: dict[tuple[int, int], Corpse] = {}

        n = self.grid_size
        self.terrain = np.full((n, n), TERRAIN_EMPTY, dtype=np.int8)
        self._place_walls()
        self.plant = np.zeros((n, n), dtype=bool)
        self._seed_plants(force_to_min=True)
        self._seed_trees(force_to_min=True)

        for _ in range(self.cfg["n_initial_agents"]):
            self._spawn_founder_agent()
        for _ in range(self.cfg["n_initial_carnivores"]):
            self._spawn_carnivore()

    def _place_walls(self):
        n = self.grid_size
        self.terrain[0, :] = TERRAIN_WALL
        self.terrain[-1, :] = TERRAIN_WALL
        self.terrain[:, 0] = TERRAIN_WALL
        self.terrain[:, -1] = TERRAIN_WALL
        interior = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)]
        n_interior_walls = int(len(interior) * self.cfg["wall_interior_density"])
        idx = self.rng.choice(len(interior), size=n_interior_walls, replace=False)
        for i in idx:
            r, c = interior[i]
            self.terrain[r, c] = TERRAIN_WALL

    def _empty_cells(self) -> list[tuple[int, int]]:
        rows, cols = np.where(self.terrain == TERRAIN_EMPTY)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]

    def _seed_plants(self, force_to_min: bool = False):
        current = int(self.plant.sum())
        target = self.cfg["min_plants"]
        if force_to_min or current < target:
            candidates = [
                (r, c) for (r, c) in self._empty_cells()
                if not self.plant[r, c] and (r, c) not in self.occupant
            ]
            self.rng.shuffle(candidates)
            for r, c in candidates[: max(0, target - current)]:
                self.plant[r, c] = True

    def _seed_trees(self, force_to_min: bool = False):
        current = int((self.terrain == TERRAIN_TREE).sum())
        target = self.cfg["min_trees"]
        if force_to_min or current < target:
            candidates = [(r, c) for (r, c) in self._empty_cells() if not self.plant[r, c]]
            self.rng.shuffle(candidates)
            for r, c in candidates[: max(0, target - current)]:
                self.terrain[r, c] = TERRAIN_TREE

    def _random_empty_cell(self) -> tuple[int, int] | None:
        candidates = [
            (r, c) for (r, c) in self._empty_cells()
            if (r, c) not in self.occupant and not self.plant[r, c] and (r, c) not in self.corpses
        ]
        if not candidates:
            return None
        i = int(self.rng.integers(0, len(candidates)))
        return candidates[i]

    def _spawn_founder_agent(self):
        cell = self._random_empty_cell()
        if cell is None:
            return
        row, col = cell
        genome = founder_genome(OBS_DIM, N_ACTIONS, self.rng, self.cfg["founder_weight_std"])
        agent = Agent(
            agent_id=self._next_agent_id,
            row=row, col=col,
            energy=self.cfg["initial_energy_agent"],
            health=self.cfg["initial_health_agent"],
            in_tree=False,
            genome=genome,
            action_weights=genome.action_weights.copy(),
            action_bias=genome.action_bias.copy(),
            generation=0,
        )
        self._next_agent_id += 1
        self.agents.append(agent)
        self.occupant[(row, col)] = agent

    def _spawn_carnivore(self):
        cell = self._random_empty_cell()
        if cell is None:
            return
        row, col = cell
        carnivore = Carnivore(
            carnivore_id=self._next_carnivore_id,
            row=row, col=col,
            energy=self.cfg["initial_energy_carnivore"],
            health=self.cfg["initial_health_carnivore"],
        )
        self._next_carnivore_id += 1
        self.carnivores.append(carnivore)
        self.occupant[(row, col)] = carnivore

    # ---- observation (agents only -- carnivores use their own hard-coded sensing) ----

    def _visual_signal(self, row: int, col: int, drow: int, dcol: int, sense_range: int) -> float:
        for dist in range(1, sense_range + 1):
            r, c = row + drow * dist, col + dcol * dist
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                break
            if (r, c) in self.occupant or (r, c) in self.corpses or self.plant[r, c] or self.terrain[r, c] != TERRAIN_EMPTY:
                return 1.0 - 0.5 * (dist - 1) / max(sense_range - 1, 1)
        return 0.0

    def _observe_agent(self, agent: Agent) -> np.ndarray:
        obs = np.zeros(OBS_DIM)
        for i, (dr, dc) in enumerate(_DIRS):
            obs[i] = self._visual_signal(agent.row, agent.col, dr, dc, self.cfg["agent_sense_range"])
        obs[4] = 1.0 if agent.in_tree else 0.0
        obs[5] = min(agent.health / self.cfg["max_health_agent"], 1.0)
        obs[6] = min(agent.energy / self.cfg["max_energy_agent"], 1.0)
        return obs

    # ---- step ----

    def step(self):
        self.current_step += 1
        coop = self.strategy in ("C", "ERLC")
        threatened_ids = self._agents_with_carnivore_nearby() if coop else frozenset()
        self._attacked_this_step: set[int] = set()

        self._step_agents()
        self._step_carnivores()

        if coop:
            self._record_evasions(threatened_ids)

        self._handle_agent_reproduction()
        self._handle_carnivore_reproduction()
        self._decay_corpses()
        self._update_plants()
        self._update_trees()
        self._regen_health()
        if self.current_step % self.cfg["carnivore_spawn_interval"] == 0:
            self._spawn_carnivore()
        self.agents = [a for a in self.agents if a.alive]
        self.carnivores = [c for c in self.carnivores if c.alive]

    def _step_agents(self):
        order = list(self.agents)
        self.rng.shuffle(order)
        learning_enabled = self.strategy in ("ERL", "L", "ERLC", "ERLK")
        coop = self.strategy in ("C", "ERLC")
        for agent in order:
            if not agent.alive:
                continue
            obs = self._observe_agent(agent)
            e_now = evaluate(obs, agent.genome.eval_weights, agent.genome.eval_bias)

            if learning_enabled and agent.prev_obs is not None:
                reinforcement = e_now - agent.prev_eval
                reinforce_update(
                    agent.action_weights, agent.action_bias,
                    agent.prev_obs, agent.prev_action, reinforcement,
                    self.cfg["lr_positive"], self.cfg["lr_negative"],
                )

            if self.strategy == "B":
                action = int(self.rng.integers(0, N_ACTIONS))
            else:
                probs = action_probs(obs, agent.action_weights, agent.action_bias)
                action = sample_action(probs, self.rng)

            self._resolve_agent_action(agent, action, track_forage=coop)

            agent.prev_obs = obs
            agent.prev_action = action
            agent.prev_eval = e_now

            if agent.alive:
                agent.energy -= self.cfg["basal_energy_cost_agent"]
                if agent.energy <= 0 or agent.health <= 0:
                    self._kill_agent(agent)

    def _resolve_agent_action(self, agent: Agent, action: int, track_forage: bool = False):
        dr, dc = _DIRS[action]
        tr, tc = agent.row + dr, agent.col + dc
        if not (0 <= tr < self.grid_size and 0 <= tc < self.grid_size):
            return  # edge of world, wall terrain there anyway (border is all wall)

        terrain = self.terrain[tr, tc]
        occupant = self.occupant.get((tr, tc))
        corpse = self.corpses.get((tr, tc))

        if terrain == TERRAIN_WALL:
            agent.health -= self.cfg["wall_damage"]
            return

        if terrain == TERRAIN_TREE:
            if occupant is None:
                self._move_agent(agent, tr, tc)
                agent.in_tree = True
            return  # occupied tree: no effect

        # terrain == EMPTY from here on
        if occupant is not None:
            if isinstance(occupant, Carnivore):
                occupant.health -= self.cfg["agent_attack_damage"]
                if occupant.health <= 0:
                    self._kill_carnivore(occupant)
            elif isinstance(occupant, Agent) and occupant is not agent:
                damage = self.cfg["agent_attack_damage"]
                if self.strategy in ("K", "ERLK"):
                    relatedness = genome_similarity(
                        agent.genome, occupant.genome, self.cfg["kinship_similarity_scale"]
                    )
                    discount = (
                        _sigmoid(agent.genome.kinship_sensitivity)
                        * self.cfg["kinship_discount_cap"]
                        * relatedness
                    )
                    damage *= (1.0 - discount)
                occupant.health -= damage
                if occupant.health <= 0:
                    self._kill_agent(occupant)
            return

        if corpse is not None:
            bite = min(self.cfg["corpse_bite_energy"], corpse.energy)
            agent.energy = min(agent.energy + bite, self.cfg["max_energy_agent"])
            corpse.energy -= bite
            if corpse.energy <= 0:
                del self.corpses[(tr, tc)]
            if track_forage:
                agent.last_forage_step = self.current_step
            return

        if self.plant[tr, tc]:
            agent.energy = min(agent.energy + self.cfg["plant_energy"], self.cfg["max_energy_agent"])
            self.plant[tr, tc] = False
            self._move_agent(agent, tr, tc)
            if track_forage:
                agent.last_forage_step = self.current_step
            return

        # empty cell, nothing there: Enter
        self._move_agent(agent, tr, tc)

    def _move_agent(self, agent: Agent, new_row: int, new_col: int):
        del self.occupant[(agent.row, agent.col)]
        agent.row, agent.col = new_row, new_col
        agent.in_tree = self.terrain[new_row, new_col] == TERRAIN_TREE
        self.occupant[(new_row, new_col)] = agent
        agent.energy -= self.cfg["move_energy_cost_agent"]

    def _kill_agent(self, agent: Agent):
        if not agent.alive:
            return
        agent.alive = False
        self.occupant.pop((agent.row, agent.col), None)
        self.corpses[(agent.row, agent.col)] = Corpse(kind="agent", energy=self.cfg["corpse_total_energy"])

    # ---- carnivores: hard-coded FSA, never affected by `strategy` ----

    def _step_carnivores(self):
        order = list(self.carnivores)
        self.rng.shuffle(order)
        for carnivore in order:
            if not carnivore.alive:
                continue
            action = self._carnivore_fsa_action(carnivore)
            self._resolve_carnivore_action(carnivore, action)
            if carnivore.alive:
                carnivore.energy -= self.cfg["basal_energy_cost_carnivore"]
                if carnivore.energy <= 0 or carnivore.health <= 0:
                    self._kill_carnivore(carnivore)

    def _carnivore_fsa_action(self, carnivore: Carnivore) -> int:
        """Fixed rule: move toward the nearest visible agent (living or dead)
        within sense range; if none visible, move randomly. Never targets a
        wall or an occupied tree (carnivores "as programmed" don't choose
        those moves -- Figure 5's footnote)."""
        best_dir, best_signal = None, 0.0
        for i, (dr, dc) in enumerate(_DIRS):
            for dist in range(1, self.cfg["carnivore_sense_range"] + 1):
                r, c = carnivore.row + dr * dist, carnivore.col + dc * dist
                if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                    break
                occ = self.occupant.get((r, c))
                if isinstance(occ, Agent) or (r, c) in self.corpses and self.corpses[(r, c)].kind == "agent":
                    signal = 1.0 - 0.5 * (dist - 1) / max(self.cfg["carnivore_sense_range"] - 1, 1)
                    if signal > best_signal:
                        best_signal, best_dir = signal, i
                    break
                if self.terrain[r, c] != TERRAIN_EMPTY:
                    break
        if best_dir is not None:
            return best_dir
        # No target visible: move randomly, but skip walls/occupied trees when easy to check.
        candidates = list(range(N_ACTIONS))
        self.rng.shuffle(candidates)
        for i in candidates:
            dr, dc = _DIRS[i]
            r, c = carnivore.row + dr, carnivore.col + dc
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size and self.terrain[r, c] != TERRAIN_WALL:
                return i
        return int(self.rng.integers(0, N_ACTIONS))

    def _resolve_carnivore_action(self, carnivore: Carnivore, action: int):
        dr, dc = _DIRS[action]
        tr, tc = carnivore.row + dr, carnivore.col + dc
        if not (0 <= tr < self.grid_size and 0 <= tc < self.grid_size):
            return
        terrain = self.terrain[tr, tc]
        if terrain in (TERRAIN_WALL, TERRAIN_TREE):
            return  # carnivores never choose these moves ("as programmed"), and can't climb

        occupant = self.occupant.get((tr, tc))
        corpse = self.corpses.get((tr, tc))

        if isinstance(occupant, Agent):
            occupant.health -= self.cfg["carnivore_attack_damage"]
            if self.strategy in ("C", "ERLC"):
                self._attacked_this_step.add(occupant.agent_id)
            if occupant.health <= 0:
                self._kill_agent(occupant)
            return
        if isinstance(occupant, Carnivore):
            return  # carnivores don't fight each other

        if corpse is not None:
            bite = min(self.cfg["corpse_bite_energy"], corpse.energy)
            carnivore.energy = min(carnivore.energy + bite, self.cfg["max_energy_carnivore"])
            corpse.energy -= bite
            if corpse.energy <= 0:
                del self.corpses[(tr, tc)]
            return

        # Empty cell (carnivores walk over plants without eating them -- Figure 5: "Enter").
        del self.occupant[(carnivore.row, carnivore.col)]
        carnivore.row, carnivore.col = tr, tc
        self.occupant[(tr, tc)] = carnivore
        carnivore.energy -= self.cfg["move_energy_cost_carnivore"]

    def _kill_carnivore(self, carnivore: Carnivore):
        if not carnivore.alive:
            return
        carnivore.alive = False
        self.occupant.pop((carnivore.row, carnivore.col), None)
        self.corpses[(carnivore.row, carnivore.col)] = Corpse(kind="carnivore", energy=self.cfg["corpse_total_energy"])

    # ---- cooperation bookkeeping (gated to "C"/"ERLC" by callers) ----

    def _agents_with_carnivore_nearby(self) -> frozenset[int]:
        """Ground-truth (not a perceptual channel) bookkeeping helper: which
        agents had a living carnivore within `agent_sense_range` (Chebyshev)
        at the START of this step, before anyone moved. O(agents *
        carnivores) -- cheap given carnivore counts stay small relative to
        agents in this world (spawn-rate-limited, not population-limited).
        """
        if not self.carnivores:
            return frozenset()
        r = self.cfg["agent_sense_range"]
        threatened = set()
        for agent in self.agents:
            if not agent.alive:
                continue
            for c in self.carnivores:
                if not c.alive:
                    continue
                if abs(c.row - agent.row) <= r and abs(c.col - agent.col) <= r:
                    threatened.add(agent.agent_id)
                    break
        return frozenset(threatened)

    def _record_evasions(self, threatened_ids: frozenset[int]):
        """An agent 'evades' this step if it was threatened at the step's
        start and survived the carnivore phase without being attacked.
        Called after `_step_carnivores` so `_attacked_this_step` is final."""
        for agent in self.agents:
            if agent.alive and agent.agent_id in threatened_ids and agent.agent_id not in self._attacked_this_step:
                agent.last_evade_step = self.current_step

    def _agent_group_is_cooperative_fit(self, agent: Agent) -> bool:
        """True if `agent`'s local group (itself + living agents within
        `cooperation_radius`, Chebyshev) has collectively demonstrated all
        three tracked competencies within `competency_window` steps, by
        ANY member -- blind to which member supplied which competency,
        mirroring Houghton (2024)'s group-fitness credit assignment.

        Implemented as a bounded box-scan over `self.occupant` (O(radius^2)
        cell lookups, radius=cooperation_radius) rather than a linear scan
        over `self.agents` (O(population)) -- the same class of fix already
        applied to `_observe`/`_try_eat` in this world's earlier
        performance pass (see RESULTS.md section 2), applied here from the
        start rather than as a later retrofit.
        """
        radius = self.cfg["cooperation_radius"]
        cutoff = self.current_step - self.cfg["competency_window"]
        foraged = agent.last_forage_step >= cutoff
        evaded = agent.last_evade_step >= cutoff
        reproduced = agent.last_reproduce_step >= cutoff
        if foraged and evaded and reproduced:
            return True
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = agent.row + dr, agent.col + dc
                if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                    continue
                other = self.occupant.get((r, c))
                if isinstance(other, Agent) and other.alive:
                    if other.last_forage_step >= cutoff:
                        foraged = True
                    if other.last_evade_step >= cutoff:
                        evaded = True
                    if other.last_reproduce_step >= cutoff:
                        reproduced = True
                    if foraged and evaded and reproduced:
                        return True
        return foraged and evaded and reproduced

    # ---- reproduction ----

    def _handle_agent_reproduction(self):
        coop = self.strategy in ("C", "ERLC")
        newborns = []
        for agent in self.agents:
            if not agent.alive:
                continue
            threshold = self.cfg["reproduction_energy_threshold_agent"]
            if coop and self._agent_group_is_cooperative_fit(agent):
                threshold *= (1.0 - self.cfg["coop_threshold_discount_frac"])
            if agent.energy < threshold:
                continue
            if len(self.agents) + len(newborns) >= self.cfg["max_population_cap"]:
                continue
            cell = self._nearest_empty_adjacent(agent.row, agent.col)
            if cell is None:
                continue
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
            if coop:
                agent.last_reproduce_step = self.current_step
            row, col = cell
            child = Agent(
                agent_id=self._next_agent_id,
                row=row, col=col,
                energy=self.cfg["initial_energy_agent"],
                health=self.cfg["initial_health_agent"],
                in_tree=False,
                genome=child_genome,
                action_weights=child_genome.action_weights.copy(),
                action_bias=child_genome.action_bias.copy(),
                generation=agent.generation + 1,
            )
            self._next_agent_id += 1
            newborns.append(child)
            self.occupant[(row, col)] = child
        self.agents.extend(newborns)

    def _handle_carnivore_reproduction(self):
        newborns = []
        for carnivore in self.carnivores:
            if not carnivore.alive or carnivore.energy < self.cfg["carnivore_reproduction_energy_threshold"]:
                continue
            if len(self.carnivores) + len(newborns) >= self.cfg["max_population_cap"]:
                continue
            cell = self._nearest_empty_adjacent(carnivore.row, carnivore.col)
            if cell is None:
                continue
            carnivore.energy -= self.cfg["carnivore_reproduction_energy_cost"]
            row, col = cell
            child = Carnivore(
                carnivore_id=self._next_carnivore_id,
                row=row, col=col,
                energy=self.cfg["initial_energy_carnivore"],
                health=self.cfg["initial_health_carnivore"],
            )
            self._next_carnivore_id += 1
            newborns.append(child)
            self.occupant[(row, col)] = child
        self.carnivores.extend(newborns)

    def _nearest_empty_adjacent(self, row: int, col: int) -> tuple[int, int] | None:
        for dr, dc in _DIRS:
            r, c = row + dr, col + dc
            if (
                0 <= r < self.grid_size and 0 <= c < self.grid_size
                and self.terrain[r, c] == TERRAIN_EMPTY
                and (r, c) not in self.occupant and (r, c) not in self.corpses and not self.plant[r, c]
            ):
                return (r, c)
        return None

    def _nearest_mate(self, agent: Agent) -> Agent | None:
        best, best_dist = None, self.cfg["mate_search_radius"] + 1
        for other in self.agents:
            if other is agent or not other.alive:
                continue
            dist = abs(other.row - agent.row) + abs(other.col - agent.col)
            if dist <= self.cfg["mate_search_radius"] and dist < best_dist:
                best, best_dist = other, dist
        return best

    # ---- world upkeep ----

    def _decay_corpses(self):
        # Corpses lose a small amount of energy each step even if unbothered
        # ("simply decay until their energy is gone").
        decay = 0.05
        for pos in list(self.corpses.keys()):
            self.corpses[pos].energy -= decay
            if self.corpses[pos].energy <= 0:
                del self.corpses[pos]

    def _update_plants(self):
        candidates = [
            (r, c) for (r, c) in self._empty_cells()
            if not self.plant[r, c] and (r, c) not in self.occupant and (r, c) not in self.corpses
        ]
        limit = int(len(self._empty_cells()) * self.cfg["plant_crowding_limit_frac"])
        current = int(self.plant.sum())
        if current < limit:
            self.rng.shuffle(candidates)
            grow_mask = self.rng.random(len(candidates)) < self.cfg["plant_growth_prob"]
            for (r, c), grow in zip(candidates, grow_mask):
                if grow and int(self.plant.sum()) < limit:
                    self.plant[r, c] = True
        self._seed_plants(force_to_min=False)

    def _update_trees(self):
        tree_cells = [(int(r), int(c)) for r, c in zip(*np.where(self.terrain == TERRAIN_TREE))]
        for (r, c) in tree_cells:
            if self.rng.random() < self.cfg["tree_death_prob"]:
                occ = self.occupant.get((r, c))
                if isinstance(occ, Agent):
                    self._kill_agent(occ)
                self.terrain[r, c] = TERRAIN_EMPTY
        empty_cells = self._empty_cells()
        self.rng.shuffle(empty_cells)
        for (r, c) in empty_cells:
            if self.plant[r, c] or (r, c) in self.occupant:
                continue
            if self.rng.random() < self.cfg["tree_birth_prob"]:
                self.terrain[r, c] = TERRAIN_TREE
        self._seed_trees(force_to_min=False)

    def _regen_health(self):
        for a in self.agents:
            if a.alive and a.health < self.cfg["max_health_agent"]:
                a.health = min(a.health + self.cfg["health_regen_agent"], self.cfg["max_health_agent"])
        for c in self.carnivores:
            if c.alive and c.health < self.cfg["max_health_carnivore"]:
                c.health = min(c.health + self.cfg["health_regen_carnivore"], self.cfg["max_health_carnivore"])

    # ---- summary ----

    def population_counts(self) -> dict[str, int]:
        return {
            "agent": sum(1 for a in self.agents if a.alive),
            "carnivore": sum(1 for c in self.carnivores if c.alive),
        }

    def genome_stats(self) -> dict[str, float]:
        genomes = [a.genome for a in self.agents if a.alive]
        if not genomes:
            return {
                "eval_weight_absmean": float("nan"),
                "action_weight_absmean": float("nan"),
                "kinship_sensitivity_mean": float("nan"),
            }
        eval_vals = np.concatenate([g.eval_weights for g in genomes])
        action_vals = np.concatenate([g.action_weights.ravel() for g in genomes])
        kinship_vals = np.array([g.kinship_sensitivity for g in genomes])
        return {
            "eval_weight_absmean": float(np.mean(np.abs(eval_vals))),
            "action_weight_absmean": float(np.mean(np.abs(action_vals))),
            # Population-mean nepotism trait -- a coarse signal only (see
            # module docstring for why this isn't fed into the validated
            # FunctionalConstraintTracker).
            "kinship_sensitivity_mean": float(np.mean(_sigmoid(kinship_vals))),
        }
