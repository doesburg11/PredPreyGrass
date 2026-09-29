"""World AL (Ackley & Littman 1991), forked from eco_evolutionary_erl_baldwin
as the base for a gradual predator-prey COEVOLUTION extension.

Step 0 (config_step0): behaviorally and RNG-identical to the world that
produced eco_evolutionary_erl_baldwin's §9 comparative study (commit
47534a0) -- the C/ERLC, K/ERLK and S/ERLS conditions (documented dead ends,
erl_baldwin RESULTS.md §12-16) are removed, and genome.py is the §9-era
version. Pinned by tests/test_step0_reproduces_study.py against §9's actual
per-seed extinction steps.

Step 1 (config_step1): carnivores stay the same fixed, hand-coded,
non-adaptive FSA, but their numbers are regulated by prey supply instead of
by immigration -- `carnivore_spawn_interval = 0` switches the paper's fixed
"one new carnivore every 200 steps" off, and carnivore reproduction is made
reachable through individual energy budgets. See config.py.

Entities (unchanged from erl_baldwin -- see its world.py for the full notes):
  - Agent: the ADAPTIVE species. Genome-initialized action + eval networks,
    local reinforcement learning during life (subject to `strategy`).
    Omnivorous: eats plants and corpses.
  - Carnivore: NON-adaptive. No genome, no network, no learning -- fixed
    rule "seek nearest visible agent", regardless of `strategy`. Eats
    corpses; reproduces when its energy reaches a threshold.
  - Plants, Trees (shelter), Walls -- as in erl_baldwin.

Observation (per agent):
  [visual_N, visual_S, visual_E, visual_W, in_tree, health_norm, energy_norm]
Action: 4 discrete choices, one per compass direction; effect determined by
the target cell's contents (Ackley & Littman's Figure 5).

Death-cause bookkeeping (NEW, no RNG use -- step-0 trajectories unaffected):
`self.deaths["agent"|"carnivore"][cause]` counts every death by cause, so a
run can say what actually regulates each population (predation vs.
starvation vs. injury). Agent causes: "carnivore", "agent_attack",
"starvation", "wounds" (health depleted at end of turn -- walls, or
earlier non-lethal attacks), "tree_fall". Carnivore causes: "agent_attack",
"starvation", "wounds". `self.carnivore_births` / `self.carnivore_immigrants`
separate reproduction from immigration.

Step 2 (`carnivore_mode`, default "fsa" = the hand-coded rule, unchanged):
  - "genome": each carnivore carries a Genome whose single-layer action
    network picks its move (evolution only, no learning). Founder weights
    are SEEDED to approximate the hand-coded rule -- pursue the strongest prey
    signal, avoid blocked cells -- plus per-founder Gaussian variation, so
    founders start competent and evolution can improve on or drift from the
    rule. (Approximate, not identical: actions are sampled from a softmax, so
    ties and near-ties are resolved stochastically, and trees count as blocked
    where the rule only avoids walls.) Reproduction is sexual like the prey's:
    crossover with the nearest carnivore within `mate_search_radius` (else a
    copy), then mutation.
  - "genome_neutral": the neutral-MARKER control. Inheritance is exactly the
    same code as "genome" (same parents, crossover, mutation, offspring
    credit, RNG draws), but the genome is not expressed: every carnivore acts
    with the same fixed canonical network (the seed weights, no founder
    variation). Genome change is therefore pure drift under the same
    demography. (Drawing donor genomes from random living carnivores would not
    be neutral -- better-surviving genomes stay in that pool longer -- which is
    why the genome is decoupled from behavior instead.)
  - "genome_nonheritable": the matched control (added after the step-2 pilot,
    whose neutral control started ahead because it expressed the noise-free
    seed). Founders express their own seed+noise genome exactly as under
    "genome"; every newborn expresses a FRESH seed+noise network drawn at
    birth. Same phenotypic variation and starting competence, but behavior is
    not inherited, so it cannot respond to selection. The genome is still
    inherited as a passive marker.
  - "mixed" (competition test): two fixed carnivore types, resident (the
    seed network) and mutant (seed network with `mixed_mutant_pursuit_weight`
    / `mixed_mutant_block_weight`). Founders and immigrants alternate type
    by id (exactly 50/50, no RNG); offspring inherit the parent's type. The
    mutant frequency over time measures within-population selection between
    the two behaviors. `self.type_births/type_steps/type_kills` hold per-type totals.
  Carnivore observation (CARN_OBS_DIM = 10): prey signal N/S/E/W (exactly
  what the hand-coded rule sees: nearest living agent or agent corpse within
  `carnivore_sense_range`, blocked by terrain), adjacent cell blocked N/S/E/W
  (wall, tree or edge), energy_norm, health_norm.
  With `carnivore_obs = "rich"` (18 inputs) the prey signal is split into
  living / sheltered-in-tree / corpse channels -- whichever the line of sight
  hits first sets its channel -- then blocked x4, energy, health. The rich seed
  weights all three prey channels equally, so it behaves exactly like the
  basic seed. Note: in a single-layer network, energy/health add the same
  amount to every action's logit, so they cannot change the chosen direction.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.genome import (
    Genome,
    crossover,
    founder_genome,
    mutate,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.networks import (
    action_probs,
    evaluate,
    reinforce_update,
    sample_action,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_coevolution.metrics import FunctionalConstraintTracker

OBS_DIM = 7  # visual_N, visual_S, visual_E, visual_W, in_tree, health_norm, energy_norm
N_ACTIONS = 4  # N, S, E, W -- no "stay"; every step targets one adjacent cell
CARN_OBS_DIM = 10  # "basic" layout: prey signal x4, adjacent blocked x4, energy_norm, health_norm (step 2)
# Carnivore input layouts (`carnivore_obs`). Row offsets of each 4-direction block.
# "rich" splits the prey signal by kind: living (attackable), sheltered in a tree
# (visible but unattackable), agent corpse (food without a fight).
CARN_OBS_LAYOUTS = {
    "basic": {"dim": 10, "prey": {"pursuit": 0}, "block": 4},
    "rich": {"dim": 18, "prey": {"living": 0, "sheltered": 4, "corpse": 8}, "block": 12},
}

TERRAIN_EMPTY = 0
TERRAIN_WALL = 1
TERRAIN_TREE = 2

_DIRS = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # N, S, E, W -- index matches action id

STRATEGIES = ("ERL", "E", "L", "F", "B")
CARNIVORE_MODES = ("fsa", "fsa_skip_sheltered", "genome", "genome_neutral", "genome_nonheritable", "mixed")


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
    # --- lineage-fitness bookkeeping (see metrics.lineage_record) ---
    born_step: int = 0
    offspring_count: int = 0


@dataclass
class Carnivore:
    carnivore_id: int
    row: int
    col: int
    energy: float
    health: float
    alive: bool = True
    # --- step 2 (carnivore_mode "genome"/"genome_neutral"); None/0 under "fsa" ---
    genome: Genome | None = None
    generation: int = 0
    born_step: int = 0
    offspring_count: int = 0
    kills: int = 0
    phenotype: Genome | None = None  # expressed network under "genome_nonheritable" only
    ctype: int = 0  # "mixed" only: 0 = resident, 1 = mutant


@dataclass
class Corpse:
    kind: str  # "agent" or "carnivore"
    energy: float


class ErlWorld:
    """`config["strategy"]` selects one of Ackley & Littman's five comparative
    conditions, applied ONLY to the adaptive Agent population (Carnivores are
    always the same fixed hard-coded rule, in every condition):
      - "ERL": genome inherited/mutated/crossed-over at reproduction, live
        action network learns during life. Default.
      - "E" (evolution alone): genome inherited/mutated, no learning.
      - "L" (learning alone): learns, but genome is cloned exactly.
      - "F" (neither): no learning, genome cloned exactly.
      - "B" (Brownian/luck alone): uniformly random actions.
    """

    def __init__(self, config: dict, rng: np.random.Generator):
        self.cfg = config
        self.rng = rng
        self.strategy = config.get("strategy", "ERL")
        assert self.strategy in STRATEGIES, self.strategy
        self.carnivore_mode = config.get("carnivore_mode", "fsa")
        assert self.carnivore_mode in CARNIVORE_MODES, self.carnivore_mode
        self.carn_layout = CARN_OBS_LAYOUTS[config.get("carnivore_obs", "basic")]
        self.carn_obs_dim = self.carn_layout["dim"]
        self.obs_dim = OBS_DIM
        self.grid_size = config["grid_size"]
        self.current_step = 0
        self._next_agent_id = 0
        self._next_carnivore_id = 0
        self.agents: list[Agent] = []
        self.carnivores: list[Carnivore] = []
        self.constraint_tracker = FunctionalConstraintTracker(self.obs_dim, N_ACTIONS)
        # Optional callback(agent, death_step) fired from `_kill_agent`, before the
        # agent is dropped from `self.agents` -- lets a caller log per-agent lineage
        # data without World doing any file IO itself.
        self.on_agent_death = None
        self.on_carnivore_death = None  # same idea, callback(carnivore, death_step)
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
        self.deaths: dict[str, Counter] = {"agent": Counter(), "carnivore": Counter()}
        self.carnivore_births = 0
        self.carnivore_immigrants = 0
        self.carnivore_kills = 0  # agents killed by a carnivore attack
        self.carnivore_steps = 0  # carnivore-steps acted, the denominator for kill rate
        self.type_births = [0, 0]  # "mixed" per-type totals: [resident, mutant]
        self.type_steps = [0, 0]
        self.type_kills = [0, 0]

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
        genome = founder_genome(self.obs_dim, N_ACTIONS, self.rng, self.cfg["founder_weight_std"])
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
            born_step=self.current_step,
        )
        self._next_agent_id += 1
        self.agents.append(agent)
        self.occupant[(row, col)] = agent

    def _spawn_carnivore(self) -> bool:
        cell = self._random_empty_cell()
        if cell is None:
            return False
        row, col = cell
        carnivore = Carnivore(
            carnivore_id=self._next_carnivore_id,
            row=row, col=col,
            energy=self.cfg["initial_energy_carnivore"],
            health=self.cfg["initial_health_carnivore"],
            genome=self._founder_carnivore_genome() if self.carnivore_mode.startswith("genome") else None,
            born_step=self.current_step,
        )
        if self.carnivore_mode == "genome_nonheritable":
            carnivore.phenotype = carnivore.genome  # founders express their own seed+noise, as under "genome"
        if self.carnivore_mode == "mixed":
            carnivore.ctype = carnivore.carnivore_id % 2
        self._next_carnivore_id += 1
        self.carnivores.append(carnivore)
        self.occupant[(row, col)] = carnivore
        return True

    def _seed_weights(self, prefix: str) -> np.ndarray:
        """Seed action weights for the current input layout, read from config keys
        `{prefix}_<channel>_weight` and `{prefix}_block_weight` (prefix
        "carnivore_seed" or "mixed_mutant"): prey channel in direction i ->
        action i, blocked in direction i -> action i."""
        weights = np.zeros((self.carn_obs_dim, N_ACTIONS))
        for channel, offset in self.carn_layout["prey"].items():
            for i in range(N_ACTIONS):
                weights[offset + i, i] = self.cfg[f"{prefix}_{channel}_weight"]
        for i in range(N_ACTIONS):
            weights[self.carn_layout["block"] + i, i] = self.cfg[f"{prefix}_block_weight"]
        return weights

    def _fixed_network(self, prefix: str) -> Genome:
        return Genome(np.zeros(self.carn_obs_dim), 0.0, self._seed_weights(prefix), np.zeros(N_ACTIONS))

    @property
    def _mixed_type_genomes(self) -> tuple[Genome, Genome]:
        """("mixed") resident = the seed network, mutant = seed with its own weights. Cached, no RNG."""
        cached = getattr(self, "_mixed_cache", None)
        if cached is None:
            cached = (self._fixed_network("carnivore_seed"), self._fixed_network("mixed_mutant"))
            self._mixed_cache = cached
        return cached

    @property
    def _canonical_carnivore_genome(self) -> Genome:
        """The seed network with no founder variation -- what every carnivore
        expresses under "genome_neutral". Built without the RNG, cached."""
        cached = getattr(self, "_canonical_cache", None)
        if cached is None:
            cached = self._fixed_network("carnivore_seed")
            self._canonical_cache = cached
        return cached

    def _founder_carnivore_genome(self) -> Genome:
        """Seeded to approximate the hand-coded rule (see `_seed_weights`), plus
        N(0, carnivore_founder_weight_std) on every weight and bias. The eval
        network is random and unused until carnivores learn (step 3)."""
        genome = founder_genome(self.carn_obs_dim, N_ACTIONS, self.rng, self.cfg["carnivore_founder_weight_std"])
        genome.action_weights += self._seed_weights("carnivore_seed")
        return genome

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
        obs = np.zeros(self.obs_dim)
        for i, (dr, dc) in enumerate(_DIRS):
            obs[i] = self._visual_signal(agent.row, agent.col, dr, dc, self.cfg["agent_sense_range"])
        obs[4] = 1.0 if agent.in_tree else 0.0
        obs[5] = min(agent.health / self.cfg["max_health_agent"], 1.0)
        obs[6] = min(agent.energy / self.cfg["max_energy_agent"], 1.0)
        return obs

    # ---- step ----

    def step(self):
        self.current_step += 1
        self._step_agents()
        self._step_carnivores()
        self._handle_agent_reproduction()
        self._handle_carnivore_reproduction()
        self._decay_corpses()
        self._update_plants()
        self._update_trees()
        self._regen_health()
        if self.immigration_active() and self.current_step % self.cfg["carnivore_spawn_interval"] == 0:
            if self._spawn_carnivore():
                self.carnivore_immigrants += 1
        self.agents = [a for a in self.agents if a.alive]
        self.carnivores = [c for c in self.carnivores if c.alive]

    def immigration_active(self) -> bool:
        """Carnivore immigration (one every `carnivore_spawn_interval` steps) is on
        while the interval is > 0 and the current step is at or before
        `carnivore_immigration_until` (absent/None = forever, as in step 0).
        Once it's off, carnivore extinction is permanent."""
        if self.cfg["carnivore_spawn_interval"] <= 0:
            return False
        until = self.cfg.get("carnivore_immigration_until")
        return until is None or self.current_step <= until

    def _step_agents(self):
        order = list(self.agents)
        self.rng.shuffle(order)
        learning_enabled = self.strategy in ("ERL", "L")
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

            self._resolve_agent_action(agent, action)

            agent.prev_obs = obs
            agent.prev_action = action
            agent.prev_eval = e_now

            if agent.alive:
                agent.energy -= self.cfg["basal_energy_cost_agent"]
                # Health first: any lethal damage this turn happened before
                # the end-of-turn basal energy charge.
                if agent.health <= 0:
                    self._kill_agent(agent, "wounds")
                elif agent.energy <= 0:
                    self._kill_agent(agent, "starvation")

    def _resolve_agent_action(self, agent: Agent, action: int):
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
                    self._kill_carnivore(occupant, "agent_attack")
            elif isinstance(occupant, Agent) and occupant is not agent:
                occupant.health -= self.cfg["agent_attack_damage"]
                if occupant.health <= 0:
                    self._kill_agent(occupant, "agent_attack")
            return

        if corpse is not None:
            bite = min(self.cfg["corpse_bite_energy"], corpse.energy)
            agent.energy = min(agent.energy + bite, self.cfg["max_energy_agent"])
            corpse.energy -= bite
            if corpse.energy <= 0:
                del self.corpses[(tr, tc)]
            return

        if self.plant[tr, tc]:
            agent.energy = min(agent.energy + self.cfg["plant_energy"], self.cfg["max_energy_agent"])
            self.plant[tr, tc] = False
            self._move_agent(agent, tr, tc)
            return

        # empty cell, nothing there: Enter
        self._move_agent(agent, tr, tc)

    def _move_agent(self, agent: Agent, new_row: int, new_col: int):
        del self.occupant[(agent.row, agent.col)]
        agent.row, agent.col = new_row, new_col
        agent.in_tree = self.terrain[new_row, new_col] == TERRAIN_TREE
        self.occupant[(new_row, new_col)] = agent
        agent.energy -= self.cfg["move_energy_cost_agent"]

    def _kill_agent(self, agent: Agent, cause: str):
        if not agent.alive:
            return
        agent.alive = False
        self.deaths["agent"][cause] += 1
        self.occupant.pop((agent.row, agent.col), None)
        self.corpses[(agent.row, agent.col)] = Corpse(kind="agent", energy=self.cfg["corpse_total_energy"])
        # Death state is committed above BEFORE notifying the callback: if the
        # installed observer raises (e.g. lineage-logging IO failure), the kill
        # itself must still have taken effect.
        if self.on_agent_death is not None:
            self.on_agent_death(agent, self.current_step)

    # ---- carnivores: hard-coded FSA or (step 2) genome network; never affected by `strategy` ----

    def _step_carnivores(self):
        order = list(self.carnivores)
        self.rng.shuffle(order)
        for carnivore in order:
            if not carnivore.alive:
                continue
            self.carnivore_steps += 1
            if self.carnivore_mode in ("fsa", "fsa_skip_sheltered"):
                action = self._carnivore_fsa_action(carnivore)
            elif self.carnivore_mode == "mixed":
                self.type_steps[carnivore.ctype] += 1
                net = self._mixed_type_genomes[carnivore.ctype]
                probs = action_probs(self._observe_carnivore(carnivore), net.action_weights, net.action_bias)
                action = sample_action(probs, self.rng)
            else:
                obs = self._observe_carnivore(carnivore)
                if self.carnivore_mode == "genome":
                    expressed = carnivore.genome
                elif self.carnivore_mode == "genome_nonheritable":
                    expressed = carnivore.phenotype
                else:
                    expressed = self._canonical_carnivore_genome
                probs = action_probs(obs, expressed.action_weights, expressed.action_bias)
                action = sample_action(probs, self.rng)
            self._resolve_carnivore_action(carnivore, action)
            if carnivore.alive:
                carnivore.energy -= self.cfg["basal_energy_cost_carnivore"]
                if carnivore.health <= 0:
                    self._kill_carnivore(carnivore, "wounds")
                elif carnivore.energy <= 0:
                    self._kill_carnivore(carnivore, "starvation")

    def _carnivore_fsa_action(self, carnivore: Carnivore) -> int:
        """Fixed rule: move toward the nearest visible agent (living or dead)
        within sense range; if none visible, move randomly. Never targets a
        wall or an occupied tree (carnivores "as programmed" don't choose
        those moves -- Figure 5's footnote)."""
        best_dir, best_signal = None, 0.0
        # "fsa_skip_sheltered" (headroom probe only): ignore agents sheltering
        # in trees, which carnivores cannot attack -- information the step-2
        # network's inputs don't contain.
        skip_sheltered = self.carnivore_mode == "fsa_skip_sheltered"
        for i, (dr, dc) in enumerate(_DIRS):
            for dist in range(1, self.cfg["carnivore_sense_range"] + 1):
                r, c = carnivore.row + dr * dist, carnivore.col + dc * dist
                if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                    break
                occ = self.occupant.get((r, c))
                if skip_sheltered and isinstance(occ, Agent) and occ.in_tree:
                    break  # headroom probe: an unattackable target is treated as an obstacle
                if isinstance(occ, Agent) or (r, c) in self.corpses and self.corpses[(r, c)].kind == "agent":
                    signal = 1.0 - 0.5 * (dist - 1) / max(self.cfg["carnivore_sense_range"] - 1, 1)
                    if signal > best_signal:
                        best_signal, best_dir = signal, i
                    break
                if self.terrain[r, c] != TERRAIN_EMPTY:
                    break
        if best_dir is not None:
            return best_dir
        # No target visible: move randomly, but skip walls when easy to check.
        candidates = list(range(N_ACTIONS))
        self.rng.shuffle(candidates)
        for i in candidates:
            dr, dc = _DIRS[i]
            r, c = carnivore.row + dr, carnivore.col + dc
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size and self.terrain[r, c] != TERRAIN_WALL:
                return i
        return int(self.rng.integers(0, N_ACTIONS))

    def _observe_carnivore(self, carnivore: Carnivore) -> np.ndarray:
        """Step 2 carnivore input -- see module docstring. The prey signal uses
        exactly the hand-coded rule's line of sight (`_carnivore_fsa_action`);
        under the "rich" layout the first prey object hit sets its own channel."""
        layout = self.carn_layout
        rich = "living" in layout["prey"]
        obs = np.zeros(self.carn_obs_dim)
        sense = self.cfg["carnivore_sense_range"]
        for i, (dr, dc) in enumerate(_DIRS):
            for dist in range(1, sense + 1):
                r, c = carnivore.row + dr * dist, carnivore.col + dc * dist
                if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                    break
                occ = self.occupant.get((r, c))
                is_agent = isinstance(occ, Agent)
                if is_agent or (r, c) in self.corpses and self.corpses[(r, c)].kind == "agent":
                    signal = 1.0 - 0.5 * (dist - 1) / max(sense - 1, 1)
                    if not rich:
                        channel = "pursuit"
                    elif is_agent:
                        channel = "sheltered" if occ.in_tree else "living"
                    else:
                        channel = "corpse"
                    obs[layout["prey"][channel] + i] = signal
                    break
                if self.terrain[r, c] != TERRAIN_EMPTY:
                    break
            r, c = carnivore.row + dr, carnivore.col + dc
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size) or self.terrain[r, c] != TERRAIN_EMPTY:
                obs[layout["block"] + i] = 1.0
        obs[-2] = min(carnivore.energy / self.cfg["max_energy_carnivore"], 1.0)
        obs[-1] = min(carnivore.health / self.cfg["max_health_carnivore"], 1.0)
        return obs

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
            if occupant.health <= 0:
                self._kill_agent(occupant, "carnivore")
                carnivore.kills += 1
                self.carnivore_kills += 1
                self.type_kills[carnivore.ctype] += 1
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

    def _kill_carnivore(self, carnivore: Carnivore, cause: str):
        if not carnivore.alive:
            return
        carnivore.alive = False
        self.deaths["carnivore"][cause] += 1
        self.occupant.pop((carnivore.row, carnivore.col), None)
        self.corpses[(carnivore.row, carnivore.col)] = Corpse(kind="carnivore", energy=self.cfg["corpse_total_energy"])
        if self.on_carnivore_death is not None:
            self.on_carnivore_death(carnivore, self.current_step)

    # ---- reproduction ----

    def _handle_agent_reproduction(self):
        newborns = []
        for agent in self.agents:
            if not agent.alive:
                continue
            if agent.energy < self.cfg["reproduction_energy_threshold_agent"]:
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
                child_genome = mutate(child_genome, self.rng, self.cfg["mutation_rate"], self.cfg["mutation_std"])
            self.constraint_tracker.record(agent.genome.flatten(), child_genome.flatten())

            agent.energy -= self.cfg["reproduction_energy_cost_agent"]
            agent.offspring_count += 1
            if mate is not None:
                # Crossover mixes ~half of each genome site from `mate` into the
                # child -- both genetic parents are credited for lineage fitness.
                mate.offspring_count += 1
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
                born_step=self.current_step,
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
            cost = self.cfg["carnivore_reproduction_energy_cost"]
            carnivore.energy -= cost
            # Step 0 gives every newborn a fixed initial_energy_carnivore
            # regardless of what the parent paid -- a net energy SOURCE whenever
            # cost < initial_energy_carnivore. Energy-conserving birth passes
            # exactly the parent's payment to the child instead.
            child_energy = cost if self.cfg.get("carnivore_energy_conserving_birth") else self.cfg["initial_energy_carnivore"]
            child_genome = None
            if self.carnivore_mode.startswith("genome"):
                child_genome = self._carnivore_child_genome(carnivore)
            row, col = cell
            child = Carnivore(
                carnivore_id=self._next_carnivore_id,
                row=row, col=col,
                energy=child_energy,
                health=self.cfg["initial_health_carnivore"],
                genome=child_genome,
                generation=carnivore.generation + 1,
                born_step=self.current_step,
                phenotype=self._founder_carnivore_genome() if self.carnivore_mode == "genome_nonheritable" else None,
                ctype=carnivore.ctype,
            )
            self.type_births[carnivore.ctype] += 1
            self._next_carnivore_id += 1
            self.carnivore_births += 1
            newborns.append(child)
            self.occupant[(row, col)] = child
        self.carnivores.extend(newborns)

    def _carnivore_child_genome(self, parent: Carnivore) -> Genome:
        """Sexual like the prey: crossover with the nearest carnivore within
        `mate_search_radius` (else a copy), then mutation. Identical under
        "genome" and "genome_neutral" -- the control differs only in whether
        the genome is expressed (see `_step_carnivores`)."""
        mate = self._nearest_carnivore_mate(parent)
        parent.offspring_count += 1
        if mate is not None:
            mate.offspring_count += 1
        genome = crossover(parent.genome, mate.genome, self.rng) if mate is not None else parent.genome.copy()
        return mutate(genome, self.rng, self.cfg["carnivore_mutation_rate"], self.cfg["carnivore_mutation_std"])

    def _nearest_carnivore_mate(self, carnivore: Carnivore) -> Carnivore | None:
        best, best_dist = None, self.cfg["mate_search_radius"] + 1
        for other in self.carnivores:
            if other is carnivore or not other.alive:
                continue
            dist = abs(other.row - carnivore.row) + abs(other.col - carnivore.col)
            if dist <= self.cfg["mate_search_radius"] and dist < best_dist:
                best, best_dist = other, dist
        return best

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
                    self._kill_agent(occ, "tree_fall")
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
            return {"eval_weight_absmean": float("nan"), "action_weight_absmean": float("nan")}
        eval_vals = np.concatenate([g.eval_weights for g in genomes])
        action_vals = np.concatenate([g.action_weights.ravel() for g in genomes])
        return {
            "eval_weight_absmean": float(np.mean(np.abs(eval_vals))),
            "action_weight_absmean": float(np.mean(np.abs(action_vals))),
        }

    def carnivore_genome_stats(self) -> dict[str, float]:
        """Population means of the two seeded behaviors (step 2): `pursuit` =
        mean prey-signal->same-direction weight, `avoid` = mean
        blocked->same-direction weight (seeded negative). NaN under "fsa"."""
        genomes = [c.genome for c in self.carnivores if c.alive and c.genome is not None]
        if not genomes:
            return {"carn_pursuit": float("nan"), "carn_avoid": float("nan"), "carn_generation": float("nan")}
        idx = np.arange(N_ACTIONS)
        prey_rows = [off + idx for off in self.carn_layout["prey"].values()]
        block_rows = self.carn_layout["block"] + idx
        return {
            "carn_pursuit": float(np.mean([np.mean([g.action_weights[rows, idx] for rows in prey_rows]) for g in genomes])),
            "carn_avoid": float(np.mean([g.action_weights[block_rows, idx].mean() for g in genomes])),
            "carn_generation": float(np.mean([c.generation for c in self.carnivores if c.alive])),
        }

    def death_counts(self) -> dict[str, int]:
        """Flat `{species}_death_{cause}` totals since reset, for CSV logging."""
        row = {}
        for species, counter in self.deaths.items():
            for cause, n in counter.items():
                row[f"{species}_death_{cause}"] = n
        return row
