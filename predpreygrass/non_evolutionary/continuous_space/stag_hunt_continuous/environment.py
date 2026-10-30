"""Continuous-space stag hunt for RLlib (DESIGN.md).

The world of base_environment_continuous with every quirk fixed (its
FIXES), and three species: predators (the hunters), rabbits (the base
prey, caught by one predator on contact) and mammoths (larger and slower,
caught only by a hunt). Rabbits graze the base grass, mammoths their own
steppe layer.

A predator's action is [move, join]: one of the 48 movement actions and a
join decision. Hunts are resolved once per decision, at its end, from the
positions then (DESIGN.md, 3.2): the contributors to a hunt are the
predators with join on that touch the mammoth and are not in a hunt
cooldown, each contributing to the nearest mammoth it touches; the
bystanders are the predators that contribute to no hunt this decision and
are within hunt_bystander_radius of a hunted mammoth's centre (each of its
nearest hunted mammoth). Ties between equally near mammoths go to the lower
agent number. With n contributors
the hunt succeeds with probability hunt_success_by_party[min(n, 4) - 1]:
the mammoth is removed and a carcass of hunt_carcass_energy goes, in equal
shares, a fraction 1 - hunt_scavenger_fraction to the contributors and the
rest to the bystanders (all to the contributors without bystanders). On a
failure the mammoth escapes; each contributor loses its share of
hunt_struggle_fraction x hunt_carcass_energy and starts a cooldown of
hunt_cooldown decisions. Every contributor pays hunt_join_cost per hunt.
"""

import math
import random

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.geometry import Torus, Vector, direction_vector, heading_degrees
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.grass import Grass

ENV_NAME = "predpreygrass_stag_hunt_continuous"
SPECIES = ("predator", "rabbit", "mammoth")
PREY = ("rabbit", "mammoth")
# Order of the animal slots in every observation (DESIGN.md, 3.4).
OBSERVED = ("predator", "mammoth", "rabbit")
DIRECTIONS = 16
SPEEDS = (1.0, 0.5, 0.0)  # full, half, stopped
MOVES = DIRECTIONS * len(SPEEDS)
SELF_SIZE = 4  # own vx, vy (/ top speed), sin and cos of own heading
SLOT_SIZE = 6  # seen flag, dx, dy, distance (/ view distance), vx, vy
FOOD_SIZE = 4  # seen flag, dx, dy, distance (/ view distance)
PARTY_SIZES = 4  # success chances for parties of 1, 2, 3 and 4 or more
JOINT_WINDOW = 2  # decisions between two predators' first touches of a mammoth
# Below this speed an animal counts as stopped and keeps its heading.
STOPPED_SPEED = 1e-6

DEFAULT_CONFIG = {
    "width": 800,
    "height": 800,
    # Physics steps per episode.
    "max_time_steps": 8000,
    # Physics steps per decision (fewer when a birth or the episode's end
    # cuts a decision short).
    "action_repeat": 8,
    "predator_count": 6,
    "rabbit_count": 8,
    "mammoth_count": 4,
    # Movement, in arena units per physics step.
    "predator_mass": 10,
    "rabbit_mass": 3,
    "mammoth_mass": 30,
    "predator_radius": 16,
    "rabbit_radius": 16,
    "mammoth_radius": 32,
    "predator_max_velocity": 5,
    "rabbit_max_velocity": 6.0,
    "mammoth_max_velocity": 4.0,
    "predator_max_acceleration": 0.6,
    "rabbit_max_acceleration": 1,
    "mammoth_max_acceleration": 0.3,
    "predator_max_steer_force": 0.6,
    "rabbit_max_steer_force": 0.6,
    "mammoth_max_steer_force": 0.6,
    # View cones (radius in arena units, angle in degrees).
    "predator_view_distance": 200,
    "rabbit_view_distance": 267,
    "mammoth_view_distance": 267,
    "predator_fov": 150,
    "rabbit_fov": 240,
    "mammoth_fov": 240,
    # Observation slots: observe_<observer>_<seen>, nearest first.
    "observe_predator_predator": 4,
    "observe_predator_mammoth": 2,
    "observe_predator_rabbit": 2,
    "observe_rabbit_predator": 3,
    "observe_rabbit_mammoth": 0,
    "observe_rabbit_rabbit": 3,
    "observe_mammoth_predator": 3,
    "observe_mammoth_mammoth": 2,
    "observe_mammoth_rabbit": 0,
    # The rabbits' grass: the base world's.
    "grass_count": 100,
    "grass_consume_radius": 12.0,
    "grass_respawn_delay": 400,
    "grass_clustered": True,
    "grass_cluster_count": 20,
    "grass_cluster_spread": 24.0,
    "grass_random_respawn": False,
    "grass_dispersal": True,
    "grass_dispersal_distance": 24.0,
    "grass_overgrazing": True,
    "grass_overgrazing_threshold": 0.0,
    "grass_overgrazing_delay": 400,
    # The mammoths' steppe: fewer, larger clusters.
    "steppe_count": 40,
    "steppe_consume_radius": 24.0,
    "steppe_respawn_delay": 400,
    "steppe_clustered": True,
    "steppe_cluster_count": 5,
    "steppe_cluster_spread": 48.0,
    "steppe_random_respawn": False,
    "steppe_dispersal": True,
    "steppe_dispersal_distance": 48.0,
    "steppe_overgrazing": True,
    "steppe_overgrazing_threshold": 0.0,
    "steppe_overgrazing_delay": 400,
    # Energy, per physics step. 75% of the cost at full speed comes from
    # movement (Taylor, Heglund & Maloiy 1982), as in the base world.
    "energy_predator_initial": 5.0,
    "energy_rabbit_initial": 3.0,
    "energy_mammoth_initial": 10.0,
    "energy_predator_max": 24.0,
    "energy_rabbit_max": 16.0,
    "energy_mammoth_max": 40.0,
    "energy_predator_resting_metabolic_cost": 0.25 * 0.15 / 8,
    "energy_rabbit_resting_metabolic_cost": 0.25 * 0.05 / 8,
    "energy_mammoth_resting_metabolic_cost": 0.25 * 0.15 / 8,
    "energy_predator_speed_cost": 0.75 * 0.15 / 8 / 5,
    "energy_rabbit_speed_cost": 0.75 * 0.05 / 8 / 5,
    "energy_mammoth_speed_cost": 0.75 * 0.15 / 8 / 4,
    "energy_grass_gain": 2.0,
    "energy_steppe_gain": 4.0,
    # Share of a caught rabbit's energy the catching predator gains.
    "energy_catch_efficiency_predator": 0.5,
    "reproduction_predator_threshold": 12.0,
    "reproduction_rabbit_threshold": 8.0,
    "reproduction_mammoth_threshold": 24.0,
    "reproduction_predator_reward": 10.0,
    "reproduction_rabbit_reward": 10.0,
    "reproduction_mammoth_reward": 10.0,
    "reproduction_max_predator": 50,
    "reproduction_max_rabbit": 100,
    "reproduction_max_mammoth": 20,
    # Agent IDs per species, never reused within an episode.
    "reproduction_predator_pool": 2000,
    "reproduction_rabbit_pool": 2000,
    "reproduction_mammoth_pool": 500,
    # Mammoth hunts (module docstring; DESIGN.md, 3.2).
    "hunt_success_by_party": (0.0, 0.4, 0.75, 0.9),
    "hunt_carcass_energy": 32.0,
    "hunt_scavenger_fraction": 0.2,
    "hunt_bystander_radius": 96.0,
    "hunt_struggle_fraction": 0.05,
    "hunt_join_cost": 0.05,
    "hunt_cooldown": 2,
}

GRASS_KEYS = (
    "count",
    "consume_radius",
    "respawn_delay",
    "clustered",
    "cluster_count",
    "cluster_spread",
    "random_respawn",
    "dispersal",
    "dispersal_distance",
    "overgrazing",
    "overgrazing_threshold",
    "overgrazing_delay",
)
BODY = ("radius", "mass", "max_velocity", "max_acceleration", "max_steer_force", "view_distance", "fov")
FOOD = {"predator": "grass", "rabbit": "grass", "mammoth": "steppe"}


def _number(value, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    if minimum is None:
        return True
    return value > minimum if strict else value >= minimum


def _integer(value, minimum):
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


class Animal:
    """One predator, rabbit or mammoth."""

    __slots__ = (
        "id",
        "species",
        "position",
        "velocity",
        "acceleration",
        "orientation",
        "energy",
        "alive",
        "ready_at",
        "radius",
        "mass",
        "max_speed",
        "max_acceleration",
        "max_steer_force",
        "view_distance",
        "fov",
    )

    def __init__(self, agent_id, species, position, velocity, energy, body):
        self.id = agent_id
        self.species = species
        self.position = position
        self.velocity = velocity
        self.acceleration = Vector(0, 0)
        self.orientation = heading_degrees(velocity)
        self.energy = energy
        self.alive = True
        # First decision at which a predator may contribute to a hunt again.
        self.ready_at = 0
        (
            self.radius,
            self.mass,
            self.max_speed,
            self.max_acceleration,
            self.max_steer_force,
            self.view_distance,
            self.fov,
        ) = body


def new_hunt_stats():
    """Per-episode counters of hunts, catches and join decisions."""
    stats = {
        "decisions": 0,  # predator decisions
        "joins": 0,  # of which with join on
        # Predator decisions ending touching a mammoth while ready, with join
        # on, and with another predator within the bystander radius of it.
        # Every contact counts, in a hunt cooldown or not; contact_ready*
        # only those of predators that could contribute.
        "contact_decisions": 0,
        "contact_joins": 0,
        "contact_ready": 0,
        "contact_ready_joins": 0,
        "contact_with_partner": 0,
        "contact_with_partner_joins": 0,
        "contact_alone": 0,
        "contact_alone_joins": 0,
        # Mammoth encounters (DESIGN.md, 4): predator decisions ending within
        # the bystander radius of a mammoth it sees; encounter episodes of a
        # mammoth (from the first predator within the bystander radius until
        # none is, or the mammoth is gone); of these, those with at least one
        # predator touching it, and joint arrivals: two predators first
        # touching it within joint_window decisions of each other.
        "encounter_decisions": 0,
        "encounter_episodes": 0,
        "touched_episodes": 0,
        "joint_arrivals": 0,
        "rabbits_caught": 0,
        "rabbit_energy": 0.0,  # gained by predators
        "mammoth_energy": 0.0,  # gained by contributors and bystanders
        "scavenger_energy": 0.0,  # of which by bystanders
        "predators_starved_in_hunts": 0,
        "grass_eaten_total": 0,
        "steppe_eaten_total": 0,
    }
    for size in range(1, PARTY_SIZES + 1):
        stats[f"hunts_party_{size}"] = 0
        stats[f"kills_party_{size}"] = 0
    return stats


class StagHuntContinuous(MultiAgentEnv):
    """RLlib multi-agent environment; see the module docstring and
    DESIGN.md. Agent IDs: predator_<i>, rabbit_<i>, mammoth_<i>."""

    def __init__(self, config=None):
        super().__init__()
        unknown = set(config or {}) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        c = dict(DEFAULT_CONFIG, **(config or {}))
        self.config = c
        for name in ("width", "height", "max_time_steps", "action_repeat"):
            if not _integer(c[name], 1):
                raise ValueError(f"{name} must be a positive integer")
        self.width, self.height = c["width"], c["height"]
        self.max_time_steps = c["max_time_steps"]
        self.action_repeat = c["action_repeat"]
        self.torus = Torus(self.width, self.height)
        self.food = {layer: Grass(self.width, self.height, **{key: c[f"{layer}_{key}"] for key in GRASS_KEYS}) for layer in ("grass", "steppe")}
        self.body = {s: tuple(c[f"{s}_{name}"] for name in BODY) for s in SPECIES}
        self.count = {s: c[f"{s}_count"] for s in SPECIES}
        self.observe = {(a, b): c[f"observe_{a}_{b}"] for a in SPECIES for b in SPECIES}
        self.initial_energy = {s: c[f"energy_{s}_initial"] for s in SPECIES}
        self.max_energy = {s: c[f"energy_{s}_max"] for s in SPECIES}
        self.resting_cost = {s: c[f"energy_{s}_resting_metabolic_cost"] for s in SPECIES}
        self.speed_cost = {s: c[f"energy_{s}_speed_cost"] for s in SPECIES}
        self.threshold = {s: c[f"reproduction_{s}_threshold"] for s in SPECIES}
        self.birth_reward = {s: c[f"reproduction_{s}_reward"] for s in SPECIES}
        self.max_population = {s: c[f"reproduction_max_{s}"] for s in SPECIES}
        self.pool = {s: c[f"reproduction_{s}_pool"] for s in SPECIES}
        self.food_gain = {"grass": c["energy_grass_gain"], "steppe": c["energy_steppe_gain"]}
        self.catch_efficiency = c["energy_catch_efficiency_predator"]
        self.success = tuple(c["hunt_success_by_party"]) if isinstance(c["hunt_success_by_party"], (list, tuple)) else None
        self.carcass = c["hunt_carcass_energy"]
        self.scavenger_fraction = c["hunt_scavenger_fraction"]
        self.bystander_radius = c["hunt_bystander_radius"]
        self.struggle_fraction = c["hunt_struggle_fraction"]
        self.join_cost = c["hunt_join_cost"]
        self.cooldown = c["hunt_cooldown"]
        self._validate()

        self.obs_size = {}
        for s in SPECIES:
            slots = sum(self.observe[(s, other)] for other in OBSERVED)
            self.obs_size[s] = SELF_SIZE + SLOT_SIZE * slots + FOOD_SIZE + 1 + (1 if s == "predator" else 0)
        self.possible_agents = [f"{s}_{i}" for s in SPECIES for i in range(self.pool[s])]
        spaces = {s: Box(-1.0, 1.0, shape=(self.obs_size[s],), dtype=np.float32) for s in SPECIES}
        actions = {"predator": MultiDiscrete([MOVES, 2]), "rabbit": Discrete(MOVES), "mammoth": Discrete(MOVES)}
        self.observation_spaces = {agent: spaces[species_of(agent)] for agent in self.possible_agents}
        self.action_spaces = {agent: actions[species_of(agent)] for agent in self.possible_agents}
        self.observation_space = Dict(self.observation_spaces)
        self.action_space = Dict(self.action_spaces)
        self._agent_ids = set(self.possible_agents)
        self.agents = []

        self.animals = {s: [] for s in SPECIES}
        self.active = []
        self.time_step = 0
        self.decision = 0
        self.birth_rng = random.Random()
        self.hunt_rng = random.Random()
        self.next_id = {}
        self.births = {}
        self.hunt_stats = new_hunt_stats()
        # Open mammoth encounter episodes: {mammoth id: {predator id: decision
        # of its first touch}}.
        self.encounters = {}
        # The last finished episode's counters, kept until the next one
        # ends, so a callback can read them after an automatic reset.
        self.episode_hunt_stats = None

    def _validate(self):
        c = self.config
        for s in SPECIES:
            if not _integer(c[f"{s}_count"], 0):
                raise ValueError(f"{s}_count must be a nonnegative integer")
            for other in SPECIES:
                if not _integer(c[f"observe_{s}_{other}"], 0):
                    raise ValueError(f"observe_{s}_{other} must be a nonnegative integer")
            for name in BODY:
                if not _number(c[f"{s}_{name}"], 0, strict=True):
                    raise ValueError(f"{s}_{name} must be finite and positive")
            if c[f"{s}_fov"] > 360:
                raise ValueError(f"{s}_fov must not exceed 360")
            # The view check tests only the nearest image.
            if c[f"{s}_view_distance"] > min(self.width, self.height) / 2:
                raise ValueError(f"{s}_view_distance must not exceed half the arena")
            for key in (f"energy_{s}_initial", f"energy_{s}_max", f"reproduction_{s}_threshold"):
                if not _number(c[key], 0, strict=True):
                    raise ValueError(f"{key} must be finite and positive")
            for key in (f"energy_{s}_resting_metabolic_cost", f"energy_{s}_speed_cost", f"reproduction_{s}_reward"):
                if not _number(c[key], 0):
                    raise ValueError(f"{key} must be finite and nonnegative")
            for key in (f"reproduction_max_{s}", f"reproduction_{s}_pool"):
                if not _integer(c[key], 1):
                    raise ValueError(f"{key} must be a positive integer")
            # A parent pays the offspring's initial energy.
            if not self.initial_energy[s] < self.threshold[s] <= self.max_energy[s]:
                raise ValueError(f"reproduction_{s}_threshold must exceed energy_{s}_initial and not exceed energy_{s}_max")
            if self.count[s] > min(self.max_population[s], self.pool[s]):
                raise ValueError(f"{s}_count must not exceed reproduction_max_{s} or the pool")
        for key in ("energy_grass_gain", "energy_steppe_gain", "energy_catch_efficiency_predator", "hunt_carcass_energy", "hunt_join_cost"):
            if not _number(c[key], 0):
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("energy_catch_efficiency_predator", "hunt_scavenger_fraction", "hunt_struggle_fraction"):
            if not _number(c[key], 0) or c[key] > 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if not _number(c["hunt_bystander_radius"], 0):
            raise ValueError("hunt_bystander_radius must be finite and nonnegative")
        if not _integer(c["hunt_cooldown"], 0):
            raise ValueError("hunt_cooldown must be a nonnegative integer")
        if self.success is None or len(self.success) != PARTY_SIZES or not all(_number(p, 0) and p <= 1 for p in self.success):
            raise ValueError(f"hunt_success_by_party must be {PARTY_SIZES} probabilities")

    # --- Episode ---------------------------------------------------------

    @property
    def predators(self):
        return self.animals["predator"]

    @property
    def rabbits(self):
        return self.animals["rabbit"]

    @property
    def mammoths(self):
        return self.animals["mammoth"]

    def living(self):
        return [a for s in SPECIES for a in self.animals[s]]

    def reset(self, *, seed=None, options=None):
        self.birth_rng = random.Random(None if seed is None else seed + 1)
        self.hunt_rng = random.Random(None if seed is None else seed + 2)
        self.food["grass"].reset(seed)
        self.food["steppe"].reset(None if seed is None else seed + 3)
        spawn = random.Random(seed)
        self.animals = {s: [] for s in SPECIES}
        # Rabbits and predators drawn as the base world's prey and predators
        # (same seed, same start), then the mammoths.
        for s, margin in (("rabbit", 20), ("predator", 0), ("mammoth", 20)):
            for i in range(self.count[s]):
                position = Vector(spawn.randint(margin, self.width - margin), spawn.randint(margin, self.height - margin))
                self.animals[s].append(self._animal(s, i, position))
        self.time_step = 0
        self.decision = 0
        self.next_id = dict(self.count)
        self.births = {s: 0 for s in SPECIES}
        self.hunt_stats = new_hunt_stats()
        self.encounters = {}
        self.active = [a.id for a in self.living()]
        self.agents = list(self.active)
        return self._observations(), {agent: {} for agent in self.active}

    def _animal(self, species, number, position):
        # Initial velocity as in the base world: predators (0, -1), prey (-1, 0).
        velocity = Vector(0, -1) if species == "predator" else Vector(-1, 0)
        return Animal(f"{species}_{number}", species, position, velocity, self.initial_energy[species], self.body[species])

    def _parse(self, agent, action):
        """(movement action, join) of an agent's action."""
        if species_of(agent) == "predator":
            values = np.asarray(action).reshape(-1)
            if values.shape != (2,) or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{agent}: a predator's action is [move, join] integers")
            move, join = int(values[0]), int(values[1])
            if join not in (0, 1):
                raise ValueError(f"{agent}: join must be 0 or 1")
        else:
            if not isinstance(action, (int, np.integer)) or isinstance(action, bool):
                raise ValueError(f"{agent}: a prey's action is an integer")
            move, join = int(action), 0
        if not 0 <= move < MOVES:
            raise ValueError(f"{agent}: movement action out of range")
        return move, join

    def step(self, action_dict):
        """One decision: each agent's movement for up to action_repeat
        physics steps (fewer after a birth or at the episode's end), then the
        hunts. Actions for agents that no longer act are ignored."""
        missing = [agent for agent in self.active if agent not in action_dict]
        if missing:
            raise ValueError(f"no action for acting agents {missing}")
        parsed = {agent: self._parse(agent, action_dict[agent]) for agent in self.active}
        moves = {agent: move for agent, (move, _) in parsed.items()}
        joins = {agent: join for agent, (_, join) in parsed.items() if species_of(agent) == "predator"}
        obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}
        totals = {"grass_eaten": {}, "births": {}}
        zeros = {s: np.zeros(self.obs_size[s], dtype=np.float32) for s in SPECIES}
        ended = False
        for _ in range(self.action_repeat):
            step_agents, step_rewards, step_terms, step_truncs, step_infos, born = self._physics_step(moves)
            for agent in step_agents:
                obs[agent] = zeros[species_of(agent)]
            for agent, reward in step_rewards.items():
                rewards[agent] = rewards.get(agent, 0.0) + reward
            terminateds.update(step_terms)
            truncateds.update(step_truncs)
            for agent, info in step_infos.items():
                infos.setdefault(agent, {}).update(info)
                for key, total in totals.items():
                    if key in info:
                        total[agent] = total.get(agent, 0) + info[key]
            if step_terms["__all__"] or step_truncs["__all__"]:
                ended = True
                break
            if born:
                # So RLlib registers each newborn while its parent is alive.
                break
            moves = {agent: move for agent, move in moves.items() if not (step_terms.get(agent) or step_truncs.get(agent))}
            if not moves:
                break
        if not ended and self.active:
            self._hunts(joins, terminateds, truncateds, infos)
            for agent in terminateds:
                if agent != "__all__" and terminateds[agent] and agent not in obs:
                    obs[agent] = zeros[species_of(agent)]
                    rewards.setdefault(agent, 0.0)
        self.decision += 1
        obs.update(self._observations())
        for key, total in totals.items():
            for agent, count in total.items():
                infos[agent][key] = count
        if terminateds.get("__all__") or truncateds.get("__all__"):
            for mammoth_id in list(self.encounters):
                self._close_encounter(mammoth_id)
            self.hunt_stats["grass_eaten_total"] = self.food["grass"].total_consumed
            self.hunt_stats["steppe_eaten_total"] = self.food["steppe"].total_consumed
            self.episode_hunt_stats = dict(self.hunt_stats, **{f"births_{s}": self.births[s] for s in SPECIES})
        self.agents = list(self.active)
        return obs, rewards, terminateds, truncateds, infos

    # --- One physics step ------------------------------------------------

    def _physics_step(self, moves):
        """Every contact from the positions at the start of the step, then
        every move; then food, energy and births. Returns (agents in this
        step, rewards, terminateds, truncateds, infos, whether anyone was
        born)."""
        agents = list(self.active)
        catches = []  # (catching predator, rabbit energy at the catch)
        contacts = [(animal, *self._contacts(animal)) for animal in self.living()]
        for animal, catcher, aways in contacts:
            self._move(animal, moves[animal.id], catcher, aways, catches)

        living = {a.id for a in self.living()}
        terminateds = {agent: agent not in living for agent in agents}
        truncateds = {agent: False for agent in agents}
        if self.time_step + 1 >= self.max_time_steps or self._extinct():
            truncateds = {agent: True for agent in agents}
        self.time_step += 1
        self.active = [agent for agent in agents if not (terminateds[agent] or truncateds[agent])]

        eaten = {}
        for layer, eaters in (("grass", self.rabbits), ("steppe", self.mammoths)):
            eaten.update(self.food[layer].advance(eaters))
        rewards = {agent: 0.0 for agent in agents}
        infos = {agent: {"grass_eaten": eaten.get(agent, 0)} for agent in agents}

        self._energy(catches, eaten, terminateds, truncateds, infos)

        born = self._reproduce(rewards, infos) if self.active else []
        for child, parent in born:
            rewards[child] = 0.0
            terminateds[child] = False
            truncateds[child] = False
            infos[child] = {"born": True, "parent": parent, "energy": self.initial_energy[species_of(child)]}
            agents.append(child)
        return (agents, rewards, *self._end_flags(terminateds, truncateds), infos, bool(born))

    def _extinct(self):
        """No predators, or no prey of either kind."""
        return not self.predators or not (self.rabbits or self.mammoths)

    def _end_flags(self, terminateds, truncateds):
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())
        if not (self.rabbits or self.mammoths):
            # Nothing is left to hunt: the episode ended, not interrupted.
            terminateds = {agent: True for agent in terminateds}
            truncateds = {agent: False for agent in truncateds}
        elif not self.active and not (terminateds["__all__"] or truncateds["__all__"]):
            # A mixed ending sets neither flag, but no agent is left.
            truncateds["__all__"] = True
        return terminateds, truncateds

    def _desired_velocity(self, move, animal):
        velocity = direction_vector(move % DIRECTIONS, DIRECTIONS)
        velocity.normalize()
        velocity.mult(animal.max_speed)
        velocity.mult(SPEEDS[move // DIRECTIONS])
        return velocity

    def _touching(self, a, b):
        return self.torus.distance(a.position, b.position) < a.radius + b.radius

    def _contacts(self, animal):
        """(catching predator or None, unit vectors pointing away from every
        animal of its own species it touches). Only rabbits are caught on
        contact."""
        catcher = None
        if animal.species == "rabbit":
            catcher = next((p for p in self.predators if self._touching(animal, p)), None)
        aways = []
        for other in self.animals[animal.species]:
            if other is not animal and self._touching(animal, other):
                away = self.torus.away_vector(animal.position, other.position)
                away.negate()
                away.normalize()
                aways.append(away)
        return catcher, aways

    def _steer(self, animal, desired):
        """Acceleration towards the desired velocity: the steering force,
        limited, over the mass, rescaled to the effective maximum
        acceleration (at most the remaining velocity error)."""
        error = desired.copy()
        error.sub(animal.velocity)
        max_acceleration = min(animal.max_acceleration, error.mag())
        animal.acceleration.mult(0)
        error.limit(animal.max_steer_force)
        error.div(animal.mass)
        animal.acceleration.add(error)
        animal.acceleration.normalize()
        animal.acceleration.mult(max_acceleration)

    def _move(self, animal, move, catcher, aways, catches):
        if catcher is not None:
            # A caught rabbit is removed before it moves.
            animal.alive = False
            catches.append((catcher, animal.energy))
            self.rabbits.remove(animal)
            return
        self._steer(animal, self._desired_velocity(move, animal))
        # A bump adds a push of top speed / 8 away from each animal touched.
        for away in aways:
            away.mult(animal.max_speed / 8)
            animal.acceleration.add(away)
        animal.velocity.add(animal.acceleration)
        animal.velocity.limit(animal.max_speed)
        animal.position.add(animal.velocity)
        animal.position.x %= self.width
        animal.position.y %= self.height
        if animal.velocity.mag() >= STOPPED_SPEED:
            animal.orientation = round(heading_degrees(animal.velocity), 1)

    def _gain(self, animal, amount):
        animal.energy = min(self.max_energy[animal.species], animal.energy + amount)

    def _remove(self, animal, terminateds, truncateds, infos, reason):
        animal.alive = False
        self.animals[animal.species].remove(animal)
        if animal.id in self.active:
            self.active.remove(animal.id)
        terminateds[animal.id] = True
        truncateds[animal.id] = False
        infos.setdefault(animal.id, {})[reason] = True

    def _energy(self, catches, eaten, terminateds, truncateds, infos):
        """Food, metabolic costs and starvation."""
        for predator, rabbit_energy in catches:
            if predator.alive:
                before = predator.energy
                self._gain(predator, self.catch_efficiency * rabbit_energy)
                self.hunt_stats["rabbits_caught"] += 1
                self.hunt_stats["rabbit_energy"] += predator.energy - before
        for prey in self.rabbits + self.mammoths:
            count = eaten.get(prey.id, 0)
            if count:
                self._gain(prey, count * self.food_gain[FOOD[prey.species]])
        living = self.living()
        for animal in living:
            animal.energy -= self.resting_cost[animal.species] + self.speed_cost[animal.species] * animal.velocity.mag()
        starved = [animal for animal in living if animal.energy <= 0]
        for animal in starved:
            self._remove(animal, terminateds, truncateds, infos, "starved")
        if starved and self._extinct():
            for agent in self.active:
                truncateds[agent] = True
            self.active.clear()
        for animal in self.living():
            infos.setdefault(animal.id, {})["energy"] = animal.energy

    def _reproduce(self, rewards, infos):
        """Births at the energy threshold. Returns [(child, parent)]."""
        population = {s: len(self.animals[s]) for s in SPECIES}
        active = set(self.active)
        born = []
        for parent in self.living():
            s = parent.species
            if (
                parent.energy < self.threshold[s]
                or parent.id not in active
                or population[s] >= self.max_population[s]
                or self.next_id[s] >= self.pool[s]
            ):
                continue
            distance = 2 * parent.radius
            angle = self.birth_rng.uniform(0, 2 * math.pi)
            position = Vector(
                (parent.position.x + distance * math.cos(angle)) % self.width,
                (parent.position.y + distance * math.sin(angle)) % self.height,
            )
            child = self._animal(s, self.next_id[s], position)
            self.next_id[s] += 1
            self.animals[s].append(child)
            self.active.append(child.id)
            parent.energy -= self.initial_energy[s]
            born.append((child.id, parent.id))
            population[s] += 1
            self.births[s] += 1
            rewards[parent.id] = rewards.get(parent.id, 0.0) + self.birth_reward[s]
            infos.setdefault(parent.id, {})["births"] = 1
            infos[parent.id]["energy"] = parent.energy
        return born

    # --- Hunts -----------------------------------------------------------

    def hunt_parties(self, joins):
        """{mammoth: (contributors, bystanders)} for this decision's hunts,
        mammoths in list order, from the current positions."""
        ready = [p for p in self.predators if p.id in joins and self.decision >= p.ready_at]
        contributors = {}
        for predator in ready:
            if not joins[predator.id]:
                continue
            mammoth = self._nearest_touched(predator)
            if mammoth is not None:
                contributors.setdefault(id(mammoth), []).append(predator)
        hunted = [m for m in self.mammoths if id(m) in contributors]
        taking_part = {id(p) for party in contributors.values() for p in party}
        bystanders = {}
        for predator in self.predators:
            if id(predator) in taking_part:
                continue
            near = [
                (self.torus.distance(predator.position, m.position), number(m), m)
                for m in hunted
                if self.torus.distance(predator.position, m.position) <= self.bystander_radius
            ]
            if near:
                _, _, mammoth = min(near, key=lambda entry: entry[:2])
                bystanders.setdefault(id(mammoth), []).append(predator)
        return {m: (contributors[id(m)], bystanders.get(id(m), [])) for m in hunted}

    def _nearest_touched(self, predator):
        """The nearest mammoth the predator touches (ties: lower agent
        number), or None."""
        touched = [(self.torus.distance(predator.position, m.position), number(m), m) for m in self.mammoths if self._touching(predator, m)]
        return min(touched, key=lambda entry: entry[:2])[2] if touched else None

    def _record(self, joins):
        """Join decisions at mammoth contact (with and without another
        predator within the bystander radius of the mammoth) and mammoth
        encounters, from the positions at the end of the decision."""
        stats = self.hunt_stats
        for predator in self.predators:
            if predator.id not in joins:
                continue
            join = joins[predator.id]
            stats["decisions"] += 1
            stats["joins"] += join
            if any(
                self.torus.distance(predator.position, m.position) <= self.bystander_radius
                and self.torus.in_view(predator, m.position, predator.view_distance, predator.fov)
                for m in self.mammoths
            ):
                stats["encounter_decisions"] += 1
            mammoth = self._nearest_touched(predator)
            if mammoth is None:
                continue
            partner = any(
                other is not predator and self.torus.distance(other.position, mammoth.position) <= self.bystander_radius for other in self.predators
            )
            stats["contact_decisions"] += 1
            stats["contact_joins"] += join
            if self.decision >= predator.ready_at:
                stats["contact_ready"] += 1
                stats["contact_ready_joins"] += join
            key = "contact_with_partner" if partner else "contact_alone"
            stats[key] += 1
            stats[f"{key}_joins"] += join
        alive = {m.id for m in self.mammoths}
        for mammoth_id in [m for m in self.encounters if m not in alive]:
            self._close_encounter(mammoth_id)
        for mammoth in self.mammoths:
            near = [p for p in self.predators if self.torus.distance(p.position, mammoth.position) <= self.bystander_radius]
            if not near:
                if mammoth.id in self.encounters:
                    self._close_encounter(mammoth.id)
                continue
            if mammoth.id not in self.encounters:
                self.encounters[mammoth.id] = {}
                stats["encounter_episodes"] += 1
            for predator in near:
                if self._touching(predator, mammoth):
                    self.encounters[mammoth.id].setdefault(predator.id, self.decision)

    def _close_encounter(self, mammoth_id):
        """End a mammoth's encounter episode: touched, and a joint arrival if
        two predators first touched it within JOINT_WINDOW decisions."""
        first = sorted(self.encounters.pop(mammoth_id).values())
        if first:
            self.hunt_stats["touched_episodes"] += 1
        if any(b - a <= JOINT_WINDOW for a, b in zip(first, first[1:])):
            self.hunt_stats["joint_arrivals"] += 1

    def _hunts(self, joins, terminateds, truncateds, infos):
        """Resolve this decision's hunts (module docstring)."""
        self._record(joins)
        stats = self.hunt_stats
        for mammoth, (party, bystanders) in self.hunt_parties(joins).items():
            n = len(party)
            size = min(n, PARTY_SIZES)
            success = self.hunt_rng.random() < self.success[size - 1]
            stats[f"hunts_party_{size}"] += 1
            for predator in party:
                predator.energy -= self.join_cost
                info = infos.setdefault(predator.id, {})
                info["hunts"] = info.get("hunts", 0) + 1
            if success:
                stats[f"kills_party_{size}"] += 1
                self._remove(mammoth, terminateds, truncateds, infos, "hunted")
                self._close_encounter(mammoth.id)
                share = self.carcass * (1 - self.scavenger_fraction if bystanders else 1.0) / n
                for predator in party:
                    before = predator.energy
                    self._gain(predator, share)
                    stats["mammoth_energy"] += predator.energy - before
                    infos[predator.id]["hunt_kills"] = infos[predator.id].get("hunt_kills", 0) + 1
                if bystanders:
                    share = self.carcass * self.scavenger_fraction / len(bystanders)
                    for predator in bystanders:
                        before = predator.energy
                        self._gain(predator, share)
                        stats["mammoth_energy"] += predator.energy - before
                        stats["scavenger_energy"] += predator.energy - before
                        info = infos.setdefault(predator.id, {})
                        info["scavenged"] = info.get("scavenged", 0) + 1
            else:
                for predator in party:
                    predator.energy -= self.struggle_fraction * self.carcass / n
                    predator.ready_at = self.decision + 1 + self.cooldown
        starved = [p for p in self.predators if p.energy <= 0]
        for predator in starved:
            self._remove(predator, terminateds, truncateds, infos, "starved")
            stats["predators_starved_in_hunts"] += 1
        if self._extinct():
            for agent in self.active:
                truncateds[agent] = True
            self.active.clear()
        if not (self.rabbits or self.mammoths) or not self.active:
            # The hunts ended the episode: recompute its end flags.
            agents = {agent: flag for agent, flag in terminateds.items() if agent != "__all__"}
            new_terminateds, new_truncateds = self._end_flags(agents, {agent: truncateds.get(agent, False) for agent in agents})
            terminateds.clear()
            terminateds.update(new_terminateds)
            truncateds.clear()
            truncateds.update(new_truncateds)
        for animal in self.living():
            infos.setdefault(animal.id, {})["energy"] = animal.energy

    # --- Observations ----------------------------------------------------

    def _observations(self):
        return {animal.id: self._observe(animal) for animal in self.living()}

    def _slots(self, observer, animals, count):
        if not count:
            return []
        seen = []
        for animal in animals:
            if animal is observer or not self.torus.in_view(observer, animal.position, observer.view_distance, observer.fov):
                continue
            dx, dy = self.torus.offset(observer.position, animal.position)
            seen.append((math.hypot(dx, dy), animal.id, dx, dy, animal))
        seen.sort(key=lambda entry: entry[:2])
        values = []
        reach = observer.view_distance
        for distance, _, dx, dy, animal in seen[:count]:
            values += [1.0, dx / reach, dy / reach, distance / reach, animal.velocity.x / animal.max_speed, animal.velocity.y / animal.max_speed]
        return values + [0.0] * (SLOT_SIZE * count - len(values))

    def _observe(self, animal):
        """Self, the nearest animals of each species in view, the nearest
        food patch in view, own energy and (predators) hunt cooldown."""
        heading = math.radians(animal.orientation)
        values = [animal.velocity.x / animal.max_speed, animal.velocity.y / animal.max_speed, math.sin(heading), math.cos(heading)]
        for other in OBSERVED:
            values += self._slots(animal, self.animals[other], self.observe[(animal.species, other)])
        obs = np.asarray(values, dtype=np.float32)
        if not np.all(np.abs(obs) <= 1.0 + 1e-4):
            raise RuntimeError(f"observation out of [-1, 1]: {obs}")
        obs = np.clip(obs, -1.0, 1.0)
        food = self.food[FOOD[animal.species]].observation(animal, animal.view_distance, animal.fov)
        tail = [np.clip(animal.energy / self.max_energy[animal.species], 0.0, 1.0)]
        if animal.species == "predator":
            tail.append(min(1.0, max(0, animal.ready_at - self.decision) / self.cooldown) if self.cooldown else 0.0)
        return np.concatenate((obs, food, np.asarray(tail, dtype=np.float32))).astype(np.float32)


def number(animal):
    """An animal's agent number (rabbit_12: 12)."""
    return int(animal.id.rsplit("_", 1)[1])


def species_of(agent_id):
    species = agent_id.rsplit("_", 1)[0]
    if species not in SPECIES:
        raise ValueError(f"unrecognised agent id {agent_id!r}")
    return species


def register() -> None:
    """Register the environment with Ray Tune under ENV_NAME."""
    register_env(ENV_NAME, StagHuntContinuous)
