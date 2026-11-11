"""Deer stag hunt with food sharing, in continuous space for RLlib
(README.md, DESIGN.md). A copy of deer_stag_hunt's environment
(non_evolutionary/continuous_space/deer_stag_hunt) with one rule added,
sharing of adult catches; with adult_share_fraction 0 it is deer_stag_hunt's
world, episode for episode (tests/test_equivalence.py). The text below is
deer_stag_hunt's (its RESULTS.md section numbers are deer_stag_hunt's),
sharing at the end.

Two species on a torus: predators (the hunters) and deer, which graze
clustered grass (grass_regrowth "instant", the default and the world of
RESULTS.md sections 1 and 2: an eaten patch is back whole after
grass_respawn_delay; "logistic": patches regrow gradually, being calibrated
in RESULTS.md section 3; grass.py). Both learn (one PPO policy per species) and are rewarded
only for births. A deer's energy sets how hard it is to catch:

- fawn (energy <= deer_fawn_energy): a lone predator touching it catches it
  for certain;
- middle (between): a lone predator catches it with a chance falling
  linearly from 1 at deer_fawn_energy to 0 at deer_adult_energy;
- a fawn or middle deer touched while another predator is within
  deer_party_radius of it is always caught, and shared evenly by every
  predator within that radius (the touchers included);
- adult (energy >= deer_adult_energy): caught only once two or more
  different predators have each touched it, while ready, within
  deer_touch_window physics steps, and shared evenly among those touchers
  only.

A failed attempt lets the deer go on, and each predator that touched it
cannot catch again for deer_catch_cooldown physics steps. A predator gains
energy_catch_efficiency_predator x its share of the deer's energy.

These are the rules with deer_catch_rule "party" (the default, the world of
RESULTS.md section 1). With deer_catch_rule "touch" (RESULTS.md section 2)
one rule covers every deer, and hunters merely near a deer play no part:

- every ready touch is a solo attempt, caught with the chance above (1 for a
  fawn, falling linearly to 0 from deer_fawn_energy to deer_adult_energy, 0
  for an adult); a caught deer goes to the toucher alone;
- a failed attempt costs the toucher deer_catch_cooldown physics steps, and
  is remembered for deer_touch_window physics steps;
- once two or more different predators have touched a deer, while ready,
  within deer_touch_window physics steps (at once, or after a failed
  attempt), it is caught, whatever its size, and shared evenly among those
  touchers only. Two touching at once draw no solo chance.

With deer_young_solo "certain" (only under "touch"; RESULTS.md section 5's
control world) a lone ready touch catches every fawn and middle deer for
certain, without a draw; adults are unchanged. The default "graded" is the
falling chance above.

Statistics keep their names: under "touch", young_caught_party counts young
deer caught by two or more touchers.

Sharing (adult_share_fraction s > 0; README.md, DESIGN.md): when an adult is
caught, every other living predator within adult_share_radius of the deer,
measured from the positions at the start of the physics step, is a
bystander. If there is at least one, the catchers split
energy_catch_efficiency_predator x (1 - s) x the deer's energy evenly, and
the bystanders split energy_catch_efficiency_predator x s x the deer's energy
evenly; without bystanders the catchers keep it all, as without sharing.
Fawns and middle deer are never shared beyond their catchers. Observations,
actions and rewards are unchanged.

This is stag_hunt_continuous's world of RESULTS.md section 8 (arm
lean_fat_r7_signal) without calls, mammoths, burrows, seasons or birth
spacing. With the same seed and the same movement actions, an episode is the
same in both, calls off (tests/test_equivalence.py). The policies' networks
differ: predators here have no unused join output and no always-zero inputs
(empty mammoth slots, hunt cooldown), so training and seeded evaluations are
not draw for draw the same as there. Rabbits are renamed deer, lean fawn and
fat adult.

A predator's action is one of the 48 movement actions (16 directions x full,
half or no speed); so is a deer's. Each decision lasts up to action_repeat
physics steps (fewer after a birth or at the episode's end).
"""

import math
import random

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.geometry import Torus, Vector, direction_vector, heading_degrees
from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.grass import Grass

ENV_NAME = "predpreygrass_deer_food_sharing"
SPECIES = ("predator", "deer")
DIRECTIONS = 16
SPEEDS = (1.0, 0.5, 0.0)  # full, half, stopped
MOVES = DIRECTIONS * len(SPEEDS)
SELF_SIZE = 4  # own vx, vy (/ top speed), sin and cos of own heading
SLOT_SIZE = 6  # seen flag, dx, dy, distance (/ view distance), vx, vy
FOOD_SIZE = 4  # seen flag, dx, dy, distance (/ view distance); logistic grass adds its biomass
# Below this speed an animal counts as stopped and keeps its heading.
STOPPED_SPEED = 1e-6
KINDS = ("fawn", "middle", "adult")

DEFAULT_CONFIG = {
    "width": 800,
    "height": 800,
    # Physics steps per episode.
    "max_time_steps": 8000,
    # Physics steps per decision.
    "action_repeat": 8,
    "predator_count": 6,
    "deer_count": 8,
    # Bodies.
    "predator_mass": 10,
    "deer_mass": 3,
    "predator_radius": 16,
    "deer_radius": 16,
    "predator_max_velocity": 5,
    "deer_max_velocity": 6.0,
    "predator_max_acceleration": 0.6,
    "deer_max_acceleration": 1,
    "predator_max_steer_force": 0.6,
    "deer_max_steer_force": 0.6,
    # Vision: view distance (units) and field of view (degrees).
    "predator_view_distance": 200,
    "deer_view_distance": 267,
    "predator_fov": 150,
    "deer_fov": 240,
    # Slots: the nearest animals of each species in view.
    "observe_predator_predator": 4,
    "observe_predator_deer": 2,
    "observe_deer_predator": 3,
    "observe_deer_deer": 3,
    # Grass (grass.py).
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
    # "instant" (default): an eaten patch is back whole after
    # grass_respawn_delay (RESULTS.md sections 1 and 2; runs saved before this
    # setting load with it, checkpoints.py). "logistic": grass regrows
    # gradually (grass.py); deer see the biomass of the nearest patch. Default
    # back to "instant" on 2026-11-09: with root reserve 0.01 logistic grass
    # halved the deer's food (RESULTS.md section 3.1); calibration pending.
    "grass_regrowth": "instant",
    "grass_growth_rate": 0.0187,
    "grass_root_reserve": 0.01,
    # "grazed" only (RESULTS.md section 4): biomass a deer bites off a patch
    # per physics step, the stubble below which it cannot graze, and how long
    # a patch kept below half grown survives before it dies and re-seeds.
    "grass_bite": 0.1,
    "grass_stubble": 0.2,
    "grass_dieback_delay": 400,
    # Energy.
    "energy_predator_initial": 5.0,
    "energy_deer_initial": 2.0,
    "energy_predator_max": 24.0,
    "energy_deer_max": 16.0,
    "energy_predator_resting_metabolic_cost": 0.25 * 0.15 / 8,
    "energy_deer_resting_metabolic_cost": 0.25 * 0.05 / 8,
    "energy_predator_speed_cost": 0.75 * 0.15 / 8 / 5,
    "energy_deer_speed_cost": 0.75 * 0.05 / 8 / 5,
    "energy_grass_gain": 2.0,
    "energy_catch_efficiency_predator": 0.5,
    # Births: at the threshold the parent pays reproduction_<species>_cost
    # (None: the child's initial energy) and gets reproduction_<species>_reward.
    "reproduction_predator_threshold": 12.0,
    "reproduction_deer_threshold": 7.0,
    "reproduction_predator_cost": None,
    "reproduction_deer_cost": 3.0,
    "reproduction_predator_reward": 10.0,
    "reproduction_deer_reward": 10.0,
    "reproduction_max_predator": 50,
    "reproduction_max_deer": 100,
    # Agent IDs per species, never reused within an episode.
    "reproduction_predator_pool": 2000,
    "reproduction_deer_pool": 2000,
    # Catching (module docstring).
    "deer_fawn_energy": 2.0,
    "deer_adult_energy": 5.0,
    "deer_party_radius": 64.0,
    "deer_catch_cooldown": 8,
    "deer_touch_window": 8,
    # "party" (default) or "touch" (module docstring).
    "deer_catch_rule": "party",
    # "graded" (default) or "certain" (module docstring; "touch" only).
    "deer_young_solo": "graded",
    # Sharing of adult catches (module docstring): the fraction of an adult's
    # energy that goes to the other predators within adult_share_radius of it.
    # 0 (default): no sharing, deer_stag_hunt's world.
    "adult_share_fraction": 0.0,
    "adult_share_radius": 128.0,
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
    "regrowth",
    "growth_rate",
    "root_reserve",
    "bite",
    "stubble",
    "dieback_delay",
)
BODY = ("radius", "mass", "max_velocity", "max_acceleration", "max_steer_force", "view_distance", "fov")


def _number(value, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    if minimum is None:
        return True
    return value > minimum if strict else value >= minimum


def _integer(value, minimum):
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


class Animal:
    """One predator or deer."""

    __slots__ = (
        "id",
        "species",
        "position",
        "velocity",
        "acceleration",
        "orientation",
        "energy",
        "alive",
        "catch_ready_at",
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
        # First physics step at which a predator may catch again.
        self.catch_ready_at = 0
        (
            self.radius,
            self.mass,
            self.max_speed,
            self.max_acceleration,
            self.max_steer_force,
            self.view_distance,
            self.fov,
        ) = body


def new_stats():
    """Per-episode counters of catches."""
    return {
        "deer_caught": 0,
        # Fawns and middle deer: caught by a lone predator, by a party (another
        # predator within the party radius), and failed lone attempts.
        "young_caught_solo": 0,
        "young_caught_party": 0,
        "young_failures": 0,
        # Adults: caught by two or more touchers within the window, and
        # failed attempts (physics steps in which ready predators touched an
        # adult with no other toucher in the window).
        "adults_caught": 0,
        "adult_failures": 0,
        # Energy gained by predators from deer of each kind at the catch
        # (fawn, middle, adult; after the catch efficiency and the energy cap).
        "energy_from_fawn": 0.0,
        "energy_from_middle": 0.0,
        "energy_from_adult": 0.0,
        # Sharing (adult_share_fraction > 0): adult catches with at least one
        # bystander, bystanders summed over them, and the energy bystanders
        # gained (after the catch efficiency and the energy cap; also counted
        # in energy_from_adult).
        "adults_shared": 0,
        "share_bystanders": 0,
        "energy_adult_to_bystanders": 0.0,
        "grass_eaten": 0,
        # Summed over physics steps.
        "predator_steps": 0,
        "deer_steps": 0,
    }


class DeerStagHunt(MultiAgentEnv):
    """RLlib multi-agent environment; see the module docstring. Agent IDs:
    predator_<i>, deer_<i>."""

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
        self.grass = Grass(self.width, self.height, **{key: c[f"grass_{key}"] for key in GRASS_KEYS})
        self.body = {s: tuple(c[f"{s}_{name}"] for name in BODY) for s in SPECIES}
        self.count = {s: c[f"{s}_count"] for s in SPECIES}
        self.observe = {(a, b): c[f"observe_{a}_{b}"] for a in SPECIES for b in SPECIES}
        self.initial_energy = {s: c[f"energy_{s}_initial"] for s in SPECIES}
        self.max_energy = {s: c[f"energy_{s}_max"] for s in SPECIES}
        self.resting_cost = {s: c[f"energy_{s}_resting_metabolic_cost"] for s in SPECIES}
        self.speed_cost = {s: c[f"energy_{s}_speed_cost"] for s in SPECIES}
        self.threshold = {s: c[f"reproduction_{s}_threshold"] for s in SPECIES}
        self.birth_cost = {s: self.initial_energy[s] if c[f"reproduction_{s}_cost"] is None else c[f"reproduction_{s}_cost"] for s in SPECIES}
        self.birth_reward = {s: c[f"reproduction_{s}_reward"] for s in SPECIES}
        self.max_population = {s: c[f"reproduction_max_{s}"] for s in SPECIES}
        self.pool = {s: c[f"reproduction_{s}_pool"] for s in SPECIES}
        self.grass_gain = c["energy_grass_gain"]
        self.catch_efficiency = c["energy_catch_efficiency_predator"]
        self.fawn_energy = c["deer_fawn_energy"]
        self.adult_energy = c["deer_adult_energy"]
        self.party_radius = c["deer_party_radius"]
        self.catch_cooldown = c["deer_catch_cooldown"]
        self.touch_window = c["deer_touch_window"]
        self.catch_rule = c["deer_catch_rule"]
        self.young_solo = c["deer_young_solo"]
        self.share_fraction = c["adult_share_fraction"]
        self.share_radius = c["adult_share_radius"]
        self._validate()

        # Predators: self, slots, food, own energy, the energy of each deer
        # slot and the own catch cooldown. Deer: self, slots, food, energy.
        self.obs_size = {}
        for s in SPECIES:
            slots = sum(self.observe[(s, other)] for other in SPECIES)
            self.obs_size[s] = SELF_SIZE + SLOT_SIZE * slots + self.grass.observation_size() + 1
        self.obs_size["predator"] += self.observe[("predator", "deer")] + 1
        self.possible_agents = [f"{s}_{i}" for s in SPECIES for i in range(self.pool[s])]
        spaces = {s: Box(-1.0, 1.0, shape=(self.obs_size[s],), dtype=np.float32) for s in SPECIES}
        self.observation_spaces = {agent: spaces[species_of(agent)] for agent in self.possible_agents}
        self.action_spaces = {agent: Discrete(MOVES) for agent in self.possible_agents}
        self.observation_space = Dict(self.observation_spaces)
        self.action_space = Dict(self.action_spaces)
        self._agent_ids = set(self.possible_agents)
        self.agents = []

        self.animals = {s: [] for s in SPECIES}
        self.active = []
        self.time_step = 0
        self.birth_rng = random.Random()
        self.catch_rng = random.Random()
        self.next_id = {}
        self.births = {}
        self.stats = new_stats()
        self.adult_touches = {}  # {deer id: {predator id: physics step of its last ready touch}}
        # The last finished episode's counters, kept until the next one
        # ends, so a callback can read them after an automatic reset.
        self.episode_stats = None
        self.extinct_at = {}  # {species: physics step at which it died out}

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
            if not self.initial_energy[s] < self.threshold[s] <= self.max_energy[s]:
                raise ValueError(f"reproduction_{s}_threshold must exceed energy_{s}_initial and not exceed energy_{s}_max")
            cost = c[f"reproduction_{s}_cost"]
            if cost is not None and not (_number(cost, 0) and self.initial_energy[s] <= cost < self.threshold[s]):
                raise ValueError(f"reproduction_{s}_cost must be None or at least energy_{s}_initial and below reproduction_{s}_threshold")
            if self.count[s] > min(self.max_population[s], self.pool[s]):
                raise ValueError(f"{s}_count must not exceed reproduction_max_{s} or the pool")
        if not _number(c["energy_grass_gain"], 0):
            raise ValueError("energy_grass_gain must be finite and nonnegative")
        if not _number(c["energy_catch_efficiency_predator"], 0) or c["energy_catch_efficiency_predator"] > 1:
            raise ValueError("energy_catch_efficiency_predator must be in [0, 1]")
        if not _number(c["deer_fawn_energy"], 0) or not _number(c["deer_party_radius"], 0):
            raise ValueError("deer_fawn_energy and deer_party_radius must be finite and nonnegative")
        if not _number(c["deer_adult_energy"], 0) or c["deer_adult_energy"] <= c["deer_fawn_energy"]:
            raise ValueError("deer_adult_energy must be finite and exceed deer_fawn_energy")
        for key in ("deer_catch_cooldown", "deer_touch_window"):
            if not _integer(c[key], 0):
                raise ValueError(f"{key} must be a nonnegative integer")
        if c["deer_catch_rule"] not in ("party", "touch"):
            raise ValueError('deer_catch_rule must be "party" or "touch"')
        if c["deer_young_solo"] not in ("graded", "certain"):
            raise ValueError('deer_young_solo must be "graded" or "certain"')
        if c["deer_young_solo"] == "certain" and c["deer_catch_rule"] != "touch":
            raise ValueError('deer_young_solo "certain" needs deer_catch_rule "touch"')
        if not _number(c["adult_share_fraction"], 0) or c["adult_share_fraction"] > 1:
            raise ValueError("adult_share_fraction must be in [0, 1]")
        if not _number(c["adult_share_radius"], 0):
            raise ValueError("adult_share_radius must be finite and nonnegative")

    # --- Episode ---------------------------------------------------------

    @property
    def predators(self):
        return self.animals["predator"]

    @property
    def deer(self):
        return self.animals["deer"]

    def living(self):
        return [a for s in SPECIES for a in self.animals[s]]

    def kind(self, deer):
        """fawn, middle or adult, by the deer's energy."""
        if deer.energy >= self.adult_energy:
            return "adult"
        return "fawn" if deer.energy <= self.fawn_energy else "middle"

    def reset(self, *, seed=None, options=None):
        # The seed offsets are stag_hunt_continuous's, so both worlds draw
        # the same numbers (tests/test_equivalence.py).
        self.birth_rng = random.Random(None if seed is None else seed + 1)
        self.grass.reset(seed)
        self.catch_rng = random.Random(None if seed is None else seed + 6)
        spawn = random.Random(seed)
        self.animals = {s: [] for s in SPECIES}
        # Deer first, then predators, as the base world's prey and predators.
        for s, margin in (("deer", 20), ("predator", 0)):
            for i in range(self.count[s]):
                position = Vector(spawn.randint(margin, self.width - margin), spawn.randint(margin, self.height - margin))
                self.animals[s].append(self._animal(s, i, position))
        self.time_step = 0
        self.next_id = dict(self.count)
        self.births = {s: 0 for s in SPECIES}
        self.stats = new_stats()
        self.adult_touches = {}
        self.extinct_at = {}
        self.active = [a.id for a in self.living()]
        self.agents = list(self.active)
        return self._observations(), {agent: {} for agent in self.active}

    def _animal(self, species, number, position):
        # Initial velocity as in the base world: predators (0, -1), prey (-1, 0).
        velocity = Vector(0, -1) if species == "predator" else Vector(-1, 0)
        return Animal(f"{species}_{number}", species, position, velocity, self.initial_energy[species], self.body[species])

    def step(self, action_dict):
        """One decision: each agent's movement for up to action_repeat
        physics steps (fewer after a birth or at the episode's end). Actions
        for agents that no longer act are ignored."""
        missing = [agent for agent in self.active if agent not in action_dict]
        if missing:
            raise ValueError(f"no action for acting agents {missing}")
        moves = {agent: self._parse(agent, action_dict[agent]) for agent in self.active}
        obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}
        totals = {"grass_eaten": {}, "births": {}}
        zeros = {s: np.zeros(self.obs_size[s], dtype=np.float32) for s in SPECIES}
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
                break
            if born:
                # So RLlib registers each newborn while its parent is alive.
                break
            moves = {agent: move for agent, move in moves.items() if not (step_terms.get(agent) or step_truncs.get(agent))}
            if not moves:
                break
        obs.update(self._observations())
        for key, total in totals.items():
            for agent, count in total.items():
                infos[agent][key] = count
        if terminateds.get("__all__") or truncateds.get("__all__"):
            self.stats["grass_eaten"] = self.grass.total_consumed
            self.episode_stats = dict(self.stats, physics_steps=self.time_step, **{f"births_{s}": self.births[s] for s in SPECIES})
        self.agents = list(self.active)
        return obs, rewards, terminateds, truncateds, infos

    def _parse(self, agent, action):
        """An agent's movement action."""
        if not isinstance(action, (int, np.integer)) or isinstance(action, bool):
            values = np.asarray(action).reshape(-1)
            if values.shape != (1,) or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{agent}: an action is one integer")
            action = values[0]
        move = int(action)
        if not 0 <= move < MOVES:
            raise ValueError(f"{agent}: movement action out of range")
        return move

    # --- One physics step ------------------------------------------------

    def _physics_step(self, moves):
        """Every contact from the positions at the start of the step, then
        every move; then grass, energy and births. Returns (agents in this
        step, rewards, terminateds, truncateds, infos, whether anyone was
        born)."""
        agents = list(self.active)
        catches = []  # (predators sharing the catch, deer energy at the catch, kind)
        self.stats["predator_steps"] += len(self.predators)
        self.stats["deer_steps"] += len(self.deer)
        contacts = [(animal, *self._contacts(animal)) for animal in self.living()]
        # Bystanders of each caught adult, from the positions at the start of
        # the step, before anything moves.
        bystanders = {animal.id: self._bystanders(animal, catcher) for animal, catcher, _ in contacts if catcher is not None and animal.species == "deer"}
        for animal, catcher, aways in contacts:
            self._move(animal, moves[animal.id], catcher, aways, catches, bystanders.get(animal.id, []))

        living = {a.id for a in self.living()}
        terminateds = {agent: agent not in living for agent in agents}
        truncateds = {agent: False for agent in agents}
        if self.time_step + 1 >= self.max_time_steps or self._extinct():
            # The survivors; an animal that died is terminated only.
            truncateds = {agent: agent in living for agent in agents}
        self.time_step += 1
        self.active = [agent for agent in agents if not (terminateds[agent] or truncateds[agent])]

        eaten = self.grass.advance(self.deer)
        rewards = {agent: 0.0 for agent in agents}
        infos = {agent: {"grass_eaten": eaten.get(agent, 0)} for agent in agents}

        self._energy(catches, eaten, terminateds, truncateds, infos)
        self._note_extinctions()

        born = self._reproduce(rewards, infos) if self.active else []
        for child, parent in born:
            rewards[child] = 0.0
            terminateds[child] = False
            truncateds[child] = False
            infos[child] = {"born": True, "parent": parent, "energy": self.initial_energy[species_of(child)]}
            agents.append(child)
        return (agents, rewards, *self._end_flags(terminateds, truncateds), infos, bool(born))

    def _note_extinctions(self):
        """The physics step at which each species present at the start died
        out (the step just taken)."""
        for s in SPECIES:
            if self.count[s] and not self.animals[s]:
                self.extinct_at.setdefault(s, self.time_step)

    def _extinct(self):
        return not self.predators or not self.deer

    def _end_flags(self, terminateds, truncateds):
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())
        if not self.deer:
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
        """(predators sharing a catch of this deer or None, unit vectors
        pointing away from every animal of its own species it touches)."""
        catcher = None
        if animal.species == "deer":
            adult = animal.energy >= self.adult_energy
            if not adult and self.catch_rule == "party":
                # Touches count only while the deer stays adult.
                self.adult_touches.pop(animal.id, None)
            touching = [p for p in self.predators if self._touching(animal, p) and p.catch_ready_at <= self.time_step]
            if touching and self.catch_rule == "touch":
                catcher = self._touch_catch(animal, touching, adult)
            elif touching and adult:
                now = self.time_step
                recent = self.adult_touches.setdefault(animal.id, {})
                for p in touching:
                    recent[p.id] = now
                for pid in [pid for pid, t in recent.items() if now - t > self.touch_window]:
                    del recent[pid]
                touchers = [p for p in self.predators if p.id in recent]
                if len(touchers) >= 2:
                    catcher = touchers
                    del self.adult_touches[animal.id]
                    self.stats["adults_caught"] += 1
                else:
                    self.stats["adult_failures"] += 1
                    for p in touching:
                        p.catch_ready_at = self.time_step + 1 + self.catch_cooldown
            elif touching:
                # The touching predators always belong to the party, whatever the radius.
                party = [p for p in self.predators if p in touching or self.torus.distance(animal.position, p.position) <= self.party_radius]
                if len(party) >= 2:
                    catcher = party
                    self.stats["young_caught_party"] += 1
                else:
                    chance = min(1.0, max(0.0, (self.adult_energy - animal.energy) / (self.adult_energy - self.fawn_energy)))
                    if self.catch_rng.random() < chance:
                        catcher = party
                        self.stats["young_caught_solo"] += 1
                    else:
                        self.stats["young_failures"] += 1
                        for p in touching:
                            p.catch_ready_at = self.time_step + 1 + self.catch_cooldown
        aways = []
        for other in self.animals[animal.species]:
            if other is not animal and self._touching(animal, other):
                away = self.torus.away_vector(animal.position, other.position)
                away.negate()
                away.normalize()
                aways.append(away)
        return catcher, aways

    def _bystanders(self, deer, catchers):
        """The predators sharing an adult catch without having caught it:
        every other living predator within adult_share_radius of the deer
        (none without sharing or for a young deer)."""
        if self.share_fraction <= 0 or self.kind(deer) != "adult":
            return []
        return [p for p in self.predators if all(p is not c for c in catchers) and self.torus.distance(deer.position, p.position) <= self.share_radius]

    def _solo_chance(self, deer):
        if self.young_solo == "certain" and deer.energy < self.adult_energy:
            return 1.0
        return min(1.0, max(0.0, (self.adult_energy - deer.energy) / (self.adult_energy - self.fawn_energy)))

    def _touch_catch(self, deer, touching, adult):
        """deer_catch_rule "touch": the predators catching the deer this
        physics step, or None (module docstring)."""
        now = self.time_step
        recent = self.adult_touches.setdefault(deer.id, {})
        for p in touching:
            recent[p.id] = now
        for pid in [pid for pid, t in recent.items() if now - t > self.touch_window]:
            del recent[pid]
        touchers = [p for p in self.predators if p.id in recent]
        if len(touchers) >= 2:
            del self.adult_touches[deer.id]
            self.stats["adults_caught" if adult else "young_caught_party"] += 1
            return touchers
        chance = self._solo_chance(deer)
        if chance >= 1.0 or (chance > 0.0 and self.catch_rng.random() < chance):
            del self.adult_touches[deer.id]
            self.stats["young_caught_solo"] += 1
            return touching
        self.stats["adult_failures" if adult else "young_failures"] += 1
        for p in touching:
            p.catch_ready_at = self.time_step + 1 + self.catch_cooldown
        return None

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

    def _move(self, animal, move, catcher, aways, catches, bystanders=()):
        if catcher is not None:
            # A caught deer is removed before it moves.
            animal.alive = False
            catches.append((catcher, animal.energy, self.kind(animal), list(bystanders)))
            self.deer.remove(animal)
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
        self.adult_touches.pop(animal.id, None)
        animal.alive = False
        self.animals[animal.species].remove(animal)
        if animal.id in self.active:
            self.active.remove(animal.id)
        terminateds[animal.id] = True
        truncateds[animal.id] = False
        infos.setdefault(animal.id, {})[reason] = True

    def _energy(self, catches, eaten, terminateds, truncateds, infos):
        """Catches, grazing, metabolic costs and starvation."""
        for hunters, deer_energy, kind, bystanders in catches:
            self.stats["deer_caught"] += 1
            food = self.catch_efficiency * deer_energy
            shares = [(hunters, food)]
            if bystanders:
                # Sharing: the catchers keep 1 - adult_share_fraction, the
                # bystanders get the rest; each group splits its part evenly.
                shares = [(hunters, food * (1 - self.share_fraction)), (bystanders, food * self.share_fraction)]
                self.stats["adults_shared"] += 1
                self.stats["share_bystanders"] += len(bystanders)
            for group, part in shares:
                for predator in group:
                    if predator.alive:
                        before = predator.energy
                        self._gain(predator, part / len(group))
                        gained = predator.energy - before
                        self.stats[f"energy_from_{kind}"] += gained
                        if group is bystanders:
                            self.stats["energy_adult_to_bystanders"] += gained
        for deer in self.deer:
            count = eaten.get(deer.id, 0)
            if count:
                self._gain(deer, count * self.grass_gain)
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
            if parent.energy < self.threshold[s] or parent.id not in active or population[s] >= self.max_population[s] or self.next_id[s] >= self.pool[s]:
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
            parent.energy -= self.birth_cost[s]
            born.append((child.id, parent.id))
            population[s] += 1
            self.births[s] += 1
            rewards[parent.id] = rewards.get(parent.id, 0.0) + self.birth_reward[s]
            infos.setdefault(parent.id, {})["births"] = 1
            infos[parent.id]["energy"] = parent.energy
        return born

    # --- Observations ----------------------------------------------------

    def _observations(self):
        return {animal.id: self._observe(animal) for animal in self.living()}

    def _slots(self, observer, animals, count, chosen=None):
        """Slot values of the nearest animals in view; appends those animals
        to chosen if given."""
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
            if chosen is not None:
                chosen.append(animal)
            values += [1.0, dx / reach, dy / reach, distance / reach, animal.velocity.x / animal.max_speed, animal.velocity.y / animal.max_speed]
        return values + [0.0] * (SLOT_SIZE * count - len(values))

    def _observe(self, animal):
        """Self, the nearest animals of each species in view, the nearest
        grass patch in view, own energy and (predators) the energy of the
        deer in their slots and their own catch cooldown."""
        heading = math.radians(animal.orientation)
        values = [animal.velocity.x / animal.max_speed, animal.velocity.y / animal.max_speed, math.sin(heading), math.cos(heading)]
        seen = {"predator": [], "deer": []}
        for other in SPECIES:
            values += self._slots(animal, self.animals[other], self.observe[(animal.species, other)], seen[other])
        obs = np.asarray(values, dtype=np.float32)
        if not np.all(np.abs(obs) <= 1.0 + 1e-4):
            raise RuntimeError(f"observation out of [-1, 1]: {obs}")
        obs = np.clip(obs, -1.0, 1.0)
        food = self.grass.observation(animal, animal.view_distance, animal.fov)
        tail = [np.clip(animal.energy / self.max_energy[animal.species], 0.0, 1.0)]
        if animal.species == "predator":
            energy = [max(0.0, d.energy) / self.max_energy["deer"] for d in seen["deer"]]
            tail += energy + [0.0] * (self.observe[("predator", "deer")] - len(energy))
            tail.append(min(1.0, max(0, animal.catch_ready_at - self.time_step) / self.catch_cooldown) if self.catch_cooldown else 0.0)
        return np.concatenate((obs, food, np.clip(np.asarray(tail, dtype=np.float32), -1.0, 1.0))).astype(np.float32)


def species_of(agent_id):
    species = agent_id.rsplit("_", 1)[0]
    if species not in SPECIES:
        raise ValueError(f"unrecognised agent id {agent_id!r}")
    return species


def register() -> None:
    """Register the environment with Ray Tune under ENV_NAME."""
    register_env(ENV_NAME, DeerStagHunt)
