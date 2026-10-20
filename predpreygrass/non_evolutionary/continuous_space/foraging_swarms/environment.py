"""Continuous-space Predator-Prey-Grass environment for RLlib (SPEC.md).

Predators and prey move on a torus with steering physics, prey graze grass
clusters that move as they are grazed, both species pay energy to live and
move, starve at zero energy and give birth at an energy threshold. Each
agent chooses a direction and a speed, and observes the nearest animals and
grass patch in its view cone, egocentrically.

With the default settings the dynamics reproduce add_walls (Aquarium 0.1.10
with its patches) in the milestone configuration exactly, including
Aquarium's quirks. Each quirk can be fixed with its own fix_* setting
(SPEC.md, section 9); FIXES turns them all on.
"""

import math
import random

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.geometry import Torus, Vector, direction_vector, heading_degrees
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.grass import Grass

ENV_NAME = "predpreygrass_continuous"
SPECIES = ("predator", "prey")
DIRECTIONS = 16
SPEEDS = (1.0, 0.5, 0.0)  # full, half, stopped
SELF_SIZE = 4  # own vx, vy (/ top speed), sin and cos of own heading
SLOT_SIZE = 6  # seen flag, dx, dy, distance (/ view distance), vx, vy
GRASS_SIZE = 4  # seen flag, dx, dy, distance (/ view distance)
# Below this speed an animal counts as stopped (fix_keep_heading_when_stopped).
STOPPED_SPEED = 1e-6

# The milestone world (runs AW-17 to AW-22).
DEFAULT_CONFIG = {
    "width": 800,
    "height": 800,
    # Physics steps; an episode lasts at most max_time_steps + 2 (SPEC.md 1).
    "max_time_steps": 8000,
    # Physics steps per decision (fewer when a birth or the episode's end
    # cuts a decision short).
    "action_repeat": 8,
    "predator_count": 6,
    "prey_count": 8,
    "predator_mass": 10,
    "prey_mass": 3,
    "predator_radius": 16,
    "prey_radius": 16,
    "predator_max_velocity": 5,
    "prey_max_velocity": 6.0,
    "predator_max_acceleration": 0.6,
    "prey_max_acceleration": 1,
    "predator_max_steer_force": 0.6,
    "prey_max_steer_force": 0.6,
    "predator_view_distance": 200,
    "prey_view_distance": 267,
    "predator_fov": 150,
    "prey_fov": 240,
    # Observation slots for the nearest predators and prey in view.
    "predator_observe_count": 1,
    "prey_observe_count": 3,
    # Social information (foraging_swarms, README): what an animal sees of
    # the members of its own species in its slots. observe_neighbors False
    # leaves those slots empty (the animals are still there and still
    # bump); observe_neighbor_movement False zeroes only their velocity.
    # The other species' slots are never changed. All True reproduces
    # base_environment_continuous exactly.
    "predator_observe_neighbors": True,
    "prey_observe_neighbors": True,
    "predator_observe_neighbor_movement": True,
    "prey_observe_neighbor_movement": True,
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
    "energy_predator_initial": 5.0,
    "energy_prey_initial": 3.0,
    "energy_predator_max": 24.0,
    "energy_prey_max": 16.0,
    # 75% of the cost at full speed comes from movement (Taylor, Heglund &
    # Maloiy 1982; add_walls README).
    "energy_predator_resting_metabolic_cost": 0.25 * 0.15 / 8,
    "energy_prey_resting_metabolic_cost": 0.25 * 0.05 / 8,
    "energy_predator_speed_cost": 0.75 * 0.15 / 8 / 5,
    "energy_prey_speed_cost": 0.75 * 0.05 / 8 / 5,
    "energy_grass_gain": 2.0,
    "energy_catch_efficiency_predator": 0.5,
    # Shared carcass (foraging_swarms, README): the catch's energy is split
    # equally among the catcher (if alive) and every other living predator
    # within this distance of the spot where the prey was caught, measured
    # after the physics step's movement. 0: the catcher alone (as in
    # base_environment_continuous).
    "energy_catch_share_radius": 0.0,
    "reproduction_predator_threshold": 12.0,
    "reproduction_prey_threshold": 8.0,
    "reproduction_predator_reward": 10.0,
    "reproduction_prey_reward": 10.0,
    "reproduction_max_predators": 50,
    "reproduction_max_prey": 100,
    "reproduction_predator_pool": 2000,
    "reproduction_prey_pool": 2000,
    # Fixes of Aquarium's quirks (SPEC.md, section 9). All off reproduces
    # add_walls exactly.
    "fix_simultaneous_movement": False,
    "fix_caught_prey_does_not_move": False,
    "fix_keep_heading_when_stopped": False,
    "fix_wrap_borders": False,
    "fix_same_collisions_for_both": False,
    "fix_all_contacts": False,
    "fix_episode_length": False,
    "fix_no_spare_spawn_draw": False,
}

# Every fix on: the cleaned-up world.
FIXES = {key: True for key in DEFAULT_CONFIG if key.startswith("fix_")}

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


class Animal:
    """One predator or prey."""

    __slots__ = (
        "id",
        "species",
        "position",
        "velocity",
        "acceleration",
        "orientation",
        "energy",
        "alive",
        "radius",
        "mass",
        "max_speed",
        "max_acceleration",
        "max_steer_force",
        "view_distance",
        "fov",
    )

    def __init__(self, agent_id, species, position, velocity, energy, settings):
        self.id = agent_id
        self.species = species
        self.position = position
        self.velocity = velocity
        self.acceleration = Vector(0, 0)
        # Not rounded until the animal first moves (SPEC.md, section 2).
        self.orientation = heading_degrees(velocity)
        self.energy = energy
        self.alive = True
        (
            self.radius,
            self.mass,
            self.max_speed,
            self.max_acceleration,
            self.max_steer_force,
            self.view_distance,
            self.fov,
        ) = settings


class PredPreyGrassContinuous(MultiAgentEnv):
    """RLlib multi-agent environment; see SPEC.md for the rules.

    Agent IDs come from a fixed pool per species (predator_0 ...
    predator_<pool - 1>, likewise prey), never reused within an episode.
    """

    def __init__(self, config=None):
        super().__init__()
        unknown = set(config or {}) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        c = dict(DEFAULT_CONFIG, **(config or {}))
        self.config = c
        for name in ("width", "height", "max_time_steps", "action_repeat"):
            if isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.width, self.height = c["width"], c["height"]
        self.max_time_steps = c["max_time_steps"]
        self.action_repeat = c["action_repeat"]
        self.torus = Torus(self.width, self.height)
        self.grass = Grass(self.width, self.height, **{key: c[f"grass_{key}"] for key in GRASS_KEYS})
        self.settings = {}
        self.count = {}
        self.observe_count = {}
        self.observe_neighbors = {}
        self.observe_neighbor_movement = {}
        self.initial_energy = {}
        self.max_energy = {}
        self.resting_cost = {}
        self.speed_cost = {}
        self.threshold = {}
        self.birth_reward = {}
        self.max_population = {}
        self.pool = {}
        for species in SPECIES:
            group = "predators" if species == "predator" else "prey"
            self.settings[species] = tuple(
                c[f"{species}_{name}"]
                for name in (
                    "radius",
                    "mass",
                    "max_velocity",
                    "max_acceleration",
                    "max_steer_force",
                    "view_distance",
                    "fov",
                )
            )
            self.count[species] = c[f"{species}_count"]
            self.observe_count[species] = c[f"{species}_observe_count"]
            self.observe_neighbors[species] = c[f"{species}_observe_neighbors"]
            self.observe_neighbor_movement[species] = c[f"{species}_observe_neighbor_movement"]
            self.initial_energy[species] = c[f"energy_{species}_initial"]
            self.max_energy[species] = c[f"energy_{species}_max"]
            self.resting_cost[species] = c[f"energy_{species}_resting_metabolic_cost"]
            self.speed_cost[species] = c[f"energy_{species}_speed_cost"]
            self.threshold[species] = c[f"reproduction_{species}_threshold"]
            self.birth_reward[species] = c[f"reproduction_{species}_reward"]
            self.max_population[species] = c[f"reproduction_max_{group}"]
            self.pool[species] = c[f"reproduction_{species}_pool"]
        self.grass_gain = c["energy_grass_gain"]
        self.catch_efficiency = c["energy_catch_efficiency_predator"]
        self.catch_share_radius = c["energy_catch_share_radius"]
        self.catch_log = []
        for key in FIXES:
            if not isinstance(c[key], bool):
                raise ValueError(f"{key} must be True or False")
        self.fix = {key[len("fix_") :]: c[key] for key in FIXES}
        self._validate()

        self.obs_size = SELF_SIZE + SLOT_SIZE * (self.observe_count["predator"] + self.observe_count["prey"]) + GRASS_SIZE + 1
        obs_space = Box(-1.0, 1.0, shape=(self.obs_size,), dtype=np.float32)
        action_space = Discrete(DIRECTIONS * len(SPEEDS))
        self.possible_agents = [f"{species}_{i}" for species in SPECIES for i in range(self.pool[species])]
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}
        self.action_spaces = {agent: action_space for agent in self.possible_agents}
        self.observation_space = Dict(self.observation_spaces)
        self.action_space = Dict(self.action_spaces)
        # For RLlib callers that still use get_agent_ids().
        self._agent_ids = set(self.possible_agents)
        self.agents = []

        self.predators = []
        self.prey = []
        self.entities = []  # movement order
        self.active = []  # agents still acting (Aquarium's agent list)
        self.time_step = 0
        self.birth_rng = random.Random()
        self.next_id = {}
        self.births = {}

    def _validate(self):
        c = self.config
        for species in SPECIES:
            group = "predators" if species == "predator" else "prey"
            for name in ("count", "observe_count"):
                if (
                    isinstance(c[f"{species}_{name}"], bool)
                    or not isinstance(c[f"{species}_{name}"], int)
                    or c[f"{species}_{name}"] < 0
                ):
                    raise ValueError(f"{species}_{name} must be a nonnegative integer")
            for name in ("observe_neighbors", "observe_neighbor_movement"):
                if not isinstance(c[f"{species}_{name}"], bool):
                    raise ValueError(f"{species}_{name} must be True or False")
            for name in (
                "radius",
                "mass",
                "max_velocity",
                "max_acceleration",
                "max_steer_force",
                "view_distance",
                "fov",
                f"energy_{species}_initial",
                f"energy_{species}_max",
                f"reproduction_{species}_threshold",
            ):
                key = name if name.startswith(("energy", "reproduction")) else f"{species}_{name}"
                value = c[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and value > 0):
                    raise ValueError(f"{key} must be finite and positive")
            for key in (
                f"energy_{species}_resting_metabolic_cost",
                f"energy_{species}_speed_cost",
                f"reproduction_{species}_reward",
            ):
                value = c[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and value >= 0):
                    raise ValueError(f"{key} must be finite and nonnegative")
            for key in (f"reproduction_max_{group}", f"reproduction_{species}_pool"):
                if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < 1:
                    raise ValueError(f"{key} must be a positive integer")
            if self.initial_energy[species] > self.max_energy[species]:
                raise ValueError(f"energy_{species}_initial must not exceed energy_{species}_max")
            # A parent pays the offspring's initial energy; at or below it a
            # parent could breed while left with nothing.
            if not self.initial_energy[species] < self.threshold[species] <= self.max_energy[species]:
                raise ValueError(
                    f"reproduction_{species}_threshold must exceed " f"energy_{species}_initial and not exceed energy_{species}_max"
                )
            if self.count[species] > min(self.max_population[species], self.pool[species]):
                raise ValueError(f"{species}_count must not exceed reproduction_max_{group} " "or the pool")
            # The observation's view check tests only the nearest image.
            if c[f"{species}_view_distance"] > min(self.width, self.height) / 2:
                raise ValueError(f"{species}_view_distance must not exceed half the arena")
        for key in ("energy_grass_gain", "energy_catch_efficiency_predator"):
            value = c[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.catch_efficiency > 1:
            raise ValueError("energy_catch_efficiency_predator must be in [0, 1]")
        value = c["energy_catch_share_radius"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and value >= 0):
            raise ValueError("energy_catch_share_radius must be finite and nonnegative")

    # --- Episode ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        self.birth_rng = random.Random(None if seed is None else seed + 1)
        self.grass.reset(seed)
        spawn = random.Random(seed)
        self.prey = []
        for i in range(self.count["prey"]):
            position = Vector(spawn.randint(20, self.width - 20), spawn.randint(20, self.height - 20))
            if not self.fix["no_spare_spawn_draw"]:
                spawn.randint(150, 250)  # Aquarium's unused replication age
            self.prey.append(self._animal("prey", i, position))
        self.predators = []
        for i in range(self.count["predator"]):
            position = Vector(spawn.randint(0, self.width), spawn.randint(0, self.height))
            self.predators.append(self._animal("predator", i, position))
        self.entities = self.prey + self.predators
        self.time_step = 0
        self.next_id = dict(self.count)
        self.births = {species: 0 for species in SPECIES}
        self.active = [a.id for a in self.predators + self.prey]
        self.agents = list(self.active)
        self.catch_log = []
        obs = self._observations()
        return obs, {agent: {} for agent in self.active}

    def _animal(self, species, number, position):
        # Initial velocity as in Aquarium: predators (0, -1), prey (-1, 0).
        velocity = Vector(0, -1) if species == "predator" else Vector(-1, 0)
        return Animal(
            f"{species}_{number}",
            species,
            position,
            velocity,
            self.initial_energy[species],
            self.settings[species],
        )

    def step(self, action_dict):
        """One decision: each agent's action for up to action_repeat physics
        steps, ending early after a birth or at the end of the episode.
        Actions for agents that no longer act are ignored."""
        missing = [agent for agent in self.active if agent not in action_dict]
        if missing:
            raise ValueError(f"no action for acting agents {missing}")
        obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}
        # This decision's catches: {"catcher", "prey", "fed", "gain"} (not part of the dynamics).
        self.catch_log = []
        actions = {agent: action_dict[agent] for agent in self.active}
        totals = {"grass_eaten": {}, "births": {}}
        zeros = np.zeros(self.obs_size, dtype=np.float32)
        for _ in range(self.action_repeat):
            step_agents, step_rewards, step_terms, step_truncs, step_infos, born = self._physics_step(actions)
            for agent in step_agents:
                obs[agent] = zeros  # replaced below for every living animal
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
            actions = {agent: action for agent, action in actions.items() if not (step_terms.get(agent) or step_truncs.get(agent))}
            if not actions:
                break
        obs.update(self._observations())
        for key, total in totals.items():
            for agent, count in total.items():
                infos[agent][key] = count
        self.agents = [agent for agent in terminateds if agent != "__all__"]
        return obs, rewards, terminateds, truncateds, infos

    # --- One physics step ------------------------------------------------

    def _physics_step(self, actions):
        """Move every animal once, then grass, energy and births (SPEC.md,
        section 3). Returns (agents in this step, rewards, terminateds,
        truncateds, infos, whether anyone was born)."""
        agents = list(self.active)
        catches = []  # (catching predator, prey energy at the catch, prey id, spot of the catch)
        if self.fix["simultaneous_movement"]:
            # Every contact from the positions at the start of the step,
            # then every move.
            moves = [(animal, *self._contacts(animal)) for animal in self.entities]
            for animal, catcher, aways in moves:
                self._move(animal, actions[animal.id], catcher, aways, catches)
        else:
            for animal in list(self.entities):
                self._move(animal, actions[animal.id], *self._contacts(animal), catches)

        living = {a.id for a in self.predators} | {a.id for a in self.prey}
        terminateds = {agent: agent not in living for agent in agents}
        truncateds = {agent: False for agent in agents}
        if self.fix["episode_length"]:
            # Exactly max_time_steps physics steps.
            out_of_time = self.time_step + 1 >= self.max_time_steps
        else:
            out_of_time = self.time_step > self.max_time_steps
        if out_of_time or not self.prey or not self.predators:
            truncateds = {agent: True for agent in agents}
        self.time_step += 1
        self.active = [agent for agent in agents if not (terminateds[agent] or truncateds[agent])]

        eaten = self.grass.advance(self.prey)
        rewards = {agent: 0.0 for agent in agents}
        infos = {
            agent: {
                "grass_eaten": eaten.get(agent, 0),
                "grass_eaten_total": self.grass.total_consumed,
            }
            for agent in agents
        }

        self._energy(catches, eaten, terminateds, truncateds, infos)

        born = self._reproduce(rewards, terminateds, truncateds, infos) if self.active else []
        for child, parent in born:
            rewards[child] = 0.0
            terminateds[child] = False
            truncateds[child] = False
            infos[child] = {"born": True, "parent": parent, "energy": self.initial_energy[child.split("_")[0]]}
            agents.append(child)

        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())
        if not self.prey:
            # Nothing is left to catch: the episode ended, not interrupted.
            terminateds = {agent: True for agent in terminateds}
            truncateds = {agent: False for agent in truncateds}
        elif not self.active and not (terminateds["__all__"] or truncateds["__all__"]):
            # A mixed ending (a starved predator terminated, surviving prey
            # truncated) sets neither flag, but no agent is left.
            truncateds["__all__"] = True
        return agents, rewards, terminateds, truncateds, infos, bool(born)

    def _desired_velocity(self, action, animal):
        if not isinstance(action, (int, np.integer)) or isinstance(action, bool) or not 0 <= action < DIRECTIONS * len(SPEEDS):
            raise ValueError("action must be an integer in the action space")
        velocity = direction_vector(int(action) % DIRECTIONS, DIRECTIONS)
        velocity.normalize()
        velocity.mult(animal.max_speed)
        velocity.mult(SPEEDS[int(action) // DIRECTIONS])
        return velocity

    def _first_contact(self, animal, others):
        """The first of others closer than the sum of radii, or None."""
        for other in others:
            if self.torus.distance(animal.position, other.position) < animal.radius + other.radius and other is not animal:
                return other
        return None

    def _contacts(self, animal):
        """(catching predator or None, unit vectors pointing away from the
        animals of its own species it bumps into), from the current
        positions."""
        catcher = self._first_contact(animal, self.predators) if animal.species == "prey" else None
        own = self.prey if animal.species == "prey" else self.predators
        if self.fix["all_contacts"]:
            others = [
                other
                for other in own
                if other is not animal and self.torus.distance(animal.position, other.position) < animal.radius + other.radius
            ]
        else:
            other = self._first_contact(animal, own)
            others = [] if other is None else [other]
        aways = []
        for other in others:
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

    def _finish_move(self, animal):
        animal.velocity.limit(animal.max_speed)
        animal.position.add(animal.velocity)
        if self.fix["wrap_borders"]:
            animal.position.x %= self.width
            animal.position.y %= self.height
        else:
            # Borders: set to the opposite edge, not wrapped (SPEC.md, 3.2).
            if animal.position.x > self.width:
                animal.position.x = 0
            elif animal.position.x < 0:
                animal.position.x = self.width
            if animal.position.y > self.height:
                animal.position.y = 0
            elif animal.position.y < 0:
                animal.position.y = self.height
        if not (self.fix["keep_heading_when_stopped"] and animal.velocity.mag() < STOPPED_SPEED):
            animal.orientation = round(heading_degrees(animal.velocity), 1)

    def _move(self, animal, action, catcher, aways, catches):
        """One animal's update (SPEC.md, 3.2 and 3.3), given its contacts."""
        desired = self._desired_velocity(action, animal)
        if catcher is not None:
            animal.alive = False
            catches.append((catcher, animal.energy, animal.id, animal.position.copy()))
            if self.fix["caught_prey_does_not_move"]:
                self.entities.remove(animal)
                self.prey.remove(animal)
                return
        self._steer(animal, desired)
        if animal.species == "prey" and not self.fix["same_collisions_for_both"]:
            if aways:
                # A bump replaces the acceleration: a unit push away.
                for away in aways:
                    animal.velocity.add(away)
            else:
                animal.velocity.add(animal.acceleration)
        else:
            # A bump adds to the acceleration: a push of top speed / 8.
            for away in aways:
                away.mult(animal.max_speed / 8)
                animal.acceleration.add(away)
            animal.velocity.add(animal.acceleration)
        self._finish_move(animal)
        if catcher is not None:
            self.entities.remove(animal)
            self.prey.remove(animal)

    def _gain(self, animal, amount):
        animal.energy = min(self.max_energy[animal.species], animal.energy + amount)

    def _energy(self, catches, eaten, terminateds, truncateds, infos):
        """Food, metabolic costs and starvation (SPEC.md, section 5)."""
        for predator, prey_energy, prey_id, spot in catches:
            fed = [predator] if predator.alive else []
            if self.catch_share_radius > 0:
                fed += [
                    other
                    for other in self.predators
                    if other is not predator and self.torus.distance(other.position, spot) <= self.catch_share_radius
                ]
            gain = self.catch_efficiency * prey_energy / len(fed) if fed else 0.0
            for animal in fed:
                self._gain(animal, gain)
            self.catch_log.append({"catcher": predator.id, "prey": prey_id, "fed": [a.id for a in fed], "gain": gain})
        for prey in self.prey:
            count = eaten.get(prey.id, 0)
            if count:
                self._gain(prey, count * self.grass_gain)
        living = self.predators + self.prey
        for animal in living:
            animal.energy = animal.energy - (
                self.resting_cost[animal.species] + self.speed_cost[animal.species] * animal.velocity.mag()
            )
        starved = [animal for animal in living if animal.energy <= 0]
        for animal in starved:
            animal.alive = False
            (self.predators if animal.species == "predator" else self.prey).remove(animal)
            self.entities.remove(animal)
            if animal.id in self.active:
                self.active.remove(animal.id)
            terminateds[animal.id] = True
            truncateds[animal.id] = False
            infos.setdefault(animal.id, {})["starved"] = True
        if starved and (not self.prey or not self.predators):
            for agent in self.active:
                truncateds[agent] = True
            self.active.clear()
        for animal in self.predators + self.prey:
            infos.setdefault(animal.id, {})["energy"] = animal.energy

    def _reproduce(self, rewards, terminateds, truncateds, infos):
        """Births at the energy threshold (SPEC.md, section 6). Returns
        [(child, parent)]."""
        population = {"predator": len(self.predators), "prey": len(self.prey)}
        active = set(self.active)
        born = []
        for parent in self.predators + self.prey:
            species = parent.species
            if (
                parent.energy < self.threshold[species]
                or parent.id not in active
                or population[species] >= self.max_population[species]
                or self.next_id[species] >= self.pool[species]
            ):
                continue
            distance = 2 * parent.radius
            angle = self.birth_rng.uniform(0, 2 * math.pi)
            position = Vector(
                (parent.position.x + distance * math.cos(angle)) % self.width,
                (parent.position.y + distance * math.sin(angle)) % self.height,
            )
            child = self._animal(species, self.next_id[species], position)
            self.next_id[species] += 1
            (self.predators if species == "predator" else self.prey).append(child)
            self.entities.append(child)
            self.active.append(child.id)
            parent.energy -= self.initial_energy[species]
            born.append((child.id, parent.id))
            population[species] += 1
            self.births[species] += 1
            rewards[parent.id] = rewards.get(parent.id, 0.0) + self.birth_reward[species]
            infos.setdefault(parent.id, {})["births"] = 1
            infos[parent.id]["energy"] = parent.energy
        return born

    # --- Observations ----------------------------------------------------

    def _observations(self):
        return {animal.id: self._observe(animal) for animal in self.predators + self.prey}

    def _slots(self, observer, species, count):
        animals = self.predators if species == "predator" else self.prey
        own_species = species == observer.species
        if own_species and not self.observe_neighbors[observer.species]:
            return [0.0] * (SLOT_SIZE * count)
        movement = not own_species or self.observe_neighbor_movement[observer.species]
        seen = []
        for animal in animals:
            if animal is observer:
                continue
            if not self.torus.in_view(observer, animal.position, observer.view_distance, observer.fov):
                continue
            dx, dy = self.torus.offset(observer.position, animal.position)
            seen.append((math.hypot(dx, dy), dx, dy, animal))
        seen.sort(key=lambda entry: entry[0])
        values = []
        reach = observer.view_distance
        for distance, dx, dy, animal in seen[:count]:
            values += [
                1.0,
                dx / reach,
                dy / reach,
                distance / reach,
                animal.velocity.x / animal.max_speed if movement else 0.0,
                animal.velocity.y / animal.max_speed if movement else 0.0,
            ]
        return values + [0.0] * (SLOT_SIZE * count - len(values))

    def _observe(self, animal):
        """SPEC.md, section 7.1."""
        heading = math.radians(animal.orientation)
        values = [
            animal.velocity.x / animal.max_speed,
            animal.velocity.y / animal.max_speed,
            math.sin(heading),
            math.cos(heading),
        ]
        values += self._slots(animal, "predator", self.observe_count["predator"])
        values += self._slots(animal, "prey", self.observe_count["prey"])
        obs = np.asarray(values, dtype=np.float32)
        # Speeds are capped at the top speed, so only rounding can overshoot.
        if not np.all(np.abs(obs) <= 1.0 + 1e-4):
            raise RuntimeError(f"observation out of [-1, 1]: {obs}")
        obs = np.clip(obs, -1.0, 1.0)
        grass = self.grass.observation(animal, animal.view_distance, animal.fov)
        energy = np.float32(np.clip(animal.energy / self.max_energy[animal.species], 0.0, 1.0))
        return np.append(np.concatenate((obs, grass)), energy)


def register() -> None:
    """Register the environment with Ray Tune under ENV_NAME."""
    register_env(ENV_NAME, PredPreyGrassContinuous)
