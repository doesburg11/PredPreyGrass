"""
Network Reciprocity environment — Step 2: fixed cooperation strategies.

Two prey subtypes with hard-coded strategies:
  cooperator_prey  — donates cooperation_cost * own_energy to every adjacent prey
                     each step. Recipients receive donation * cooperation_benefit_multiplier.
  defector_prey    — keeps all energy; never donates.

Predators are unchanged from base_environment and can be trained with PPO.

Spatial mechanism (Nowak & May 1992): because offspring spawn adjacent to parents,
cooperators that start near each other form clusters. Within the cluster each agent
both donates and receives, netting positive from the multiplier. Defectors at the
cluster boundary exploit edge cooperators, but the cluster interior is protected.
Whether clusters persist depends on the cost/multiplier ratio — tune via config.

Step 3 will add a clustering metric; Step 4 will replace fixed strategies with
learned policies.
"""
from predpreygrass.non_evolutionary.network_reciprocity.config.config_env import config_env

import numpy as np
from numpy.typing import NDArray
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, Dict, List, Tuple


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config = config or config_env

        self.verbose_engagement = config.get("verbose_engagement", False)
        self.verbose_movement = config.get("verbose_movement", False)
        self.verbose_spawning = config.get("verbose_spawning", False)
        self.verbose_cooperation = config.get("verbose_cooperation", False)

        self.max_steps = config.get("max_steps", 10000)

        # Rewards
        self.reward_predator_catch_prey = config.get("reward_predator_catch_prey", 0.0)
        self.reward_prey_eat_grass = config.get("reward_prey_eat_grass", 0.0)
        self.reward_predator_step = config.get("reward_predator_step", 0.0)
        self.reward_prey_step = config.get("reward_prey_step", 0.0)
        self.penalty_prey_caught = config.get("penalty_prey_caught", 0.0)
        self.reproduction_reward_predator = config.get("reproduction_reward_predator", 10.0)
        self.reproduction_reward_prey = config.get("reproduction_reward_prey", 10.0)

        # Energy settings
        self.energy_loss_per_step_predator = config.get("energy_loss_per_step_predator", 0.15)
        self.energy_loss_per_step_prey = config.get("energy_loss_per_step_prey", 0.05)
        self.predator_creation_energy_threshold = config.get("predator_creation_energy_threshold", 12.0)
        self.prey_creation_energy_threshold = config.get("prey_creation_energy_threshold", 8.0)

        # Cooperation parameters
        self.cooperation_cost = config.get("cooperation_cost", 0.05)
        self.cooperation_benefit_multiplier = config.get("cooperation_benefit_multiplier", 1.5)

        # Agent counts
        self.n_possible_predators = config.get("n_possible_predators", 50)
        self.n_possible_cooperator_prey = config.get("n_possible_cooperator_prey", 50)
        self.n_possible_defector_prey = config.get("n_possible_defector_prey", 50)
        self.n_initial_active_predator = config.get("n_initial_active_predator", 6)
        self.n_initial_active_cooperator_prey = config.get("n_initial_active_cooperator_prey", 10)
        self.n_initial_active_defector_prey = config.get("n_initial_active_defector_prey", 10)

        self.initial_energy_predator = config.get("initial_energy_predator", 5.0)
        self.initial_energy_prey = config.get("initial_energy_prey", 3.0)

        # Grid and Observation Settings
        self.grid_size = config.get("grid_size", 25)
        self.num_obs_channels = config.get("num_obs_channels", 4)
        self.predator_obs_range = config.get("predator_obs_range", 7)
        self.prey_obs_range = config.get("prey_obs_range", 9)

        # Grass settings
        self.initial_num_grass = config.get("initial_num_grass", 100)
        self.initial_energy_grass = config.get("initial_energy_grass", 2.0)
        self.energy_gain_per_step_grass = config.get("energy_gain_per_step_grass", 0.04)

        self.cumulative_rewards: Dict[AgentID, float] = {}

        self.possible_agents: List[AgentID] = (
            [f"predator_{i}" for i in range(self.n_possible_predators)]
            + [f"cooperator_prey_{i}" for i in range(self.n_possible_cooperator_prey)]
            + [f"defector_prey_{i}" for i in range(self.n_possible_defector_prey)]
        )
        self.agents: List[AgentID] = (
            [f"predator_{i}" for i in range(self.n_initial_active_predator)]
            + [f"cooperator_prey_{i}" for i in range(self.n_initial_active_cooperator_prey)]
            + [f"defector_prey_{i}" for i in range(self.n_initial_active_defector_prey)]
        )

        self.grass_agents: List[AgentID] = [f"grass_{k}" for k in range(self.initial_num_grass)]

        # Observation spaces — both prey subtypes share the same shape
        predator_obs_shape = (self.num_obs_channels, self.predator_obs_range, self.predator_obs_range)
        prey_obs_shape = (self.num_obs_channels, self.prey_obs_range, self.prey_obs_range)
        predator_obs_space = gymnasium.spaces.Box(low=0.0, high=100.0, shape=predator_obs_shape, dtype=np.float64)
        prey_obs_space = gymnasium.spaces.Box(low=0.0, high=100.0, shape=prey_obs_shape, dtype=np.float64)

        self.observation_spaces = {
            agent: predator_obs_space if "predator" in agent else prey_obs_space
            for agent in self.possible_agents
        }

        self.action_to_move_tuple: Dict[int, Tuple[int, int]] = {
            0: (-1, -1), 1: (-1, 0), 2: (-1, 1),
            3: (0, -1),  4: (0, 0),  5: (0, 1),
            6: (1, -1),  7: (1, 0),  8: (1, 1),
        }
        action_space = gymnasium.spaces.Discrete(len(self.action_to_move_tuple))
        self.action_spaces = {agent: action_space for agent in self.possible_agents}

        self.agent_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.predator_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.prey_positions: Dict[AgentID, Tuple[int, int]] = {}  # all prey regardless of type
        self.grass_positions: Dict[AgentID, Tuple[int, int]] = {}

        self.agent_energies: Dict[AgentID, float] = {}
        self.grass_energies: Dict[AgentID, float] = {}

        self.grid_world_state_shape: Tuple[int, int, int] = (
            self.num_obs_channels, self.grid_size, self.grid_size,
        )
        self.initial_grid_world_state: NDArray[np.float64] = np.zeros(self.grid_world_state_shape, dtype=np.float64)
        self.grid_world_state: NDArray[np.float64] = self.initial_grid_world_state.copy()
        self.num_actions = len(self.action_to_move_tuple)
        self.agents_just_ate: set = set()

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.rng = np.random.default_rng(seed)
        self.grid_world_state = self.initial_grid_world_state.copy()

        self.possible_agents = (
            [f"predator_{i}" for i in range(self.n_possible_predators)]
            + [f"cooperator_prey_{i}" for i in range(self.n_possible_cooperator_prey)]
            + [f"defector_prey_{i}" for i in range(self.n_possible_defector_prey)]
        )
        self.agents = (
            [f"predator_{i}" for i in range(self.n_initial_active_predator)]
            + [f"cooperator_prey_{i}" for i in range(self.n_initial_active_cooperator_prey)]
            + [f"defector_prey_{i}" for i in range(self.n_initial_active_defector_prey)]
        )

        self.agent_positions = {}
        self.predator_positions = {}
        self.prey_positions = {}
        self.agent_energies = {}
        self.cumulative_rewards = {agent: 0.0 for agent in self.agents}

        n_prey = self.n_initial_active_cooperator_prey + self.n_initial_active_defector_prey
        total_entities = self.n_initial_active_predator + n_prey + self.initial_num_grass
        all_positions = self._generate_unique_positions(self.grid_size, total_entities, seed)

        predator_positions = all_positions[: self.n_initial_active_predator]
        prey_positions = all_positions[
            self.n_initial_active_predator : self.n_initial_active_predator + n_prey
        ]
        grass_positions = all_positions[self.n_initial_active_predator + n_prey :]

        prey_list = (
            [f"cooperator_prey_{i}" for i in range(self.n_initial_active_cooperator_prey)]
            + [f"defector_prey_{i}" for i in range(self.n_initial_active_defector_prey)]
        )

        for i, agent in enumerate([f"predator_{i}" for i in range(self.n_initial_active_predator)]):
            pos = predator_positions[i]
            self.agent_positions[agent] = pos
            self.predator_positions[agent] = pos
            self.agent_energies[agent] = self.initial_energy_predator
            self.grid_world_state[1, *pos] = self.initial_energy_predator

        for i, agent in enumerate(prey_list):
            pos = prey_positions[i]
            self.agent_positions[agent] = pos
            self.prey_positions[agent] = pos
            self.agent_energies[agent] = self.initial_energy_prey
            self.grid_world_state[2, *pos] = self.initial_energy_prey

        self.grass_positions = {}
        self.grass_energies = {}
        for i, grass in enumerate(self.grass_agents):
            pos = grass_positions[i]
            self.grass_positions[grass] = pos
            self.grass_energies[grass] = self.initial_energy_grass
            self.grid_world_state[3, *pos] = self.initial_energy_grass

        self.current_num_predators = self.n_initial_active_predator
        self.current_num_cooperator_prey = self.n_initial_active_cooperator_prey
        self.current_num_defector_prey = self.n_initial_active_defector_prey

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        return observations, {}

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        # Step 0: truncation
        if self.current_step >= self.max_steps:
            for agent in self.possible_agents:
                if agent in self.agents:
                    observations[agent] = self._get_observation(agent)
                else:
                    obs_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
                    observations[agent] = np.zeros(
                        (self.num_obs_channels, obs_range, obs_range), dtype=np.float64
                    )
                rewards[agent] = 0.0
                truncations[agent] = True
                terminations[agent] = False
            truncations["__all__"] = True
            terminations["__all__"] = False
            return observations, rewards, terminations, truncations, infos

        self.agents_just_ate.clear()

        # Step 1: energy decay
        for agent in action_dict:
            if "predator" in agent:
                self.agent_energies[agent] -= self.energy_loss_per_step_predator
                self.grid_world_state[1, *self.agent_positions[agent]] = self.agent_energies[agent]
            elif "prey" in agent:
                self.agent_energies[agent] -= self.energy_loss_per_step_prey
                self.grid_world_state[2, *self.agent_positions[agent]] = self.agent_energies[agent]

        # Step 2: grass regeneration
        for grass, pos in self.grass_positions.items():
            self.grass_energies[grass] = min(
                self.grass_energies[grass] + self.energy_gain_per_step_grass,
                self.initial_energy_grass,
            )
            self.grid_world_state[3, *pos] = self.grass_energies[grass]

        # Step 3: movement
        for agent, action in action_dict.items():
            if agent not in self.agent_positions:
                continue
            old_pos = self.agent_positions[agent]
            new_pos = self._get_move(agent, action)
            self.agent_positions[agent] = new_pos
            if "predator" in agent:
                self.predator_positions[agent] = new_pos
                self.grid_world_state[1, *old_pos] = 0
                self.grid_world_state[1, *new_pos] = self.agent_energies[agent]
            elif "prey" in agent:
                self.prey_positions[agent] = new_pos
                self.grid_world_state[2, *old_pos] = 0
                self.grid_world_state[2, *new_pos] = self.agent_energies[agent]

            if self.verbose_movement:
                print(f"[MOVE] {agent}: {old_pos} -> {new_pos}")

        # Step 4: cooperation donations (cooperator_prey only)
        self._apply_prey_cooperation()

        # Step 5: engagements, terminations
        for agent in self.agents:
            if agent not in self.agent_positions:
                continue

            if self.agent_energies[agent] <= 0:
                observations[agent] = self._get_observation(agent)
                rewards[agent] = 0.0
                terminations[agent] = True
                truncations[agent] = False
                self._remove_agent_from_grid(agent)
                continue

            if "predator" in agent:
                caught_prey = next(
                    (p for p, pos in self.agent_positions.items()
                     if "prey" in p and np.array_equal(self.agent_positions[agent], pos)),
                    None,
                )
                if caught_prey:
                    if self.verbose_engagement:
                        print(f"[ENGAGE] {agent} caught {caught_prey}")
                    self.agents_just_ate.add(agent)
                    self.agent_energies[agent] += self.agent_energies[caught_prey]
                    self.grid_world_state[1, *self.agent_positions[agent]] = self.agent_energies[agent]

                    observations[caught_prey] = self._get_observation(caught_prey)
                    rewards[caught_prey] = self.penalty_prey_caught
                    self.cumulative_rewards[caught_prey] = self.cumulative_rewards.get(caught_prey, 0) + rewards[caught_prey]
                    terminations[caught_prey] = True
                    truncations[caught_prey] = False
                    self._remove_agent_from_grid(caught_prey)

                    rewards[agent] = self.reward_predator_catch_prey
                else:
                    rewards[agent] = self.reward_predator_step

                observations[agent] = self._get_observation(agent)
                self.cumulative_rewards[agent] = self.cumulative_rewards.get(agent, 0) + rewards[agent]
                terminations[agent] = False
                truncations[agent] = False

            elif "prey" in agent:
                if terminations.get(agent):
                    continue
                prey_pos = self.agent_positions[agent]
                caught_grass = next(
                    (g for g, pos in self.grass_positions.items()
                     if np.array_equal(prey_pos, pos)),
                    None,
                )
                if caught_grass:
                    if self.verbose_engagement:
                        print(f"[ENGAGE] {agent} ate grass at {prey_pos}")
                    self.agents_just_ate.add(agent)
                    self.agent_energies[agent] += self.grass_energies[caught_grass]
                    self.grid_world_state[2, *prey_pos] = self.agent_energies[agent]
                    self.grid_world_state[3, *self.grass_positions[caught_grass]] = 0
                    self.grass_energies[caught_grass] = 0
                    rewards[agent] = self.reward_prey_eat_grass
                else:
                    rewards[agent] = self.reward_prey_step

                observations[agent] = self._get_observation(agent)
                self.cumulative_rewards[agent] = self.cumulative_rewards.get(agent, 0) + rewards[agent]
                terminations[agent] = False
                truncations[agent] = False

        # Step 6: remove terminated agents
        for agent in self.agents[:]:
            if terminations.get(agent):
                if self.verbose_engagement:
                    print(f"[TERMINATED] {agent}")
                self.agents.remove(agent)

        # Step 7: reproduction (offspring inherit parent's strategy)
        for agent in self.agents[:]:
            if "predator" in agent:
                if self.agent_energies.get(agent, 0) >= self.predator_creation_energy_threshold:
                    self._spawn_offspring(agent, prefix="predator_", grid_channel=1)

            elif "cooperator_prey" in agent:
                if self.agent_energies.get(agent, 0) >= self.prey_creation_energy_threshold:
                    self._spawn_offspring(agent, prefix="cooperator_prey_", grid_channel=2)

            elif "defector_prey" in agent:
                if self.agent_energies.get(agent, 0) >= self.prey_creation_energy_threshold:
                    self._spawn_offspring(agent, prefix="defector_prey_", grid_channel=2)

        # Step 8: refresh observations
        for agent in self.agents:
            if agent in self.agent_positions:
                observations[agent] = self._get_observation(agent)

        self.current_num_predators = sum(1 for a in self.agents if "predator" in a)
        self.current_num_cooperator_prey = sum(1 for a in self.agents if "cooperator_prey" in a)
        self.current_num_defector_prey = sum(1 for a in self.agents if "defector_prey" in a)
        current_num_prey = self.current_num_cooperator_prey + self.current_num_defector_prey

        terminations["__all__"] = current_num_prey <= 0 or self.current_num_predators <= 0

        observations = {a: observations[a] for a in self.agents if a in observations}
        rewards = {a: rewards[a] for a in self.agents if a in rewards}
        terminations = {a: terminations[a] for a in self.agents if a in terminations}
        truncations = {a: truncations.get(a, False) for a in self.agents}
        truncations["__all__"] = False

        terminations["__all__"] = current_num_prey <= 0 or self.current_num_predators <= 0

        self.agents.sort()
        self.current_step += 1

        return observations, rewards, terminations, truncations, infos

    # -------------------------------------------------------------------------
    # Cooperation mechanic
    # -------------------------------------------------------------------------

    def _apply_prey_cooperation(self):
        """
        cooperator_prey donate cooperation_cost * own_energy to every adjacent prey.
        The recipient gains donation * cooperation_benefit_multiplier.
        Defection is individually rational (isolated cooperators lose energy),
        but cooperator clusters net positive via the multiplier.
        """
        for agent in list(self.agents):
            if "cooperator_prey" not in agent:
                continue
            if agent not in self.agent_positions:
                continue
            for neighbour in self._get_adjacent_prey(agent):
                donation = self.cooperation_cost * self.agent_energies[agent]
                self.agent_energies[agent] -= donation
                self.agent_energies[neighbour] += donation * self.cooperation_benefit_multiplier
                self.grid_world_state[2, *self.agent_positions[agent]] = self.agent_energies[agent]
                self.grid_world_state[2, *self.agent_positions[neighbour]] = self.agent_energies[neighbour]
                if self.verbose_cooperation:
                    print(
                        f"[COOP] {agent} donates {donation:.3f} to {neighbour} "
                        f"(recipient gains {donation * self.cooperation_benefit_multiplier:.3f})"
                    )

    def _get_adjacent_prey(self, agent: AgentID) -> List[AgentID]:
        """Return all prey (any type) in the Moore neighbourhood of agent."""
        x, y = self.agent_positions[agent]
        adjacent = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbour_pos = (x + dx, y + dy)
                for other, pos in self.prey_positions.items():
                    if pos == neighbour_pos and other in self.agent_energies:
                        adjacent.append(other)
        return adjacent

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _spawn_offspring(self, parent: AgentID, prefix: str, grid_channel: int):
        candidates = [a for a in self.possible_agents if a not in self.agents and a.startswith(prefix)]
        if not candidates:
            if self.verbose_spawning:
                print(f"[SPAWN] No slots available for {prefix}")
            return
        new_agent = candidates[0]
        occupied = set(self.agent_positions.values())
        new_pos = self._find_available_spawn_position(self.agent_positions[parent], occupied)
        if new_pos is None:
            return

        self.agents.append(new_agent)
        self.agent_positions[new_agent] = new_pos
        self.agent_energies[new_agent] = self.initial_energy_prey if "prey" in new_agent else self.initial_energy_predator
        self.agent_energies[parent] -= self.agent_energies[new_agent]
        self.grid_world_state[grid_channel, *new_pos] = self.agent_energies[new_agent]
        self.grid_world_state[grid_channel, *self.agent_positions[parent]] = self.agent_energies[parent]
        if "prey" in new_agent:
            self.prey_positions[new_agent] = new_pos
        else:
            self.predator_positions[new_agent] = new_pos

        rewards_parent = self.reproduction_reward_prey if "prey" in parent else self.reproduction_reward_predator
        self.cumulative_rewards[parent] = self.cumulative_rewards.get(parent, 0) + rewards_parent
        self.cumulative_rewards[new_agent] = 0.0

        if self.verbose_spawning:
            print(f"[SPAWN] {new_agent} born at {new_pos} from {parent}")

    def _remove_agent_from_grid(self, agent: AgentID):
        if agent not in self.agent_positions:
            return
        pos = self.agent_positions[agent]
        if "predator" in agent:
            self.grid_world_state[1, *pos] = 0
            self.predator_positions.pop(agent, None)
        elif "prey" in agent:
            self.grid_world_state[2, *pos] = 0
            self.prey_positions.pop(agent, None)
        del self.agent_positions[agent]
        del self.agent_energies[agent]

    def _get_move(self, agent: AgentID, action: int) -> Tuple[int, int]:
        agent_type_nr = 1 if "predator" in agent else 2
        x, y = self.agent_positions[agent]
        dx, dy = self.action_to_move_tuple[action]
        new_pos = tuple(np.clip((x + dx, y + dy), 0, self.grid_size - 1))
        if self.grid_world_state[agent_type_nr, *new_pos] > 0:
            new_pos = (x, y)
        return new_pos

    def _get_observation(self, agent: AgentID) -> NDArray[np.float64]:
        obs_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp, obs_range)
        observation = np.zeros((self.num_obs_channels, obs_range, obs_range), dtype=np.float64)
        observation[0].fill(1)
        observation[0, xolo:xohi, yolo:yohi] = 0
        observation[1:, xolo:xohi, yolo:yohi] = self.grid_world_state[1:, xlo:xhi, ylo:yhi]
        return observation

    def _obs_clip(self, x, y, observation_range):
        offset = (observation_range - 1) // 2
        xld, xhd = x - offset, x + offset
        yld, yhd = y - offset, y + offset
        xlo = np.clip(xld, 0, self.grid_size - 1)
        xhi = np.clip(xhd, 0, self.grid_size - 1)
        ylo = np.clip(yld, 0, self.grid_size - 1)
        yhi = np.clip(yhd, 0, self.grid_size - 1)
        xolo = abs(np.clip(xld, -offset, 0))
        yolo = abs(np.clip(yld, -offset, 0))
        xohi = xolo + (xhi - xlo)
        yohi = yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def _find_available_spawn_position(self, reference_pos, occupied_positions):
        x, y = reference_pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            pos = (x + dx, y + dy)
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size and pos not in occupied_positions:
                return pos
        free = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) not in occupied_positions
        ]
        return free[np.random.randint(len(free))] if free else None

    @staticmethod
    def _generate_unique_positions(grid_size: int, num_positions: int, seed=None):
        if num_positions > grid_size * grid_size:
            raise ValueError("Cannot place more unique positions than grid cells.")
        rng = np.random.default_rng(seed)
        positions: set = set()
        while len(positions) < num_positions:
            pos = tuple(rng.integers(0, grid_size, size=2))
            positions.add(pos)
        return list(positions)

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def get_cooperation_stats(self) -> dict:
        """
        Return current population counts and a simple spatial clustering score.
        The clustering score is the average fraction of each cooperator's Moore
        neighbours that are also cooperators. 1.0 = fully clustered, 0.0 = isolated.
        """
        n_coop = self.current_num_cooperator_prey
        n_defect = self.current_num_defector_prey
        total_prey = n_coop + n_defect

        clustering_score = 0.0
        if n_coop > 0:
            scores = []
            for agent in self.agents:
                if "cooperator_prey" not in agent or agent not in self.agent_positions:
                    continue
                neighbours = self._get_adjacent_prey(agent)
                if neighbours:
                    coop_neighbours = sum(1 for n in neighbours if "cooperator_prey" in n)
                    scores.append(coop_neighbours / len(neighbours))
            clustering_score = float(np.mean(scores)) if scores else 0.0

        return {
            "step": self.current_step,
            "cooperators": n_coop,
            "defectors": n_defect,
            "total_prey": total_prey,
            "predators": self.current_num_predators,
            "cooperator_fraction": n_coop / total_prey if total_prey > 0 else 0.0,
            "cooperator_clustering": clustering_score,
        }

    # -------------------------------------------------------------------------
    # Snapshot (for step-back in visualiser)
    # -------------------------------------------------------------------------

    def get_state_snapshot(self):
        return {
            "current_step": self.current_step,
            "agent_positions": self.agent_positions.copy(),
            "agent_energies": self.agent_energies.copy(),
            "predator_positions": self.predator_positions.copy(),
            "prey_positions": self.prey_positions.copy(),
            "grass_positions": self.grass_positions.copy(),
            "grass_energies": self.grass_energies.copy(),
            "grid_world_state": self.grid_world_state.copy(),
            "agents": self.agents.copy(),
            "cumulative_rewards": self.cumulative_rewards.copy(),
            "current_num_predators": self.current_num_predators,
            "current_num_cooperator_prey": self.current_num_cooperator_prey,
            "current_num_defector_prey": self.current_num_defector_prey,
            "agents_just_ate": self.agents_just_ate.copy(),
        }

    def restore_state_snapshot(self, snapshot):
        self.current_step = snapshot["current_step"]
        self.agent_positions = snapshot["agent_positions"].copy()
        self.agent_energies = snapshot["agent_energies"].copy()
        self.predator_positions = snapshot["predator_positions"].copy()
        self.prey_positions = snapshot["prey_positions"].copy()
        self.grass_positions = snapshot["grass_positions"].copy()
        self.grass_energies = snapshot["grass_energies"].copy()
        self.grid_world_state = snapshot["grid_world_state"].copy()
        self.agents = snapshot["agents"].copy()
        self.cumulative_rewards = snapshot["cumulative_rewards"].copy()
        self.current_num_predators = snapshot["current_num_predators"]
        self.current_num_cooperator_prey = snapshot["current_num_cooperator_prey"]
        self.current_num_defector_prey = snapshot["current_num_defector_prey"]
        self.agents_just_ate = snapshot["agents_just_ate"].copy()
