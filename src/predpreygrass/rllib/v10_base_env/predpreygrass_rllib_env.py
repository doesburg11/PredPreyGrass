"""
Predator-Prey Grass RLlib Environment
v_10_base_env
Modualized code for all steps in the step function
(like experimental_6 from v7_modular)
"""

# external libraries
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, Dict, List, Tuple
import numpy as np
import math
import os
import json


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()

        self._initialization()
        self._load_config(config)
        self._create_learning_agent_lists()
        self._create_grid_world()
        self._create_spaces()

        # Non-learning agents (grass); not included in 'possible_agents' or 'agents'
        self.grass_agents: List[AgentID] = [f"grass_{k}" for k in range(self.initial_num_grass)]

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)

        self._initialization()
        self.current_step = 0
        self.rng = np.random.default_rng(seed)
        self.grid_world_state = np.zeros(self.grid_world_state_shape, dtype=np.float64)
        # Reset learning agent
        self._create_learning_agent_lists()

        # create grass agent list
        self.grass_agents = [f"grass_{k}" for k in range(self.initial_num_grass)]

        # Generate random positions for learning agents on grid
        n_active_entities = len(self.agents) + len(self.grass_agents)
        active_entities_positions = self._generate_random_positions(self.grid_size, n_active_entities, seed=seed)
        # print(f"Active entities positions: {active_entities_positions}")
        # print(f"Number of active entities: {n_active_entities}")

        # Assign positions to learning agents
        predator_positions = active_entities_positions[: len([a for a in self.agents if "predator" in a])]
        # print(f"Predator positions: {predator_positions}")
        # print(f"Number of predators: {len(predator_positions)}")
        prey_positions = active_entities_positions[
            len(predator_positions) : len(predator_positions) + len([a for a in self.agents if "prey" in a])
        ]
        # print(f"Prey positions: {prey_positions}")
        # print(f"Number of prey: {len(prey_positions)}")
        grass_positions = active_entities_positions[len(predator_positions) + len(prey_positions) :]
        # print(f"Grass positions: {grass_positions}")

        # Assign predators
        for i, agent in enumerate([a for a in self.agents if "predator" in a]):
            pos = predator_positions[i]
            self.predator_positions[agent] = pos
            self._place_agent(agent, pos, self.initial_energy_predator, self.initial_hydration_predator, layer_index=1)

        # Assign prey
        for i, agent in enumerate([a for a in self.agents if "prey" in a]):
            pos = prey_positions[i]
            self.prey_positions[agent] = pos
            self._place_agent(agent, pos, self.initial_energy_prey, self.initial_hydration_prey, layer_index=2)

        # Assign random grass positions and energy
        for i, grass in enumerate(self.grass_agents):
            pos = grass_positions[i]
            self.grass_positions[grass] = pos
            self.grass_energies[grass] = self.initial_energy_grass
            self.grid_world_state[3, *pos] = self.initial_energy_grass

        # Track counts
        self.active_num_predators = len(self.predator_positions)
        self.active_num_prey = len(self.prey_positions)
        # Generate observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        return observations, {}

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        # step 0: check for truncation
        truncation_result = self._check_truncation_and_early_return(observations, rewards, terminations, truncations, infos)
        if truncation_result is not None:
            return truncation_result

        # Step 1: If not truncated; process energy depletion due to time steps and update age
        self._apply_homeostatic_loss(action_dict)

        # Step 2: Update ages of all agents who act
        self._apply_age_update(action_dict)

        # Step 3: Regenerate grass energy
        self._regenerate_grass_energy()

        # Step 4: process agent movements
        self._process_agent_movements(action_dict)

        # Step 5: Handle agent engagements
        for agent in self.agents:
            if agent not in self.agent_positions:
                continue
            starvation = self.agent_energies[agent] <= 0
            if starvation:
                self._handle_homeostatic_depletion(agent, starvation, observations, rewards, terminations, truncations)
            elif "predator" in agent:
                self._handle_predator_engagement(agent, observations, rewards, terminations, truncations)
            elif "prey" in agent:
                self._handle_prey_engagement(agent, observations, rewards, terminations, truncations)

        # Step 6: Handle agent removals
        for agent in self.agents[:]:
            if terminations[agent]:
                self._log(self.verbose_termination, f"[TERMINATED] Agent {agent} terminated!", "red")
                self.agents.remove(agent)

        # Step 7: Spawning of new agents
        for agent in self.agents[:]:
            if "predator" in agent:
                self._handle_predator_reproduction(agent, rewards, observations, terminations, truncations)
            elif "prey" in agent:
                self._handle_prey_reproduction(agent, rewards, observations, terminations, truncations)

        # Step 8: Generate observations for all agents AFTER all engagements in the step
        for agent in self.agents:
            if agent in self.agent_positions:
                observations[agent] = self._get_observation(agent)

        # Global termination and truncation
        terminations["__all__"] = self.active_num_prey <= 0 or self.active_num_predators <= 0

        # output only observations, rewards for active agents
        observations = {agent: observations[agent] for agent in self.agents if agent in observations}
        rewards = {agent: rewards[agent] for agent in self.agents if agent in rewards}
        terminations = {agent: terminations[agent] for agent in self.agents if agent in terminations}
        truncations = {agent: truncations[agent] for agent in self.agents if agent in truncations}
        truncations["__all__"] = False  # already handled at the beginning of the step

        # Global termination and truncation
        terminations["__all__"] = self.active_num_prey <= 0 or self.active_num_predators <= 0

        self.agents.sort()  # Sort agents

        # Increment step counter
        self.current_step += 1

        self._export_grid_to_file(self.grid_world_state, self.current_step)

        return observations, rewards, terminations, truncations, infos

    def _get_movement_energy_cost(self, agent, current_position, new_position):
        """
        Calculate energy cost for movement based on distance and a configurable factor.
        """
        distance_factor = self.config.get("move_energy_cost_factor", 0.1)
        # print(f"Distance factor: {distance_factor}")
        current_energy = self.agent_energies[agent]
        # print(f"Current energy: {current_energy}")
        # distance gigh speed =[0.00,1.00, 1.41, 2.00, 2.24, 2.83]
        distance = math.sqrt((new_position[0] - current_position[0]) ** 2 + (new_position[1] - current_position[1]) ** 2)
        # print (f"Distance: {distance}")
        energy_cost = distance * distance_factor * current_energy
        return energy_cost

    def _get_move(self, agent: AgentID, action: int) -> Tuple[int, int]:
        """
        Get the new position of the agent based on the action and its speed.
        """
        current_position = self.agent_positions[agent]

        # Choose the appropriate movement dictionary based on agent speed
        if "speed_1" in agent:
            move_vector = self.action_to_move_tuple_speed_1_agents[action]
        elif "speed_2" in agent:
            move_vector = self.action_to_move_tuple_speed_2_agents[action]
        else:
            raise ValueError(f"Unknown speed for agent: {agent}")

        new_position = (
            current_position[0] + move_vector[0],
            current_position[1] + move_vector[1],
        )
        # Clip new position to stay within grid bounds
        new_position = tuple(np.clip(new_position, 0, self.grid_size - 1))

        agent_type_nr = 1 if "predator" in agent else 2
        if self.grid_world_state[agent_type_nr, *new_position] > 0:
            # Collision with another same-type agent — stay in place
            new_position = current_position

        crossing = (current_position not in self.river_cells and new_position in self.river_cells) or (
            current_position in self.river_cells and new_position not in self.river_cells
        )
        if crossing:
            y = new_position[1] if current_position[1] == new_position[1] else current_position[1]
            river_width = sum((x, y) in self.river_cells for x in range(self.grid_size))
            if river_width > 1:
                return current_position

        return new_position

    def _get_observation(self, agent):
        """
        Generate an observation for the agent.
        """
        observation_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp, observation_range)
        observation = np.zeros(
            (self.num_obs_channels, observation_range, observation_range),
            dtype=np.float64,
        )
        observation[0].fill(1)
        observation[0, xolo:xohi, yolo:yohi] = 0
        observation[1:, xolo:xohi, yolo:yohi] = self.grid_world_state[1:, xlo:xhi, ylo:yhi]

        return observation

    def _obs_clip(self, x, y, observation_range):
        """
        Clip the observation window to the boundaries of the grid_world_state.
        """
        observation_offset = (observation_range - 1) // 2
        xld, xhd = x - observation_offset, x + observation_offset
        yld, yhd = y - observation_offset, y + observation_offset
        xlo, xhi = np.clip(xld, 0, self.grid_size - 1), np.clip(xhd, 0, self.grid_size - 1)
        ylo, yhi = np.clip(yld, 0, self.grid_size - 1), np.clip(yhd, 0, self.grid_size - 1)
        xolo, yolo = abs(np.clip(xld, -observation_offset, 0)), abs(np.clip(yld, -observation_offset, 0))
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def _print_grid_from_positions(self):
        print(f"\nCurrent Grid State (IDs):  predators: {self.active_num_predators} prey: {self.active_num_prey}  \n")

        # Initialize empty grids (not transposed yet)
        predator_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        prey_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grass_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Populate Predator Grid
        for agent, pos in self.predator_positions.items():
            x, y = pos
            parts = agent.split("_")  # ['speed', '1', 'predator', '11']
            speed = parts[1]
            agent_num = parts[3]
            predator_grid[y][x] = f"{speed}_{agent_num}".center(5)

        # Populate Prey Grid
        for agent, pos in self.prey_positions.items():
            x, y = pos
            parts = agent.split("_")  # ['speed', '1', 'prey', '11']
            speed = parts[1]
            agent_num = parts[3]
            prey_grid[y][x] = f"{speed}_{agent_num}".center(5)

        # Populate Grass Grid
        for agent, pos in self.grass_positions.items():
            x, y = pos
            agent_num = int(agent.split("_")[1])
            grass_grid[y][x] = f"G{agent_num:02d}".center(5)

        # Transpose the grids (rows become columns)
        predator_grid = list(map(list, zip(*predator_grid)))
        prey_grid = list(map(list, zip(*prey_grid)))
        grass_grid = list(map(list, zip(*grass_grid)))

        # Print Headers
        print(
            f"{'Predators'.center(self.grid_size * 6)}   {'Prey'.center(self.grid_size * 6)}   {'Grass'.center(self.grid_size * 6)}"
        )
        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

        # Print Transposed Grids
        for x in range(self.grid_size):  # Now iterating over transposed rows (original columns)
            predator_row = " ".join(predator_grid[x])
            prey_row = " ".join(prey_grid[x])
            grass_row = " ".join(grass_grid[x])
            print(f"{predator_row}     {prey_row}     {grass_row}")

        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

    def _print_grid_from_state(self):
        print(
            f"\nCurrent Grid State (Energy & Hydration Levels):  predators: {self.active_num_predators} prey: {self.active_num_prey} \n"
        )

        # Initialize empty grids
        predator_energy_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        prey_energy_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grass_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        water_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        predator_hydration_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        prey_hydration_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Fill the grids
        for agent, pos in self.predator_positions.items():
            x, y = pos
            energy = self.agent_energies.get(agent, 0)
            hydration = self.agent_hydration.get(agent, 0)
            predator_energy_grid[y][x] = f"{energy:4.2f}".center(5)
            predator_hydration_grid[y][x] = f"{hydration:4.2f}".center(5)

        for agent, pos in self.prey_positions.items():
            x, y = pos
            energy = self.agent_energies.get(agent, 0)
            hydration = self.agent_hydration.get(agent, 0)
            prey_energy_grid[y][x] = f"{energy:4.2f}".center(5)
            prey_hydration_grid[y][x] = f"{hydration:4.2f}".center(5)

        for agent, pos in self.grass_positions.items():
            x, y = pos
            energy = self.grass_energies.get(agent, 0)
            grass_grid[y][x] = f"{energy:4.2f}".center(5)

        for x, y in self.river_cells:
            water_grid[y][x] = f"{1.00:4.2f}".center(5)

        # Transpose grids for column-wise display
        def transpose(g):
            return [[g[x][y] for x in range(self.grid_size)] for y in range(self.grid_size)]

        predator_energy_grid = transpose(predator_energy_grid)
        prey_energy_grid = transpose(prey_energy_grid)
        grass_grid = transpose(grass_grid)
        water_grid = transpose(water_grid)
        predator_hydration_grid = transpose(predator_hydration_grid)
        prey_hydration_grid = transpose(prey_hydration_grid)

        # Print headers
        print(
            f"{'Pred_E'.center(self.grid_size * 6)}   {'Prey_E'.center(self.grid_size * 6)}   {'Grass'.center(self.grid_size * 6)}   {'Water'.center(self.grid_size * 6)}"
        )
        print(
            "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6
        )

        for row in range(self.grid_size):
            pe = " ".join(predator_energy_grid[row])
            pr = " ".join(prey_energy_grid[row])
            gr = " ".join(grass_grid[row])
            wa = " ".join(water_grid[row])
            print(f"{pe}     {pr}     {gr}     {wa}")

        print("\n" + "-" * (self.grid_size * 6 * 4))

        print(f"{'Pred_H'.center(self.grid_size * 6)}   {'Prey_H'.center(self.grid_size * 6)}")
        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

        for row in range(self.grid_size):
            ph = " ".join(predator_hydration_grid[row])
            prh = " ".join(prey_hydration_grid[row])
            print(f"{ph}     {prh}")

        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

    def _print_movement_table(
        self,
        action_dict,
        predator_position_after_action,
        prey_new_unresolved_positions,
        resolved_positions,
        colliding_predator_agents,
        colliding_prey_agents,
    ):
        """
        Prints the movement table for predators and prey, including actions, positions, and energy levels.
        """

        print("\nPredator Position Table:")
        print(
            "{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<20}".format(
                "Agent", "Tuple", "Energy", "Array", "Action", "Action Array", "New", "Resolved"
            )
        )
        print("-" * 120)

        for i, (agent, position) in enumerate(self.predator_positions.items()):
            array_position = np.array(position)
            action_number = action_dict[agent]
            action_array = np.array(self.action_to_move_tuple[action_number])
            new_position = predator_position_after_action[i]
            resolved_position = resolved_positions[agent]  # Position after collision resolution
            energy = self.agent_energies[agent]

            print(
                "{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<20}".format(
                    agent,
                    str(position),
                    f"{energy:.2f}",
                    str(array_position),
                    action_number,
                    str(action_array),
                    str(new_position),
                    str(resolved_position),
                )
            )

        print("-" * 120)
        print()
        print("Colliding Predators:", colliding_predator_agents)
        print()

        print("\nPrey Position Table:")
        print(
            "{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<20}".format(
                "Agent", "Tuple", "Energy", "Array", "Action", "Action Array", "New", "Resolved"
            )
        )
        print("-" * 120)

        for i, (agent, position) in enumerate(self.prey_positions.items()):
            array_position = np.array(position)
            action_number = action_dict[agent]
            action_array = np.array(self.action_to_move_tuple[action_number])
            new_position = prey_new_unresolved_positions[i]
            resolved_position = resolved_positions[agent]  # Position after collision resolution
            energy = self.agent_energies[agent]

            print(
                "{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<20}".format(
                    agent,
                    str(position),
                    f"{energy:.2f}",
                    str(array_position),
                    action_number,
                    str(action_array),
                    str(new_position),
                    str(resolved_position),
                )
            )

        print("-" * 120)
        print()
        print("Colliding Prey:", colliding_prey_agents)
        print()

    def _find_available_spawn_position(self, reference_position, occupied_positions):
        """
        Finds an available position for spawning a new agent.
        Tries to spawn near the parent agent first before selecting a random free position.
        """
        # Get all occupied positions
        # occupied_positions = set(self.agent_positions.values()) | set(self.grass_positions.values())

        x, y = reference_position  # Parent agent's position
        potential_positions = [
            (x + dx, y + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
            if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size  # Stay in bounds
        ]
        # Filter for unoccupied positions
        valid_positions = [pos for pos in potential_positions if pos not in occupied_positions]

        if valid_positions:
            return valid_positions[0]  # Prefer adjacent position if available

        # Fallback: Find any random unoccupied position
        active_agents_positions = {(i, j) for i in range(self.grid_size) for j in range(self.grid_size)}
        free_positions = list(active_agents_positions - occupied_positions)

        if free_positions:
            return free_positions[self.rng.integers(len(free_positions))]

        return None  # No available position found

    def _log(self, verbose: bool, message: str, color: str = None):
        """
        Log with sharp 90° box-drawing borders (Unicode), optional color.

        Args:
            verbose (bool): Whether to log this message (per category).
            message (str): Message text (can be multi-line).
            color (str, optional): One of red, green, yellow, blue, magenta, cyan.
        """
        if not getattr(self, "debug_mode", True):
            return
        if not verbose:
            return

        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "reset": "\033[0m",
        }

        prefix = colors.get(color, "")
        suffix = colors["reset"] if color else ""

        lines = message.strip().split("\n")
        max_width = max(len(line) for line in lines)
        border = "─" * (max_width + 2)

        print(f"┌{border}┐")
        for line in lines:
            print(f"│ {prefix}{line.ljust(max_width)}{suffix} │")
        print(f"└{border}┘")

    def _check_truncation_and_early_return(self, observations, rewards, terminations, truncations, infos):
        """
        If the max step limit is reached, populate outputs and return early.

        Returns:
            A 5-tuple (obs, rewards, terminations, truncations, infos) if truncated.
            Otherwise, returns None.
        """
        if self.current_step >= self.max_steps:
            for agent in self.possible_agents:
                if agent in self.agents:  # Active agents get observation
                    observations[agent] = self._get_observation(agent)
                else:
                    obs_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
                    observations[agent] = np.zeros((self.num_obs_channels, obs_range, obs_range), dtype=np.float64)

                rewards[agent] = 0.0
                truncations[agent] = True
                terminations[agent] = False

            truncations["__all__"] = True
            terminations["__all__"] = False
            return observations, rewards, terminations, truncations, infos

        return None

    def _apply_homeostatic_loss(self, action_dict):
        """
        Apply fixed per-step energy and hydration loss to in a restig state, based on agent type.
        """
        for agent in action_dict:
            if agent not in self.agent_positions:
                continue

            old_energy = self.agent_energies[agent]
            # old_water = self.agent_hydration[agent]

            if "predator" in agent:
                energy_decay = self.energy_loss_per_step_predator
                # ater_decay = self.dehydration_per_step_predator
                layer = 1
            elif "prey" in agent:
                energy_decay = self.energy_loss_per_step_prey
                # water_decay = self.dehydration_per_step_prey
                layer = 2
            else:
                continue

            self.agent_energies[agent] -= energy_decay
            self.grid_world_state[layer, *self.agent_positions[agent]] = self.agent_energies[agent]

            self._log(
                self.verbose_decay,
                f"[ENERGY DECAY] {agent} energy: {round(old_energy, 2)} -> {round(self.agent_energies[agent], 2)}",
                "red",
            )

    def _apply_age_update(self, action_dict):
        """
        Increment the age of each active agent by one step.
        """
        for agent in action_dict:
            internal_id = self.agent_internal_ids.get(agent)
            if internal_id is not None:
                self.agent_ages[internal_id] += 1

    def _regenerate_grass_energy(self):
        for grass, pos in self.grass_positions.items():
            self.grass_energies[grass] = min(
                self.grass_energies[grass] + self.energy_gain_per_step_grass, self.initial_energy_grass
            )
            self.grid_world_state[3, *pos] = self.grass_energies[grass]

    def _process_agent_movements(self, action_dict):
        """
        Process movement, energy cost, and grid updates for all agents.
        """
        for agent, action in action_dict.items():
            if agent in self.agent_positions:
                old_position = self.agent_positions[agent]
                new_position = self._get_move(agent, action)
                self.agent_positions[agent] = new_position
                move_cost = self._get_movement_energy_cost(agent, old_position, new_position)
                self.agent_energies[agent] -= move_cost
                if "predator" in agent:
                    self.predator_positions[agent] = new_position
                    self.grid_world_state[1, *old_position] = 0
                    self.grid_world_state[1, *new_position] = self.agent_energies[agent]
                elif "prey" in agent:
                    self.prey_positions[agent] = new_position
                    self.grid_world_state[2, *old_position] = 0
                    self.grid_world_state[2, *new_position] = self.agent_energies[agent]

                self._log(
                    self.verbose_movement,
                    f"[MOVE] {agent} moved: {tuple(map(int, old_position))} -> {tuple(map(int, new_position))}. "
                    f"Move energy: {move_cost:.2f} Energy level: {self.agent_energies[agent]:.2f}\n",
                    "blue",
                )

    def _handle_homeostatic_depletion(self, agent, starvation, observations, rewards, terminations, truncations):
        self._log(
            self.verbose_decay, f"[DECAY] {agent} at {self.agent_positions[agent]} starved {starvation} and is removed.", "red"
        )
        observations[agent] = self._get_observation(agent)
        rewards[agent] = 0
        terminations[agent] = True
        truncations[agent] = False

        layer = 1 if "predator" in agent else 2
        self.grid_world_state[layer, *self.agent_positions[agent]] = 0

        if "predator" in agent:
            self.active_num_predators -= 1
            del self.predator_positions[agent]
            internal_id = self.agent_internal_ids[agent]
            if starvation:
                self.death_cause_predator[internal_id] = "starved"
        else:
            self.active_num_prey -= 1
            del self.prey_positions[agent]
            internal_id = self.agent_internal_ids[agent]
            if starvation:
                self.death_cause_prey[internal_id] = "starved"

        self._log(
            self.verbose_death_cause,
            f"[CAUSE OF DEATH] {agent} : {self.death_cause_predator[internal_id] if 'predator' in agent else self.death_cause_prey[internal_id]}",
            "red",
        )

        del self.agent_positions[agent]
        del self.agent_energies[agent]
        del self.agent_hydration[agent]

    def _handle_predator_engagement(self, agent, observations, rewards, terminations, truncations):
        predator_position = self.agent_positions[agent]
        # Check if predator caught prey
        caught_prey = next(
            (prey for prey, pos in self.agent_positions.items() if "prey" in prey and np.array_equal(predator_position, pos)), None
        )

        if caught_prey:
            self._log(
                self.verbose_engagement, f"[ENGAGE] {agent} caught {caught_prey} at {tuple(map(int, predator_position))}", "white"
            )

            rewards[agent] = self.reward_predator_catch_prey
            self.cumulative_rewards.setdefault(agent, 0)
            self.cumulative_rewards[agent] += rewards[agent]

            self.agent_energies[agent] += self.agent_energies[caught_prey]
            self.grid_world_state[1, *predator_position] = self.agent_energies[agent]

            observations[caught_prey] = self._get_observation(caught_prey)
            rewards[caught_prey] = self.penalty_prey_caught
            self.cumulative_rewards.setdefault(caught_prey, 0.0)
            self.cumulative_rewards[caught_prey] += rewards[caught_prey]

            internal_id = self.agent_internal_ids[caught_prey]
            self.death_cause_prey[internal_id] = "eaten"

            terminations[caught_prey] = True
            truncations[caught_prey] = False
            self.active_num_prey -= 1
            self.grid_world_state[2, *self.agent_positions[caught_prey]] = 0
            del self.agent_positions[caught_prey]
            del self.prey_positions[caught_prey]
            del self.agent_energies[caught_prey]
            del self.agent_hydration[caught_prey]
            self._log(
                self.verbose_death_cause, f"[CAUSE OF DEATH] {caught_prey} {self.death_cause_prey[internal_id]} by {agent}", "red"
            )
        else:
            rewards[agent] = self.reward_predator_step

        observations[agent] = self._get_observation(agent)
        self.cumulative_rewards.setdefault(agent, 0)
        self.cumulative_rewards[agent] += rewards[agent]
        terminations[agent] = False
        truncations[agent] = False

    def _handle_prey_engagement(self, agent, observations, rewards, terminations, truncations):
        if terminations.get(agent):
            return
        prey_position = self.agent_positions[agent]
        # Check if predator steps into in river
        if self.grid_world_state[4, *prey_position] > 0:
            self._log(self.verbose_engagement, f"[ENGAGE] {agent} stepped into river at {tuple(map(int, prey_position))}", "blue")
            self.agent_energies[agent] = max(self.agent_energies[agent] - self.energy_loss_staying_in_river_prey, 0)
            self.grid_world_state[2, *prey_position] = self.agent_energies[agent]
            # no hydration, just a bad move
            # self.agent_hydration[agent] = min(self.agent_hydration[agent] + 1, self.max_hydration_prey)
        # Check if prey can drink water in adjacent cell (Moore neighborhood)
        if self._is_water_nearby(prey_position):
            self._log(self.verbose_engagement, f"[ENGAGE] {agent} drank water at {tuple(map(int, prey_position))}", "blue")
            # rewards[agent] = self.reward_prey_drink_water
            # self.cumulative_rewards.setdefault(agent, 0)
            # self.cumulative_rewards[agent] += rewards[agent]
            self.agent_hydration[agent] = min(
                self.agent_hydration[agent] + 1, self.max_hydration_predator
            )  # TODO remove hardcoded value
        # Check if prey caught grass
        caught_grass = next(
            (g for g, pos in self.grass_positions.items() if "grass" in g and np.array_equal(prey_position, pos)), None
        )

        if caught_grass:
            self._log(self.verbose_engagement, f"[ENGAGE] {agent} caught grass at {tuple(map(int, prey_position))}", "white")
            rewards[agent] = self.reward_prey_eat_grass
            self.cumulative_rewards.setdefault(agent, 0)
            self.cumulative_rewards[agent] += rewards[agent]

            self.agent_energies[agent] += self.grass_energies[caught_grass]
            self.grid_world_state[2, *prey_position] = self.agent_energies[agent]

            self.grid_world_state[3, *prey_position] = 0
            self.grass_energies[caught_grass] = 0
        else:
            rewards[agent] = self.reward_prey_step

        observations[agent] = self._get_observation(agent)
        self.cumulative_rewards.setdefault(agent, 0)
        self.cumulative_rewards[agent] += rewards[agent]
        terminations[agent] = False
        truncations[agent] = False

    def _handle_predator_reproduction(self, agent, rewards, observations, terminations, truncations):
        if self.agent_energies[agent] >= self.predator_creation_energy_threshold:
            parent_speed = int(agent.split("_")[1])  # from "speed_1_predator_3"
            # Mutation: chance (self.mutation_rate_predator) to switch speed
            if self.rng.random() < self.mutation_rate_predator:
                new_speed = 2 if parent_speed == 1 else 1
            else:
                new_speed = parent_speed

            # Find available new agent ID
            potential_new_ids = [
                f"speed_{new_speed}_predator_{i}"
                for i in range(self.config.get(f"n_possible_speed_{new_speed}_predators", 25))
                if f"speed_{new_speed}_predator_{i}" not in self.agents
            ]
            if not potential_new_ids:
                # Always grant reproduction reward, even if no slot available
                rewards[agent] = self.reproduction_reward_predator
                self.cumulative_rewards.setdefault(agent, 0)
                self.cumulative_rewards[agent] += rewards[agent]
                self._log(
                    self.verbose_reproduction, f"[REPRODUCTION] No available predator slots at speed {new_speed} for spawning" "red"
                )
                return
                # TODO continue is left out because it's it a lop anymmore, check outside of function
                # if potential_new_ids still available
            new_agent = potential_new_ids[0]
            self.agents.append(new_agent)

            self.agent_internal_ids[new_agent] = self.agent_instance_counter
            self.agent_ages[self.agent_instance_counter] = 0
            self.agent_instance_counter += 1

            # Spawn position
            occupied_positions = set(self.agent_positions.values())
            new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)

            self.agent_positions[new_agent] = new_position
            self.predator_positions[new_agent] = new_position
            self.agent_energies[new_agent] = self.initial_energy_predator
            self.agent_hydration[new_agent] = self.initial_hydration_predator
            self.agent_energies[agent] -= self.initial_energy_predator

            self.grid_world_state[1, *new_position] = self.initial_energy_predator
            self.grid_world_state[1, *self.agent_positions[agent]] = self.agent_energies[agent]

            self.active_num_predators += 1

            # Rewards and tracking
            rewards[new_agent] = 0
            rewards[agent] = self.reproduction_reward_predator
            self.cumulative_rewards[new_agent] = 0
            self.cumulative_rewards[agent] += rewards[agent]

            observations[new_agent] = self._get_observation(new_agent)
            terminations[new_agent] = False
            truncations[new_agent] = False
            self._log(
                self.verbose_reproduction,
                f"[REPRODUCTION] Predator {agent} spawned {new_agent} at {tuple(map(int, new_position))}",
                "green",
            )

    def _handle_prey_reproduction(self, agent, rewards, observations, terminations, truncations):
        if self.agent_energies[agent] >= self.prey_creation_energy_threshold:
            parent_speed = int(agent.split("_")[1])  # from "speed_1_prey_6"

            # Mutation: 10% chance to switch speed
            if self.rng.random() < self.mutation_rate_prey:
                new_speed = 2 if parent_speed == 1 else 1
            else:
                new_speed = parent_speed

            # Find available new agent ID
            potential_new_ids = [
                f"speed_{new_speed}_prey_{i}"
                for i in range(self.config.get(f"n_possible_speed_{new_speed}_prey", 25))
                if f"speed_{new_speed}_prey_{i}" not in self.agents
            ]
            if not potential_new_ids:
                # Always grant reproduction reward, even if no slot available
                rewards[agent] = self.reproduction_reward_prey
                self.cumulative_rewards.setdefault(agent, 0)
                self.cumulative_rewards[agent] += rewards[agent]
                self._log(
                    self.verbose_reproduction, f"[REPRODUCTION] No available prey slots at speed {new_speed} for spawning", "red"
                )
                return

            new_agent = potential_new_ids[0]
            self.agents.append(new_agent)

            self.agent_internal_ids[new_agent] = self.agent_instance_counter
            self.agent_ages[self.agent_instance_counter] = 0
            self.agent_instance_counter += 1

            # Spawn position
            occupied_positions = set(self.agent_positions.values())
            new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)

            self.agent_positions[new_agent] = new_position
            self.prey_positions[new_agent] = new_position
            self.agent_energies[new_agent] = self.initial_energy_prey
            self.agent_hydration[new_agent] = self.initial_hydration_prey
            self.agent_energies[agent] -= self.initial_energy_prey

            self.grid_world_state[2, *new_position] = self.initial_energy_prey
            self.grid_world_state[2, *self.agent_positions[agent]] = self.agent_energies[agent]

            self.active_num_prey += 1

            # Rewards and tracking
            rewards[new_agent] = 0
            rewards[agent] = self.reproduction_reward_prey
            self.cumulative_rewards[new_agent] = 0
            self.cumulative_rewards[agent] += rewards[agent]

            observations[new_agent] = self._get_observation(new_agent)
            terminations[new_agent] = False
            truncations[new_agent] = False
            self._log(
                self.verbose_reproduction,
                f"[REPRODUCTION] Prey {agent} spawned {new_agent} at {tuple(map(int, new_position))}",
                "green",
            )

    def _load_config(self, config):
        """
        Load the environment configuration from a dictionary.
        Args:
            config (dict): Configuration dictionary.
        """
        # Check if config is provided
        if config is None:
            raise ValueError("Environment config must be provided explicitly.")
        self.config = config

        # Set specific verbose flags based on debug mode
        self.debug_mode = config.get("debug_mode", False)

        # Verbosity
        self.verbose_movement = config.get("verbose_movement", self.debug_mode)
        self.verbose_decay = config.get("verbose_decay", self.debug_mode)
        self.verbose_reproduction = config.get("verbose_reproduction", self.debug_mode)
        self.verbose_engagement = config.get("verbose_engagement", self.debug_mode)
        self.verbose_termination = config.get("verbose_termination", self.debug_mode)
        self.verbose_death_cause = config.get("verbose_death_cause", self.debug_mode)

        # epsiode
        self.max_steps = config.get("max_steps", 10000)
        self.rng = np.random.default_rng(config.get("seed", 42))

        # Learning agents
        self.n_possible_speed_1_predator = config.get("n_possible_speed_1_predator", 25)
        self.n_possible_speed_2_predator = config.get("n_possible_speed_2_predator", 25)
        self.n_possible_speed_1_prey = config.get("n_possible_speed_1_prey", 25)
        self.n_possible_speed_2_prey = config.get("n_possible_speed_2_prey", 25)

        self.n_initial_active_speed_1_predator = config.get("n_initial_active_speed_1_predator", 6)
        self.n_initial_active_speed_2_predator = config.get("n_initial_active_speed_2_predator", 0)
        self.n_initial_active_speed_1_prey = config.get("n_initial_active_speed_1_prey", 8)
        self.n_initial_active_speed_2_prey = config.get("n_initial_active_speed_2_prey", 0)

        # Rewards
        self.reward_predator_catch_prey = config.get("reward_predator_catch_prey", 0.0)
        self.reward_prey_eat_grass = config.get("reward_prey_eat_grass", 0.0)
        self.reward_predator_step = config.get("reward_predator_step", 0.0)
        self.reward_prey_step = config.get("reward_prey_step", 0.0)
        self.penalty_prey_caught = config.get("penalty_prey_caught", 0.0)
        self.reproduction_reward_predator = config.get("reproduction_reward_predator", 10.0)
        self.reproduction_reward_prey = config.get("reproduction_reward_prey", 10.0)

        # Energy
        self.energy_loss_per_step_predator = config.get("energy_loss_per_step_predator", 0.15)
        self.energy_loss_per_step_prey = config.get("energy_loss_per_step_prey", 0.05)
        self.energy_loss_staying_in_river_predator = config.get("energy_loss_staying_in_river", 0.15)
        self.energy_loss_staying_in_river_prey = config.get("energy_loss_staying_in_river", 0.05)
        self.predator_creation_energy_threshold = config.get("predator_creation_energy_threshold", 12.0)
        self.prey_creation_energy_threshold = config.get("prey_creation_energy_threshold", 8.0)
        self.initial_energy_predator = config.get("initial_energy_predator", 5.0)
        self.initial_energy_prey = config.get("initial_energy_prey", 3.0)

        # Water
        self.initial_hydration_predator = config.get("initial_hydration_predator", 3.0)
        self.initial_hydration_prey = config.get("initial_hydration_prey", 2.0)
        self.dehydration_per_step_predator = config.get("dehydration_per_step_predator", 0.1)
        self.dehydration_per_step_prey = config.get("dehydration_per_step_prey", 0.05)
        self.max_hydration_predator = config.get("max_hydration_predator", 4.0)
        self.max_hydration_prey = config.get("max_hydration_prey", 3.0)

        # Grid & obs
        self.grid_size = config.get("grid_size", 10)
        self.num_obs_channels = config.get("num_obs_channels", 5)
        self.predator_obs_range = config.get("predator_obs_range", 7)
        self.prey_obs_range = config.get("prey_obs_range", 9)

        # Action range
        self.speed_1_act_range = config.get("speed_1_action_range", 3)
        self.speed_2_act_range = config.get("speed_2_action_range", 5)

        # Evolution
        self.mutation_rate_predator = config.get("mutation_rate_predator", 0.1)
        self.mutation_rate_prey = config.get("mutation_rate_prey", 0.1)

        # River
        self.river_max_width = config.get("river_max_width", 1)
        self.n_steps_river_change = config.get("n_steps_river_change", 10)

        # Grass
        self.initial_num_grass = config.get("initial_num_grass", 25)
        self.initial_energy_grass = config.get("initial_energy_grass", 2.0)
        self.energy_gain_per_step_grass = config.get("energy_gain_per_step_grass", 0.2)

    def _create_learning_agent_lists(self):
        # create list of al possible learning agents
        self.agents = []
        self.possible_agents = []

        for role in ["predator", "prey"]:
            speeds = getattr(self, f"{role}_speeds")
            for speed in speeds:
                # Add all possible agents
                n_possible = getattr(self, f"n_possible_speed_{speed}_{role}")
                for i in range(n_possible):
                    agent_id = f"speed_{speed}_{role}_{i}"
                    self.possible_agents.append(agent_id)

                # Add initial active agents (subset)
                n_initial = getattr(self, f"n_initial_active_speed_{speed}_{role}")
                for i in range(n_initial):
                    agent_id = f"speed_{speed}_{role}_{i}"
                    self.agents.append(agent_id)
                    self.agent_internal_ids[agent_id] = self.agent_instance_counter
                    self.agent_ages[self.agent_instance_counter] = 0
                    self.agent_instance_counter += 1

    def _create_spaces(self):
        # Spaces
        predator_obs_shape = (self.num_obs_channels, self.predator_obs_range, self.predator_obs_range)
        prey_obs_shape = (self.num_obs_channels, self.prey_obs_range, self.prey_obs_range)

        # Define observation space per agent
        predator_obs_space = gymnasium.spaces.Box(low=0.0, high=100.0, shape=predator_obs_shape, dtype=np.float64)
        prey_obs_space = gymnasium.spaces.Box(low=0.0, high=100.0, shape=prey_obs_shape, dtype=np.float64)
        # Generate action maps for both speed levels
        n_speed_1_actions = self.speed_1_act_range**2
        n_speed_2_actions = self.speed_2_act_range**2

        # Create action space objects
        speed_1_action_space = gymnasium.spaces.Discrete(n_speed_1_actions)
        speed_2_action_space = gymnasium.spaces.Discrete(n_speed_2_actions)

        # Assign spaces per agent ID
        self.observation_spaces = {}
        self.action_spaces = {}

        for agent in self.possible_agents:
            if "predator" in agent:
                self.observation_spaces[agent] = predator_obs_space
            elif "prey" in agent:
                self.observation_spaces[agent] = prey_obs_space

            # Set action space depending on speed string in agent ID
            if "speed_1" in agent:
                self.action_spaces[agent] = speed_1_action_space
            elif "speed_2" in agent:
                self.action_spaces[agent] = speed_2_action_space

    def _create_grid_world(self):
        # Initialize grid_world_state
        self.grid_world_state_shape = (self.num_obs_channels, self.grid_size, self.grid_size)
        self.grid_world_state = np.zeros(self.grid_world_state_shape, dtype=np.float64)

        def _generate_action_map(range_size: int) -> dict[int, tuple[int, int]]:
            """
            Generate a mapping of action indices to movement vectors.
            The movement vectors are defined as (dx, dy) pairs, where dx and dy
            are the changes in x and y coordinates, respectively.

            Args:
                range_size (int): The size of the action range.
            Returns:
                dict[int, tuple[int, int]]: A dictionary mapping action indices to movement vectors.
            """
            delta = (range_size - 1) // 2
            return {
                i: (dx, dy)
                for i, (dx, dy) in enumerate((dx, dy) for dx in range(-delta, delta + 1) for dy in range(-delta, delta + 1))
            }

        # Save both dictionaries for later lookup
        self.action_to_move_tuple_speed_1_agents = _generate_action_map(self.speed_1_act_range)
        self.action_to_move_tuple_speed_2_agents = _generate_action_map(self.speed_2_act_range)

    def _initialization(self):
        """
        Initialize the environment variables
        """
        # intiailisation
        self.agent_instance_counter: int = 0
        self.cumulative_rewards: Dict[AgentID, float] = {}
        self.agent_internal_ids: Dict[AgentID, int] = {}  # Maps agent_id (e.g., 'speed_1_prey_0') -> internal ID
        self.agent_ages: Dict[AgentID, int] = {}
        self.death_cause_prey: Dict[int, str] = {}  # key = internal ID, value = "eaten" or "starved"
        self.death_cause_predator: Dict[int, str] = {}  # key = internal ID, value = "eaten" or "starved"
        self.predator_speeds = [1, 2]  # TODO: make configurable
        self.prey_speeds = [1, 2]
        # Initialize grid_world_state and agent positions
        self.agent_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.predator_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.prey_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.grass_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.agent_energies: Dict[AgentID, float] = {}
        self.agent_hydration: Dict[AgentID, float] = {}
        self.grass_energies: Dict[AgentID, float] = {}
        # River configuration
        self.river_cells = set()

    def _generate_random_positions(self, grid_size: int, num_positions: int, seed: int = None) -> List[Tuple[int, int]]:
        """
        Generate unique random positions on a grid.

        Args:
            grid_size (int): The size of the grid (assumes square grid).
            num_positions (int): The number of unique positions to generate.

        Returns:
            List[Tuple[int, int]]: A list of unique (x, y) position tuples.
        """
        if num_positions > grid_size * grid_size:
            raise ValueError("Cannot place more unique positions than grid cells.")

        rng = np.random.default_rng(seed)
        positions = set()

        while len(positions) < num_positions:
            pos = tuple(rng.integers(0, grid_size, size=2))
            positions.add(pos)  # Ensures uniqueness because positions is a set

        return list(positions)

    def _is_water_nearby(self, agent_id):
        """
        Check if water is in the Moore neighborhood (8 adjacent tiles) of the given agent.
        Returns True if water is found, False otherwise.
        """
        if agent_id not in self.agent_positions:
            return False  # Agent not on grid

        x, y = self.agent_positions[agent_id]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip center tile
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if (nx, ny) in self.river_cells:
                        return True
        return False

    def _get_available_positions_from_layer(self, layer_index: int, occupied_threshold: float = 0.0) -> List[Tuple[int, int]]:
        """
        Return a list of (x, y) positions where the given layer is unoccupied.

        Args:
            layer_index (int): Index into grid_world_state (e.g., 1=predator, 2=prey, 3=grass, 4=water).
            occupied_threshold (float): Values > this threshold are considered occupied.

        Returns:
            List[Tuple[int, int]]: List of available (x, y) coordinates.
        """
        occupied_mask = self.grid_world_state[layer_index] > occupied_threshold
        available_mask = ~occupied_mask  # Invert mask
        available_positions = np.argwhere(available_mask)
        return [tuple(pos) for pos in available_positions]

    def _get_occupied_positions_from_layers(self, layer_indices: List[int], threshold: float = 0.0) -> set[Tuple[int, int]]:
        """
        Combine masks from multiple layers to find all occupied cells.

        Args:
            layer_indices (List[int]): List of layer indices to check (e.g., [1, 2, 3]).
            threshold (float): Minimum value considered as occupied.

        Returns:
            Set[Tuple[int, int]]: Set of (x, y) positions that are occupied in any of the layers.
        """
        occupied = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for idx in layer_indices:
            occupied |= self.grid_world_state[idx] > threshold
        return set(map(tuple, np.argwhere(occupied)))

    def _get_spawnable_positions(
        self,
        center: Tuple[int, int],
        avoid_layers: List[int],
        radius: int = 1,
        exclude_center: bool = True,
    ) -> List[Tuple[int, int]]:
        """
        Return a list of spawnable positions within a square region (Moore-like neighborhood)
        around a center point, excluding any positions occupied in the given grid layers.

        Args:
            center: (x, y) position to search around.
            avoid_layers: List of grid layers to treat as occupied (e.g., [1, 2, 3]).
            radius: Neighborhood radius (1 = Moore neighborhood, 2 = wider area).
            exclude_center: Whether to exclude the center cell itself.

        Returns:
            List of (x, y) tuples that are unoccupied and within radius.
        """
        cx, cy = center
        occupied = self._get_occupied_positions_from_layers(avoid_layers)

        candidates = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    pos = (nx, ny)
                    if exclude_center and pos == center:
                        continue
                    if pos not in occupied:
                        candidates.append(pos)

        return candidates

    def _place_agent(
        self, agent_id: str, position: Tuple[int, int], energy: float, hydration: float = None, layer_index: int = None
    ):
        """
        Assign an agent to a position on the grid and set its energy and hydration values.

        Args:
            agent_id (str): The agent's identifier (e.g., "speed_1_predator_0").
            position (Tuple[int, int]): (x, y) grid position to place the agent.
            energy (float): Initial energy level to assign.
            hydration (float, optional): Initial hydration level to assign.
            layer_index (int, optional): Grid layer index to mark the energy value (1=predator, 2=prey).
        """
        self.agent_positions[agent_id] = position
        self.agent_energies[agent_id] = energy

        if hydration is not None:
            self.agent_hydration[agent_id] = hydration

        if layer_index is not None:
            self.grid_world_state[layer_index, *position] = energy

    def _export_grid_to_json(self, grid_state: np.ndarray, step: int, export_dir: str = "unity_json_exports"):
        """Save the 3D grid_world_state as a JSON file for external visualization (e.g., in Godot)."""
        os.makedirs(export_dir, exist_ok=True)

        # Convert to nested Python list for JSON serialization
        grid_as_list = grid_state.tolist()

        filename = f"grid_step_{step:05d}.json"
        path = os.path.join(export_dir, filename)

        with open(path, "w") as f:
            json.dump(grid_as_list, f)

        print(f"[✓] Exported grid to JSON: {path}")

    def _export_grid_to_file(self, grid_state, step, export_dir="unity_viewer_exports"):
        """
        Export the grid state to a JSON file, rotating each channel 90° counter-clockwise
        to align with Unity's coordinate system (origin bottom-left).
        Assumes grid_state shape: [channels, width, height] (CHW).
        """
        os.makedirs(export_dir, exist_ok=True)
        filepath = os.path.join(export_dir, f"grid_step_{step:05d}.json")

        # Rotate each 2D layer
        rotated_layers = [np.rot90(grid_state[i], k=3) for i in range(grid_state.shape[0])]

        # Convert to list-of-lists format
        grid_as_list = [layer.tolist() for layer in rotated_layers]

        # Save as JSON
        with open(filepath, "w") as f:
            json.dump(grid_as_list, f)
