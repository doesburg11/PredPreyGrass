"""
Predator-Prey Grass RLlib Environment
"""
# external libraries
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, Tuple
import numpy as np
import math


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            raise ValueError("Environment config must be provided explicitly.")
        self.config = config
        self._initialize_from_config()

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self._init_reset_variables(seed)

        total_entities = len(self.agents) + len(self.grass_agents)
        all_positions = self._generate_random_positions(self.grid_size, total_entities, seed)

        predator_list = [a for a in self.agents if "predator" in a]
        prey_list = [a for a in self.agents if "prey" in a]

        predator_positions = all_positions[: len(predator_list)]
        prey_positions = all_positions[len(predator_list) : len(predator_list) + len(prey_list)]
        grass_positions = all_positions[len(predator_list) + len(prey_list) :]

        for i, agent in enumerate(predator_list):
            pos = predator_positions[i]
            self.agent_positions[agent] = self.predator_positions[agent] = pos
            self.agent_energies[agent] = self.initial_energy_predator
            self.grid_world_state[1, *pos] = self.initial_energy_predator
            self.cumulative_rewards[agent] = 0

        for i, agent in enumerate(prey_list):
            pos = prey_positions[i]
            self.agent_positions[agent] = self.prey_positions[agent] = pos
            self.agent_energies[agent] = self.initial_energy_prey
            self.grid_world_state[2, *pos] = self.initial_energy_prey
            self.cumulative_rewards[agent] = 0

        for i, grass in enumerate(self.grass_agents):
            pos = grass_positions[i]
            self.grass_positions[grass] = pos
            self.grass_energies[grass] = self.initial_energy_grass
            self.grid_world_state[3, *pos] = self.initial_energy_grass

        self.active_num_predators = len(self.predator_positions)
        self.active_num_prey = len(self.prey_positions)
        self.current_num_grass = len(self.grass_positions)

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        return observations, {}

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        # For stepwise display eating in grid
        self.agents_just_ate.clear()

        # step 0: check for truncation
        truncation_result = self._check_truncation_and_early_return(observations, rewards, terminations, truncations, infos)
        if truncation_result is not None:
            return truncation_result

        # Step 1: If not truncated; process energy depletion due to time steps and update age
        self._apply_energy_decay_per_step(action_dict)

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
            if self.agent_energies[agent] <= 0:
                self._handle_energy_depletion(agent, observations, rewards, terminations, truncations)
            elif "predator" in agent:
                self._handle_predator_engagement(agent, observations, rewards, terminations, truncations)
            elif "prey" in agent:
                self._handle_prey_engagement(agent, observations, rewards, terminations, truncations)

        # Step 6: Handle agent removals
        for agent in self.agents[:]:
            if terminations[agent]:
                self._log(self.verbose_engagement, f"[TERMINATED] Agent {agent} terminated!", "red")
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
        print(f"\nCurrent Grid State (Energy Levels):  predators: {self.active_num_predators} prey: {self.active_num_prey} \n")

        # Initialize empty grids
        predator_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        prey_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grass_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Fill the grid (storing values in original order)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                predator_energy = self.grid_world_state[1, x, y]
                prey_energy = self.grid_world_state[2, x, y]
                grass_energy = self.grid_world_state[3, x, y]

                if predator_energy > 0:
                    predator_grid[y][x] = f"{predator_energy:4.2f}".center(5)
                if prey_energy > 0:
                    prey_grid[y][x] = f"{prey_energy:4.2f}".center(5)
                if grass_energy > 0:
                    grass_grid[y][x] = f"{grass_energy:4.2f}".center(5)

        # Transpose the grids (swap rows and columns)
        predator_grid = [[predator_grid[x][y] for x in range(self.grid_size)] for y in range(self.grid_size)]
        prey_grid = [[prey_grid[x][y] for x in range(self.grid_size)] for y in range(self.grid_size)]
        grass_grid = [[grass_grid[x][y] for x in range(self.grid_size)] for y in range(self.grid_size)]

        # Print Headers
        print(
            f"{'Predator '.center(self.grid_size * 6)}   {'Prey'.center(self.grid_size * 6)}   {'Grass'.center(self.grid_size * 6)}"
        )
        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

        # Print Transposed Grids (rows become columns)
        for x in range(self.grid_size):  # Now iterating over transposed rows (original columns)
            predator_row = " ".join(predator_grid[x])
            prey_row = " ".join(prey_grid[x])
            grass_row = " ".join(grass_grid[x])
            print(f"{predator_row}     {prey_row}     {grass_row}")

        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

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
        all_positions = {(i, j) for i in range(self.grid_size) for j in range(self.grid_size)}
        free_positions = list(all_positions - occupied_positions)

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

    def _apply_energy_decay_per_step(self, action_dict):
        """
        Apply fixed per-step energy decay to all agents based on type.
        """
        for agent in action_dict:
            if agent not in self.agent_positions:
                continue

            old_energy = self.agent_energies[agent]

            if "predator" in agent:
                decay = self.energy_loss_per_step_predator
                layer = 1
            elif "prey" in agent:
                decay = self.energy_loss_per_step_prey
                layer = 2
            else:
                continue

            self.agent_energies[agent] -= decay
            self.grid_world_state[layer, *self.agent_positions[agent]] = self.agent_energies[agent]

            self._log(
                self.verbose_decay,
                f"[DECAY] {agent} energy: {round(old_energy, 2)} -> {round(self.agent_energies[agent], 2)}",
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
        """
        Increase energy of all grass patches, capped at initial energy value.
        """
        for grass, pos in self.grass_positions.items():
            new_energy = min(self.grass_energies[grass] + self.energy_gain_per_step_grass, self.initial_energy_grass)
            self.grass_energies[grass] = new_energy
            self.grid_world_state[3, *pos] = new_energy

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

    def _handle_energy_depletion(self, agent, observations, rewards, terminations, truncations):
        self._log(self.verbose_decay, f"[DECAY] {agent} at {self.agent_positions[agent]} ran out of energy and is removed.", "red")
        observations[agent] = self._get_observation(agent)
        rewards[agent] = 0
        terminations[agent] = True
        truncations[agent] = False

        layer = 1 if "predator" in agent else 2
        self.grid_world_state[layer, *self.agent_positions[agent]] = 0

        if "predator" in agent:
            self.active_num_predators -= 1
            del self.predator_positions[agent]
        else:
            self.active_num_prey -= 1
            del self.prey_positions[agent]
            internal_id = self.agent_internal_ids[agent]
            self.death_cause_prey[internal_id] = "starved"

        del self.agent_positions[agent]
        del self.agent_energies[agent]

    def _handle_predator_engagement(self, agent, observations, rewards, terminations, truncations):
        predator_position = self.agent_positions[agent]
        caught_prey = next(
            (prey for prey, pos in self.agent_positions.items() if "prey" in prey and np.array_equal(predator_position, pos)), None
        )

        if caught_prey:
            self._log(
                self.verbose_engagement, f"[ENGAGE] {agent} caught {caught_prey} at {tuple(map(int, predator_position))}", "white"
            )
            self.agents_just_ate.add(agent)  # Show green ring for next 1 step

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
        caught_grass = next(
            (g for g, pos in self.grass_positions.items() if "grass" in g and np.array_equal(prey_position, pos)), None
        )

        if caught_grass:
            self._log(self.verbose_engagement, f"[ENGAGE] {agent} caught grass at {tuple(map(int, prey_position))}", "white")
            self.agents_just_ate.add(agent)  # Show green ring for next 1 step
            # Reward prey for eating grass
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
            "active_num_predators": self.active_num_predators,
            "active_num_prey": self.active_num_prey,
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
        self.active_num_predators = snapshot["active_num_predators"]
        self.active_num_prey = snapshot["active_num_prey"]
        self.agents_just_ate = snapshot["agents_just_ate"].copy()

    def _initialize_from_config(self):
        cfg = self.config
        self.debug_mode = cfg.get("debug_mode", False)
        self.verbose_movement = cfg.get("verbose_movement", self.debug_mode)
        self.verbose_decay = cfg.get("verbose_decay", self.debug_mode)
        self.verbose_reproduction = cfg.get("verbose_reproduction", self.debug_mode)
        self.verbose_engagement = cfg.get("verbose_engagement", self.debug_mode)

        self.max_steps = cfg.get("max_steps", 10000)
        self.rng = np.random.default_rng(cfg.get("seed", 42))

        # Rewards
        self.reward_predator_catch_prey = cfg.get("reward_predator_catch_prey", 0.0)
        self.reward_prey_eat_grass = cfg.get("reward_prey_eat_grass", 0.0)
        self.reward_predator_step = cfg.get("reward_predator_step", 0.0)
        self.reward_prey_step = cfg.get("reward_prey_step", 0.0)
        self.penalty_prey_caught = cfg.get("penalty_prey_caught", 0.0)
        self.reproduction_reward_predator = cfg.get("reproduction_reward_predator", 10.0)
        self.reproduction_reward_prey = cfg.get("reproduction_reward_prey", 10.0)

        # Energy settings
        self.energy_loss_per_step_predator = cfg.get("energy_loss_per_step_predator", 0.15)
        self.energy_loss_per_step_prey = cfg.get("energy_loss_per_step_prey", 0.05)
        self.predator_creation_energy_threshold = cfg.get("predator_creation_energy_threshold", 12.0)
        self.prey_creation_energy_threshold = cfg.get("prey_creation_energy_threshold", 8.0)

        # Learning agents
        self.n_possible_speed_1_predators = cfg.get("n_possible_speed_1_predators", 25)
        self.n_possible_speed_2_predators = cfg.get("n_possible_speed_2_predators", 25)
        self.n_possible_speed_1_prey = cfg.get("n_possible_speed_1_prey", 25)
        self.n_possible_speed_2_prey = cfg.get("n_possible_speed_2_prey", 25)

        self.n_initial_active_speed_1_predator = cfg.get("n_initial_active_speed_1_predator", 6)
        self.n_initial_active_speed_2_predator = cfg.get("n_initial_active_speed_2_predator", 0)
        self.n_initial_active_speed_1_prey = cfg.get("n_initial_active_speed_1_prey", 8)
        self.n_initial_active_speed_2_prey = cfg.get("n_initial_active_speed_2_prey", 0)

        self.initial_energy_predator = cfg.get("initial_energy_predator", 5.0)
        self.initial_energy_prey = cfg.get("initial_energy_prey", 3.0)

        # Grid and Observation Settings
        self.grid_size = cfg.get("grid_size", 10)
        self.num_obs_channels = cfg.get("num_obs_channels", 4)
        self.predator_obs_range = cfg.get("predator_obs_range", 7)
        self.prey_obs_range = cfg.get("prey_obs_range", 5)

        # Grass settings
        self.initial_num_grass = cfg.get("initial_num_grass", 25)
        self.initial_energy_grass = cfg.get("initial_energy_grass", 2.0)
        self.energy_gain_per_step_grass = cfg.get("energy_gain_per_step_grass", 0.2)

        # Mutation
        self.mutation_rate_predator = cfg.get("mutation_rate_predator", 0.1)
        self.mutation_rate_prey = cfg.get("mutation_rate_prey", 0.1)

        # Agent tracking
        self.agent_instance_counter = 0
        self.agent_internal_ids = {}
        self.agent_ages = {}
        self.cumulative_rewards = {}
        self.death_cause_prey = {}
        self.predator_speeds = [1, 2]
        self.prey_speeds = [1, 2]
        self.agents_just_ate = set()

        # Build agent lists
        self.possible_agents = (
            [f"speed_1_predator_{i}" for i in range(self.n_possible_speed_1_predators)]
            + [f"speed_2_predator_{i}" for i in range(self.n_possible_speed_2_predators)]
            + [f"speed_1_prey_{i}" for i in range(self.n_possible_speed_1_prey)]
            + [f"speed_2_prey_{i}" for i in range(self.n_possible_speed_2_prey)]
        )

        self.agents = (
            [f"speed_1_predator_{i}" for i in range(self.n_initial_active_speed_1_predator)]
            + [f"speed_2_predator_{i}" for i in range(self.n_initial_active_speed_2_predator)]
            + [f"speed_1_prey_{i}" for i in range(self.n_initial_active_speed_1_prey)]
            + [f"speed_2_prey_{i}" for i in range(self.n_initial_active_speed_2_prey)]
        )

        self.grass_agents = [f"grass_{i}" for i in range(self.initial_num_grass)]

        # Action range and movement mapping
        self.speed_1_act_range = cfg.get("speed_1_action_range", 3)
        self.speed_2_act_range = cfg.get("speed_2_action_range", 5)

        def _generate_action_map(range_size: int):
            delta = (range_size - 1) // 2
            return {
                i: (dx, dy)
                for i, (dx, dy) in enumerate((dx, dy) for dx in range(-delta, delta + 1) for dy in range(-delta, delta + 1))
            }

        self.action_to_move_tuple_speed_1_agents = _generate_action_map(self.speed_1_act_range)
        self.action_to_move_tuple_speed_2_agents = _generate_action_map(self.speed_2_act_range)

        # Action/Observation spaces
        self.observation_spaces = {}
        self.action_spaces = {}

        pred_shape = (self.num_obs_channels, self.predator_obs_range, self.predator_obs_range)
        prey_shape = (self.num_obs_channels, self.prey_obs_range, self.prey_obs_range)

        pred_obs_space = gymnasium.spaces.Box(low=0, high=100.0, shape=pred_shape, dtype=np.float64)
        prey_obs_space = gymnasium.spaces.Box(low=0, high=100.0, shape=prey_shape, dtype=np.float64)

        for agent in self.possible_agents:
            if "predator" in agent:
                self.observation_spaces[agent] = pred_obs_space
            elif "prey" in agent:
                self.observation_spaces[agent] = prey_obs_space

            if "speed_1" in agent:
                self.action_spaces[agent] = gymnasium.spaces.Discrete(self.speed_1_act_range**2)
            elif "speed_2" in agent:
                self.action_spaces[agent] = gymnasium.spaces.Discrete(self.speed_2_act_range**2)

        # Grid and state memory
        self.agent_positions = {}
        self.predator_positions = {}
        self.prey_positions = {}
        self.grass_positions = {}
        self.agent_energies = {}
        self.grass_energies = {}

        self.grid_world_state_shape = (self.num_obs_channels, self.grid_size, self.grid_size)
        self.initial_grid_world_state = np.zeros(self.grid_world_state_shape, dtype=np.float64)
        self.grid_world_state = self.initial_grid_world_state.copy()

    def _init_reset_variables(self, seed):
        self.current_step = 0
        self.rng = np.random.default_rng(seed)

        self.grid_world_state = self.initial_grid_world_state.copy()
        self.agent_positions = {}
        self.predator_positions = {}
        self.prey_positions = {}
        self.grass_positions = {}
        self.agent_energies = {}
        self.grass_energies = {}

        self.agent_instance_counter = 0
        self.agent_internal_ids = {}
        self.agent_ages = {}
        self.death_cause_prey = {}
        self.agents_just_ate = set()
        self.cumulative_rewards = {}

        self.agents = []
        self.possible_agents = []

        for speed in [1, 2]:
            for i in range(self.config.get(f"n_possible_speed_{speed}_predators", 0)):
                self.possible_agents.append(f"speed_{speed}_predator_{i}")
            for i in range(self.config.get(f"n_initial_active_speed_{speed}_predator", 0)):
                aid = f"speed_{speed}_predator_{i}"
                self.agents.append(aid)
                self.agent_internal_ids[aid] = self.agent_instance_counter
                self.agent_ages[self.agent_instance_counter] = 0
                self.agent_instance_counter += 1

            for i in range(self.config.get(f"n_possible_speed_{speed}_prey", 0)):
                self.possible_agents.append(f"speed_{speed}_prey_{i}")
            for i in range(self.config.get(f"n_initial_active_speed_{speed}_prey", 0)):
                aid = f"speed_{speed}_prey_{i}"
                self.agents.append(aid)
                self.agent_internal_ids[aid] = self.agent_instance_counter
                self.agent_ages[self.agent_instance_counter] = 0
                self.agent_instance_counter += 1

        self.grass_agents = [f"grass_{i}" for i in range(self.initial_num_grass)]

    def _generate_random_positions(self, grid_size: int, num_positions: int, seed=None):
        """
        Generate unique random positions on a grid using a local RNG seeded per reset,
        to ensure consistent and reproducible placement across runs.

        Args:
            grid_size (int): Size of the square grid.
            num_positions (int): Number of unique positions to generate.
            seed (int or None): Seed for local RNG (passed from reset()).

        Returns:
            List[Tuple[int, int]]: Unique (x, y) positions.
        """
        if num_positions > grid_size * grid_size:
            raise ValueError("Cannot place more unique positions than grid cells.")

        rng = np.random.default_rng(seed)
        positions = set()

        while len(positions) < num_positions:
            pos = tuple(rng.integers(0, grid_size, size=2))
            positions.add(pos)

        return list(positions)
