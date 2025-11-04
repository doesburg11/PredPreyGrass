"""
Predator-Prey Grass RLlib Environment

Additional features:
- Added refactored reset function
- Fixed overlapping wall & grass placement

"""
# external libraries (Ray required)
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import math
import time


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            raise ValueError("Environment config must be provided explicitly.")
        self.config = config
        self._initialize_from_config()  # import config variables

        self.possible_agents = self._build_possible_agent_ids()

        self.observation_spaces = {agent_id: self._build_observation_space(agent_id) for agent_id in self.possible_agents}

        self.action_spaces = {agent_id: self._build_action_space(agent_id) for agent_id in self.possible_agents}

        # Precompute LOS masks for each obs range (assuming static walls for now)
        # This must be done after config and grid/wall initialization
        self.los_mask_predator = self._precompute_los_mask(self.predator_obs_range)
        self.los_mask_prey = self._precompute_los_mask(self.prey_obs_range)

    def _precompute_los_mask(self, observation_range):
        offset = (observation_range - 1) // 2
        mask = np.zeros((observation_range, observation_range), dtype=np.float32)
        center = (offset, offset)
        for dx in range(-offset, offset + 1):
            for dy in range(-offset, offset + 1):
                tx, ty = center[0] + dx, center[1] + dy
                # Convert window offset to global grid offset as needed
                if self._line_of_sight_clear(center, (tx, ty)):
                    mask[tx, ty] = 1.0
        return mask

    def _initialize_from_config(self):
        config = self.config
        self.debug_mode = config["debug_mode"]
        self.verbose_movement = config["verbose_movement"]
        self.verbose_decay = config["verbose_decay"]
        self.verbose_reproduction = config["verbose_reproduction"]
        self.verbose_engagement = config["verbose_engagement"]

        self.max_steps = config["max_steps"]
        # RNG will be initialized during reset to ensure per-episode reproducibility

        # Rewards dictionaries
        self.reproduction_reward_predator = config["reproduction_reward_predator"]
        self.reproduction_reward_prey = config["reproduction_reward_prey"]

        # Energy delta settings
        self.energy_loss_per_step_predator = config["energy_loss_per_step_predator"]
        self.energy_loss_per_step_prey = config["energy_loss_per_step_prey"]
        self.predator_creation_energy_threshold = config["predator_creation_energy_threshold"]
        self.prey_creation_energy_threshold = config["prey_creation_energy_threshold"]
        self.move_energy_cost_predator = config["move_energy_cost_predator"]
        self.move_energy_cost_prey = config["move_energy_cost_prey"]

        # Learning agents
        self.n_possible_predators = config["n_possible_predators"]
        self.n_possible_prey = config["n_possible_prey"]

        self.n_initial_active_predator = config["n_initial_active_predator"]
        self.n_initial_active_prey = config["n_initial_active_prey"]

        self.initial_energy_predator = config["initial_energy_predator"]
        self.initial_energy_prey = config["initial_energy_prey"]

        # Grid and Observation Settings
        self.grid_size = config["grid_size"]
        self.num_obs_channels = config["num_obs_channels"]
        self.predator_obs_range = config["predator_obs_range"]
        self.prey_obs_range = config["prey_obs_range"]
        # Optional extra observation channel (appended as last channel) showing
        # line-of-sight visibility (1 = visible, 0 = occluded by at least one wall).
        # When disabled, observation tensors retain their original channel count.
        self.include_visibility_channel = config["include_visibility_channel"]
        # Movement restriction: if True, agents may only move to target cells with unobstructed LOS (no wall between current and target).
        self.respect_los_for_movement = config["respect_los_for_movement"]
        # If True, dynamic observation channels (predators/prey/grass) are masked so that
        # entities behind walls (no line-of-sight) appear as 0 even if within square range.
        # Works independently of include_visibility_channel; if that is False we still mask
        # but do not append the visibility channel itself.
        self.mask_observation_with_visibility = config["mask_observation_with_visibility"]

        # Grass settings
        self.initial_num_grass = config["initial_num_grass"]
        self.initial_energy_grass = config["initial_energy_grass"]
        self.energy_gain_per_step_grass = config["energy_gain_per_step_grass"]
        self.max_energy_grass = config["max_energy_grass"]

        # Walls (static obstacles)
        self.manual_wall_positions = config["manual_wall_positions"]
        self.wall_positions = set()

        # Mutation
        self.mutation_rate_predator = config["mutation_rate_predator"]
        self.mutation_rate_prey = config["mutation_rate_prey"]

        # Action range and movement mapping
        self.act_range = config["action_range"]

    def _init_reset_variables(self, seed):
        # Agent tracking
        self.current_step = 0
        # Seed RNG for this episode: prefer provided reset seed, else fallback to config seed, else default
        if seed is None:
            seed = self.config["seed"]
        self.rng = np.random.default_rng(seed)

        self.agent_positions = {}
        self.predator_positions = {}
        self.prey_positions = {}
        self.grass_positions = {}
        self.agent_energies = {}
        self.grass_energies = {}
        self.agent_ages = {}
        self.agent_parents = {}
        self.unique_agents = {}  # list of unique agent IDs
        self.unique_agent_stats = {}
        self.per_step_agent_data = []  # One entry per step; each is {agent_id: {position, energy, ...}}
        self._per_agent_step_deltas = {}  # Internal temp storage to track energy deltas during step
        self.agent_offspring_counts = {}
        self.agent_live_offspring_ids = {}

        self.agents_just_ate = set()
        self.cumulative_rewards = {}

        # Per-step infos accumulator and last-move diagnostics
        self._pending_infos = {}
        self._last_move_block_reason = {}
        # Global counters (optional diagnostics)
        self.los_rejected_moves_total = 0
        self.los_rejected_moves_by_type = {"predator": 0, "prey": 0}

        self.agent_activation_counts = {agent_id: 0 for agent_id in self.possible_agents}
        self.agent_ages = {agent_id: 0 for agent_id in self.possible_agents}
        self.death_agents_stats = {}
        self.death_cause_prey = {}

        self.agent_last_reproduction = {}

        # aggregates per step
        self.active_num_predators = 0
        self.active_num_prey = 0

        self.agents = []
        # create active agents list based on config
        for agent_type in ["predator", "prey"]:
            key = f"n_initial_active_{agent_type}"
            count = self.config[key]
            for i in range(count):
                agent_id = f"{agent_type}_{i}"
                self.agents.append(agent_id)
                self._register_new_agent(agent_id)

        self.grass_agents = [f"grass_{i}" for i in range(self.initial_num_grass)]

        self.grid_world_state_shape = (self.num_obs_channels, self.grid_size, self.grid_size)
        self.initial_grid_world_state = np.zeros(self.grid_world_state_shape, dtype=np.float32)
        self.grid_world_state = self.initial_grid_world_state.copy()

        def _generate_action_map(range_size: int):
            delta = (range_size - 1) // 2
            return {
                i: (dx, dy)
                for i, (dx, dy) in enumerate((dx, dy) for dx in range(-delta, delta + 1) for dy in range(-delta, delta + 1))
            }

        self.action_to_move_tuple_agents = _generate_action_map(self.act_range)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self._init_reset_variables(seed)

        self._create_and_place_grid_world_entities()

        self.active_num_predators = len(self.predator_positions)
        self.active_num_prey = len(self.prey_positions)
        self.current_num_grass = len(self.grass_positions)

        self.current_step = 0

        self.potential_new_ids = list(set(self.possible_agents) - set(self.agents))        
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        return observations, {}

    #-------- Reset placement methods grid world entities --------
    def _create_and_place_grid_world_entities(self):
        """
        Place and create all entities (walls, predators, prey, grass) into the grid world state.
        """
        self.wall_positions = self._create_wall_positions()
        predator_list, prey_list, predator_positions, prey_positions, grass_positions = self._sample_agent_and_grass_positions()
        self._place_walls(self.wall_positions)
        self._place_predators(predator_list, predator_positions)
        self._place_prey(prey_list, prey_positions)
        self._place_grass(grass_positions)
  
    # -------- Reset wall placement methods --------
    def _create_wall_positions(self):
        """
        Compute wall positions in the environment according to the placement mode and return as a set.
        """
        wall_positions = set()
        raw_positions = self.manual_wall_positions or []
        added = 0
        for pos in raw_positions:
            x, y = pos
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                if (x, y) not in wall_positions:
                    wall_positions.add((x, y))
                    added += 1
            else:
                if self.debug_mode:
                    print(f"[Walls] Skipping out-of-bounds {(x, y)}")
        if added == 0 and self.manual_wall_positions:
            if self.debug_mode:
                print("[Walls] No valid manual wall positions provided; resulting set is empty.")
        # If manual_wall_positions is empty, leave wall_positions empty (explicit)
        return wall_positions

    # -------- Reset other entity placement methods --------
    def _sample_agent_and_grass_positions(self):
        """
        Sample free positions for all entities and return lists for placement.
        Returns:
            predator_list, prey_list, predator_positions, prey_positions, grass_positions
        """
        num_grid_cells = self.grid_size * self.grid_size
        num_agents_and_grass = len(self.agents) + len(self.grass_agents)
        free_non_wall_indices = [i for i in range(num_grid_cells) if (i // self.grid_size, i % self.grid_size) not in self.wall_positions]

        chosen_grid_indices = self.rng.choice(free_non_wall_indices, size=num_agents_and_grass, replace=False)
        chosen_positions = [(i // self.grid_size, i % self.grid_size) for i in chosen_grid_indices]

        predator_list = [a for a in self.agents if "predator" in a]
        prey_list = [a for a in self.agents if "prey" in a]

        predator_positions = chosen_positions[: len(predator_list)]
        prey_positions = chosen_positions[len(predator_list) : len(predator_list) + len(prey_list)]
        grass_positions = chosen_positions[len(predator_list) + len(prey_list) :]

        return predator_list, prey_list, predator_positions, prey_positions, grass_positions

    def _place_walls(self, wall_positions):
        """
        Place walls into the grid world state.
        """
        self.wall_positions = wall_positions
        self.grid_world_state[0, :, :] = 0.0  # Clear wall channel
        for (wx, wy) in wall_positions:
            self.grid_world_state[0, wx, wy] = 1.0

    #-------- Placement method for predators --------
    def _place_predators(self, predator_list, predator_positions):
        self.predator_positions = {}
        for i, agent in enumerate(predator_list):
            pos = predator_positions[i]
            self.agent_positions[agent] = pos
            self.predator_positions[agent] = pos
            self.agent_energies[agent] = self.initial_energy_predator
            self.grid_world_state[1, *pos] = self.initial_energy_predator
            self.cumulative_rewards[agent] = 0.0

    #-------- Placement method for prey --------
    def _place_prey(self, prey_list, prey_positions):
        self.prey_positions = {}
        for i, agent in enumerate(prey_list):
            pos = prey_positions[i]
            self.agent_positions[agent] = pos
            self.prey_positions[agent] = pos
            self.agent_energies[agent] = self.initial_energy_prey
            self.grid_world_state[2, *pos] = self.initial_energy_prey
            self.cumulative_rewards[agent] = 0.0

    #-------- Placement method for grass --------
    def _place_grass(self, grass_positions):
        self.grass_positions = {}
        self.grass_energies = {}
        for i, grass in enumerate(self.grass_agents):
            pos = grass_positions[i]
            self.grass_positions[grass] = pos
            self.grass_energies[grass] = self.initial_energy_grass
            self.grid_world_state[3, *pos] = self.initial_energy_grass
        self.grass_pos_map = {tuple(pos): grass_id for grass_id, pos in self.grass_positions.items()}

    def step(self, action_dict):
        self.observations, self.rewards, self.terminations, self.truncations, self.infos = {}, {}, {}, {}, {}
        self.inactive_agents = list(set(self.possible_agents) - set(self.agents))
        self.inactive_predators = [agent for agent in self.inactive_agents if "predator" in agent]
        self.inactive_preys = [agent for agent in self.inactive_agents if "prey" in agent]

        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        

        # Step 1: Update age and process energy depletion due to time steps
        for agent in self.agents[:]:
            self.agent_ages[agent] += 1
            agent_channel = 1 if "predator" in agent else 2
            decay = self.energy_loss_per_step_predator if "predator" in agent else self.energy_loss_per_step_prey
            self.grid_world_state[agent_channel, *self.agent_positions[agent]] -= decay
            self.agent_energies[agent] -= decay
            if self.agent_energies[agent] <= 0:
                self.rewards[agent] = 0
                self.terminations[agent] = True
                self.truncations[agent] = False
                self.infos[agent] = {
                    "death_cause": "starvation",
                    "age_of_death": self.agent_ages[agent],
                    "position_of_death": self.agent_positions[agent],
                    "agent_activation_count": self.agent_activation_counts[agent],
                    }
                self.grid_world_state[agent_channel, *self.agent_positions[agent]] = 0.0
                if "predator" in agent:
                    self.active_num_predators -= 1
                    del self.predator_positions[agent]
                elif "prey" in agent:
                    self.active_num_prey -= 1
                    del self.prey_positions[agent]
                print(f"[Step] {agent} died of starvation at step {self.current_step}.")
                del self.agent_positions[agent]
                del self.agent_energies[agent]
                del self.agent_ages[agent]
                self.agents.remove(agent)   

        # Step 2: Regenerate grass energy on same spot
        for grass in self.grass_positions:
            self.grass_energies[grass] = min(
                self.grass_energies[grass] + self.energy_gain_per_step_grass, 
                self.max_energy_grass
                )
        
        # Step 3: Process agent movements
        for agent, action in action_dict.items():
            if agent not in self.agents:
                continue  # Skip agents that already have been terminated by starvation
            current_pos = self.agent_positions[agent]
            agent_channel = 1 if "predator" in agent else 2
            move_tuple = self.action_to_move_tuple_agents[action]
            if move_tuple[0] == 0 and move_tuple[1] == 0:
                # No movement, skip move_energy cost
                continue
            else:
                move_energy = self.move_energy_cost_predator if agent_channel == 1 else self.move_energy_cost_prey
                self.grid_world_state[agent_channel, *current_pos] -= move_energy
                self.agent_energies[agent] -= move_energy
            candidate_pos = (current_pos[0] + move_tuple[0], current_pos[1] + move_tuple[1])
            if not (0 <= candidate_pos[0] < self.grid_size and 0 <= candidate_pos[1] < self.grid_size):
                # Out of bounds; stay in place
                continue
            if self.grid_world_state[agent_channel, *candidate_pos] == 0.0: # cell is free (no wall or other same type agent)
                self.agent_positions[agent] = candidate_pos 
                if agent_channel == 1:
                    self.predator_positions[agent] = candidate_pos
                else:
                    self.prey_positions[agent] = candidate_pos
                self.grid_world_state[agent_channel, *current_pos] = 0.0
                self.grid_world_state[agent_channel, *candidate_pos] = self.agent_energies[agent]
            # else: blocked move; stay in place

        # Build a mapping of prey positions to prey IDs for quick lookup
        self.prey_pos_map = {tuple(pos): prey_id for prey_id, pos in self.agent_positions.items() if "prey" in prey_id}
 
        # Step 4: Handle agent engagements
        # Process predators first, so only one can eat a prey per cell
        for predator, predator_pos in self.predator_positions.items():
            caught_prey = self.prey_pos_map.get(predator_pos, None)
            if caught_prey is not None:
                self.grid_world_state[2, *predator_pos] = 0.0
                self.grid_world_state[1, *predator_pos] = self.agent_energies[predator]
                self.agent_energies[predator] += self.agent_energies[caught_prey]
                self.terminations[caught_prey] = True
                self.infos[caught_prey] = {
                    "death_cause": "caught_by_predator", 
                    "age_of_death": self.agent_ages[caught_prey],
                    "position_of_death": predator_pos,
                    "agent_activation_count": self.agent_activation_counts[caught_prey],   
                }
                print(f"[Eaten] {predator} ate {caught_prey} at step {self.current_step} with energy {round(self.agent_energies[caught_prey], 2)}.")
                self.agent_activation_counts[caught_prey] += 1
                del self.agent_positions[caught_prey]
                del self.prey_positions[caught_prey]
                del self.agent_energies[caught_prey]
                del self.agent_ages[caught_prey]
                self.agents.remove(caught_prey)
                self.active_num_prey -= 1
                

        for prey, prey_pos in self.prey_positions.items():
            caught_grass = self.grass_pos_map.get(prey_pos, None)
            if caught_grass is not None:
                print(f"[Eaten] {prey} ate {caught_grass} at step {self.current_step} with energy {round(self.grass_energies[caught_grass], 2)}.")
                self.grid_world_state[3, *prey_pos] = 0.0
                self.grid_world_state[2, *prey_pos] += self.grass_energies[caught_grass]
                self.agent_energies[prey] += self.grass_energies[caught_grass]


        # Step 5: Spawning of new agents
        # Iterate over a snapshot so newborns don't reproduce in the same step
        agents_snapshot = list(self.agents)
        for agent in agents_snapshot:
            if (
                "predator" in agent 
                and 
                self.agent_energies[agent] >= self.predator_creation_energy_threshold 
                and self.inactive_predators
            ):
                new_predator_id = self.inactive_predators.pop() 
                # Spawn position
                occupied_positions = set(self.agent_positions.values())
                new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)
                if new_position is None:
                    print(f"[Reproduction] No available spawn position for new predator {new_predator_id}.")
                    continue  # No available position found; skip reproduction this step
                else:
                    print(f"[Reproduction] {agent} reproduces new predator {new_predator_id} at step {self.current_step}.") 
                    # new predator placement
                    self.agent_activation_counts[new_predator_id] += 1
                    self.agent_positions[new_predator_id] = new_position
                    self.predator_positions[new_predator_id] = new_position
                    self.agent_energies[new_predator_id] = self.initial_energy_predator
                    self.agent_ages[new_predator_id] = 0
                    self.grid_world_state[1, *new_position] = self.initial_energy_predator
                    
                    # parent adjustments
                    self.agent_energies[agent] = self.agent_energies[agent] - self.initial_energy_predator
                    self.grid_world_state[1, *self.agent_positions[agent]] = self.agent_energies[agent]
                    self.rewards[agent] += self.reproduction_reward_predator

                    self.agents.append(new_predator_id)
                    self.active_num_predators += 1
                    print(f"[Reproduction] Number of active predators: {self.active_num_predators}")

               

        # Snapshot again to exclude the just spawned predators from reproducing now
        agents_snapshot = list(self.agents)
        for agent in agents_snapshot:
            if (
                "prey" in agent
                and self.agent_energies[agent] >= self.prey_creation_energy_threshold
                and self.inactive_preys
            ):
                new_prey_id = self.inactive_preys.pop()
                # Spawn position
                occupied_positions = set(self.agent_positions.values())
                new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)
                if new_position is None:
                    print(f"[Reproduction] No available spawn position for new prey {new_prey_id}.")
                    continue  # No available position found; skip reproduction this step
                else:
                    print(f"[Reproduction] {agent} reproduces new prey {new_prey_id} at step {self.current_step}.") 
                    # new prey placement
                    self.agent_activation_counts[new_prey_id] += 1
                    self.agent_positions[new_prey_id] = new_position
                    self.prey_positions[new_prey_id] = new_position
                    self.agent_energies[new_prey_id] = self.initial_energy_prey
                    self.agent_ages[new_prey_id] = 0
                    self.grid_world_state[2, *new_position] = self.initial_energy_prey
                    
                    # parent adjustments
                    self.agent_energies[agent] = self.agent_energies[agent] - self.initial_energy_prey
                    self.grid_world_state[2, *self.agent_positions[agent]] = self.agent_energies[agent]
                    self.rewards[agent] += self.reproduction_reward_prey

                    self.agents.append(new_prey_id)
                    self.active_num_prey += 1
                    print(f"[Reproduction] Number of active prey: {self.active_num_prey}")




        # Step 8: Generate observations for all agents AFTER all engagements in the step
        for agent in self.agents:
            if agent in self.agent_positions:
                self.observations[agent] = self._get_observation(agent)

        # --- Ensure per_step_agent_data is always appended ---
        step_data = {}
        for agent in self.agents:
            if agent in self.agent_positions:
                step_data[agent] = {
                    "position": self.agent_positions[agent],
                    "energy": self.agent_energies[agent],
                    "age": self.agent_ages[agent],
                    # Add more fields as needed
                }
        self.per_step_agent_data.append(step_data)

        # Increment step counter
        self.current_step += 1

        # step 9: Check for truncation
        if self.current_step >= self.max_steps:
            for agent in self.possible_agents:
                self.truncations[agent] = True

        # print(f"[Step] Step {self.current_step} completed. Active predators: {self.active_num_predators}, Active prey: {self.active_num_prey}.")
        # print(f"[Step] Active agents: {self.agents}")
        # print(f"[Step] Inactive agents: {self.inactive_agents}")
        # print(f"[Step] Rewards: {self.rewards}")
        # print(f"[Step] Terminations: {self.terminations}")

        self.truncations["__all__"] = False  # already handled at the beginning of the step        # Global termination and truncation
        self.terminations["__all__"] = self.active_num_prey <= 0 or self.active_num_predators <= 0

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos


    def _get_move(self, agent, action: int):
        """
        Get the new position of the agent based on the action and its type.
        """
        action = int(action)

        # Choose the appropriate movement dictionary based on agent type
        move_vector = self.action_to_move_tuple_agents[action]

        current_position = self.agent_positions[agent]
        new_position = (
            current_position[0] + move_vector[0],
            current_position[1] + move_vector[1],
        )

        # Clip new position to stay within grid bounds
        new_position = tuple(np.clip(new_position, 0, self.grid_size - 1))

        agent_type_nr = 1 if "predator" in agent else 2
        # Default: no block
        self._last_move_block_reason[agent] = None
        # Block entry into wall cells
        if new_position in self.wall_positions:
            new_position = current_position
            self._last_move_block_reason[agent] = "wall"
        elif self.grid_world_state[agent_type_nr, *new_position] > 0:
            new_position = current_position
            self._last_move_block_reason[agent] = "occupied"
        elif self.respect_los_for_movement and new_position != current_position:
            # Block diagonal moves if either adjacent orthogonal cell is a wall (no corner cutting)
            dx = new_position[0] - current_position[0]
            dy = new_position[1] - current_position[1]
            if abs(dx) == 1 and abs(dy) == 1:
                ortho1 = (current_position[0] + dx, current_position[1])
                ortho2 = (current_position[0], current_position[1] + dy)
                if ortho1 in self.wall_positions or ortho2 in self.wall_positions:
                    new_position = current_position
                    self._last_move_block_reason[agent] = "corner_cut"
            # Perform LOS check; if blocked by any wall (excluding endpoints) cancel move.
            elif not self._line_of_sight_clear(current_position, new_position):
                new_position = current_position
                self._last_move_block_reason[agent] = "los"

        return new_position

    def _line_of_sight_clear(self, start, end):
        """Return True if straight line between start and end (inclusive endpoints) has no wall strictly between.

        Uses a simple integer Bresenham traversal. Walls at the destination do not count here
        because destination walls are already handled earlier; we exclude start/end when checking.
        """
        (x0, y0), (x1, y1) = start, end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        if dx >= dy:
            err = dx / 2.0
            while x != x1:
                if (x, y) not in (start, end) and (x, y) in self.wall_positions:
                    return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if (x, y) not in (start, end) and (x, y) in self.wall_positions:
                    return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        # Check final cell (excluded by earlier condition); not necessary for movement blocking beyond destination.
        return True

    def _get_observation(self, agent):
        """
        Generate an observation for the agent.
        """
        observation_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp, observation_range)
        channels = self.num_obs_channels + (1 if self.include_visibility_channel else 0)
        observation = np.zeros((channels, observation_range, observation_range), dtype=np.float32)
        # Channel 0: walls (binary). Slice from the pre-painted global walls channel for this window.
        observation[0, xolo:xohi, yolo:yohi] = self.grid_world_state[0, xlo:xhi, ylo:yhi]
        # Copy dynamic channels (predators, prey, grass) into fixed locations 1..num_obs_channels-1 first
        observation[1:self.num_obs_channels, xolo:xohi, yolo:yohi] = self.grid_world_state[1:, xlo:xhi, ylo:yhi]

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

    def _build_possible_agent_ids(self):
        """
        Build the list of possible agents based on the configuration.
        This is called during reset to ensure the agent list is up-to-date.
        """
        agent_ids = []
        for i in range(self.n_possible_predators):
            agent_ids.append(f"predator_{i}")
        for i in range(self.n_possible_prey):
            agent_ids.append(f"prey_{i}")
        return agent_ids

    def _build_observation_space(self, agent_id):
        """
        Build the observation space for a specific agent.
        """
        extra = 1 if getattr(self, "include_visibility_channel", False) else 0
        if "predator" in agent_id:
            predator_shape = (self.num_obs_channels + extra, self.predator_obs_range, self.predator_obs_range)
            obs_space = gymnasium.spaces.Box(low=0, high=100.0, shape=predator_shape, dtype=np.float32)

        elif "prey" in agent_id:
            prey_shape = (self.num_obs_channels + extra, self.prey_obs_range, self.prey_obs_range)
            obs_space = gymnasium.spaces.Box(low=0, high=100.0, shape=prey_shape, dtype=np.float32)

        else:
            raise ValueError(f"Unknown agent type in ID: {agent_id}")

        return obs_space

    def _build_action_space(self, agent_id):
        """
        Build the action space for a specific agent.
        """
        action_space = gymnasium.spaces.Discrete(self.act_range**2)

        return action_space

    def _register_new_agent(self, agent_id: str, parent_unique_id: str = None):
        reuse_index = self.agent_activation_counts[agent_id]
        unique_id = f"{agent_id}_{reuse_index}"
        self.unique_agents[agent_id] = unique_id
        self.agent_activation_counts[agent_id] += 1

        self.agent_ages[agent_id] = 0
        self.agent_offspring_counts[agent_id] = 0
        self.agent_live_offspring_ids[agent_id] = []

        self.agent_parents[agent_id] = parent_unique_id

        self.agent_last_reproduction[agent_id] = -self.config["reproduction_cooldown_steps"]

        self.unique_agent_stats[unique_id] = {
            "birth_step": self.current_step,
            "parent": parent_unique_id,
            "offspring_count": 0,
            "distance_traveled": 0.0,
            "times_ate": 0,
            "energy_gained": 0.0,
            "energy_spent": 0.0,
            "avg_energy_sum": 0.0,
            "avg_energy_steps": 0,
            "cumulative_reward": 0.0,
            "policy_group": "_".join(agent_id.split("_")[:3]),
            "mutated": False,
            "death_step": None,
            "death_cause": None,
            # removed final_energy
            "avg_energy": None,
        }

    def get_total_energy_by_type(self):
        """
        Returns a dict with total energy by category:
        - 'predator': total predator energy
        - 'prey': total prey energy
        - 'grass': total grass energy
        """
        energy_totals = {
            "predator": 0.0,
            "prey": 0.0,
            "grass": sum(self.grass_energies.values()),
        }

        for agent_id, energy in self.agent_energies.items():
            if "predator" in agent_id:
                energy_totals["predator"] += energy
            elif "prey" in agent_id:
                energy_totals["prey"] += energy
        return energy_totals

    def get_total_offspring_by_type(self):
        """
        Returns a dict of total offspring counts by agent type.
        Example:
        {
            "prey": 34,
            ...
        }
        """
        counts = {
            "predator": 0,
            "prey": 0,
        }
        for stats in self.unique_agent_stats.values():
            group = stats["policy_group"]
            if group in counts:
                counts[group] += stats.get("offspring_count", 0)
        return counts

    def get_total_energy_spent_by_type(self):
        """
        Returns a dict of total energy spent by agent type.
        Example:
        {
            "prey": 192.3,
            ...
        }
        """
        energy_spent = {
            "predator": 0.0,
            "prey": 0.0,
        }
        for stats in self.unique_agent_stats.values():
            group = stats["policy_group"]
            if group in energy_spent:
                energy_spent[group] += stats.get("energy_spent", 0.0)
        return energy_spent

    def _get_type_specific(self, key: str, agent_id: str):
        raw_val = getattr(self, f"{key}_config", 0.0)
        if isinstance(raw_val, dict):
            for k in raw_val:
                if agent_id.startswith(k):
                    return raw_val[k]
            raise KeyError(f"Type-specific key '{agent_id}' not found under '{key}'")
        return raw_val
