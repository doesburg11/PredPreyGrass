"""
Predator-Prey Grass RLlib Environment

Additional features:

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

        # --- Carcass state tracking ---
        # Carcass positions: {carcass_id: (x, y)}
        self.carcass_positions = {}
        # Carcass energies: {carcass_id: energy}
        self.carcass_energies = {}
        # Carcass ages: {carcass_id: age in steps}
        self.carcass_ages = {}
        # Carcass unique id counter
        self.carcass_counter = 0

        # Add carcass channel to grid state (one extra channel)
        self.carcass_channel_idx = self.num_obs_channels

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
        self.reward_predator_catch_prey_config = config["reward_predator_catch_prey"]
        self.reward_prey_eat_grass_config = config["reward_prey_eat_grass"]

        self.reward_predator_step_config = config["reward_predator_step"]
        self.reward_prey_step_config = config["reward_prey_step"]
        self.penalty_prey_caught_config = config["penalty_prey_caught"]
        self.reproduction_reward_predator_config = config["reproduction_reward_predator"]
        self.reproduction_reward_prey_config = config["reproduction_reward_prey"]

        # Energy settings
        self.energy_loss_per_step_predator = config["energy_loss_per_step_predator"]
        self.energy_loss_per_step_prey = config["energy_loss_per_step_prey"]
        self.predator_creation_energy_threshold = config["predator_creation_energy_threshold"]
        self.prey_creation_energy_threshold = config["prey_creation_energy_threshold"]

        # Learning agents
        self.n_possible_type_1_predators = config["n_possible_type_1_predators"]
        self.n_possible_type_2_predators = config["n_possible_type_2_predators"]
        self.n_possible_type_1_prey = config["n_possible_type_1_prey"]
        self.n_possible_type_2_prey = config["n_possible_type_2_prey"]

        self.n_initial_active_type_1_predator = config["n_initial_active_type_1_predator"]
        self.n_initial_active_type_2_predator = config["n_initial_active_type_2_predator"]
        self.n_initial_active_type_1_prey = config["n_initial_active_type_1_prey"]
        self.n_initial_active_type_2_prey = config["n_initial_active_type_2_prey"]

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
        # Walls (static obstacles)
        self.num_walls = config["num_walls"]
        # New: wall placement mode: 'random' (default) or 'manual'.
        # When 'manual', positions come from manual_wall_positions (list of (x,y)).
        self.wall_placement_mode = config["wall_placement_mode"]
        self.manual_wall_positions = config["manual_wall_positions"]
        self.wall_positions = set()

        # Mutation
        self.mutation_rate_predator = config["mutation_rate_predator"]
        self.mutation_rate_prey = config["mutation_rate_prey"]

        # Action range and movement mapping
        self.type_1_act_range = config["type_1_action_range"]
        self.type_2_act_range = config["type_2_action_range"]


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
        self.death_agents_stats = {}
        self.death_cause_prey = {}

        self.agent_last_reproduction = {}

        # aggregates per step
        self.active_num_predators = 0
        self.active_num_prey = 0

        self.agents = []
        # create active agents list based on config
        for agent_type in ["predator", "prey"]:
            for type in [1, 2]:
                key = f"n_initial_active_type_{type}_{agent_type}"
                count = self.config[key]
                for i in range(count):
                    agent_id = f"type_{type}_{agent_type}_{i}"
                    self.agents.append(agent_id)
                    self._register_new_agent(agent_id)

        self.grass_agents = [f"grass_{i}" for i in range(self.initial_num_grass)]

        # Add an extra channel for carcasses
        self.grid_world_state_shape = (self.num_obs_channels + 1, self.grid_size, self.grid_size)
        self.initial_grid_world_state = np.zeros(self.grid_world_state_shape, dtype=np.float32)
        self.grid_world_state = self.initial_grid_world_state.copy()

        def _generate_action_map(range_size: int):
            delta = (range_size - 1) // 2
            return {
                i: (dx, dy)
                for i, (dx, dy) in enumerate((dx, dy) for dx in range(-delta, delta + 1) for dy in range(-delta, delta + 1))
            }

        self.action_to_move_tuple_type_1_agents = _generate_action_map(self.type_1_act_range)
        self.action_to_move_tuple_type_2_agents = _generate_action_map(self.type_2_act_range)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self._init_reset_variables(seed)

        # --- Place walls first ---
        self.wall_positions = set()
        max_cells = self.grid_size * self.grid_size
        if self.wall_placement_mode not in ("random", "manual"):
            raise ValueError("wall_placement_mode must be 'random' or 'manual'")

        if self.wall_placement_mode == "manual":
            # Manual mode: use provided coordinates; ignore duplicates/out-of-bounds
            raw_positions = self.manual_wall_positions or []
            added = 0
            for pos in raw_positions:
                try:
                    x, y = map(int, pos)
                except Exception:
                    if self.debug_mode:
                        print(f"[Walls] Skipping non-integer position {pos}")
                    continue
                if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                    if self.debug_mode:
                        print(f"[Walls] Skipping out-of-bounds {(x,y)}")
                    continue
                if (x, y) in self.wall_positions:
                    continue
                self.wall_positions.add((x, y))
                added += 1
            # Optional: if manual list empty, fallback to random to avoid empty wall layer unless explicitly desired
            if added == 0 and self.manual_wall_positions:
                if self.debug_mode:
                    print("[Walls] No valid manual wall positions provided; resulting set is empty.")
            if added == 0 and not self.manual_wall_positions:
                # Keep behavior consistent: if user sets mode manual but no list, leave empty (explicit)
                pass
        else:  # random
            if self.num_walls >= max_cells:
                raise ValueError("num_walls must be less than total grid cells")
            if self.num_walls > 0:
                wall_indices = self.rng.choice(max_cells, size=self.num_walls, replace=False)
                for idx in wall_indices:
                    gx = idx // self.grid_size
                    gy = idx % self.grid_size
                    self.wall_positions.add((gx, gy))

        total_entities = len(self.agents) + len(self.grass_agents)
        # PATCH: allow any config where number of free cells >= number of agents (even if zero grass).
        # Use actual number of placed walls (manual or random) for free-cell check.
        free_cells = max_cells - len(self.wall_positions)
        if total_entities > free_cells:
            raise ValueError(
                f"Too many agents+grass ({total_entities}) for free cells ({free_cells}) given {len(self.wall_positions)} walls on {self.grid_size}x{self.grid_size} grid"
            )
        free_indices = [i for i in range(max_cells) if (i // self.grid_size, i % self.grid_size) not in self.wall_positions]
        if total_entities > 0:
            chosen = self.rng.choice(free_indices, size=total_entities, replace=False)
            all_positions = [(i // self.grid_size, i % self.grid_size) for i in chosen]
        else:
            all_positions = []

        predator_list = [a for a in self.agents if "predator" in a]
        prey_list = [a for a in self.agents if "prey" in a]

        predator_positions = all_positions[: len(predator_list)]
        prey_positions = all_positions[len(predator_list) : len(predator_list) + len(prey_list)]
        grass_positions = all_positions[len(predator_list) + len(prey_list) :]

        # Paint walls into channel 0
        for (wx, wy) in self.wall_positions:
            self.grid_world_state[0, wx, wy] = 1.0

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
        # Print the shape of the first agent's observation and its expected space
        if observations:
            first_agent = next(iter(observations))
            actual = observations[first_agent].shape
            expected = self.observation_spaces[first_agent].shape if first_agent in self.observation_spaces else None
            print(f"[RESET DEBUG] First agent {first_agent}: actual obs shape {actual}, expected {expected}")
        return observations, {}

    def step(self, action_dict):
        t0 = time.perf_counter()
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        # For stepwise display eating in grid
        self.agents_just_ate.clear()
        # Reset per-step infos
        self._pending_infos = {}

        # step 0: Check for truncation
        t_trunc0 = time.perf_counter()
        truncation_result = self._check_truncation_and_early_return(observations, rewards, terminations, truncations, infos)
        t_trunc1 = time.perf_counter()
        trunc_check = t_trunc1 - t_trunc0
        if truncation_result is not None:
            t_total = time.perf_counter() - t0
            print(f"[PROFILE] step: trunc_check={trunc_check:.6f}s, decay=0.000000s, age=0.000000s, grass=0.000000s, move=0.000000s, engage=0.000000s, repro=0.000000s, obs=0.000000s, total={t_total:.6f}s")
            return truncation_result

        # Step 1: If not truncated; process energy depletion due to time steps and update age
        t_decay0 = time.perf_counter()
        self._apply_energy_decay_per_step(action_dict)
        t_decay1 = time.perf_counter()
        decay = t_decay1 - t_decay0

        # Step 2: Update ages of all agents who act
        t_age0 = time.perf_counter()
        self._apply_age_update(action_dict)
        t_age1 = time.perf_counter()
        age = t_age1 - t_age0

        # Step 3: Regenerate grass energy
        t_grass0 = time.perf_counter()
        self._regenerate_grass_energy()
        t_grass1 = time.perf_counter()
        grass = t_grass1 - t_grass0

        # Step 4: process agent movements
        t_move0 = time.perf_counter()
        self._process_agent_movements(action_dict)
        t_move1 = time.perf_counter()
        move = t_move1 - t_move0

        # Step 5: Handle agent engagements (optimized)
        t_engage0 = time.perf_counter()
        # Precompute position-to-agent mappings for prey and grass for O(1) lookup
        prey_pos_map = {tuple(pos): prey for prey, pos in self.agent_positions.items() if "prey" in prey}
        grass_pos_map = {tuple(pos): grass for grass, pos in self.grass_positions.items()}
        engage_subsections = [
            "log", "just_ate", "reward", "gain", "cap", "stats", "grid", "prey_reward", "prey_stats", "del", "grass", "total"
        ]
        engage_totals = {k: 0.0 for k in engage_subsections}
        engage_counts = 0
        for agent in self.agents:
            if agent not in self.agent_positions:
                continue
            if self.agent_energies[agent] <= 0:
                self._handle_energy_decay(agent, observations, rewards, terminations, truncations)
            elif "predator" in agent:
                timings = self._handle_predator_engagement(agent, observations, rewards, terminations, truncations, prey_pos_map=prey_pos_map)
                if isinstance(timings, dict):
                    for k in engage_subsections:
                        if k in timings:
                            engage_totals[k] += timings[k]
                    engage_counts += 1
            elif "prey" in agent:
                timings = self._handle_prey_engagement(agent, observations, rewards, terminations, truncations, grass_pos_map=grass_pos_map)
                if isinstance(timings, dict):
                    for k in engage_subsections:
                        if k in timings:
                            engage_totals[k] += timings[k]
                    engage_counts += 1
        t_engage1 = time.perf_counter()
        engage = t_engage1 - t_engage0
        if engage_counts > 0 and getattr(self, "debug_mode", False):
            print('[PROFILE-ENGAGE-SUMMARY] ' + ' '.join(f'{k}={engage_totals[k]:.6f}' for k in engage_subsections))

        # Step 6: Handle agent removals
        for agent in self.agents[:]:
            if terminations.get(agent, False):
                self._log(self.verbose_engagement, f"[TERMINATED] Agent {agent} terminated!", "red")
                self.agents.remove(agent)
                uid = self.unique_agents[agent]
                self.death_agents_stats[uid] = {
                    **self.unique_agent_stats[uid],
                    "lifetime": self.agent_ages[agent],
                    "parent": self.agent_parents[agent],
                }
                del self.unique_agents[agent]

        # Step 7: Spawning of new agents
        t_repro0 = time.perf_counter()
        for agent in self.agents[:]:
            if "predator" in agent:
                self._handle_predator_reproduction(agent, rewards, observations, terminations, truncations)
            elif "prey" in agent:
                self._handle_prey_reproduction(agent, rewards, observations, terminations, truncations)
        t_repro1 = time.perf_counter()
        repro = t_repro1 - t_repro0

        # Step 8: Generate observations for all agents AFTER all engagements in the step
        t_obs0 = time.perf_counter()
        for agent in self.agents:
            if agent in self.agent_positions:
                observations[agent] = self._get_observation(agent)
        t_obs1 = time.perf_counter()
        obs = t_obs1 - t_obs0

        # output only observations, rewards for active agents
        observations = {agent: observations[agent] for agent in self.agents if agent in observations}
        rewards = {agent: rewards[agent] for agent in self.agents if agent in rewards}
        terminations = {agent: terminations[agent] for agent in self.agents if agent in terminations}
        truncations = {agent: truncations[agent] for agent in self.agents if agent in truncations}
        truncations["__all__"] = False  # already handled at the beginning of the step        # Global termination and truncation
        terminations["__all__"] = self.active_num_prey <= 0 or self.active_num_predators <= 0

        # Provide infos accumulated during the step
        infos = {agent: self._pending_infos.get(agent, {}) for agent in self.agents}

        # Sort agents for debugging
        self.agents.sort()

        step_data = {}

        for agent in self.agents:
            pos = self.agent_positions[agent]
            energy = self.agent_energies[agent]
            deltas = self._per_agent_step_deltas.get(agent, {"decay": 0.0, "move": 0.0, "eat": 0.0, "repro": 0.0})

            step_data[agent] = {
                "position": pos,
                "energy": energy,
                "energy_decay": deltas["decay"],
                "energy_movement": deltas["move"],
                "energy_eating": deltas["eat"],
                "energy_reproduction": deltas["repro"],
                "age": self.agent_ages[agent],
                "offspring_count": self.agent_offspring_counts[agent],
                "offspring_ids": self.agent_live_offspring_ids[agent],
            }

        self.per_step_agent_data.append(step_data)
        self._per_agent_step_deltas.clear()

        # Increment step counter
        self.current_step += 1

        # Profiling summary for this step
        t1 = time.perf_counter()
        t_total = t1 - t0
        if getattr(self, "debug_mode", False):
            print(f"[PROFILE] step: trunc_check={trunc_check:.6f}s, decay={decay:.6f}s, age={age:.6f}s, grass={grass:.6f}s, move={move:.6f}s, engage={engage:.6f}s, repro={repro:.6f}s, obs={obs:.6f}s, total={t_total:.6f}s")
        return observations, rewards, terminations, truncations, infos

    def _get_movement_energy_cost(self, agent, current_position, new_position):
        """
        Calculate energy cost for movement based on distance and a configurable factor.
        """
        distance_factor = self.config["move_energy_cost_factor"]
        # print(f"Distance factor: {distance_factor}")
        current_energy = self.agent_energies[agent]
        # print(f"Current energy: {current_energy}")
        # distance gigh type =[0.00,1.00, 1.41, 2.00, 2.24, 2.83]
        distance = math.sqrt((new_position[0] - current_position[0]) ** 2 + (new_position[1] - current_position[1]) ** 2)
        # print (f"Distance: {distance}")
        energy_cost = distance * distance_factor * current_energy
        return energy_cost

    def _get_move(self, agent, action: int):
        """
        Get the new position of the agent based on the action and its type.
        """
        action = int(action)

        # Choose the appropriate movement dictionary based on agent type
        if "type_1" in agent:
            move_vector = self.action_to_move_tuple_type_1_agents[action]
        elif "type_2" in agent:
            move_vector = self.action_to_move_tuple_type_2_agents[action]
        else:
            raise ValueError(f"Unknown type for agent: {agent}")

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
        import time
        obs_t0 = time.perf_counter()
        """
        Generate an observation for the agent.
        """
        observation_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp, observation_range)
        # Number of channels in the grid (including carcass channel)
        grid_channels = self.grid_world_state.shape[0]
        channels = grid_channels + (1 if self.include_visibility_channel else 0)
        observation = np.zeros((channels, observation_range, observation_range), dtype=np.float32)
        # Channel 0: walls (binary)
        dx = min(xhi - xlo, xohi - xolo)
        dy = min(yhi - ylo, yohi - yolo)
        observation[0, xolo:xolo+dx, yolo:yolo+dy] = self.grid_world_state[0, xlo:xlo+dx, ylo:ylo+dy]
        # Copy all other channels (predators, prey, grass, carcass) into observation
        observation[1:grid_channels, xolo:xolo+dx, yolo:yolo+dy] = self.grid_world_state[1:, xlo:xlo+dx, ylo:ylo+dy]

        need_visibility_mask = self.include_visibility_channel or self.mask_observation_with_visibility
        visibility_mask = None
        if need_visibility_mask:
            visibility_mask = np.zeros((observation_range, observation_range), dtype=np.float32)

            los_times = []
            for lx in range(observation_range):
                for ly in range(observation_range):
                    gx = xlo + (lx - xolo)
                    gy = ylo + (ly - yolo)
                    if not (0 <= gx < self.grid_size and 0 <= gy < self.grid_size):
                        continue
                    los_t0 = time.perf_counter()
                    clear = self._line_of_sight_clear((xp, yp), (gx, gy))
                    los_t1 = time.perf_counter()
                    los_times.append(((xp, yp), (gx, gy), los_t1 - los_t0))
                    visibility_mask[lx, ly] = 1.0 if clear else 0.0
            # Log slowest LOS pair for this agent
            if los_times:
                slowest = max(los_times, key=lambda t: t[2])
                if slowest[2] > 0.0005:  # Only log if >0.5ms
                    if getattr(self, "debug_mode", False):
                        print(f"[PROFILE-OBS-LOS] agent={agent} from={slowest[0]} to={slowest[1]} los_time={slowest[2]:.6f}s")

            if self.mask_observation_with_visibility:
                # Multiply dynamic channels (exclude channel 0 walls, and exclude visibility channel if it'll be appended later)
                for c in range(1, self.num_obs_channels):
                    observation[c] *= visibility_mask

            if self.include_visibility_channel:
                vis_idx = channels - 1
                observation[vis_idx] = visibility_mask

        obs_t1 = time.perf_counter()
        obs_time = obs_t1 - obs_t0
        if obs_time > 0.002 and getattr(self, "debug_mode", False):  # Only log if >2ms
            print(f"[PROFILE-OBS-AGENT] agent={agent} obs_time={obs_time:.6f}s")
        # Debug: compare actual observation shape to declared observation space
        expected_shape = self.observation_spaces[agent].shape if hasattr(self, 'observation_spaces') and agent in self.observation_spaces else None
        print(f"[OBS DEBUG] Agent {agent}: actual obs shape {observation.shape}, expected {expected_shape}")
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
            "white": "\033[97m",
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
                else:  # Inactive agents get empty observation
                    obs_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
                    channels = self.num_obs_channels + (1 if self.include_visibility_channel else 0)
                    observations[agent] = np.zeros((channels, obs_range, obs_range), dtype=np.float32)

                rewards[agent] = 0.0
                truncations[agent] = True
                terminations[agent] = False

            truncations["__all__"] = True
            terminations["__all__"] = False
            return observations, rewards, terminations, truncations, infos

        return None

    def _apply_energy_decay_per_step(self, action_dict):
        """
        Apply fixed per-step energy decay to all active (alive) agents based on type.
        """
        for agent in list(self.agent_positions.keys()):

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
            self._per_agent_step_deltas[agent] = {
                "decay": -decay,
                "move": 0.0,
                "eat": 0.0,
                "repro": 0.0,
            }

            self.grid_world_state[layer, *self.agent_positions[agent]] = self.agent_energies[agent]

            self._log(
                self.verbose_decay,
                f"[DECAY] {agent} energy: {round(old_energy, 2)} -> {round(self.agent_energies[agent], 2)}",
                "red",
            )

    def _apply_age_update(self, action_dict):
        """
        Increment the age of each active (alive) agent by one step.
        """
        for agent in self.agent_positions.keys():
            self.agent_ages[agent] += 1

    def _regenerate_grass_energy(self):
        """
        Increase energy of all grass patches, capped at initial energy value.
        """
        # Cap energy to maximum allowed for grass
        max_energy_grass = self.config["max_energy_grass"]
        for grass, pos in self.grass_positions.items():
            new_energy = min(self.grass_energies[grass] + self.energy_gain_per_step_grass, max_energy_grass)
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
                # Populate per-step infos and counters for LOS rejections
                reason = self._last_move_block_reason.get(agent)
                agent_kind = "predator" if "predator" in agent else "prey"
                if reason == "los":
                    self.los_rejected_moves_total += 1
                    self.los_rejected_moves_by_type[agent_kind] += 1
                    self._pending_infos.setdefault(agent, {})["los_rejected"] = 1
                    self._pending_infos[agent]["move_blocked_reason"] = reason
                else:
                    # Ensure key exists for easier aggregation
                    self._pending_infos.setdefault(agent, {})["los_rejected"] = 0
                    if reason:
                        self._pending_infos[agent]["move_blocked_reason"] = reason
                move_cost = self._get_movement_energy_cost(agent, old_position, new_position)
                self.agent_energies[agent] -= move_cost
                self._per_agent_step_deltas[agent]["move"] = -move_cost

                uid = self.unique_agents[agent]
                self.unique_agent_stats[uid]["distance_traveled"] += np.linalg.norm(np.array(new_position) - np.array(old_position))
                self.unique_agent_stats[uid]["energy_spent"] += move_cost
                self.unique_agent_stats[uid]["avg_energy_sum"] += self.agent_energies[agent]
                self.unique_agent_stats[uid]["avg_energy_steps"] += 1

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

    def _handle_energy_decay(self, agent, observations, rewards, terminations, truncations):
        self._log(self.verbose_decay, f"[DECAY] {agent} at {self.agent_positions[agent]} ran out of energy and is removed.", "red")
        observations[agent] = self._get_observation(agent)
        rewards[agent] = 0
        terminations[agent] = True
        truncations[agent] = False

        layer = 1 if "predator" in agent else 2
        self.grid_world_state[layer, *self.agent_positions[agent]] = 0
        uid = self.unique_agents[agent]
        stat = self.unique_agent_stats[uid]
        stat["death_step"] = self.current_step

        stat["death_cause"] = "starved"  # or "eaten"
    # removed final_energy: not used
        steps = max(stat["avg_energy_steps"], 1)
        stat["avg_energy"] = stat["avg_energy_sum"] / steps
        # Fix: Always use the agent's own cumulative reward, not another agent's value
        stat["cumulative_reward"] = self.cumulative_rewards.get(agent, 0.0)

        self.death_agents_stats[uid] = stat

        if "predator" in agent:
            self.active_num_predators -= 1
            del self.predator_positions[agent]
        else:
            self.active_num_prey -= 1
            del self.prey_positions[agent]

        del self.agent_positions[agent]
        del self.agent_energies[agent]

    def _handle_predator_engagement(self, agent, observations, rewards, terminations, truncations, prey_pos_map=None):
        predator_position = tuple(self.agent_positions[agent])
        if prey_pos_map is not None:
            caught_prey = prey_pos_map.get(predator_position, None)
            # Guard against stale map entries (prey already removed by another predator earlier this step)
            if caught_prey is not None and caught_prey not in self.agent_positions:
                caught_prey = None
        else:
            # fallback to old method if not provided
            caught_prey = next(
                (prey for prey, pos in self.agent_positions.items() if "prey" in prey and np.array_equal(predator_position, pos)), None
            )

        if caught_prey:
            t0 = time.perf_counter()
            self._log(self.verbose_engagement, f"[ENGAGE] {agent} caught {caught_prey} at {tuple(map(int, predator_position))}", "white")
            t_log = time.perf_counter()
            self.agents_just_ate.add(agent)
            t_just_ate = time.perf_counter()
            rewards[agent] = self._get_type_specific("reward_predator_catch_prey", agent)
            self.cumulative_rewards.setdefault(agent, 0)
            self.cumulative_rewards[agent] += rewards[agent]
            t_reward = time.perf_counter()
            prey_energy = self.agent_energies[caught_prey]
            max_intake = self.config["max_energy_gain_per_prey"]
            raw_gain = min(prey_energy, max_intake)
            efficiency = self.config["energy_transfer_efficiency"]
            gain = raw_gain * efficiency
            self.agent_energies[agent] += gain
            self._per_agent_step_deltas[agent]["eat"] = gain
            t_gain = time.perf_counter()
            max_energy = self.config["max_energy_predator"]
            self.agent_energies[agent] = min(self.agent_energies[agent], max_energy)
            t_cap = time.perf_counter()
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["times_ate"] += 1
            self.unique_agent_stats[uid]["energy_gained"] += gain
            self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]
            t_stats = time.perf_counter()
            self.grid_world_state[1, *predator_position] = self.agent_energies[agent]
            t_grid = time.perf_counter()
            observations[caught_prey] = self._get_observation(caught_prey)
            rewards[caught_prey] = self._get_type_specific("penalty_prey_caught", caught_prey)
            self.cumulative_rewards.setdefault(caught_prey, 0.0)
            self.cumulative_rewards[caught_prey] += rewards[caught_prey]
            t_prey_reward = time.perf_counter()
            terminations[caught_prey] = True
            truncations[caught_prey] = False
            self.active_num_prey -= 1
            prey_pos = self.agent_positions[caught_prey]
            self.grid_world_state[2, *prey_pos] = 0
            uid_prey = self.unique_agents[caught_prey]
            stat = self.unique_agent_stats[uid_prey]
            stat["death_step"] = self.current_step
            stat["death_cause"] = "eaten"
            steps = max(stat["avg_energy_steps"], 1)
            stat["avg_energy"] = stat["avg_energy_sum"] / steps
            stat["cumulative_reward"] = self.cumulative_rewards.get(caught_prey, 0.0)
            self.death_agents_stats[uid_prey] = stat
            t_prey_stats = time.perf_counter()
            # --- Carcass creation logic ---
            leftover_energy = prey_energy - raw_gain
            if leftover_energy > 0:
                # Create a new carcass at the prey's position
                carcass_id = f"carcass_{self.carcass_counter}"
                self.carcass_counter += 1
                self.carcass_positions[carcass_id] = prey_pos
                self.carcass_energies[carcass_id] = leftover_energy
                self.carcass_ages[carcass_id] = 0
                # Paint carcass energy into the grid's carcass channel
                self.grid_world_state[self.carcass_channel_idx, *prey_pos] = leftover_energy
            else:
                # Clear any carcass at this position (shouldn't be needed, but for safety)
                self.grid_world_state[self.carcass_channel_idx, *prey_pos] = 0
            if prey_pos_map is not None:
                prey_pos_map.pop(predator_position, None)
            del self.agent_positions[caught_prey]
            del self.prey_positions[caught_prey]
            del self.agent_energies[caught_prey]
            t_del = time.perf_counter()
            if getattr(self, "debug_mode", False):
                print(f"[PROFILE-ENGAGE] pred: log={1e3*(t_log-t0):.3f}ms just_ate={1e3*(t_just_ate-t_log):.3f}ms reward={1e3*(t_reward-t_just_ate):.3f}ms gain={1e3*(t_gain-t_reward):.3f}ms cap={1e3*(t_cap-t_gain):.3f}ms stats={1e3*(t_stats-t_cap):.3f}ms grid={1e3*(t_grid-t_stats):.3f}ms prey_reward={1e3*(t_prey_reward-t_grid):.3f}ms prey_stats={1e3*(t_prey_stats-t_prey_reward):.3f}ms del={1e3*(t_del-t_prey_stats):.3f}ms total={1e3*(t_del-t0):.3f}ms")
            return {
                "log": t_log-t0, "just_ate": t_just_ate-t_log, "reward": t_reward-t_just_ate, "gain": t_gain-t_reward,
                "cap": t_cap-t_gain, "stats": t_stats-t_cap, "grid": t_grid-t_stats, "prey_reward": t_prey_reward-t_grid,
                "prey_stats": t_prey_stats-t_prey_reward, "del": t_del-t_prey_stats, "total": t_del-t0
            }
        else:
            # Check for carcass at predator's position
            predator_position = tuple(self.agent_positions[agent])
            carcass_id = None
            for cid, pos in self.carcass_positions.items():
                if tuple(pos) == predator_position:
                    carcass_id = cid
                    break
            if carcass_id is not None:
                t0 = time.perf_counter()
                self._log(self.verbose_engagement, f"[ENGAGE] {agent} consumes carcass {carcass_id} at {predator_position}", "magenta")
                t_log = time.perf_counter()
                self.agents_just_ate.add(agent)
                t_just_ate = time.perf_counter()
                rewards[agent] = self._get_type_specific("reward_predator_catch_prey", agent)  # Optionally use a separate reward for carcass
                self.cumulative_rewards.setdefault(agent, 0)
                self.cumulative_rewards[agent] += rewards[agent]
                t_reward = time.perf_counter()
                carcass_energy = self.carcass_energies[carcass_id]
                max_intake = self.config["max_energy_gain_per_prey"]
                raw_gain = min(carcass_energy, max_intake)
                efficiency = self.config["energy_transfer_efficiency"]
                gain = raw_gain * efficiency
                self.agent_energies[agent] += gain
                self._per_agent_step_deltas[agent]["eat"] = gain
                t_gain = time.perf_counter()
                max_energy = self.config["max_energy_predator"]
                self.agent_energies[agent] = min(self.agent_energies[agent], max_energy)
                t_cap = time.perf_counter()
                uid = self.unique_agents[agent]
                self.unique_agent_stats[uid]["times_ate"] += 1
                self.unique_agent_stats[uid]["energy_gained"] += gain
                self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]
                t_stats = time.perf_counter()
                self.grid_world_state[1, *predator_position] = self.agent_energies[agent]
                # Subtract energy from carcass
                leftover = carcass_energy - raw_gain
                if leftover > 0:
                    self.carcass_energies[carcass_id] = leftover
                    self.grid_world_state[self.carcass_channel_idx, *predator_position] = leftover
                else:
                    # Remove carcass
                    del self.carcass_positions[carcass_id]
                    del self.carcass_energies[carcass_id]
                    del self.carcass_ages[carcass_id]
                    self.grid_world_state[self.carcass_channel_idx, *predator_position] = 0
                t_carcass = time.perf_counter()
                if getattr(self, "debug_mode", False):
                    print(f"[PROFILE-ENGAGE] pred: carcass log={1e3*(t_log-t0):.3f}ms just_ate={1e3*(t_just_ate-t_log):.3f}ms reward={1e3*(t_reward-t_just_ate):.3f}ms gain={1e3*(t_gain-t_reward):.3f}ms cap={1e3*(t_cap-t_gain):.3f}ms stats={1e3*(t_stats-t_cap):.3f}ms carcass={1e3*(t_carcass-t_stats):.3f}ms total={1e3*(t_carcass-t0):.3f}ms")
                return {
                    "log": t_log-t0, "just_ate": t_just_ate-t_log, "reward": t_reward-t_just_ate, "gain": t_gain-t_reward,
                    "cap": t_cap-t_gain, "stats": t_stats-t_cap, "carcass": t_carcass-t_stats, "total": t_carcass-t0
                }
            else:
                t0 = time.perf_counter()
                rewards[agent] = self._get_type_specific("reward_predator_step", agent)
                uid = self.unique_agents[agent]
                self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]
                t1 = time.perf_counter()
                if getattr(self, "debug_mode", False):
                    print(f"[PROFILE-ENGAGE] pred: no-catch reward+stats={1e3*(t1-t0):.3f}ms")
                return {"log": 0.0, "just_ate": 0.0, "reward": t1-t0, "gain": 0.0, "cap": 0.0, "stats": 0.0, "grid": 0.0, "prey_reward": 0.0, "prey_stats": 0.0, "del": 0.0, "total": t1-t0}

    def _handle_prey_engagement(self, agent, observations, rewards, terminations, truncations, grass_pos_map=None):
        import time
        if terminations.get(agent):
            return

        prey_position = tuple(self.agent_positions[agent])
        if grass_pos_map is not None:
            caught_grass = grass_pos_map.get(prey_position, None)
        else:
            caught_grass = next(
                (g for g, pos in self.grass_positions.items() if "grass" in g and np.array_equal(prey_position, pos)), None
            )

        if caught_grass:
            t0 = time.perf_counter()
            self._log(self.verbose_engagement, f"[ENGAGE] {agent} caught grass at {tuple(map(int, prey_position))}", "white")
            t_log = time.perf_counter()
            self.agents_just_ate.add(agent)
            t_just_ate = time.perf_counter()
            rewards[agent] = self._get_type_specific("reward_prey_eat_grass", agent)
            self.cumulative_rewards.setdefault(agent, 0)
            self.cumulative_rewards[agent] += rewards[agent]
            t_reward = time.perf_counter()
            raw_gain = min(self.grass_energies[caught_grass], self.config["max_energy_gain_per_grass"])
            efficiency = self.config["energy_transfer_efficiency"]
            gain = raw_gain * efficiency
            self.agent_energies[agent] += gain
            self._per_agent_step_deltas[agent]["eat"] = gain
            t_gain = time.perf_counter()
            max_energy = self.config["max_energy_prey"]
            self.agent_energies[agent] = min(self.agent_energies[agent], max_energy)
            t_cap = time.perf_counter()
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["times_ate"] += 1
            self.unique_agent_stats[uid]["energy_gained"] += gain
            self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]
            t_stats = time.perf_counter()
            self.grid_world_state[2, *prey_position] = self.agent_energies[agent]
            t_grid = time.perf_counter()
            self.grid_world_state[3, *prey_position] = 0
            self.grass_energies[caught_grass] = 0
            t_grass = time.perf_counter()
            if getattr(self, "debug_mode", False):
                print(f"[PROFILE-ENGAGE] prey: log={1e3*(t_log-t0):.3f}ms just_ate={1e3*(t_just_ate-t_log):.3f}ms reward={1e3*(t_reward-t_just_ate):.3f}ms gain={1e3*(t_gain-t_reward):.3f}ms cap={1e3*(t_cap-t_gain):.3f}ms stats={1e3*(t_stats-t_cap):.3f}ms grid={1e3*(t_grid-t_stats):.3f}ms grass={1e3*(t_grass-t_grid):.3f}ms total={1e3*(t_grass-t0):.3f}ms")
            return {
                "log": t_log-t0, "just_ate": t_just_ate-t_log, "reward": t_reward-t_just_ate, "gain": t_gain-t_reward,
                "cap": t_cap-t_gain, "stats": t_stats-t_cap, "grid": t_grid-t_stats, "grass": t_grass-t_grid, "total": t_grass-t0
            }
        else:
            t0 = time.perf_counter()
            rewards[agent] = self._get_type_specific("reward_prey_step", agent)
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]
            t1 = time.perf_counter()
            if getattr(self, "debug_mode", False):
                print(f"[PROFILE-ENGAGE] prey: no-catch reward+stats={1e3*(t1-t0):.3f}ms")
            return {"log": 0.0, "just_ate": 0.0, "reward": t1-t0, "gain": 0.0, "cap": 0.0, "stats": 0.0, "grid": 0.0, "grass": 0.0, "total": t1-t0}

    def _handle_predator_reproduction(self, agent, rewards, observations, terminations, truncations):
        cooldown = self.config["reproduction_cooldown_steps"]
        if self.current_step - self.agent_last_reproduction.get(agent, -cooldown) < cooldown:
            return

        chance_key = "reproduction_chance_predator" if "predator" in agent else "reproduction_chance_prey"
        if self.rng.random() > self.config[chance_key]:
            return

        if self.agent_energies[agent] >= self.predator_creation_energy_threshold:
            parent_type = int(agent.split("_")[1])  # from "type_1_predator_3"

            # Mutation: chance (self.mutation_rate_predator) to switch type
            mutated = self.rng.random() < self.mutation_rate_predator  # or _prey
            if mutated:
                new_type = 2 if parent_type == 1 else 1
            else:
                new_type = parent_type

            # Find available new agent ID
            potential_new_ids = [
                f"type_{new_type}_predator_{i}"
                for i in range(self.config[f"n_possible_type_{new_type}_predators"])
                if f"type_{new_type}_predator_{i}" not in self.agents
            ]
            if not potential_new_ids:
                # Always grant reproduction reward, even if no slot available
                rewards[agent] = self._get_type_specific("reproduction_reward_predator", agent)
                self.cumulative_rewards.setdefault(agent, 0)
                self.cumulative_rewards[agent] += rewards[agent]
                self._log(
                    self.verbose_reproduction,
                    f"[REPRODUCTION] No available predator slots at type {new_type} for spawning",
                    "red",
                )
                return

            new_agent = potential_new_ids[0]
            self.agents.append(new_agent)
            self._per_agent_step_deltas[new_agent] = {
                "decay": 0.0,
                "move": 0.0,
                "eat": 0.0,
                "repro": 0.0,
            }
            # And after successful reproduction, store for cooldown
            self.agent_last_reproduction[agent] = self.current_step

            self._register_new_agent(new_agent, parent_unique_id=self.unique_agents[agent])
            child_uid = self.unique_agents[new_agent]
            self.agent_live_offspring_ids[agent].append(child_uid)
            self.agent_offspring_counts[agent] += 1

            self.unique_agent_stats[self.unique_agents[new_agent]]["mutated"] = mutated
            self.unique_agent_stats[self.unique_agents[agent]]["offspring_count"] += 1

            # Spawn position
            occupied_positions = set(self.agent_positions.values())
            new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)

            self.agent_positions[new_agent] = new_position
            self.predator_positions[new_agent] = new_position

            repro_eff = self.config["reproduction_energy_efficiency"]
            energy_given = self.initial_energy_predator * repro_eff
            self.agent_energies[new_agent] = energy_given
            self.agent_energies[agent] -= self.initial_energy_predator
            self._per_agent_step_deltas[agent]["repro"] = -self.initial_energy_predator

            # Write the child's actual starting energy (after reproduction efficiency) into the grid
            self.grid_world_state[1, *new_position] = energy_given
            self.grid_world_state[1, *self.agent_positions[agent]] = self.agent_energies[agent]

            self.active_num_predators += 1

            # Rewards and tracking
            rewards[new_agent] = 0
            rewards[agent] = self._get_type_specific("reproduction_reward_predator", agent)

            self.cumulative_rewards[new_agent] = 0
            self.cumulative_rewards[agent] += rewards[agent]
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]

            observations[new_agent] = self._get_observation(new_agent)
            terminations[new_agent] = False
            truncations[new_agent] = False
            self._log(
                self.verbose_reproduction,
                f"[REPRODUCTION] Predator {agent} spawned {new_agent} at {tuple(map(int, new_position))}",
                "green",
            )

    def _handle_prey_reproduction(self, agent, rewards, observations, terminations, truncations):
        cooldown = self.config["reproduction_cooldown_steps"]
        if self.current_step - self.agent_last_reproduction.get(agent, -cooldown) < cooldown:
            return

        chance_key = "reproduction_chance_predator" if "predator" in agent else "reproduction_chance_prey"
        if self.rng.random() > self.config[chance_key]:
            return

        if self.agent_energies[agent] >= self.prey_creation_energy_threshold:
            parent_type = int(agent.split("_")[1])  # from "type_1_prey_6"

            # Mutation: 10% chance to switch type
            mutated = self.rng.random() < self.mutation_rate_prey
            if mutated:
                new_type = 2 if parent_type == 1 else 1
            else:
                new_type = parent_type

            # Find available new agent ID
            potential_new_ids = [
                f"type_{new_type}_prey_{i}"
                for i in range(self.config[f"n_possible_type_{new_type}_prey"])
                if f"type_{new_type}_prey_{i}" not in self.agents
            ]
            if not potential_new_ids:
                # Always grant reproduction reward, even if no slot available
                rewards[agent] = self._get_type_specific("reproduction_reward_prey", agent)
                self.cumulative_rewards.setdefault(agent, 0)
                self.cumulative_rewards[agent] += rewards[agent]
                self._log(
                    self.verbose_reproduction, f"[REPRODUCTION] No available prey slots at type {new_type} for spawning", "red"
                )
                return

            new_agent = potential_new_ids[0]
            self.agents.append(new_agent)
            self._per_agent_step_deltas[new_agent] = {
                "decay": 0.0,
                "move": 0.0,
                "eat": 0.0,
                "repro": 0.0,
            }

            # And after successful reproduction, store for cooldown
            self.agent_last_reproduction[agent] = self.current_step

            self._register_new_agent(new_agent, parent_unique_id=self.unique_agents[agent])
            child_uid = self.unique_agents[new_agent]
            self.agent_live_offspring_ids[agent].append(child_uid)

            self.agent_offspring_counts[agent] += 1
            self.unique_agent_stats[self.unique_agents[new_agent]]["mutated"] = mutated
            self.unique_agent_stats[self.unique_agents[agent]]["offspring_count"] += 1

            # Spawn position
            occupied_positions = set(self.agent_positions.values())
            new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)

            self.agent_positions[new_agent] = new_position
            self.prey_positions[new_agent] = new_position

            repro_eff = self.config["reproduction_energy_efficiency"]
            energy_given = self.initial_energy_prey * repro_eff
            self.agent_energies[new_agent] = energy_given
            self.agent_energies[agent] -= self.initial_energy_prey
            self._per_agent_step_deltas[agent]["repro"] = -self.initial_energy_prey

            # Write the child's actual starting energy (after reproduction efficiency) into the grid
            self.grid_world_state[2, *new_position] = energy_given
            self.grid_world_state[2, *self.agent_positions[agent]] = self.agent_energies[agent]

            self.active_num_prey += 1

            # Rewards and tracking
            rewards[new_agent] = 0
            rewards[agent] = self._get_type_specific("reproduction_reward_prey", agent)
            self.cumulative_rewards[new_agent] = 0
            self.cumulative_rewards[agent] += rewards[agent]
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]

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
            "unique_agents": self.unique_agents.copy(),
            "agent_activation_counts": self.agent_activation_counts.copy(),
            "agent_ages": self.agent_ages.copy(),
            "death_cause_prey": self.death_cause_prey.copy(),
            "agent_last_reproduction": self.agent_last_reproduction.copy(),
            "per_step_agent_data": self.per_step_agent_data.copy(),  # ← aligned with rest
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
        self.unique_agents = snapshot["unique_agents"].copy()
        self.agent_activation_counts = snapshot["agent_activation_counts"].copy()
        self.agent_ages = snapshot["agent_ages"].copy()
        self.death_cause_prey = snapshot["death_cause_prey"].copy()
        self.agent_last_reproduction = snapshot["agent_last_reproduction"].copy()
        self.per_step_agent_data = snapshot["per_step_agent_data"].copy()

    def _build_possible_agent_ids(self):
        """
        Build the list of possible agents based on the configuration.
        This is called during reset to ensure the agent list is up-to-date.
        """
        agent_ids = []
        for i in range(self.n_possible_type_1_predators):
            agent_ids.append(f"type_1_predator_{i}")
        for i in range(self.n_possible_type_2_predators):
            agent_ids.append(f"type_2_predator_{i}")
        for i in range(self.n_possible_type_1_prey):
            agent_ids.append(f"type_1_prey_{i}")
        for i in range(self.n_possible_type_2_prey):
            agent_ids.append(f"type_2_prey_{i}")
        return agent_ids  # ✅ ← this was missing

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
        if "type_1" in agent_id:
            action_space = gymnasium.spaces.Discrete(self.type_1_act_range**2)
        elif "type_2" in agent_id:
            action_space = gymnasium.spaces.Discrete(self.type_2_act_range**2)
        else:
            raise ValueError(f"Unknown agent type in ID: {agent_id}")

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
        - 'type_1_predator', 'type_2_predator'
        - 'type_1_prey', 'type_2_prey'
        """
        energy_totals = {
            "predator": 0.0,
            "prey": 0.0,
            "grass": sum(self.grass_energies.values()),
            "type_1_predator": 0.0,
            "type_2_predator": 0.0,
            "type_1_prey": 0.0,
            "type_2_prey": 0.0,
        }

        for agent_id, energy in self.agent_energies.items():
            if "predator" in agent_id:
                energy_totals["predator"] += energy
                if "type_1" in agent_id:
                    energy_totals["type_1_predator"] += energy
                elif "type_2" in agent_id:
                    energy_totals["type_2_predator"] += energy
            elif "prey" in agent_id:
                energy_totals["prey"] += energy
                if "type_1" in agent_id:
                    energy_totals["type_1_prey"] += energy
                elif "type_2" in agent_id:
                    energy_totals["type_2_prey"] += energy

        return energy_totals

    def get_total_offspring_by_type(self):
        """
        Returns a dict of total offspring counts by agent type.
        Example:
        {
            "type_1_prey": 34,
            "type_2_prey": 12,
            ...
        }
        """
        counts = {
            "type_1_predator": 0,
            "type_2_predator": 0,
            "type_1_prey": 0,
            "type_2_prey": 0,
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
            "type_1_prey": 192.3,
            "type_2_prey": 83.1,
            ...
        }
        """
        energy_spent = {
            "type_1_predator": 0.0,
            "type_2_predator": 0.0,
            "type_1_prey": 0.0,
            "type_2_prey": 0.0,
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
