"""
Predator-Prey Grass RLlib Environment

Additional features:
-kinship rewards for parents when offspring survive time steps
"""
# external libraries (Ray required)
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from collections import deque
from typing import Optional


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            raise ValueError("Environment config must be provided explicitly.")
        self.config = config
        self._initialize_from_config()

        # RLlib necessities in constructor
        self.possible_agents = self._build_possible_agent_ids()

        self.observation_spaces = {agent_id: self._build_observation_space(agent_id) for agent_id in self.possible_agents}

        self.action_spaces = {agent_id: self._build_action_space(agent_id) for agent_id in self.possible_agents}


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
        self.max_energy_grass = self.config["max_energy_grass"]


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
        self.include_visibility_channel = config["include_visibility_channel"]
        # Movement restriction: if True, agents may only move to target cells with unobstructed LOS (no wall between current and target).
        self.respect_los_for_movement = config["respect_los_for_movement"] # TODO is this even a problem in a Moore neighborhood?
        self.mask_observation_with_visibility = config["mask_observation_with_visibility"]

        # Grass settings
        self.initial_num_grass = config["initial_num_grass"]
        self.initial_energy_grass = config["initial_energy_grass"]
        self.energy_gain_per_step_grass = config["energy_gain_per_step_grass"]
        # Walls (static obstacles)
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
        self.agent_stats_live = {}
        self.agent_stats_completed = {}
        self.per_step_agent_data = []  # One entry per step; each is {agent_id: {position, energy, ...}}
        self._per_agent_step_deltas = {}  # Internal temp storage to track energy deltas during step
        self._next_lifetime_id = 0
        self.agent_parents = {}
        self.agent_offspring_counts = {}
        self.agent_live_offspring_ids = {}
        # Track all agent IDs that have ever been active in this episode to prevent reuse
        self.used_agent_ids = set()
        # Capacity block counters (episode-level)
        self.reproduction_blocked_due_to_capacity_predator = 0
        self.reproduction_blocked_due_to_capacity_prey = 0
        # Episode-level spawn counters
        self.spawned_predators = 0
        self.spawned_prey = 0


        self.agents_just_ate = set()

        # Per-step infos accumulator and last-move diagnostics
        self._pending_infos = {}
        self._last_move_block_reason = {}
        # Global counters (optional diagnostics)
        self.los_rejected_moves_total = 0
        self.los_rejected_moves_by_type = {"predator": 0, "prey": 0}

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
                    # _register_new_agent already adds to used_agent_ids

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

        self.action_to_move_tuple_type_1_agents = _generate_action_map(self.type_1_act_range)
        self.action_to_move_tuple_type_2_agents = _generate_action_map(self.type_2_act_range)

        # Initialize per-type available ID pools (never reuse within an episode)
        self._init_available_id_pools()
        # Print-once guard for termination debug logs (per episode)
        self._printed_termination_ids = set()
        # Precompute LOS masks for each obs range (assuming static walls for now)
        # This must be done after config and grid/wall initialization
        self.los_mask_predator = self._precompute_los_mask(self.predator_obs_range)
        self.los_mask_prey = self._precompute_los_mask(self.prey_obs_range)

    def _init_available_id_pools(self):
        """Build deques of never-used IDs per species/type for O(1) allocation.

        Pools are initialized with all possible IDs from config, then filtered to exclude
        any IDs that are already used in this episode (initial actives). IDs are never
        returned to the pool until reset.
        """
        pools = {
            "type_1_predator": deque(),
            "type_2_predator": deque(),
            "type_1_prey": deque(),
            "type_2_prey": deque(),
        }

        # Populate in deterministic order
        for i in range(self.n_possible_type_1_predators):
            aid = f"type_1_predator_{i}"
            if aid not in self.used_agent_ids:
                pools["type_1_predator"].append(aid)
        for i in range(self.n_possible_type_2_predators):
            aid = f"type_2_predator_{i}"
            if aid not in self.used_agent_ids:
                pools["type_2_predator"].append(aid)
        for i in range(self.n_possible_type_1_prey):
            aid = f"type_1_prey_{i}"
            if aid not in self.used_agent_ids:
                pools["type_1_prey"].append(aid)
        for i in range(self.n_possible_type_2_prey):
            aid = f"type_2_prey_{i}"
            if aid not in self.used_agent_ids:
                pools["type_2_prey"].append(aid)

        self._available_id_pools = pools

    def _alloc_new_id(self, species: str, type_nr: int):
        """Allocate a fresh agent ID from the per-type pool or return None if exhausted.

        Ensures the returned ID has not been used earlier in this episode and is not currently active.
        """
        key = f"type_{type_nr}_{species}"
        dq = self._available_id_pools.get(key)
        if dq is None:
            return None
        while dq:
            cand = dq.popleft()
            if cand not in self.used_agent_ids and cand not in self.agents:
                return cand
        return None

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

    def step(self, action_dict):
        self.observations, self.rewards, self.terminations, self.truncations, self.infos = {}, {}, {}, {}, {}
        self.agents_just_ate.clear()  # For stepwise display eating in grid
        self._pending_infos = {}  # Reset per-step self.infos

        # Step 1: Process energy depletion due to time steps and update age 
        self._apply_time_step_update()

        # Step 2: Regenerate grass energy
        self._regenerate_grass_energy()

        # Step 3: process agent movements
        self._process_agent_movements(action_dict)

        # Step 4: Handle agent engagements (optimized scans with snapshots)
        # 4a) Starvation first: snapshot of energies to avoid repeated dict lookups
        energies = self.agent_energies
        handle_starv = self._handle_energy_starvation
        for agent, energy in tuple(energies.items()):
            if energy <= 0:
                handle_starv(agent)

        # 4b) Prey engagements over active prey only (skip terminated)
        prey_snapshot = tuple(self.prey_positions.keys())
        for agent in prey_snapshot:
            if self.terminations.get(agent):
                continue
            self._handle_prey_engagement(agent)

        # 4c) Predator engagements over active predators only (skip terminated)
        predator_snapshot = tuple(self.predator_positions.keys())
        for agent in predator_snapshot:
            if self.terminations.get(agent):
                continue
            self._handle_predator_engagement(agent)

        # Step 5: Handle agent removals 
        # Collect all agents marked terminated and still present in the active maps
        terminations = self.terminations
        agent_positions = self.agent_positions
        to_remove = [a for a, t in terminations.items() if t and a in agent_positions]

        if to_remove:
            to_remove_set = set(to_remove)
            energies = self.agent_energies
            predator_pos = self.predator_positions
            prey_pos = self.prey_positions

            for agent in to_remove:
                agent_positions.pop(agent, None)
                energies.pop(agent, None)
                predator_pos.pop(agent, None)
                prey_pos.pop(agent, None)

            # Rebuild active agent list without repeated O(n) list.remove calls
            self.agents = [a for a in self.agents if a not in to_remove_set]

        # Step 7: Spawning of new agents (cooldown and chance removed; energy-only)
        # Use snapshots of active predators/prey to avoid iterating over newly spawned agents in this step
        predator_snapshot = tuple(self.predator_positions.keys())
        prey_snapshot = tuple(self.prey_positions.keys())
        energies = self.agent_energies
        pred_thr = self.predator_creation_energy_threshold
        prey_thr = self.prey_creation_energy_threshold

        for agent in predator_snapshot:
            if energies[agent] >= pred_thr:
                self._handle_predator_reproduction(agent)

        for agent in prey_snapshot:
            if energies[agent] >= prey_thr:
                self._handle_prey_reproduction(agent)

        # Step 8: Assemble return dicts.
        # Generate observations for all still-active agents AFTER engagements and reproduction.
        get_obs = self._get_observation
        active_obs = {agent: get_obs(agent) for agent in self.agents}
        self.observations.update(active_obs)  # Preserve any earlier terminal snapshots

        # Union of all agent IDs referenced this step.
        all_ids = set(self.observations) | set(self.terminations) | set(self.rewards) | set(self.truncations)

        # Guarantee observation presence for terminated agents that didn't have a snapshot captured earlier.
        for aid in all_ids:
            if aid not in self.observations:
                if "predator" in aid:
                    obs_range = self.predator_obs_range
                elif "prey" in aid:
                    obs_range = self.prey_obs_range
                else:
                    continue  # Skip non-learning entities if any
                channels = self.num_obs_channels + (1 if self.include_visibility_channel else 0)
                self.observations[aid] = np.zeros((channels, obs_range, obs_range), dtype=np.float32)

        # Fill defaults for missing reward/termination/truncation keys (without erasing True terminations).
        self.rewards = {aid: self.rewards.get(aid, 0.0) for aid in all_ids}
        self.terminations = {aid: self.terminations.get(aid, False) for aid in all_ids}
        self.truncations = {aid: self.truncations.get(aid, False) for aid in all_ids}
        self.terminations["__all__"] = self.active_num_prey <= 0 or self.active_num_predators <= 0
        self.truncations["__all__"] = False
        self.infos = {aid: self._pending_infos.get(aid, {}) for aid in all_ids}
        step_data = {}
        alive_agents = set(self.agents)
        for agent in self.agents:
            kin_reward = self._get_type_specific("kin_kick_back_predator", agent) \
                if "predator" in agent else self._get_type_specific("kin_kick_back_prey", agent)
            pos = self.agent_positions[agent]
            energy = self.agent_energies[agent]
            deltas = self._per_agent_step_deltas.get(agent, {"decay": 0.0, "move": 0.0, "eat": 0.0, "repro": 0.0})
            parent = self.agent_parents.get(agent)
            # Kin survival reward: if parent is alive, reward the parent
            if parent is not None and parent in alive_agents:
                self.rewards[parent] = self.rewards.get(parent, 0.0) + kin_reward
                parent_record = self.agent_stats_live.get(parent)
                if parent_record is not None:
                    parent_record["cumulative_reward"] += kin_reward
                    parent_record["kin_kickbacks"] = parent_record.get("kin_kickbacks", 0) + 1
            step_data[agent] = {
                "position": pos,
                "energy": energy,
                "energy_decay": deltas["decay"],
                "energy_movement": deltas["move"],
                "energy_eating": deltas["eat"],
                "energy_reproduction": deltas["repro"],
                "age": self.agent_ages[agent],
                "offspring_count": self.agent_offspring_counts[agent],
                "offspring_ids": self.agent_live_offspring_ids.get(agent, []),
                "parent": parent,
            }

        self.per_step_agent_data.append(step_data)
        self._per_agent_step_deltas.clear()

        # Increment step counter
        self.current_step += 1

        if self.current_step >= self.max_steps:
            # Final time-limit step: include active agents as truncated=True, and
            # any agents that terminated this step as termination=True with final obs.
            obs, rews, terms, truncs, infos = {}, {}, {}, {}, {}

            terminated_this_step = {aid for aid, t in self.terminations.items() if t}
            active_now = set(self.agents)

            # Active agents -> truncated=True, terminated=False
            for agent in active_now:
                obs[agent] = self._get_observation(agent)
                rews[agent] = self.rewards.get(agent, 0.0)
                truncs[agent] = True
                terms[agent] = False
                infos[agent] = self._pending_infos.get(agent, {})

            # Agents that died this step -> termination=True, truncated=False
            for agent in terminated_this_step:
                # Preserve any earlier captured final obs; else generate a zero fallback
                if agent in self.observations:
                    obs[agent] = self.observations[agent]
                else:
                    if "predator" in agent:
                        rng = self.predator_obs_range
                    elif "prey" in agent:
                        rng = self.prey_obs_range
                    else:
                        continue
                    channels = self.num_obs_channels + (1 if self.include_visibility_channel else 0)
                    obs[agent] = np.zeros((channels, rng, rng), dtype=np.float32)

                rews[agent] = self.rewards.get(agent, 0.0)
                terms[agent] = True
                truncs[agent] = False
                infos[agent] = self._pending_infos.get(agent, {})

            truncs["__all__"] = True
            terms["__all__"] = False

            # Overwrite step-assembled dicts to only contain final-step relevant agents.
            self.observations, self.rewards = obs, rews
            self.terminations, self.truncations, self.infos = terms, truncs, infos

            for agent_id in list(self.agent_stats_live.keys()):
                self._finalize_agent_record(agent_id, cause="time_limit")

            return self.observations, self.rewards, self.terminations, self.truncations, self.infos

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def _apply_time_step_update(self):
        """
        Apply all per-step updates (energy decay, age increment).
        """
        for agent in self.agents:
            layer = 1 if "predator" in agent else 2
            if layer == 1:
                energy_decay = self.energy_loss_per_step_predator
                self.agent_energies[agent] -= energy_decay
                self.grid_world_state[layer, *self.agent_positions[agent]] = self.agent_energies[agent]
            elif layer == 2:
                energy_decay = self.energy_loss_per_step_prey
                self.agent_energies[agent] -= energy_decay
                self.grid_world_state[layer, *self.agent_positions[agent]] = self.agent_energies[agent]

            self.agent_ages[agent] += 1
            self._per_agent_step_deltas[agent] = {
                "decay": -energy_decay,
                "move": 0.0,
                "eat": 0.0,
                "repro": 0.0,
            }

    def _regenerate_grass_energy(self):
        """
        Increase energy of all grass patches, capped at initial energy value.
        """
        for grass, pos in self.grass_positions.items():
            old_energy = self.grass_energies[grass]
            new_energy = min(old_energy + self.energy_gain_per_step_grass, self.max_energy_grass)
            self.grass_energies[grass] = new_energy
            self.grid_world_state[3, pos[0], pos[1]] = new_energy

    def _process_agent_movements(self, action_dict):
        """
        Process movement and grid updates for all agents (non-vectorized, simple loop).
        """
        for agent in action_dict.keys():
            old_position = self.agent_positions[agent]
            action = action_dict[agent]
            new_position = self._get_move(agent, action)
            if "predator" in agent:
                self.predator_positions[agent] = tuple(new_position)
                self.grid_world_state[1, old_position[0], old_position[1]] = 0
                self.grid_world_state[1, new_position[0], new_position[1]] = self.agent_energies[agent]
            elif "prey" in agent:
                self.prey_positions[agent] = tuple(new_position)
                self.grid_world_state[2, old_position[0], old_position[1]] = 0
                self.grid_world_state[2, new_position[0], new_position[1]] = self.agent_energies[agent]
            record = self.agent_stats_live.get(agent)
            if record is not None:
                record["avg_energy_sum"] += self.agent_energies[agent]
                record["avg_energy_steps"] += 1
                record["distance_traveled"] += float(np.linalg.norm(np.array(new_position) - np.array(old_position)))
            self.agent_positions[agent] = tuple(new_position)

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
        t0 = time.perf_counter()
        # Generate an observation for the agent.
        obs_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp, obs_range)
        channels = self.num_obs_channels + (1 if self.include_visibility_channel else 0)
        # Allocate observation tensor
        obs = np.zeros((channels, obs_range, obs_range), dtype=np.float32)
        gws = self.grid_world_state  # local reference
        # Channel 0: walls
        obs[0, xolo:xohi, yolo:yohi] = gws[0, xlo:xhi, ylo:yhi]
        # Dynamic channels (predators, prey, grass)
        obs[1:self.num_obs_channels, xolo:xohi, yolo:yohi] = gws[1:, xlo:xhi, ylo:yhi]

        if self.include_visibility_channel or self.mask_observation_with_visibility:
            # Use precomputed LOS mask without copying (treated read-only)
            visibility_mask = self.los_mask_predator if "predator" in agent else self.los_mask_prey
            if self.mask_observation_with_visibility:
                # Apply mask to dynamic channels (exclude channel 0 walls)
                for c in range(1, self.num_obs_channels):
                    obs[c] *= visibility_mask
            if self.include_visibility_channel:
                obs[channels - 1] = visibility_mask

        dt = time.perf_counter() - t0
        if dt > 0.002 and getattr(self, "debug_mode", False):  # log only if >2ms
            print(f"[PROFILE-OBS-AGENT] agent={agent} obs_time={dt:.6f}s")
        return obs

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
        wall_positions = self.wall_positions
        valid_positions = [pos for pos in potential_positions if pos not in occupied_positions and pos not in wall_positions]

        if valid_positions:
            return valid_positions[0]  # Prefer adjacent position if available

        # Fallback: Find any random unoccupied position
        all_positions = {
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) not in wall_positions
        }
        free_positions = list(all_positions - occupied_positions)

        if free_positions:
            return free_positions[self.rng.integers(len(free_positions))]

        return None  # No available position found

    def _handle_energy_starvation(self, agent):
        self.observations[agent] = self._get_observation(agent)
        self.rewards[agent] = 0
        self.terminations[agent] = True
        self.truncations[agent] = False

        layer = 1 if "predator" in agent else 2
        self.grid_world_state[layer, *self.agent_positions[agent]] = 0
        self._finalize_agent_record(agent, cause="starved")

        if "predator" in agent:
            self.active_num_predators -= 1
        else:
            self.active_num_prey -= 1
        #del self.agent_ages[agent]

    def _handle_predator_engagement(self, agent):
        predator_position = tuple(self.agent_positions[agent])
        caught_prey = next(
            (prey for prey, pos in self.agent_positions.items() if "prey" in prey and np.array_equal(predator_position, pos)), None
        )
        if caught_prey:
            # attribution predator
            self.agents_just_ate.add(agent)
            self.rewards[agent] = self._get_type_specific("reward_predator_catch_prey", agent)
            # cumulative_reward is tracked directly in agent_stats_live
            energy_gain = min(self.agent_energies[caught_prey], self.config["max_energy_gain_per_prey"])
            self.agent_energies[agent] +=  energy_gain
            self.grid_world_state[1, *predator_position] = energy_gain
            self._per_agent_step_deltas[agent]["eat"] =  energy_gain
            predator_record = self.agent_stats_live.get(agent)
            if predator_record is not None:
                predator_record["times_ate"] += 1
                predator_record["energy_gained"] += energy_gain
                predator_record["cumulative_reward"] += self.rewards[agent]
            # attribution prey
            # Capture a final observation for the caught prey at the moment of termination
            # so RLlib registers the terminal step properly.
            self.observations[caught_prey] = self._get_observation(caught_prey)
            self.terminations[caught_prey] = True
            penalty = self._get_type_specific("penalty_prey_caught", caught_prey)
            self.rewards[caught_prey] = penalty
            self.truncations[caught_prey] = False
            self.active_num_prey -= 1
            self.grid_world_state[2, *self.agent_positions[caught_prey]] = 0
            prey_record = self.agent_stats_live.get(caught_prey)
            if prey_record is not None:
                prey_record["death_cause"] = "eaten"
                # Add penalty to existing cumulative_reward (do not overwrite)
                prey_record["cumulative_reward"] += penalty
            self._finalize_agent_record(caught_prey, cause="eaten")
        else:
            self.rewards[agent] = self._get_type_specific("reward_predator_step", agent)
            predator_record = self.agent_stats_live.get(agent)
            if predator_record is not None:
                predator_record["cumulative_reward"] += self.rewards[agent]

    def _handle_prey_engagement(self, agent):
        if self.terminations.get(agent):
            return
        prey_position = tuple(self.agent_positions[agent])
        caught_grass = next(
            (g for g, pos in self.grass_positions.items() if "grass" in g and np.array_equal(prey_position, pos)), None
        )
        if caught_grass:
            # attribution prey
            self.agents_just_ate.add(agent)
            self.rewards[agent] = self._get_type_specific("reward_prey_eat_grass", agent)
            # cumulative_reward is tracked directly in agent_stats_live
            energy_gain = min(self.grass_energies[caught_grass], self.config["max_energy_gain_per_grass"])
            self.agent_energies[agent] += energy_gain
            self._per_agent_step_deltas[agent]["eat"] = energy_gain
            self.grid_world_state[2, *prey_position] = self.agent_energies[agent]
            self.grid_world_state[3, *prey_position] = 0
            prey_record = self.agent_stats_live.get(agent)
            if prey_record is not None:
                prey_record["times_ate"] += 1
                prey_record["energy_gained"] += energy_gain
                prey_record["cumulative_reward"] += self.rewards[agent]
            self.grass_energies[caught_grass] = 0
        else:
            self.rewards[agent] = self._get_type_specific("reward_prey_step", agent)
            prey_record = self.agent_stats_live.get(agent)
            if prey_record is not None:
                prey_record["cumulative_reward"] += self.rewards[agent]

    def _handle_predator_reproduction(self, agent):
        # Cooldown removed: reproduction now only gated by energy + random chance handled before call.
        # Chance removed as well: reproduction attempts occur whenever energy threshold is met.
        

        if self.agent_energies[agent] >= self.predator_creation_energy_threshold:
            parent_type = int(agent.split("_")[1])  # from "type_1_predator_3"

            # Mutation: chance (self.mutation_rate_predator) to switch type
            mutated = self.rng.random() < self.mutation_rate_predator  # or _prey
            if mutated:
                new_type = 2 if parent_type == 1 else 1
            else:
                new_type = parent_type

            # Find available new agent ID using pool allocator
            new_agent = self._alloc_new_id("predator", new_type)
            if not new_agent:
                # Capacity exhausted: record metric + info flag and still award reproduction reward.
                self.reproduction_blocked_due_to_capacity_predator += 1
                self.rewards[agent] = self._get_type_specific("reproduction_reward_predator", agent)
                # cumulative_reward is tracked directly in agent_stats_live
                self._pending_infos.setdefault(agent, {})["reproduction_blocked_due_to_capacity"] = True
                self._pending_infos[agent]["reproduction_blocked_due_to_capacity_count_predator"] = self.reproduction_blocked_due_to_capacity_predator
                # Print immediately to console as requested (not gated by verbose flags)
                print(
                    f"[CAPACITY] Predator reproduction blocked at step {self.current_step}: "
                    f"type={new_type}, agent={agent}, total_blocked={self.reproduction_blocked_due_to_capacity_predator}"
                )
                return

            self.agents.append(new_agent)
            self._per_agent_step_deltas[new_agent] = {
                "decay": 0.0,
                "move": 0.0,
                "eat": 0.0,
                "repro": 0.0,
            }
            # And after successful reproduction, store for cooldown
            self.agent_last_reproduction[agent] = self.current_step

            self._register_new_agent(new_agent, parent_agent_id=agent, mutated=mutated)
            self.agent_live_offspring_ids[agent].append(new_agent)
            self.agent_offspring_counts[agent] += 1
            parent_record = self.agent_stats_live.get(agent)
            if parent_record is not None:
                parent_record["offspring_count"] += 1

            # Count successful predator spawns
            self.spawned_predators += 1

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

            # self.rewards and tracking
            self.rewards[new_agent] = 0
            self.rewards[agent] = self._get_type_specific("reproduction_reward_predator", agent)

            # cumulative_reward is tracked directly in agent_stats_live
            if parent_record is not None:
                parent_record["cumulative_reward"] += self.rewards[agent]

            self.observations[new_agent] = self._get_observation(new_agent)
            self.terminations[new_agent] = False
            self.truncations[new_agent] = False

    def _handle_prey_reproduction(self, agent):
        # Cooldown removed: reproduction now only gated by energy + random chance handled before call.
        # Chance removed as well: reproduction attempts occur whenever energy threshold is met.
        

        if self.agent_energies[agent] >= self.prey_creation_energy_threshold:
            parent_type = int(agent.split("_")[1])  # from "type_1_prey_6"

            # Mutation: 10% chance to switch type
            mutated = self.rng.random() < self.mutation_rate_prey
            if mutated:
                new_type = 2 if parent_type == 1 else 1
            else:
                new_type = parent_type

            # Find available new agent ID using pool allocator
            new_agent = self._alloc_new_id("prey", new_type)
            if not new_agent:
                # Capacity exhausted: record metric + info flag and still award reproduction reward.
                self.reproduction_blocked_due_to_capacity_prey += 1
                self.rewards[agent] = self._get_type_specific("reproduction_reward_prey", agent)
                # cumulative_reward is tracked directly in agent_stats_live
                self._pending_infos.setdefault(agent, {})["reproduction_blocked_due_to_capacity"] = True
                self._pending_infos[agent]["reproduction_blocked_due_to_capacity_count_prey"] = self.reproduction_blocked_due_to_capacity_prey
                # Print immediately to console as requested (not gated by verbose flags)
                print(
                    f"[CAPACITY] Prey reproduction blocked at step {self.current_step}: "
                    f"type={new_type}, agent={agent}, total_blocked={self.reproduction_blocked_due_to_capacity_prey}"
                )
                return

            self.agents.append(new_agent)
            self._per_agent_step_deltas[new_agent] = {
                "decay": 0.0,
                "move": 0.0,
                "eat": 0.0,
                "repro": 0.0,
            }

            # And after successful reproduction, store for cooldown
            self.agent_last_reproduction[agent] = self.current_step

            self._register_new_agent(new_agent, parent_agent_id=agent, mutated=mutated)
            self.agent_live_offspring_ids[agent].append(new_agent)

            self.agent_offspring_counts[agent] += 1
            parent_record = self.agent_stats_live.get(agent)
            if parent_record is not None:
                parent_record["offspring_count"] += 1
            # Debug: print cumulative_reward before reproduction reward
            if parent_record is not None:
                print(f"[DEBUG] Before reproduction: {agent} cumulative_reward={parent_record['cumulative_reward']}")
            # Count successful prey spawns
            self.spawned_prey += 1

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

            # self.rewards and tracking
            self.rewards[new_agent] = 0
            self.rewards[agent] = self._get_type_specific("reproduction_reward_prey", agent)
            # cumulative_reward is tracked directly in agent_stats_live
            if parent_record is not None:
                parent_record["cumulative_reward"] += self.rewards[agent]
                print(f"[DEBUG] After reproduction: {agent} cumulative_reward={parent_record['cumulative_reward']} (added {self.rewards[agent]})")

            self.observations[new_agent] = self._get_observation(new_agent)
            self.terminations[new_agent] = False
            self.truncations[new_agent] = False

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
            # cumulative_rewards is now stored directly in agent_stats_live/agent_stats_completed
            "active_num_predators": self.active_num_predators,
            "active_num_prey": self.active_num_prey,
            "agents_just_ate": self.agents_just_ate.copy(),
            "agent_stats_live": {aid: self._copy_agent_record(stats) for aid, stats in self.agent_stats_live.items()},
            "agent_stats_completed": {
                aid: self._copy_agent_record(stats) for aid, stats in self.agent_stats_completed.items()
            },
            "agent_ages": self.agent_ages.copy(),
            "death_cause_prey": self.death_cause_prey.copy(),
            "agent_last_reproduction": self.agent_last_reproduction.copy(),
            "agent_parents": self.agent_parents.copy(),
            "agent_offspring_counts": self.agent_offspring_counts.copy(),
            "agent_live_offspring_ids": {
                aid: list(ids) for aid, ids in self.agent_live_offspring_ids.items()
            },
            "used_agent_ids": list(self.used_agent_ids),
            "per_step_agent_data": self.per_step_agent_data.copy(),  # ← aligned with rest
        }

    def restore_state_snapshot(self, snapshot):
    # cumulative_rewards is now stored directly in agent_stats_live/agent_stats_completed
        self.current_step = snapshot["current_step"]
        self.agent_positions = snapshot["agent_positions"].copy()
        self.agent_energies = snapshot["agent_energies"].copy()
        self.predator_positions = snapshot["predator_positions"].copy()
        self.prey_positions = snapshot["prey_positions"].copy()
        self.grass_positions = snapshot["grass_positions"].copy()
        self.grass_energies = snapshot["grass_energies"].copy()
        self.grid_world_state = snapshot["grid_world_state"].copy()
        self.agents = snapshot["agents"].copy()
        self.active_num_predators = snapshot["active_num_predators"]
        self.active_num_prey = snapshot["active_num_prey"]
        self.agents_just_ate = snapshot["agents_just_ate"].copy()
        self.agent_stats_live = {aid: self._copy_agent_record(stats) for aid, stats in snapshot["agent_stats_live"].items()}
        self.agent_stats_completed = {
            aid: self._copy_agent_record(stats) for aid, stats in snapshot["agent_stats_completed"].items()
        }
        # Rebuild offspring lists in auxiliary index and ensure shared references
        self.agent_live_offspring_ids = {}
        for agent_id, record in self.agent_stats_live.items():
            offspring_list = list(record.get("offspring_ids", []))
            record["offspring_ids"] = offspring_list
            self.agent_live_offspring_ids[agent_id] = offspring_list
        self.agent_ages = snapshot["agent_ages"].copy()
        self.death_cause_prey = snapshot["death_cause_prey"].copy()
        self.agent_last_reproduction = snapshot["agent_last_reproduction"].copy()
        self.agent_parents = snapshot.get("agent_parents", {}).copy()
        self.agent_offspring_counts = snapshot.get("agent_offspring_counts", {}).copy()
        stored_live_offspring = snapshot.get("agent_live_offspring_ids", {})
        if stored_live_offspring:
            for agent_id, offspring_ids in stored_live_offspring.items():
                copied = list(offspring_ids)
                if agent_id in self.agent_stats_live:
                    self.agent_stats_live[agent_id]["offspring_ids"] = copied
                    self.agent_live_offspring_ids[agent_id] = copied
        self.used_agent_ids = set(snapshot.get("used_agent_ids", []))
        self.per_step_agent_data = snapshot["per_step_agent_data"].copy()
        # No longer need to restore cumulative_rewards separately

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
        if "type_1" in agent_id:
            action_space = gymnasium.spaces.Discrete(self.type_1_act_range**2)
        elif "type_2" in agent_id:
            action_space = gymnasium.spaces.Discrete(self.type_2_act_range**2)
        else:
            raise ValueError(f"Unknown agent type in ID: {agent_id}")

        return action_space

    def _register_new_agent(self, agent_id: str, parent_agent_id: Optional[str] = None, *, mutated: bool = False):
        if agent_id in self.agent_stats_live or agent_id in self.agent_stats_completed:
            raise ValueError(f"Agent id {agent_id} already registered in this episode.")

        self.used_agent_ids.add(agent_id)
        self.agent_ages[agent_id] = 0
        self.agent_offspring_counts[agent_id] = 0
        self.agent_live_offspring_ids[agent_id] = []
        self.agent_parents[agent_id] = parent_agent_id
        self.agent_last_reproduction[agent_id] = -self.config["reproduction_cooldown_steps"]
        self.agent_stats_live[agent_id] = {
            "agent_id": agent_id,
            "birth_step": self.current_step,
            "parent": parent_agent_id,
            "offspring_count": 0,
            "offspring_ids": self.agent_live_offspring_ids[agent_id],
            "distance_traveled": 0.0,
            "times_ate": 0,
            "energy_gained": 0.0,
            "avg_energy_sum": 0.0,
            "avg_energy_steps": 0,
            "cumulative_reward": 0.0,
            "policy_group": "_".join(agent_id.split("_")[:3]),
            "mutated": mutated,
            "death_step": None,
            "death_cause": None,
            "avg_energy": 0.0,
            "kin_kickbacks": 0,  # Number of kin reward kickbacks received
        }

        return self.agent_stats_live[agent_id]

    def _copy_agent_record(self, record: dict) -> dict:
        copied = record.copy()
        copied["offspring_ids"] = list(record.get("offspring_ids", []))
        return copied

    def _finalize_agent_record(self, agent_id: str, cause: Optional[str] = None):
        # Debug: print cumulative_reward and full record for prey with offspring > 0, before and after finalization
        record = self.agent_stats_live.get(agent_id)
        if record is not None and "prey" in agent_id and record.get("offspring_count", 0) > 0:
            print(f"[DEBUG] (Before finalize) {agent_id} record: {record}")
        record = self.agent_stats_live.pop(agent_id, None)
        if record is None:
            return
        # Only add the final reward if not forcibly set (e.g., by engagement logic)
        if not record.pop("_cumulative_reward_forced", False):
            reward = self.rewards.get(agent_id, 0.0)
            record["cumulative_reward"] += reward
        if cause is not None:
            record["death_cause"] = cause
        if record.get("death_step") is None:
            record["death_step"] = self.current_step
        record["offspring_count"] = self.agent_offspring_counts.get(agent_id, record.get("offspring_count", 0))
        steps = max(record.get("avg_energy_steps", 0), 1)
        record["avg_energy"] = record.get("avg_energy_sum", 0.0) / steps
        if "prey" in agent_id and record.get("offspring_count", 0) > 0:
            print(f"[DEBUG] (After finalize) {agent_id} record: {record}")
        # Copy kin_kickbacks to completed record
        record["kin_kickbacks"] = record.get("kin_kickbacks", 0)
        self.agent_stats_completed[agent_id] = record
        self.agent_live_offspring_ids.pop(agent_id, None)

    def _iter_all_agent_records(self):
        for agent_id, record in self.agent_stats_live.items():
            yield agent_id, record
        for agent_id, record in self.agent_stats_completed.items():
            yield agent_id, record

    def get_all_agent_stats(self) -> dict[str, dict]:
        """Return copies of all agent records keyed by agent_id."""
        stats_by_id = {}
        for agent_id, record in self._iter_all_agent_records():
            copied = self._copy_agent_record(record)
            stats_by_id[agent_id] = copied
        return stats_by_id

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
        for _, stats in self._iter_all_agent_records():
            group = stats.get("policy_group")
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
        for _, stats in self._iter_all_agent_records():
            group = stats.get("policy_group")
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

    #-------- Placement method for prey --------
    def _place_prey(self, prey_list, prey_positions):
        self.prey_positions = {}
        for i, agent in enumerate(prey_list):
            pos = prey_positions[i]
            self.agent_positions[agent] = pos
            self.prey_positions[agent] = pos
            self.agent_energies[agent] = self.initial_energy_prey
            self.grid_world_state[2, *pos] = self.initial_energy_prey

    #-------- Placement method for grass --------
    def _place_grass(self, grass_positions):
        self.grass_positions = {}
        self.grass_energies = {}
        for i, grass in enumerate(self.grass_agents):
            pos = grass_positions[i]
            self.grass_positions[grass] = pos
            self.grass_energies[grass] = self.initial_energy_grass
            self.grid_world_state[3, *pos] = self.initial_energy_grass


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

