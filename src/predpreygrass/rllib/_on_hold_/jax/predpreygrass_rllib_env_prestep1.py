"""Pre-Step1 baseline environment (original logic before micro-optimizations).
Copied from original version (no position indices, always per-step agent sort, linear scans in engagements).
"""
import gymnasium
try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    from ray.rllib.utils.typing import AgentID, Tuple
except ImportError:  # fallback so benchmarking works without ray installed
    from typing import Hashable as AgentID  # type: ignore
    from typing import Tuple as _TTuple
    class MultiAgentEnv:  # type: ignore
        pass
    Tuple = _TTuple
import numpy as np
import math


class PredPreyGrassPreStep1(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            raise ValueError("Environment config must be provided explicitly.")
        self.config = config
        self._initialize_from_config()
        self.possible_agents = self._build_possible_agent_ids()
        self.observation_spaces = {a: self._build_observation_space(a) for a in self.possible_agents}
        self.action_spaces = {a: self._build_action_space(a) for a in self.possible_agents}

    def _initialize_from_config(self):
        c = self.config
        self.debug_mode = c.get("debug_mode", False)
        self.verbose_movement = c.get("verbose_movement", self.debug_mode)
        self.verbose_decay = c.get("verbose_decay", self.debug_mode)
        self.verbose_reproduction = c.get("verbose_reproduction", self.debug_mode)
        self.verbose_engagement = c.get("verbose_engagement", self.debug_mode)
        self.max_steps = c.get("max_steps", 10000)
        self.rng = np.random.default_rng(c.get("seed", 42))
        self.reward_predator_catch_prey_config = c.get("reward_predator_catch_prey", 0.0)
        self.reward_prey_eat_grass_config = c.get("reward_prey_eat_grass", 0.0)
        self.reward_predator_step_config = c.get("reward_predator_step", 0.0)
        self.reward_prey_step_config = c.get("reward_prey_step", 0.0)
        self.penalty_prey_caught_config = c.get("penalty_prey_caught", 0.0)
        self.reproduction_reward_predator_config = c.get("reproduction_reward_predator", 10.0)
        self.reproduction_reward_prey_config = c.get("reproduction_reward_prey", 10.0)
        self.energy_loss_per_step_predator = c.get("energy_loss_per_step_predator", 0.15)
        self.energy_loss_per_step_prey = c.get("energy_loss_per_step_prey", 0.05)
        self.predator_creation_energy_threshold = c.get("predator_creation_energy_threshold", 12.0)
        self.prey_creation_energy_threshold = c.get("prey_creation_energy_threshold", 8.0)
        self.n_possible_type_1_predators = c.get("n_possible_type_1_predators", 25)
        self.n_possible_type_2_predators = c.get("n_possible_type_2_predators", 25)
        self.n_possible_type_1_prey = c.get("n_possible_type_1_prey", 25)
        self.n_possible_type_2_prey = c.get("n_possible_type_2_prey", 25)
        self.n_initial_active_type_1_predator = c.get("n_initial_active_type_1_predator", 6)
        self.n_initial_active_type_2_predator = c.get("n_initial_active_type_2_predator", 0)
        self.n_initial_active_type_1_prey = c.get("n_initial_active_type_1_prey", 8)
        self.n_initial_active_type_2_prey = c.get("n_initial_active_type_2_prey", 0)
        self.initial_energy_predator = c.get("initial_energy_predator", 5.0)
        self.initial_energy_prey = c.get("initial_energy_prey", 3.0)
        self.grid_size = c.get("grid_size", 10)
        self.num_obs_channels = c.get("num_obs_channels", 4)
        self.predator_obs_range = c.get("predator_obs_range", 7)
        self.prey_obs_range = c.get("prey_obs_range", 5)
        self.initial_num_grass = c.get("initial_num_grass", 25)
        self.initial_energy_grass = c.get("initial_energy_grass", 2.0)
        self.energy_gain_per_step_grass = c.get("energy_gain_per_step_grass", 0.2)
        self.mutation_rate_predator = c.get("mutation_rate_predator", 0.1)
        self.mutation_rate_prey = c.get("mutation_rate_prey", 0.1)
        self.type_1_act_range = c.get("type_1_action_range", 3)
        self.type_2_act_range = c.get("type_2_action_range", 5)

    def _init_reset_variables(self, seed):
        self.current_step = 0
        self.rng = np.random.default_rng(seed)
        self.agent_positions = {}
        self.predator_positions = {}
        self.prey_positions = {}
        self.grass_positions = {}
        self.agent_energies = {}
        self.grass_energies = {}
        self.agent_ages = {}
        self.agent_parents = {}
        self.unique_agents = {}
        self.unique_agent_stats = {}
        self.per_step_agent_data = []
        self._per_agent_step_deltas = {}
        self.agent_offspring_counts = {}
        self.agent_live_offspring_ids = {}
        self.agents_just_ate = set()
        self.cumulative_rewards = {}
        self.agent_activation_counts = {a: 0 for a in self.possible_agents}
        self.death_agents_stats = {}
        self.death_cause_prey = {}
        self.agent_last_reproduction = {}
        self.active_num_predators = 0
        self.active_num_prey = 0
        self.agents = []
        for agent_type in ["predator", "prey"]:
            for t in [1, 2]:
                key = f"n_initial_active_type_{t}_{agent_type}"
                count = self.config.get(key, 0)
                for i in range(count):
                    aid = f"type_{t}_{agent_type}_{i}"
                    self.agents.append(aid)
                    self._register_new_agent(aid)
        self.grass_agents = [f"grass_{i}" for i in range(self.initial_num_grass)]
        self.grid_world_state_shape = (self.num_obs_channels, self.grid_size, self.grid_size)
        self.initial_grid_world_state = np.zeros(self.grid_world_state_shape, dtype=np.float32)
        self.grid_world_state = self.initial_grid_world_state.copy()
        def _generate_action_map(range_size: int):
            delta = (range_size - 1) // 2
            return {i: (dx, dy) for i, (dx, dy) in enumerate((dx, dy) for dx in range(-delta, delta + 1) for dy in range(-delta, delta + 1))}
        self.action_to_move_tuple_type_1_agents = _generate_action_map(self.type_1_act_range)
        self.action_to_move_tuple_type_2_agents = _generate_action_map(self.type_2_act_range)

    def reset(self, *, seed=None, options=None):
        if hasattr(super(), 'reset'):
            try:
                super().reset(seed=seed)
            except TypeError:
                super().reset()
        self._init_reset_variables(seed)
        total_entities = len(self.agents) + len(self.grass_agents)
        all_positions = self._generate_random_positions(self.grid_size, total_entities, seed)
        predator_list = [a for a in self.agents if "predator" in a]
        prey_list = [a for a in self.agents if "prey" in a]
        predator_positions = all_positions[: len(predator_list)]
        prey_positions = all_positions[len(predator_list): len(predator_list) + len(prey_list)]
        grass_positions = all_positions[len(predator_list) + len(prey_list):]
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
        self.current_grass_energy = sum(self.grass_energies.values())
        obs = {agent: self._get_observation(agent) for agent in self.agents}
        return obs, {}

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        self.agents_just_ate.clear()
        truncation_result = self._check_truncation_and_early_return(observations, rewards, terminations, truncations, infos)
        if truncation_result is not None:
            return truncation_result
        self._apply_energy_decay_per_step(action_dict)
        self._apply_age_update(action_dict)
        self._regenerate_grass_energy()
        self._process_agent_movements(action_dict)
        for agent in self.agents:
            if agent not in self.agent_positions:
                continue
            if self.agent_energies[agent] <= 0:
                self._handle_energy_decay(agent, observations, rewards, terminations, truncations)
            elif "predator" in agent:
                self._handle_predator_engagement(agent, observations, rewards, terminations, truncations)
            elif "prey" in agent:
                self._handle_prey_engagement(agent, observations, rewards, terminations, truncations)
        for agent in self.agents[:]:
            if terminations.get(agent):
                self.agents.remove(agent)
                uid = self.unique_agents[agent]
                self.death_agents_stats[uid] = {**self.unique_agent_stats[uid], "lifetime": self.agent_ages[agent], "parent": self.agent_parents[agent]}
                del self.unique_agents[agent]
        for agent in self.agents[:]:
            if "predator" in agent:
                self._handle_predator_reproduction(agent, rewards, observations, terminations, truncations)
            elif "prey" in agent:
                self._handle_prey_reproduction(agent, rewards, observations, terminations, truncations)
        for agent in self.agents:
            if agent in self.agent_positions:
                observations[agent] = self._get_observation(agent)
        observations = {a: observations[a] for a in self.agents if a in observations}
        rewards = {a: rewards[a] for a in self.agents if a in rewards}
        terminations = {a: terminations[a] for a in self.agents if a in terminations}
        truncations = {a: truncations[a] for a in self.agents if a in truncations}
        truncations["__all__"] = False
        terminations["__all__"] = self.active_num_prey <= 0 or self.active_num_predators <= 0
        self.agents.sort()  # always sort baseline
        step_data = {}
        for agent in self.agents:
            pos = self.agent_positions[agent]
            energy = self.agent_energies[agent]
            deltas = self._per_agent_step_deltas[agent]
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
        self.current_step += 1
        return observations, rewards, terminations, truncations, infos

    def _get_movement_energy_cost(self, agent, current_position, new_position):
        distance_factor = self.config.get("move_energy_cost_factor", 0.01)
        current_energy = self.agent_energies[agent]
        distance = math.sqrt((new_position[0]-current_position[0])**2 + (new_position[1]-current_position[1])**2)
        return distance * distance_factor * current_energy

    def _get_move(self, agent, action: int):
        action = int(action)
        if "type_1" in agent:
            move_vector = self.action_to_move_tuple_type_1_agents[action]
        elif "type_2" in agent:
            move_vector = self.action_to_move_tuple_type_2_agents[action]
        else:
            raise ValueError(f"Unknown type for agent: {agent}")
        cp = self.agent_positions[agent]
        np_ = (cp[0] + move_vector[0], cp[1] + move_vector[1])
        np_ = tuple(np.clip(np_, 0, self.grid_size - 1))
        layer = 1 if "predator" in agent else 2
        if self.grid_world_state[layer, *np_] > 0:
            np_ = cp
        return np_

    def _get_observation(self, agent):
        rng = self.predator_obs_range if "predator" in agent else self.prey_obs_range
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp, rng)
        obs = np.zeros((self.num_obs_channels, rng, rng), dtype=np.float32)
        obs[0].fill(1)
        obs[0, xolo:xohi, yolo:yohi] = 0
        obs[1:, xolo:xohi, yolo:yohi] = self.grid_world_state[1:, xlo:xhi, ylo:yhi]
        return obs

    def _obs_clip(self, x, y, r):
        off = (r - 1)//2
        xld, xhd = x-off, x+off
        yld, yhd = y-off, y+off
        xlo, xhi = np.clip(xld, 0, self.grid_size - 1), np.clip(xhd, 0, self.grid_size - 1)
        ylo, yhi = np.clip(yld, 0, self.grid_size - 1), np.clip(yhd, 0, self.grid_size - 1)
        xolo, yolo = abs(np.clip(xld, -off, 0)), abs(np.clip(yld, -off, 0))
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi+1, ylo, yhi+1, xolo, xohi+1, yolo, yohi+1

    def _check_truncation_and_early_return(self, observations, rewards, terminations, truncations, infos):
        if self.current_step >= self.max_steps:
            for agent in self.possible_agents:
                if agent in self.agents:
                    observations[agent] = self._get_observation(agent)
                else:
                    rng = self.predator_obs_range if "predator" in agent else self.prey_obs_range
                    observations[agent] = np.zeros((self.num_obs_channels, rng, rng), dtype=np.float32)
                rewards[agent] = 0.0
                truncations[agent] = True
                terminations[agent] = False
            truncations["__all__"] = True
            terminations["__all__"] = False
            return observations, rewards, terminations, truncations, infos
        return None

    def _apply_energy_decay_per_step(self, action_dict):
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
            self._per_agent_step_deltas[agent] = {"decay": -decay, "move": 0.0, "eat": 0.0, "repro": 0.0}
            self.grid_world_state[layer, *self.agent_positions[agent]] = self.agent_energies[agent]

    def _apply_age_update(self, action_dict):
        for agent in action_dict:
            self.agent_ages[agent] += 1

    def _regenerate_grass_energy(self):
        max_energy_grass = self.config.get("max_energy_grass", float("inf"))
        for g, pos in self.grass_positions.items():
            new_energy = min(self.grass_energies[g] + self.energy_gain_per_step_grass, max_energy_grass)
            self.grass_energies[g] = new_energy
            self.grid_world_state[3, *pos] = new_energy

    def _process_agent_movements(self, action_dict):
        for agent, action in action_dict.items():
            if agent in self.agent_positions:
                old_pos = self.agent_positions[agent]
                new_pos = self._get_move(agent, action)
                self.agent_positions[agent] = new_pos
                move_cost = self._get_movement_energy_cost(agent, old_pos, new_pos)
                self.agent_energies[agent] -= move_cost
                self._per_agent_step_deltas[agent]["move"] = -move_cost
                uid = self.unique_agents[agent]
                self.unique_agent_stats[uid]["distance_traveled"] += np.linalg.norm(np.array(new_pos) - np.array(old_pos))
                self.unique_agent_stats[uid]["energy_spent"] += move_cost
                self.unique_agent_stats[uid]["avg_energy_sum"] += self.agent_energies[agent]
                self.unique_agent_stats[uid]["avg_energy_steps"] += 1
                if "predator" in agent:
                    self.predator_positions[agent] = new_pos
                    self.grid_world_state[1, *old_pos] = 0
                    self.grid_world_state[1, *new_pos] = self.agent_energies[agent]
                elif "prey" in agent:
                    self.prey_positions[agent] = new_pos
                    self.grid_world_state[2, *old_pos] = 0
                    self.grid_world_state[2, *new_pos] = self.agent_energies[agent]

    def _handle_energy_decay(self, agent, observations, rewards, terminations, truncations):
        observations[agent] = self._get_observation(agent)
        rewards[agent] = 0
        terminations[agent] = True
        truncations[agent] = False
        layer = 1 if "predator" in agent else 2
        self.grid_world_state[layer, *self.agent_positions[agent]] = 0
        uid = self.unique_agents[agent]
        stat = self.unique_agent_stats[uid]
        stat["death_step"] = self.current_step
        stat["death_cause"] = "starved"
        stat["final_energy"] = self.agent_energies[agent]
        steps = max(stat["avg_energy_steps"], 1)
        stat["avg_energy"] = stat["avg_energy_sum"] / steps
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

    def _handle_predator_engagement(self, agent, observations, rewards, terminations, truncations):
        predator_position = self.agent_positions[agent]
        caught_prey = next((p for p, pos in self.agent_positions.items() if "prey" in p and np.array_equal(predator_position, pos)), None)
        if caught_prey:
            rewards[agent] = self._get_type_specific("reward_predator_catch_prey", agent)
            self.cumulative_rewards.setdefault(agent, 0)
            self.cumulative_rewards[agent] += rewards[agent]
            raw_gain = min(self.agent_energies[caught_prey], self.config.get("max_energy_gain_per_prey", float("inf")))
            eff = self.config.get("energy_transfer_efficiency", 1.0)
            gain = raw_gain * eff
            self.agent_energies[agent] += gain
            self._per_agent_step_deltas[agent]["eat"] = gain
            max_e = self.config.get("max_energy_predator", float("inf"))
            self.agent_energies[agent] = min(self.agent_energies[agent], max_e)
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["times_ate"] += 1
            self.unique_agent_stats[uid]["energy_gained"] += self.agent_energies[caught_prey]
            self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]
            self.grid_world_state[1, *predator_position] = self.agent_energies[agent]
            observations[caught_prey] = self._get_observation(caught_prey)
            rewards[caught_prey] = self._get_type_specific("penalty_prey_caught", caught_prey)
            self.cumulative_rewards.setdefault(caught_prey, 0.0)
            self.cumulative_rewards[caught_prey] += rewards[caught_prey]
            terminations[caught_prey] = True
            truncations[caught_prey] = False
            self.active_num_prey -= 1
            self.grid_world_state[2, *self.agent_positions[caught_prey]] = 0
            uidp = self.unique_agents[caught_prey]
            stat = self.unique_agent_stats[uidp]
            stat["death_step"] = self.current_step
            stat["death_cause"] = "eaten"
            stat["final_energy"] = self.agent_energies[agent]
            steps = max(stat["avg_energy_steps"], 1)
            stat["avg_energy"] = stat["avg_energy_sum"] / steps
            stat["cumulative_reward"] = self.cumulative_rewards.get(agent, 0.0)
            self.death_agents_stats[uidp] = stat
            del self.agent_positions[caught_prey]
            del self.prey_positions[caught_prey]
            del self.agent_energies[caught_prey]
        else:
            rewards[agent] = self._get_type_specific("reward_predator_step", agent)
        observations[agent] = self._get_observation(agent)
        self.cumulative_rewards.setdefault(agent, 0)
        self.cumulative_rewards[agent] += rewards[agent]
        terminations[agent] = False
        truncations[agent] = False

    def _handle_prey_engagement(self, agent, observations, rewards, terminations, truncations):
        if terminations.get(agent):
            return
        prey_position = self.agent_positions[agent]
        caught_grass = next((g for g, pos in self.grass_positions.items() if "grass" in g and np.array_equal(prey_position, pos)), None)
        if caught_grass:
            rewards[agent] = self._get_type_specific("reward_prey_eat_grass", agent)
            self.cumulative_rewards.setdefault(agent, 0)
            self.cumulative_rewards[agent] += rewards[agent]
            raw_gain = min(self.grass_energies[caught_grass], self.config.get("max_energy_gain_per_grass", float("inf")))
            eff = self.config.get("energy_transfer_efficiency", 1.0)
            gain = raw_gain * eff
            self.agent_energies[agent] += gain
            self._per_agent_step_deltas[agent]["eat"] = gain
            max_e = self.config.get("max_energy_prey", float("inf"))
            self.agent_energies[agent] = min(self.agent_energies[agent], max_e)
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["times_ate"] += 1
            self.unique_agent_stats[uid]["energy_gained"] += self.grass_energies[caught_grass]
            self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]
            self.grid_world_state[2, *prey_position] = self.agent_energies[agent]
            self.grid_world_state[3, *prey_position] = 0
            self.grass_energies[caught_grass] = 0
        else:
            rewards[agent] = self._get_type_specific("reward_prey_step", agent)
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]
        observations[agent] = self._get_observation(agent)
        self.cumulative_rewards.setdefault(agent, 0)
        self.cumulative_rewards[agent] += rewards[agent]
        terminations[agent] = False
        truncations[agent] = False

    def _handle_predator_reproduction(self, agent, rewards, observations, terminations, truncations):
        cooldown = self.config.get("reproduction_cooldown_steps", 10)
        if self.current_step - self.agent_last_reproduction.get(agent, -cooldown) < cooldown:
            return
        chance_key = "reproduction_chance_predator" if "predator" in agent else "reproduction_chance_prey"
        if self.rng.random() > self.config.get(chance_key, 1.0):
            return
        if self.agent_energies[agent] >= self.predator_creation_energy_threshold:
            parent_type = int(agent.split("_")[1])
            mutated = self.rng.random() < self.mutation_rate_predator
            new_type = 2 if mutated and parent_type == 1 else (1 if mutated and parent_type == 2 else parent_type)
            potential_new_ids = [
                f"type_{new_type}_predator_{i}" for i in range(self.config.get(f"n_possible_type_{new_type}_predators", 25))
                if f"type_{new_type}_predator_{i}" not in self.agents
            ]
            if not potential_new_ids:
                rewards[agent] = self._get_type_specific("reproduction_reward_predator", agent)
                self.cumulative_rewards.setdefault(agent, 0)
                self.cumulative_rewards[agent] += rewards[agent]
                return
            new_agent = potential_new_ids[0]
            self.agents.append(new_agent)
            self._per_agent_step_deltas[new_agent] = {"decay": 0.0, "move": 0.0, "eat": 0.0, "repro": 0.0}
            self.agent_last_reproduction[agent] = self.current_step
            self._register_new_agent(new_agent, parent_unique_id=self.unique_agents[agent])
            child_uid = self.unique_agents[new_agent]
            self.agent_live_offspring_ids[agent].append(child_uid)
            self.agent_offspring_counts[agent] += 1
            self.unique_agent_stats[self.unique_agents[new_agent]]["mutated"] = mutated
            self.unique_agent_stats[self.unique_agents[agent]]["offspring_count"] += 1
            occupied_positions = set(self.agent_positions.values())
            new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)
            self.agent_positions[new_agent] = new_position
            self.predator_positions[new_agent] = new_position
            repro_eff = self.config.get("reproduction_energy_efficiency", 1.0)
            energy_given = self.initial_energy_predator * repro_eff
            self.agent_energies[new_agent] = energy_given
            self.agent_energies[agent] -= self.initial_energy_predator
            self._per_agent_step_deltas[agent]["repro"] = -self.initial_energy_predator
            self.grid_world_state[1, *new_position] = self.initial_energy_predator
            self.grid_world_state[1, *self.agent_positions[agent]] = self.agent_energies[agent]
            self.active_num_predators += 1
            rewards[new_agent] = 0
            rewards[agent] = self._get_type_specific("reproduction_reward_predator", agent)
            self.cumulative_rewards[new_agent] = 0
            self.cumulative_rewards[agent] += rewards[agent]
            observations[new_agent] = self._get_observation(new_agent)
            terminations[new_agent] = False
            truncations[new_agent] = False

    def _handle_prey_reproduction(self, agent, rewards, observations, terminations, truncations):
        cooldown = self.config.get("reproduction_cooldown_steps", 10)
        if self.current_step - self.agent_last_reproduction.get(agent, -cooldown) < cooldown:
            return
        chance_key = "reproduction_chance_predator" if "predator" in agent else "reproduction_chance_prey"
        if self.rng.random() > self.config.get(chance_key, 1.0):
            return
        if self.agent_energies[agent] >= self.prey_creation_energy_threshold:
            parent_type = int(agent.split("_")[1])
            mutated = self.rng.random() < self.mutation_rate_prey
            new_type = 2 if mutated and parent_type == 1 else (1 if mutated and parent_type == 2 else parent_type)
            potential_new_ids = [
                f"type_{new_type}_prey_{i}" for i in range(self.config.get(f"n_possible_type_{new_type}_prey", 25))
                if f"type_{new_type}_prey_{i}" not in self.agents
            ]
            if not potential_new_ids:
                rewards[agent] = self._get_type_specific("reproduction_reward_prey", agent)
                self.cumulative_rewards.setdefault(agent, 0)
                self.cumulative_rewards[agent] += rewards[agent]
                return
            new_agent = potential_new_ids[0]
            self.agents.append(new_agent)
            self._per_agent_step_deltas[new_agent] = {"decay": 0.0, "move": 0.0, "eat": 0.0, "repro": 0.0}
            self.agent_last_reproduction[agent] = self.current_step
            self._register_new_agent(new_agent, parent_unique_id=self.unique_agents[agent])
            child_uid = self.unique_agents[new_agent]
            self.agent_live_offspring_ids[agent].append(child_uid)
            self.agent_offspring_counts[agent] += 1
            self.unique_agent_stats[self.unique_agents[new_agent]]["mutated"] = mutated
            self.unique_agent_stats[self.unique_agents[agent]]["offspring_count"] += 1
            occupied_positions = set(self.agent_positions.values())
            new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)
            self.agent_positions[new_agent] = new_position
            self.prey_positions[new_agent] = new_position
            repro_eff = self.config.get("reproduction_energy_efficiency", 1.0)
            energy_given = self.initial_energy_prey * repro_eff
            self.agent_energies[new_agent] = energy_given
            self.agent_energies[agent] -= self.initial_energy_prey
            self._per_agent_step_deltas[agent]["repro"] = -self.initial_energy_prey
            self.grid_world_state[2, *new_position] = self.initial_energy_prey
            self.grid_world_state[2, *self.agent_positions[agent]] = self.agent_energies[agent]
            self.active_num_prey += 1
            rewards[new_agent] = 0
            rewards[agent] = self._get_type_specific("reproduction_reward_prey", agent)
            self.cumulative_rewards[new_agent] = 0
            self.cumulative_rewards[agent] += rewards[agent]
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]
            observations[new_agent] = self._get_observation(new_agent)
            terminations[new_agent] = False
            truncations[new_agent] = False

    def _generate_random_positions(self, grid_size, num_positions, seed=None):
        if num_positions > grid_size * grid_size:
            raise ValueError("Too many positions")
        rng = np.random.default_rng(seed)
        pos = set()
        while len(pos) < num_positions:
            pos.add(tuple(rng.integers(0, grid_size, size=2)))
        return list(pos)

    def _find_available_spawn_position(self, ref, occupied):
        x,y = ref
        candidates = [(x+dx, y+dy) for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)] if 0 <= x+dx < self.grid_size and 0 <= y+dy < self.grid_size]
        free = [p for p in candidates if p not in occupied]
        if free:
            return free[0]
        all_positions = {(i,j) for i in range(self.grid_size) for j in range(self.grid_size)}
        free_positions = list(all_positions - occupied)
        return free_positions[self.rng.integers(len(free_positions))] if free_positions else None

    def _register_new_agent(self, agent_id: str, parent_unique_id: str = None):
        reuse_index = self.agent_activation_counts[agent_id]
        unique_id = f"{agent_id}_{reuse_index}"
        self.unique_agents[agent_id] = unique_id
        self.agent_activation_counts[agent_id] += 1
        self.agent_ages[agent_id] = 0
        self.agent_offspring_counts[agent_id] = 0
        self.agent_live_offspring_ids[agent_id] = []
        self.agent_parents[agent_id] = parent_unique_id
        self.agent_last_reproduction[agent_id] = -self.config.get("reproduction_cooldown_steps", 10)
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
            "final_energy": None,
            "avg_energy": None,
        }

    def _build_possible_agent_ids(self):
        ids = []
        for i in range(self.n_possible_type_1_predators):
            ids.append(f"type_1_predator_{i}")
        for i in range(self.n_possible_type_2_predators):
            ids.append(f"type_2_predator_{i}")
        for i in range(self.n_possible_type_1_prey):
            ids.append(f"type_1_prey_{i}")
        for i in range(self.n_possible_type_2_prey):
            ids.append(f"type_2_prey_{i}")
        return ids

    def _build_observation_space(self, agent_id):
        if "predator" in agent_id:
            shape = (self.num_obs_channels, self.predator_obs_range, self.predator_obs_range)
        elif "prey" in agent_id:
            shape = (self.num_obs_channels, self.prey_obs_range, self.prey_obs_range)
        else:
            raise ValueError("Unknown agent id")
        return gymnasium.spaces.Box(low=0, high=100.0, shape=shape, dtype=np.float32)

    def _build_action_space(self, agent_id):
        if "type_1" in agent_id:
            return gymnasium.spaces.Discrete(self.type_1_act_range ** 2)
        elif "type_2" in agent_id:
            return gymnasium.spaces.Discrete(self.type_2_act_range ** 2)
        raise ValueError("Unknown agent id")

    def _get_type_specific(self, key: str, agent_id: str):
        raw = getattr(self, f"{key}_config", 0.0)
        if isinstance(raw, dict):
            for k in raw:
                if agent_id.startswith(k):
                    return raw[k]
            raise KeyError(f"Type-specific key '{agent_id}' not found under '{key}'")
        return raw
