"""
Predator-Prey Grass RLlib Environment

Additional features:
- lineage rewards for predators and prey
- limited fertility age per agent type
- limited age per agent type
"""
# external libraries (Ray required)
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from collections import deque
from typing import Optional

from predpreygrass.eco_evolutionary.utils.genome import Genome, founder_genome, mutate_genome


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
        self.observation_space = gymnasium.spaces.Dict({str(aid): space for aid, space in self.observation_spaces.items()})
        self.action_space = gymnasium.spaces.Dict({str(aid): space for aid, space in self.action_spaces.items()})
        self.observation_space_struct = self.observation_spaces
        self.action_space_struct = self.action_spaces

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
        self.reproduction_reward_predator_config = config["reproduction_reward_predator"]
        self.reproduction_reward_prey_config = config["reproduction_reward_prey"]
        # Lineage survival reward coefficient (can be scalar or per-policy dict)
        self.lineage_reward_coeff_config = config["lineage_reward_coeff"]
        default_fertility_caps = {
            "predator": 80,
            "prey": 60,
        }
        configured_caps = config.get("max_fertility_age")
        if isinstance(configured_caps, dict):
            self.max_fertility_age_config = {**default_fertility_caps, **configured_caps}
        else:
            self.max_fertility_age_config = default_fertility_caps
        default_age_caps = {
            "predator": 120,
            "prey": 100,
        }
        configured_age_caps = config.get("max_agent_age")
        if isinstance(configured_age_caps, dict):
            merged_caps = default_age_caps.copy()
            for key, limit in configured_age_caps.items():
                # Treat None as "use default" to keep protective caps active unless explicitly overridden
                if limit is not None:
                    merged_caps[key] = limit
            self.max_agent_age_config = merged_caps
        else:
            self.max_agent_age_config = default_age_caps

        default_carcass_only_caps = {
            "predator": None,
        }
        configured_carcass_caps = config.get("carcass_only_predator_age")
        if isinstance(configured_carcass_caps, dict):
            self.carcass_only_predator_age_config = {**default_carcass_only_caps, **configured_carcass_caps}
        else:
            self.carcass_only_predator_age_config = default_carcass_only_caps

        # Energy settings
        self.energy_loss_per_step_predator = config["energy_loss_per_step_predator"]
        self.energy_loss_per_step_prey = config["energy_loss_per_step_prey"]
        self.movement_energy_cost_per_cell_predator = config.get("movement_energy_cost_per_cell_predator", 0.0)
        self.movement_energy_cost_per_cell_prey = config.get("movement_energy_cost_per_cell_prey", 0.0)
        self.movement_speed_cost_exponent = config.get("movement_speed_cost_exponent", 2.0)
        self.predator_creation_energy_threshold = config["predator_creation_energy_threshold"]
        self.prey_creation_energy_threshold = config["prey_creation_energy_threshold"]
        self.max_energy_grass = self.config["max_energy_grass"]


        # Learning agents
        self.n_possible_predators = config["n_possible_predators"]
        self.n_possible_prey = config["n_possible_prey"]

        self.n_initial_active_predators = config["n_initial_active_predators"]
        self.n_initial_active_prey = config["n_initial_active_prey"]

        self.initial_energy_predator = config["initial_energy_predator"]
        self.initial_energy_prey = config["initial_energy_prey"]

        # Grid and Observation Settings
        self.grid_size = config["grid_size"]
        self.num_obs_channels = config["num_obs_channels"]
        self.predator_obs_range = config["predator_obs_range"]
        self.prey_obs_range = config["prey_obs_range"]

        # Grass settings
        self.initial_num_grass = config["initial_num_grass"]
        self.initial_energy_grass = config["initial_energy_grass"]
        self.energy_gain_per_step_grass = config["energy_gain_per_step_grass"]

        # Action range and movement mapping
        self.action_range = config["action_range"]
        self.genome_enabled = config.get("genome_enabled", True)
        self.genome_config = {
            "founder_genome": config.get("founder_genome", {}),
            "genome_mutation": config.get("genome_mutation", {}),
            "trait_bounds": config.get("trait_bounds", {}),
        }
        self.speed_distance_threshold = config.get("speed_distance_threshold", 1.5)
        self.slow_max_move_distance = config.get("slow_max_move_distance", 1)
        self.fast_max_move_distance = config.get("fast_max_move_distance", 2)

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
        # Per-step return dicts
        self.observations, self.rewards, self.terminations, self.truncations, self.infos = {}, {}, {}, {}, {}
        # Ages (in steps) for all currently active agents
        self.agent_ages = {}
        # Per-agent event log for detailed post-hoc analysis
        # Structure per agent_id:
        # {
        #   "agent_id": str,
        #   "birth_step": int,
        #   "death_step": Optional[int],
        #   "parent_id": Optional[str],
        #   "death_cause": Optional[str],
        #   "eating_events": [
        #       {"t": int, "id_eaten": str, "alive_before_bite": bool,
        #        "bite_size": float, "energy_after": float}, ...],
        #   "reproduction_events": [
        #       {"t": int, "child_id": str}, ...],
        #   "reward_events": [
        #       {"t": int, "reproduction_reward": float,
        #        "lineage_reward": float,
        #        "cumulative_reward": float}, ...],
        #   "fertility_events": [],
        #   "lifecycle_events": [],
        # }
        self.agent_event_log = {}
        self.agent_stats_live = {}
        self.agent_stats_completed = {}
        self.per_step_agent_data = []  # One entry per step; each is {agent_id: {position, energy, ...}}
        self._per_agent_step_deltas = {}  # Internal temp storage to track energy deltas during step
        self._next_lifetime_id = 0
        self.agent_parents = {}
        self.agent_genomes = {}
        self.agent_offspring_counts = {}
        self.agent_live_offspring_ids = {}
        # Lineage tracking: stores parent-child relationships and live-descendant counts
        self.lineage_tracker = {}
        # Track all agent IDs that have ever been active in this episode to prevent reuse
        self.used_agent_ids = set()
        # Capacity block counters (episode-level)
        self.reproduction_blocked_due_to_capacity_predator = 0
        self.reproduction_blocked_due_to_capacity_prey = 0
        self.reproduction_blocked_due_to_fertility_predator = 0
        self.reproduction_blocked_due_to_fertility_prey = 0
        self.carcass_only_live_prey_blocks_predator = 0
        # Episode-level spawn counters
        self.spawned_predators = 0
        self.spawned_prey = 0


        self.agents_just_ate = set()
        # Prey that have been bitten at least once and are now "dead meat":
        # cannot move, cannot eat grass, cannot reproduce, and should not
        # contribute lineage survival rewards as surviving parents.
        self.dead_prey = set()

        # Per-step infos accumulator and last-move diagnostics
        self._pending_infos = {}
        self._last_move_block_reason = {}

        self.death_cause_prey = {}

        self.agent_last_reproduction = {}

        # Episode-level debug counters for predator rewards
        self.debug_predator_total_reward = 0.0
        self.debug_predator_repro_events = 0
        self.debug_predator_lineage_events = 0

        # aggregates per step
        self.active_num_predators = 0
        self.active_num_prey = 0

        self.agents = []
        # create active agents list based on config
        for i in range(self.n_initial_active_predators):
            agent_id = f"predator_{i}"
            self.agents.append(agent_id)
            self._register_new_agent(agent_id, is_founder=True)
        for i in range(self.n_initial_active_prey):
            agent_id = f"prey_{i}"
            self.agents.append(agent_id)
            self._register_new_agent(agent_id, is_founder=True)

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

        self.action_to_move_tuple_agents = _generate_action_map(self.action_range)

        # Initialize per-type available ID pools (never reuse within an episode)
        self._init_available_id_pools()
        # Print-once guard for termination debug logs (per episode)
        self._printed_termination_ids = set()

    def _init_available_id_pools(self):
        """Build deques of never-used IDs per species/type for O(1) allocation.

        Pools are initialized with all possible IDs from config, then filtered to exclude
        any IDs that are already used in this episode (initial actives). IDs are never
        returned to the pool until reset.
        """
        pools = {"predator": deque(), "prey": deque()}

        # Populate in deterministic order
        for i in range(self.n_possible_predators):
            aid = f"predator_{i}"
            if aid not in self.used_agent_ids:
                pools["predator"].append(aid)
        for i in range(self.n_possible_prey):
            aid = f"prey_{i}"
            if aid not in self.used_agent_ids:
                pools["prey"].append(aid)

        self._available_id_pools = pools

    def _alloc_new_id(self, species: str):
        """Allocate a fresh agent ID from the species pool or return None if exhausted.

        Ensures the returned ID has not been used earlier in this episode and is not currently active.
        """
        dq = self._available_id_pools.get(species)
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
        self.observation_space_struct = self.observation_spaces
        self.action_space_struct = self.action_spaces
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        return observations, {}

    def step(self, action_dict):
        acted_ids = {str(agent) for agent in action_dict}
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

        # Step 6: Spawning of new agents (cooldown and chance removed; energy-only)
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

        # Step 6.5: Apply lineage survival rewards based on live descendant changes
        self._apply_lineage_survival_rewards()

        # Step 8: Assemble return dicts. Observations contain only agents that
        # should act next. Rewards/done flags may include agents that ended on
        # this step.
        live_ids: set[str] = {str(a) for a in self.agents}
        reward_full = dict(self.rewards)
        term_full = dict(self.terminations)
        trunc_full = dict(self.truncations)
        ended_ids: set[str] = {str(aid) for aid, flag in term_full.items() if flag} | {str(aid) for aid, flag in trunc_full.items() if flag}

        missing_acted_ids = (acted_ids - live_ids - ended_ids) & set(self.possible_agents)
        for agent in missing_acted_ids:
            # RLlib requires every agent that acted and disappeared to receive a
            # final done flag plus final observation. This catches carcass/cleanup
            # edge cases where lower-level bookkeeping removed an acted agent
            # before the final return dict was assembled.
            reward_full.setdefault(agent, self.rewards.get(agent, 0.0))
            term_full[agent] = True
            trunc_full[agent] = False
        ended_ids |= missing_acted_ids

        episode_done = self.active_num_prey <= 0 or self.active_num_predators <= 0
        if episode_done:
            # Extinction is a natural terminal condition, not a time-limit truncation.
            for agent in live_ids:
                term_full[agent] = True
                trunc_full[agent] = False
            ended_ids |= live_ids
            self._attach_final_cumulative_rewards_for_live_agents()
            self._attach_episode_training_metrics()

        output_ids = live_ids | ended_ids
        self.rewards = {aid: reward_full.get(aid, 0.0) for aid in output_ids}
        self.terminations = {aid: term_full.get(aid, False) for aid in output_ids}
        self.truncations = {aid: trunc_full.get(aid, False) for aid in output_ids}
        self.infos = {aid: self._pending_infos.get(aid, {}) for aid in output_ids}

        if "__all__" in self._pending_infos:
            self.infos["__all__"] = self._pending_infos["__all__"]
        self.terminations["__all__"] = episode_done
        self.truncations["__all__"] = False

        next_actor_ids = live_ids - ended_ids if not episode_done else set()
        get_obs = self._get_observation
        final_observations = {}
        for agent in ended_ids:
            if agent in self.observations:
                final_observations[agent] = self.observations[agent]
            elif agent in self.agent_positions:
                final_observations[agent] = get_obs(agent)
            elif agent in acted_ids:
                final_observations[agent] = self._empty_observation(agent)
        self.observations = final_observations
        self.observations.update({agent: get_obs(agent) for agent in next_actor_ids})

        step_data = {}
        for agent in self.agents:
            pos = self.agent_positions[agent]
            energy = self.agent_energies[agent]
            deltas = self._per_agent_step_deltas.get(agent, {"decay": 0.0, "move": 0.0, "eat": 0.0, "repro": 0.0})
            parent = self.agent_parents.get(agent)
            age = self.agent_ages[agent]
            step_data[agent] = {
                "position": pos,
                "energy": energy,
                "energy_decay": deltas["decay"],
                "energy_movement": deltas["move"],
                "energy_eating": deltas["eat"],
                "energy_reproduction": deltas["repro"],
                "age": age,
                "offspring_count": self.agent_offspring_counts[agent],
                "offspring_ids": self.agent_live_offspring_ids.get(agent, []),
                "parent": parent,
            }

        self.per_step_agent_data.append(step_data)
        self._per_agent_step_deltas.clear()

        # Increment step counter
        self.current_step += 1

        if self.current_step >= self.max_steps and not episode_done:
            # Final time-limit step: active agents are truncated. Agents that
            # died earlier in this step remain terminated. RLlib requires a
            # final observation for truncated agents for value bootstrapping.
            active_now: set[str] = {str(a) for a in self.agents}
            final_ids = (set(self.rewards) | set(self.terminations) | set(self.truncations) | active_now) - {"__all__"}
            final_ids |= (acted_ids - active_now) & set(self.possible_agents)

            obs, rews, terms, truncs, infos = {}, {}, {}, {}, {}
            for agent in active_now:
                truncs[agent] = True
                terms[agent] = False
                obs[agent] = self._get_observation(agent)

            for agent in final_ids:
                rews[agent] = self.rewards.get(agent, 0.0)
                terms[agent] = self.terminations.get(agent, terms.get(agent, False))
                truncs[agent] = self.truncations.get(agent, truncs.get(agent, False))
                if agent in active_now:
                    terms[agent] = False
                    truncs[agent] = True
                elif agent in acted_ids and not terms[agent] and not truncs[agent]:
                    terms[agent] = True
                info = self._pending_infos.get(agent, {}).copy()
                record = self.agent_stats_live.get(agent) or self.agent_stats_completed.get(agent)
                if record is not None:
                    info["final_cumulative_reward"] = record.get("cumulative_reward", 0.0)
                infos[agent] = info
                if (terms[agent] or truncs[agent]) and agent not in obs:
                    if agent in self.observations:
                        obs[agent] = self.observations[agent]
                    elif agent in self.agent_positions:
                        obs[agent] = self._get_observation(agent)
                    elif agent in acted_ids:
                        obs[agent] = self._empty_observation(agent)

            truncs["__all__"] = True
            terms["__all__"] = False

            # Overwrite step-assembled dicts to only contain final-step relevant agents.
            self.observations, self.rewards = obs, rews
            self.terminations, self.truncations, self.infos = terms, truncs, infos

            for agent_id in list(self.agent_stats_live.keys()):
                self._finalize_agent_record(agent_id, cause="time_limit")

            infos["__all__"] = {"training_metrics": self._build_episode_training_metrics()}
            self.agents = []

            return self.observations, self.rewards, self.terminations, self.truncations, self.infos

        if episode_done:
            self.agents = []

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def _policy_group(self, agent_id: str) -> str:
        if "predator" in agent_id:
            return "predator"
        if "prey" in agent_id:
            return "prey"
        raise ValueError(f"Unknown agent type in ID: {agent_id}")

    def _get_agent_genome(self, agent_id: str) -> Optional[Genome]:
        return self.agent_genomes.get(agent_id)

    def _get_max_move_distance(self, agent_id: str) -> int:
        genome = self._get_agent_genome(agent_id)
        if genome is None:
            return int(self.slow_max_move_distance)
        if float(genome.speed) >= float(self.speed_distance_threshold):
            return int(self.fast_max_move_distance)
        return int(self.slow_max_move_distance)

    def _get_speed_cost_factor(self, agent_id: str) -> float:
        genome = self._get_agent_genome(agent_id)
        if genome is None:
            return 1.0
        return float(genome.speed) ** float(self.movement_speed_cost_exponent)

    def _get_movement_energy_cost(self, agent_id: str, old_position, new_position) -> float:
        distance = float(np.linalg.norm(np.array(new_position) - np.array(old_position)))
        if distance <= 0:
            return 0.0
        if "predator" in agent_id:
            cost_per_cell = float(self.movement_energy_cost_per_cell_predator)
        else:
            cost_per_cell = float(self.movement_energy_cost_per_cell_prey)
        return cost_per_cell * distance * self._get_speed_cost_factor(agent_id)

    def _inherit_genome(self, agent_id: str, parent_agent_id: Optional[str], *, is_founder: bool) -> Optional[Genome]:
        if not self.genome_enabled:
            return None
        if is_founder or parent_agent_id is None or parent_agent_id not in self.agent_genomes:
            return founder_genome(self._policy_group(agent_id), self.genome_config, self.rng)
        return mutate_genome(self.agent_genomes[parent_agent_id], self.genome_config, self.rng)

    def _apply_time_step_update(self):
        """
        Apply all per-step updates (energy decay, age increment).
        """
        aged_out_agents = []
        for raw_agent in list(self.agents):
            agent = str(raw_agent)
            is_predator = "predator" in agent
            layer = 0 if is_predator else 1

            # Basal metabolism always applies while the agent/carcass exists.
            # Locomotion cost is charged later from actual distance moved.
            if is_predator:
                energy_decay = self.energy_loss_per_step_predator
            else:
                energy_decay = self.energy_loss_per_step_prey

            self.agent_energies[agent] -= energy_decay
            self.grid_world_state[layer, *self.agent_positions[agent]] = self.agent_energies[agent]

            # Freeze age for dead prey (carcasses): they no longer accrue lifetime.
            if not ("prey" in agent and agent in self.dead_prey):
                self.agent_ages[agent] += 1
                self._maybe_mark_fertility_expired(agent)
                if self._agent_age_exceeded(agent):
                    aged_out_agents.append(agent)

            self._per_agent_step_deltas[agent] = {
                "decay": -energy_decay,
                "move": 0.0,
                "eat": 0.0,
                "repro": 0.0,
            }

        for aged_agent in aged_out_agents:
            self._terminate_agent_due_to_age(aged_agent)

    def _regenerate_grass_energy(self):
        """
        Increase energy of all grass patches, capped at initial energy value.
        """
        for grass, pos in self.grass_positions.items():
            old_energy = self.grass_energies[grass]
            new_energy = min(old_energy + self.energy_gain_per_step_grass, self.max_energy_grass)
            self.grass_energies[grass] = new_energy
            self.grid_world_state[2, pos[0], pos[1]] = new_energy

    def _process_agent_movements(self, action_dict):
        """
        Process movement and grid updates for all agents (non-vectorized, simple loop).
        """
        for agent in action_dict.keys():
            if agent not in self.agent_positions or self.terminations.get(agent):
                continue
            # Dead prey (bitten at least once) do not move anymore.
            if "prey" in agent and agent in self.dead_prey:
                continue
            old_position = self.agent_positions[agent]
            action = action_dict[agent]
            new_position = self._get_move(agent, action)
            movement_cost = self._get_movement_energy_cost(agent, old_position, new_position)
            self.agent_energies[agent] -= movement_cost
            self._per_agent_step_deltas.setdefault(
                agent,
                {"decay": 0.0, "move": 0.0, "eat": 0.0, "repro": 0.0},
            )
            self._per_agent_step_deltas[agent]["move"] -= movement_cost
            if "predator" in agent:
                self.predator_positions[agent] = tuple(new_position)
                self.grid_world_state[0, old_position[0], old_position[1]] = 0
                self.grid_world_state[0, new_position[0], new_position[1]] = self.agent_energies[agent]
            elif "prey" in agent:
                self.prey_positions[agent] = tuple(new_position)
                self.grid_world_state[1, old_position[0], old_position[1]] = 0
                self.grid_world_state[1, new_position[0], new_position[1]] = self.agent_energies[agent]
            record = self.agent_stats_live.get(agent)
            if record is not None:
                record["avg_energy_sum"] += self.agent_energies[agent]
                record["avg_energy_steps"] += 1
                record["distance_traveled"] += float(np.linalg.norm(np.array(new_position) - np.array(old_position)))
                record["movement_energy_spent"] += movement_cost
            self.agent_positions[agent] = tuple(new_position)

    def _get_move(self, agent, action: int):
        """
        Get the new position of the agent based on the action and its type.
        """
        action = int(action)

        move_vector = self.action_to_move_tuple_agents[action]

        current_position = self.agent_positions[agent]
        max_distance = self._get_max_move_distance(agent)
        move_distance = max(abs(move_vector[0]), abs(move_vector[1]))
        if move_distance > max_distance:
            move_vector = (
                int(np.sign(move_vector[0])) * max_distance,
                int(np.sign(move_vector[1])) * max_distance,
            )
        new_position = (
            current_position[0] + move_vector[0],
            current_position[1] + move_vector[1],
        )

        # Clip new position to stay within grid bounds
        new_position = tuple(np.clip(new_position, 0, self.grid_size - 1))

        agent_layer = 0 if "predator" in agent else 1
        # Default: no block
        self._last_move_block_reason[agent] = None
        if self.grid_world_state[agent_layer, *new_position] > 0:
            new_position = current_position
            self._last_move_block_reason[agent] = "occupied"

        return new_position

    def _get_observation(self, agent):
        # Generate an observation for the agent.
        obs_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp, obs_range)
        # Allocate observation tensor
        obs = np.zeros((self.num_obs_channels, obs_range, obs_range), dtype=np.float32)
        gws = self.grid_world_state  # local reference
        # Channels: predators, prey, grass. The grid edge is represented by
        # clipping the copied window and leaving out-of-grid observation cells at zero.
        obs[:, xolo:xohi, yolo:yohi] = gws[:, xlo:xhi, ylo:yhi]
        return obs

    def _empty_observation(self, agent):
        obs_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
        return np.zeros((self.num_obs_channels, obs_range, obs_range), dtype=np.float32)

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
        all_positions = {
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
        }
        free_positions = list(all_positions - occupied_positions)

        if free_positions:
            return free_positions[self.rng.integers(len(free_positions))]

        return None  # No available position found

    def _handle_energy_starvation(self, agent):
        # Ensure dead-prey bookkeeping is cleaned up if a carcass-starved prey dies.
        if "prey" in agent:
            self.dead_prey.discard(agent)
        self._handle_lineage_death(agent)
        self.observations[agent] = self._get_observation(agent)
        self.rewards[agent] = 0
        self.terminations[agent] = True
        self.truncations[agent] = False

        layer = 0 if "predator" in agent else 1
        self.grid_world_state[layer, *self.agent_positions[agent]] = 0
        self._finalize_agent_record(agent, cause="starved")

        if "predator" in agent:
            self.active_num_predators -= 1
        else:
            self.active_num_prey -= 1
        #del self.agent_ages[agent]

    def _handle_predator_engagement(self, agent):
        self._per_agent_step_deltas.setdefault(
            agent,
            {
                "decay": 0.0,
                "move": 0.0,
                "eat": 0.0,
                "repro": 0.0,
            },
        )
        predator_position = tuple(self.agent_positions[agent])
        caught_prey = next(
            (prey for prey, pos in self.agent_positions.items() if "prey" in prey and np.array_equal(predator_position, pos)), None
        )
        if caught_prey:
            was_dead_before = caught_prey in self.dead_prey
            if (not was_dead_before) and self._predator_requires_carcass_only(agent):
                self._record_carcass_only_block(agent, caught_prey)
                return
            # attribution predator
            self.agents_just_ate.add(agent)
            self.rewards[agent] = self._get_role_specific("reward_predator_catch_prey", agent)
            # Debug: track predator reward contributions
            self.debug_predator_total_reward += float(self.rewards[agent])
            # cumulative_reward is tracked directly in agent_stats_live
            prey_energy = float(self.agent_energies[caught_prey])
            intake_cap = float(self.config.get("max_energy_gain_per_prey", float("inf")))
            bite = min(prey_energy, intake_cap)
            energy_gain = bite

            # Predator energy update (no hard cap unless configured)
            self.agent_energies[agent] += energy_gain
            self.grid_world_state[0, *predator_position] = self.agent_energies[agent]
            self._per_agent_step_deltas[agent]["eat"] = energy_gain
            predator_record = self.agent_stats_live.get(agent)
            if predator_record is not None:
                predator_record["times_ate"] += 1
                predator_record["energy_gained"] += energy_gain
                predator_record["cumulative_reward"] += self.rewards[agent]
            # attribution prey
            remaining_prey_energy = prey_energy - bite
            if remaining_prey_energy > 0.0:
                # Prey survives with reduced energy but becomes dead meat:
                # it will no longer move, eat grass or reproduce.
                self.agent_energies[caught_prey] = remaining_prey_energy
                prey_pos = self.agent_positions[caught_prey]
                self.grid_world_state[1, *prey_pos] = remaining_prey_energy
                self.dead_prey.add(caught_prey)
                # Align death_step with age logic: the first fatal bite is
                # considered the time of death, even though the carcass may
                # persist as food. Freeze death_step here if not already set.
                prey_record = self.agent_stats_live.get(caught_prey)
                if prey_record is not None and prey_record.get("death_step") is None:
                    prey_record["death_step"] = int(self.current_step)
                if not was_dead_before:
                    self._handle_lineage_death(caught_prey)
                # Do not mark termination; prey continues into next step
            else:
                # Fully eaten prey: keep original termination path
                # Capture a final observation for the caught prey at the moment of termination
                # so RLlib registers the terminal step properly.
                if not was_dead_before:
                    self._handle_lineage_death(caught_prey)
                self.observations[caught_prey] = self._get_observation(caught_prey)
                self.terminations[caught_prey] = True
                penalty = self._get_role_specific("penalty_prey_caught", caught_prey)
                self.rewards[caught_prey] = penalty
                self.truncations[caught_prey] = False
                self.active_num_prey -= 1
                self.grid_world_state[1, *self.agent_positions[caught_prey]] = 0
                self.dead_prey.add(caught_prey)
                prey_record = self.agent_stats_live.get(caught_prey)
                if prey_record is not None:
                    prey_record["death_cause"] = "eaten"
                    # Add penalty to existing cumulative_reward (do not overwrite)
                    prey_record["cumulative_reward"] += penalty
                self._finalize_agent_record(caught_prey, cause="eaten")
            # Log predator eating event (whether or not prey fully eaten)
            # alive_before_bite=True for the first ever bite on this prey id,
            # alive_before_bite=False for all subsequent (carcass) bites.
            evt = self.agent_event_log.get(agent)
            if evt is not None:
                evt.setdefault("eating_events", []).append(
                    {
                        "t": int(self.current_step),
                        "id_eaten": caught_prey,
                        "alive_before_bite": (not was_dead_before),
                        "bite_size": float(bite),
                        "energy_after": float(self.agent_energies[agent]),
                    }
                )
        else:
            self.rewards[agent] = self._get_role_specific("reward_predator_step", agent)
            # Debug: track predator step rewards
            self.debug_predator_total_reward += float(self.rewards[agent])
            predator_record = self.agent_stats_live.get(agent)
            if predator_record is not None:
                predator_record["cumulative_reward"] += self.rewards[agent]

    def _handle_prey_engagement(self, agent):
        if self.terminations.get(agent):
            return
        # Dead prey never eat grass; they are carcass-like resources only.
        if agent in self.dead_prey:
            self.rewards[agent] = self._get_role_specific("reward_prey_step", agent)
            prey_record = self.agent_stats_live.get(agent)
            if prey_record is not None:
                prey_record["cumulative_reward"] += self.rewards[agent]
            return
        prey_position = tuple(self.agent_positions[agent])
        caught_grass = next(
            (g for g, pos in self.grass_positions.items() if "grass" in g and np.array_equal(prey_position, pos)), None
        )
        if caught_grass:
            # attribution prey
            self.agents_just_ate.add(agent)
            self.rewards[agent] = self._get_role_specific("reward_prey_eat_grass", agent)
            # cumulative_reward is tracked directly in agent_stats_live
            grass_energy = float(self.grass_energies[caught_grass])
            intake_cap = float(self.config.get("max_energy_gain_per_grass", float("inf")))
            bite = min(grass_energy, intake_cap)
            energy_gain = bite

            self.agent_energies[agent] += energy_gain
            self._per_agent_step_deltas[agent]["eat"] = energy_gain
            self.grid_world_state[1, *prey_position] = self.agent_energies[agent]

            remaining_grass_energy = grass_energy - bite
            if remaining_grass_energy > 0.0:
                self.grass_energies[caught_grass] = remaining_grass_energy
                self.grid_world_state[2, *prey_position] = remaining_grass_energy
            else:
                self.grass_energies[caught_grass] = 0.0
                self.grid_world_state[2, *prey_position] = 0.0
            prey_record = self.agent_stats_live.get(agent)
            if prey_record is not None:
                prey_record["times_ate"] += 1
                prey_record["energy_gained"] += energy_gain
                prey_record["cumulative_reward"] += self.rewards[agent]
            # Log prey eating event
            evt = self.agent_event_log.get(agent)
            if evt is not None:
                evt.setdefault("eating_events", []).append(
                    {
                        "t": int(self.current_step),
                        "id_eaten": caught_grass,
                        "alive_before_bite": True,
                        "bite_size": float(bite),
                        "energy_after": float(self.agent_energies[agent]),
                    }
                )
        else:
            self.rewards[agent] = self._get_role_specific("reward_prey_step", agent)
            prey_record = self.agent_stats_live.get(agent)
            if prey_record is not None:
                prey_record["cumulative_reward"] += self.rewards[agent]

    def _apply_lineage_survival_rewards(self):
        """Grant lineage survival rewards based on live-descendant deltas."""
        if not self.lineage_tracker:
            return
        for agent_id, record in list(self.lineage_tracker.items()):
            if agent_id not in self.agent_stats_live:
                # Only living agents accrue lineage rewards
                continue
            # Ensure reward key exists even if no lineage delta occurs
            self.rewards.setdefault(agent_id, 0.0)
            coeff = self._get_role_specific("lineage_reward_coeff", agent_id)
            current = record.get("live_descendants", 0)
            previous = record.get("prev_live_descendants", 0)
            delta = current - previous
            if delta == 0:
                record["prev_live_descendants"] = current
                continue
            reward = coeff * float(delta)
            stats_record = self.agent_stats_live.get(agent_id)
            # Always update lineage_reward_total for visibility, even if reward is zero
            if stats_record is not None:
                stats_record["lineage_reward_total"] = stats_record.get("lineage_reward_total", 0.0) + reward
            if reward != 0:
                self.rewards[agent_id] = self.rewards.get(agent_id, 0.0) + reward
                if stats_record is not None:
                    stats_record["cumulative_reward"] += reward
                if "predator" in agent_id:
                    self.debug_predator_total_reward += float(reward)
                    self.debug_predator_lineage_events += 1
            evt = self.agent_event_log.get(agent_id)
            if evt is not None:
                evt.setdefault("reward_events", []).append(
                    {
                        "t": int(self.current_step),
                        "reproduction_reward": 0.0,
                        "lineage_reward": float(reward),
                        "cumulative_reward": float(stats_record.get("cumulative_reward", 0.0))
                        if stats_record is not None
                        else float(self.rewards[agent_id]),
                    }
                )
            record["prev_live_descendants"] = current

    def _get_fertility_limit(self, agent_id: str):
        caps = getattr(self, "max_fertility_age_config", None)
        if caps is None:
            return None
        if isinstance(caps, dict):
            for prefix, limit in caps.items():
                if agent_id.startswith(prefix):
                    return limit
            return None
        return caps

    def _maybe_mark_fertility_expired(self, agent_id: str, limit=None):
        if limit is None:
            limit = self._get_fertility_limit(agent_id)
        if limit is None:
            return
        if isinstance(limit, (int, float)) and limit < 0:
            return
        age = self.agent_ages.get(agent_id, 0)
        if isinstance(limit, (int, float)):
            if age < limit:
                return
        else:
            return

        record = self.agent_stats_live.get(agent_id)
        if record is not None and record.get("fertility_expired_step") is not None:
            return
        if record is not None:
            record["fertility_expired_step"] = self.current_step
        evt = self.agent_event_log.get(agent_id)
        if evt is not None:
            evt.setdefault("fertility_events", []).append(
                {
                    "t": int(self.current_step),
                    "event": "fertility_expired",
                    "age": int(age),
                }
            )

    def _agent_is_fertile(self, agent_id: str) -> bool:
        limit = self._get_fertility_limit(agent_id)
        if limit is None:
            return True
        if isinstance(limit, (int, float)) and limit < 0:
            return True
        age = self.agent_ages.get(agent_id, 0)
        if not isinstance(limit, (int, float)):
            return True
        if age < limit:
            return True
        self._maybe_mark_fertility_expired(agent_id, limit)
        return False

    def _record_fertility_block_event(self, agent_id: str):
        evt = self.agent_event_log.get(agent_id)
        if evt is None:
            return
        evt.setdefault("fertility_events", []).append(
            {
                "t": int(self.current_step),
                "event": "reproduction_blocked",
                "reason": "fertility_cap",
            }
        )

    def _get_max_age_limit(self, agent_id: str):
        caps = getattr(self, "max_agent_age_config", None)
        if caps is None:
            return None
        if isinstance(caps, dict):
            for prefix, limit in caps.items():
                if agent_id.startswith(prefix):
                    return limit
            return None
        return caps

    def _agent_age_exceeded(self, agent_id: str) -> bool:
        limit = self._get_max_age_limit(agent_id)
        if limit is None:
            return False
        if isinstance(limit, (int, float)) and limit >= 0:
            return self.agent_ages.get(agent_id, 0) >= limit
        return False

    def _get_carcass_only_age_limit(self, agent_id: str):
        caps = getattr(self, "carcass_only_predator_age_config", None)
        if caps is None:
            return None
        if isinstance(caps, dict):
            for prefix, limit in caps.items():
                if agent_id.startswith(prefix):
                    return limit
            return None
        return caps

    def _get_initial_age(self, agent_id: str, *, is_founder: bool) -> int:
        if not is_founder:
            return 0
        if "predator" in agent_id:
            limit = self._get_carcass_only_age_limit(agent_id)
            if isinstance(limit, (int, float)) and limit is not None and limit >= 0:
                return int(limit)
        return 0

    def _predator_requires_carcass_only(self, agent_id: str) -> bool:
        if "predator" not in agent_id:
            return False
        limit = self._get_carcass_only_age_limit(agent_id)
        if limit is None:
            return False
        if isinstance(limit, (int, float)) and limit >= 0:
            return self.agent_ages.get(agent_id, 0) < limit
        return False

    def _record_carcass_only_block(self, predator_id: str, prey_id: str):
        self.rewards[predator_id] = self._get_role_specific("reward_predator_step", predator_id)
        record = self.agent_stats_live.get(predator_id)
        if record is not None:
            record["cumulative_reward"] += self.rewards[predator_id]
            record["carcass_only_blocks"] = record.get("carcass_only_blocks", 0) + 1
            block_count = record["carcass_only_blocks"]
        else:
            block_count = None
        info = self._pending_infos.setdefault(predator_id, {})
        info["carcass_only_live_prey_blocked"] = True
        if block_count is not None:
            info["carcass_only_block_count"] = block_count
        self.carcass_only_live_prey_blocks_predator += 1
        evt = self.agent_event_log.get(predator_id)
        if evt is not None:
            evt.setdefault("diet_events", []).append(
                {
                    "t": int(self.current_step),
                    "event": "carcass_only_block",
                    "prey_id": prey_id,
                    "age": int(self.agent_ages.get(predator_id, 0)),
                }
            )

    def _terminate_agent_due_to_age(self, agent: str):
        if self.terminations.get(agent) or agent not in self.agent_positions:
            return
        layer = 0 if "predator" in agent else 1
        if "prey" in agent:
            self.dead_prey.discard(agent)
            self.active_num_prey = max(self.active_num_prey - 1, 0)
        else:
            self.active_num_predators = max(self.active_num_predators - 1, 0)
        self._handle_lineage_death(agent)
        self.observations[agent] = self._get_observation(agent)
        self.rewards[agent] = self.rewards.get(agent, 0.0)
        self.terminations[agent] = True
        self.truncations[agent] = False
        self.grid_world_state[layer, *self.agent_positions[agent]] = 0
        record = self.agent_stats_live.get(agent)
        if record is not None:
            record["death_cause"] = "max_age"
            record["age_expired_step"] = int(self.current_step)
        info = self._pending_infos.setdefault(agent, {})
        info["terminated_due_to_age"] = True
        evt = self.agent_event_log.get(agent)
        if evt is not None:
            evt.setdefault("lifecycle_events", []).append(
                {
                    "t": int(self.current_step),
                    "event": "max_age_reached",
                    "age": int(self.agent_ages.get(agent, 0)),
                }
            )
        self._finalize_agent_record(agent, cause="max_age")

    def _handle_predator_reproduction(self, agent):
        # Cooldown removed: reproduction now only gated by energy + random chance handled before call.
        # Chance removed as well: reproduction attempts occur whenever energy threshold is met.
        
        self._per_agent_step_deltas.setdefault(
            agent,
            {"decay": 0.0, "move": 0.0, "eat": 0.0, "repro": 0.0},
        )

        if not self._agent_is_fertile(agent):
            self.reproduction_blocked_due_to_fertility_predator += 1
            info = self._pending_infos.setdefault(agent, {})
            info["reproduction_blocked_due_to_fertility"] = True
            info["reproduction_blocked_due_to_fertility_count_predator"] = self.reproduction_blocked_due_to_fertility_predator
            record = self.agent_stats_live.get(agent)
            if record is not None:
                record["fertility_blocked_attempts"] = record.get("fertility_blocked_attempts", 0) + 1
            self._record_fertility_block_event(agent)
            return

        if self.agent_energies[agent] >= self.predator_creation_energy_threshold:
            # Find available new agent ID using pool allocator
            new_agent = self._alloc_new_id("predator")
            if not new_agent:
                msg = (
                    f"[PredPreyGrass] Predator ID pool exhausted "
                    f"at step {self.current_step}. Active predators: {self.active_num_predators}; "
                    f"configured capacity: {self.n_possible_predators}. Exiting."
                )
                print(msg, flush=True)
                raise SystemExit(msg)

            self.agents.append(new_agent)
            self._per_agent_step_deltas[new_agent] = {
                "decay": 0.0,
                "move": 0.0,
                "eat": 0.0,
                "repro": 0.0,
            }
            # And after successful reproduction, store for cooldown
            self.agent_last_reproduction[agent] = self.current_step

            self._register_new_agent(new_agent, parent_agent_id=agent)
            self.agent_live_offspring_ids[agent].append(new_agent)
            self.agent_offspring_counts[agent] += 1
            parent_record = self.agent_stats_live.get(agent)
            if parent_record is not None:
                parent_record["offspring_count"] += 1
                # Log reproduction event for parent
                evt = self.agent_event_log.get(agent)
                if evt is not None:
                    evt.setdefault("reproduction_events", []).append(
                        {"t": int(self.current_step), "child_id": new_agent}
                    )

            # Count successful predator spawns
            self.spawned_predators += 1

            # Spawn position
            occupied_positions = set(self.agent_positions.values())
            new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)
            if new_position is None:
                raise RuntimeError(f"No free spawn position available for predator offspring of {agent}.")

            self.agent_positions[new_agent] = new_position
            self.predator_positions[new_agent] = new_position

            offspring_energy = float(self.initial_energy_predator)
            self.agent_energies[new_agent] = offspring_energy
            self.agent_energies[agent] -= offspring_energy
            self._per_agent_step_deltas[agent]["repro"] = -offspring_energy

            # Write the child's actual starting energy (after reproduction efficiency) into the grid
            self.grid_world_state[0, *new_position] = offspring_energy
            self.grid_world_state[0, *self.agent_positions[agent]] = self.agent_energies[agent]

            self.active_num_predators += 1

            # self.rewards and tracking
            self.rewards[new_agent] = 0
            self.rewards[agent] = self._get_role_specific("reproduction_reward_predator", agent)
            # Debug: successful reproduction reward
            self.debug_predator_total_reward += float(self.rewards[agent])
            self.debug_predator_repro_events += 1

            # cumulative_reward is tracked directly in agent_stats_live
            if parent_record is not None:
                parent_record["cumulative_reward"] += self.rewards[agent]
                # Log direct reproduction reward event
                evt = self.agent_event_log.get(agent)
                if evt is not None:
                    evt.setdefault("reward_events", []).append(
                        {
                            "t": int(self.current_step),
                            "reproduction_reward": float(self.rewards[agent]),
                            "lineage_reward": 0.0,
                            "cumulative_reward": float(parent_record.get("cumulative_reward", 0.0)),
                        }
                    )

            self.observations[new_agent] = self._get_observation(new_agent)
            self.terminations[new_agent] = False
            self.truncations[new_agent] = False

    def _handle_prey_reproduction(self, agent):
        # Cooldown removed: reproduction now only gated by energy + random chance handled before call.
        # Chance removed as well: reproduction attempts occur whenever energy threshold is met.
        # Dead prey (carcass-like) cannot reproduce.
        self._per_agent_step_deltas.setdefault(
            agent,
            {"decay": 0.0, "move": 0.0, "eat": 0.0, "repro": 0.0},
        )
        if agent in self.dead_prey:
            return

        if not self._agent_is_fertile(agent):
            self.reproduction_blocked_due_to_fertility_prey += 1
            info = self._pending_infos.setdefault(agent, {})
            info["reproduction_blocked_due_to_fertility"] = True
            info["reproduction_blocked_due_to_fertility_count_prey"] = self.reproduction_blocked_due_to_fertility_prey
            record = self.agent_stats_live.get(agent)
            if record is not None:
                record["fertility_blocked_attempts"] = record.get("fertility_blocked_attempts", 0) + 1
            self._record_fertility_block_event(agent)
            return

        if self.agent_energies[agent] >= self.prey_creation_energy_threshold:
            # Find available new agent ID using pool allocator
            new_agent = self._alloc_new_id("prey")
            if not new_agent:
                msg = (
                    f"[PredPreyGrass] Prey ID pool exhausted "
                    f"at step {self.current_step}. Active prey: {self.active_num_prey}; "
                    f"configured capacity: {self.n_possible_prey}. Exiting."
                )
                print(msg, flush=True)
                raise SystemExit(msg)

            self.agents.append(new_agent)
            self._per_agent_step_deltas[new_agent] = {
                "decay": 0.0,
                "move": 0.0,
                "eat": 0.0,
                "repro": 0.0,
            }

            # And after successful reproduction, store for cooldown
            self.agent_last_reproduction[agent] = self.current_step

            self._register_new_agent(new_agent, parent_agent_id=agent)
            self.agent_live_offspring_ids[agent].append(new_agent)

            self.agent_offspring_counts[agent] += 1
            parent_record = self.agent_stats_live.get(agent)
            if parent_record is not None:
                parent_record["offspring_count"] += 1
                # Log reproduction event for parent
                evt = self.agent_event_log.get(agent)
                if evt is not None:
                    evt.setdefault("reproduction_events", []).append(
                        {"t": int(self.current_step), "child_id": new_agent}
                    )
            # Count successful prey spawns
            self.spawned_prey += 1

            # Spawn position
            occupied_positions = set(self.agent_positions.values())
            new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)
            if new_position is None:
                raise RuntimeError(f"No free spawn position available for prey offspring of {agent}.")

            self.agent_positions[new_agent] = new_position
            self.prey_positions[new_agent] = new_position

            offspring_energy = float(self.initial_energy_prey)
            self.agent_energies[new_agent] = offspring_energy
            self.agent_energies[agent] -= offspring_energy
            self._per_agent_step_deltas[agent]["repro"] = -offspring_energy

            # Write the child's actual starting energy (after reproduction efficiency) into the grid
            self.grid_world_state[1, *new_position] = offspring_energy
            self.grid_world_state[1, *self.agent_positions[agent]] = self.agent_energies[agent]

            self.active_num_prey += 1

            # self.rewards and tracking
            self.rewards[new_agent] = 0
            self.rewards[agent] = self._get_role_specific("reproduction_reward_prey", agent)
            # cumulative_reward is tracked directly in agent_stats_live
            if parent_record is not None:
                parent_record["cumulative_reward"] += self.rewards[agent]
                # Log direct reproduction reward event
                evt = self.agent_event_log.get(agent)
                if evt is not None:
                    evt.setdefault("reward_events", []).append(
                        {
                            "t": int(self.current_step),
                            "reproduction_reward": float(self.rewards[agent]),
                            "lineage_reward": 0.0,
                            "cumulative_reward": float(parent_record.get("cumulative_reward", 0.0)),
                        }
                    )

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
            "agent_genomes": {
                aid: genome.to_dict() if genome is not None else None
                for aid, genome in self.agent_genomes.items()
            },
            "agent_offspring_counts": self.agent_offspring_counts.copy(),
            "agent_live_offspring_ids": {
                aid: list(ids) for aid, ids in self.agent_live_offspring_ids.items()
            },
            "lineage_tracker": {
                aid: {
                    "parent_id": entry.get("parent_id"),
                    "children_ids": list(entry.get("children_ids", set())),
                    "live_descendants": entry.get("live_descendants", 0),
                    "prev_live_descendants": entry.get("prev_live_descendants", 0),
                    "is_alive_descendant": entry.get("is_alive_descendant", False),
                }
                for aid, entry in self.lineage_tracker.items()
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
        # Restore lineage tracker (children lists back to sets)
        self.lineage_tracker = {}
        for aid, entry in snapshot.get("lineage_tracker", {}).items():
            self.lineage_tracker[aid] = {
                "parent_id": entry.get("parent_id"),
                "children_ids": set(entry.get("children_ids", [])),
                "live_descendants": entry.get("live_descendants", 0),
                "prev_live_descendants": entry.get("prev_live_descendants", 0),
                "is_alive_descendant": entry.get("is_alive_descendant", False),
            }
        self.agent_ages = snapshot["agent_ages"].copy()
        self.death_cause_prey = snapshot["death_cause_prey"].copy()
        self.agent_last_reproduction = snapshot["agent_last_reproduction"].copy()
        self.agent_parents = snapshot.get("agent_parents", {}).copy()
        active_genome_traits = set(Genome.__dataclass_fields__)
        self.agent_genomes = {
            aid: Genome(**{trait: genome_data.get(trait, 1.0) for trait in active_genome_traits})
            for aid, genome_data in snapshot.get("agent_genomes", {}).items()
            if genome_data is not None
        }
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
        for i in range(self.n_possible_predators):
            agent_ids.append(f"predator_{i}")
        for i in range(self.n_possible_prey):
            agent_ids.append(f"prey_{i}")
        return agent_ids 

    def _build_observation_space(self, agent_id):
        """
        Build the observation space for a specific agent.
        """
        if "predator" in agent_id:
            predator_shape = (self.num_obs_channels, self.predator_obs_range, self.predator_obs_range)
            obs_space = gymnasium.spaces.Box(low=0, high=100.0, shape=predator_shape, dtype=np.float32)

        elif "prey" in agent_id:
            prey_shape = (self.num_obs_channels, self.prey_obs_range, self.prey_obs_range)
            obs_space = gymnasium.spaces.Box(low=0, high=100.0, shape=prey_shape, dtype=np.float32)

        else:
            raise ValueError(f"Unknown agent type in ID: {agent_id}")

        return obs_space

    def _build_action_space(self, agent_id):
        """
        Build the action space for a specific agent.
        """
        if "predator" in agent_id or "prey" in agent_id:
            action_space = gymnasium.spaces.Discrete(self.action_range**2)
        else:
            raise ValueError(f"Unknown agent type in ID: {agent_id}")

        return action_space

    # -------- Lineage tracking helpers --------
    def _ensure_lineage_entry(self, agent_id: str, parent_agent_id: Optional[str]):
        if agent_id in self.lineage_tracker:
            entry = self.lineage_tracker[agent_id]
            # Parent might be None for founders; only update if provided
            if parent_agent_id is not None:
                entry["parent_id"] = parent_agent_id
            return entry
        entry = {
            "parent_id": parent_agent_id,
            "children_ids": set(),
            "live_descendants": 0,
            "prev_live_descendants": 0,
            "is_alive_descendant": False,
        }
        self.lineage_tracker[agent_id] = entry
        return entry

    def _propagate_lineage_delta(self, ancestor_id: Optional[str], delta: int):
        if delta == 0:
            return
        current = ancestor_id
        while current is not None:
            ancestor = self.lineage_tracker.get(current)
            if ancestor is None:
                break
            ancestor["live_descendants"] = ancestor.get("live_descendants", 0) + delta
            current = ancestor.get("parent_id")

    def _set_lineage_alive_flag(self, agent_id: str, alive: bool):
        entry = self.lineage_tracker.get(agent_id)
        if entry is None:
            return
        if entry.get("is_alive_descendant", False) == alive:
            return
        entry["is_alive_descendant"] = alive
        delta = 1 if alive else -1
        self._propagate_lineage_delta(entry.get("parent_id"), delta)

    def _handle_lineage_birth(self, agent_id: str, parent_agent_id: Optional[str]):
        self._ensure_lineage_entry(agent_id, parent_agent_id)
        if parent_agent_id is not None and parent_agent_id in self.lineage_tracker:
            self.lineage_tracker[parent_agent_id]["children_ids"].add(agent_id)
        self._set_lineage_alive_flag(agent_id, True)

    def _handle_lineage_death(self, agent_id: str):
        if agent_id not in self.lineage_tracker:
            return
        self._set_lineage_alive_flag(agent_id, False)

    def _register_new_agent(self, agent_id: str, parent_agent_id: Optional[str] = None, *, is_founder: bool = False):
        if agent_id in self.agent_stats_live or agent_id in self.agent_stats_completed:
            raise ValueError(f"Agent id {agent_id} already registered in this episode.")

        self.used_agent_ids.add(agent_id)
        self.agent_ages[agent_id] = self._get_initial_age(agent_id, is_founder=is_founder)
        self.agent_offspring_counts[agent_id] = 0
        self.agent_live_offspring_ids[agent_id] = []
        self.agent_parents[agent_id] = parent_agent_id
        genome = self._inherit_genome(agent_id, parent_agent_id, is_founder=is_founder)
        if genome is not None:
            self.agent_genomes[agent_id] = genome
            genome_dict = genome.to_dict()
        else:
            genome_dict = None
        # Initialize event-log entry
        self.agent_event_log[agent_id] = {
            "agent_id": agent_id,
            "birth_step": self.current_step,
            "death_step": None,
            "parent_id": parent_agent_id,
            "death_cause": None,
            "eating_events": [],
            "reproduction_events": [],
            "reward_events": [],
            "fertility_events": [],
            "diet_events": [],
            "lifecycle_events": [],
            "genome": genome_dict,
        }
        self.agent_stats_live[agent_id] = {
            "agent_id": agent_id,
            "birth_step": self.current_step,
            "parent": parent_agent_id,
            "offspring_count": 0,
            "offspring_ids": self.agent_live_offspring_ids[agent_id],
            "distance_traveled": 0.0,
            "movement_energy_spent": 0.0,
            "times_ate": 0,
            "energy_gained": 0.0,
            "avg_energy_sum": 0.0,
            "avg_energy_steps": 0,
            "cumulative_reward": 0.0,
            "lineage_reward_total": 0.0,
            "policy_group": self._policy_group(agent_id),
            "genome": genome_dict,
            "death_step": None,
            "death_cause": None,
            "avg_energy": 0.0,
            "max_fertility_age": self._get_fertility_limit(agent_id),
            "fertility_expired_step": None,
            "fertility_blocked_attempts": 0,
            "max_age": self._get_max_age_limit(agent_id),
            "age_expired_step": None,
            "carcass_only_blocks": 0,
        }

        # Initialize lineage tracking after stats to ensure helper has context
        self._handle_lineage_birth(agent_id, parent_agent_id)

        return self.agent_stats_live[agent_id]

    def _copy_agent_record(self, record: dict) -> dict:
        copied = record.copy()
        copied["offspring_ids"] = list(record.get("offspring_ids", []))
        return copied

    def _finalize_agent_record(self, agent_id: str, cause: Optional[str] = None):
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
        record["lineage_reward_total"] = record.get("lineage_reward_total", 0.0)
        final_total = record.get("cumulative_reward", 0.0)
        # Ensure callbacks can access the exact per-agent totals via infos regardless of termination path.
        info = self._pending_infos.setdefault(agent_id, {})
        # Avoid overwriting if a branch (e.g., time-limit) already attached the same value.
        info.setdefault("final_cumulative_reward", final_total)
        birth_step = record.get("birth_step", self.current_step)
        death_step = record.get("death_step", self.current_step)
        lifetime = max(int(death_step - birth_step), 0)
        info.setdefault("lifetime_steps", lifetime)
        info.setdefault("parent_id", record.get("parent"))
        # Mirror core lifecycle fields into the event log record for convenience
        evt = self.agent_event_log.get(agent_id)
        if evt is not None:
            evt["death_step"] = death_step
            evt["death_cause"] = record.get("death_cause")
            evt.setdefault("parent_id", record.get("parent"))
        if "predator" in agent_id:
            info.setdefault("species", "predator")
        elif "prey" in agent_id:
            info.setdefault("species", "prey")
        self.agent_stats_completed[agent_id] = record
        self.agent_live_offspring_ids.pop(agent_id, None)

    def export_agent_event_log(self, path: str) -> None:
        """Export the per-agent event log to a JSON file.

        Designed for evaluation scripts. Writes a dict keyed by agent_id,
        where each value is that agent's event-log record.
        """
        import json

        def _convert(obj):
            if isinstance(obj, (int, float, str)) or obj is None:
                return obj
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            try:
                return obj.item()
            except Exception:
                return str(obj)

        payload = {aid: _convert(rec) for aid, rec in self.agent_event_log.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _attach_final_cumulative_rewards_for_live_agents(self):
        """Ensure all agents that remain alive at episode end expose their cumulative rewards via infos."""
        for agent_id, record in self.agent_stats_live.items():
            info = self._pending_infos.setdefault(agent_id, {})
            info.setdefault("final_cumulative_reward", record.get("cumulative_reward", 0.0))
            birth_step = record.get("birth_step", self.current_step)
            lifetime = max(int(self.current_step - birth_step), 0)
            info.setdefault("lifetime_steps", lifetime)
            info.setdefault("parent_id", record.get("parent"))
            if "predator" in agent_id:
                info.setdefault("species", "predator")
            elif "prey" in agent_id:
                info.setdefault("species", "prey")

    def _build_episode_training_metrics(self) -> dict[str, float]:
        """Build lightweight scalar genome/behavior summaries for RLlib callbacks."""
        metrics = {}
        threshold = float(self.speed_distance_threshold)
        grouped_records = {"predator": [], "prey": []}
        for agent_id, record in self._iter_all_agent_records():
            if agent_id.startswith("predator"):
                grouped_records["predator"].append(record)
            elif agent_id.startswith("prey"):
                grouped_records["prey"].append(record)

        for species, records in grouped_records.items():
            speeds = [
                float(record["genome"]["speed"])
                for record in records
                if isinstance(record.get("genome"), dict) and "speed" in record["genome"]
            ]
            if speeds:
                p25, p50, p75 = np.percentile(speeds, [25, 50, 75])
                metrics[f"{species}_speed_mean"] = float(np.mean(speeds))
                metrics[f"{species}_speed_p25"] = float(p25)
                metrics[f"{species}_speed_p50"] = float(p50)
                metrics[f"{species}_speed_p75"] = float(p75)
                metrics[f"{species}_fraction_fast"] = float(np.mean(np.array(speeds) >= threshold))
            else:
                metrics[f"{species}_fraction_fast"] = 0.0

            if records:
                metrics[f"{species}_distance_traveled_mean"] = float(
                    np.mean([float(record.get("distance_traveled", 0.0)) for record in records])
                )
                metrics[f"{species}_movement_energy_spent_mean"] = float(
                    np.mean([float(record.get("movement_energy_spent", 0.0)) for record in records])
                )
                metrics[f"{species}_offspring_count_mean"] = float(
                    np.mean([float(record.get("offspring_count", 0.0)) for record in records])
                )
                metrics[f"{species}_agent_count"] = float(len(records))
            else:
                metrics[f"{species}_distance_traveled_mean"] = 0.0
                metrics[f"{species}_movement_energy_spent_mean"] = 0.0
                metrics[f"{species}_offspring_count_mean"] = 0.0
                metrics[f"{species}_agent_count"] = 0.0

        return metrics

    def _attach_episode_training_metrics(self):
        info = self._pending_infos.setdefault("__all__", {})
        info["training_metrics"] = self._build_episode_training_metrics()

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
        """Returns a dict of total offspring counts by role."""
        counts = {"predator": 0, "prey": 0}
        for _, stats in self._iter_all_agent_records():
            group = stats.get("policy_group")
            if group in counts:
                counts[group] += stats.get("offspring_count", 0)
        return counts

    def get_total_energy_spent_by_type(self):
        """Returns a dict of total energy spent by role."""
        energy_spent = {"predator": 0.0, "prey": 0.0}
        for _, stats in self._iter_all_agent_records():
            group = stats.get("policy_group")
            if group in energy_spent:
                energy_spent[group] += stats.get("energy_spent", 0.0)
        return energy_spent

    def _get_role_specific(self, key: str, agent_id: str):
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
        Place and create all entities (predators, prey, grass) into the open grid world state.
        """
        predator_list, prey_list, predator_positions, prey_positions, grass_positions = self._sample_agent_and_grass_positions()
        self._place_predators(predator_list, predator_positions)
        self._place_prey(prey_list, prey_positions)
        self._place_grass(grass_positions)

    # -------- Reset other entity placement methods --------
    def _sample_agent_and_grass_positions(self):
        """
        Sample free positions for all entities and return lists for placement.
        Returns:
            predator_list, prey_list, predator_positions, prey_positions, grass_positions
        """
        num_grid_cells = self.grid_size * self.grid_size
        num_agents_and_grass = len(self.agents) + len(self.grass_agents)

        chosen_grid_indices = self.rng.choice(num_grid_cells, size=num_agents_and_grass, replace=False)
        chosen_positions = [(i // self.grid_size, i % self.grid_size) for i in chosen_grid_indices]

        predator_list = [str(a) for a in self.agents if "predator" in str(a)]
        prey_list = [str(a) for a in self.agents if "prey" in str(a)]

        predator_positions = chosen_positions[: len(predator_list)]
        prey_positions = chosen_positions[len(predator_list) : len(predator_list) + len(prey_list)]
        grass_positions = chosen_positions[len(predator_list) + len(prey_list) :]

        return predator_list, prey_list, predator_positions, prey_positions, grass_positions

    #-------- Placement method for predators --------
    def _place_predators(self, predator_list, predator_positions):
        self.predator_positions = {}
        for i, agent in enumerate(predator_list):
            pos = predator_positions[i]
            self.agent_positions[agent] = pos
            self.predator_positions[agent] = pos
            self.agent_energies[agent] = self.initial_energy_predator
            self.grid_world_state[0, *pos] = self.initial_energy_predator

    #-------- Placement method for prey --------
    def _place_prey(self, prey_list, prey_positions):
        self.prey_positions = {}
        for i, agent in enumerate(prey_list):
            pos = prey_positions[i]
            self.agent_positions[agent] = pos
            self.prey_positions[agent] = pos
            self.agent_energies[agent] = self.initial_energy_prey
            self.grid_world_state[1, *pos] = self.initial_energy_prey

    #-------- Placement method for grass --------
    def _place_grass(self, grass_positions):
        self.grass_positions = {}
        self.grass_energies = {}
        for i, grass in enumerate(self.grass_agents):
            pos = grass_positions[i]
            self.grass_positions[grass] = pos
            self.grass_energies[grass] = self.initial_energy_grass
            self.grid_world_state[2, *pos] = self.initial_energy_grass
