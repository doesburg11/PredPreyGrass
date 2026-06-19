"""
Predator-Prey Grass RLlib Environment

Additions:
        - Static wall obstacles: `num_walls` random cells sampled at reset (default 20).
            * Stored in `self.wall_positions` (set of (x,y)).
            * Painted into observation channel 0 (binary 1=wall).
            * Agents (predator, prey, grass placement) avoid wall cells at reset.
            * Movement into a wall cell is disallowed (agent stays in place, 
              still pays movement energy cost as computed for attempted move).
"""
# external libraries (Ray optional for lightweight diagnostics)
from typing import TYPE_CHECKING

import gymnasium
import math
import numpy as np


class _FallbackMultiAgentEnv:
    """Minimal stub for wall/placement local tests."""

    def __init__(self) -> None:
        pass


if TYPE_CHECKING:

    class MultiAgentEnv:
        """Static-analysis stub; runtime uses Ray's MultiAgentEnv when available."""

        possible_agents: list[str]
        agents: list[str]

        def __init__(self) -> None:
            pass

        def reset(self, *, seed=None, options=None) -> object:
            pass

else:
    try:
        from ray.rllib.env.multi_agent_env import MultiAgentEnv
    except Exception:  # pragma: no cover

        MultiAgentEnv = _FallbackMultiAgentEnv


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            raise ValueError("Environment config must be provided explicitly.")
        self.config = config
        self._initialize_from_config()  # import config variables

        self.possible_agents: list[str] = self._build_possible_agent_ids()

        self.observation_spaces = {agent_id: self._build_observation_space(agent_id) for agent_id in self.possible_agents}

        self.action_spaces = {agent_id: self._build_action_space(agent_id) for agent_id in self.possible_agents}

    def close(self) -> None:
        """Release environment-owned resources."""
        pass

    def _initialize_from_config(self):
        config = self.config
        self.debug_mode = config.get("debug_mode", False)
        self.verbose_movement = config.get("verbose_movement", self.debug_mode)
        self.verbose_decay = config.get("verbose_decay", self.debug_mode)
        self.verbose_reproduction = config.get("verbose_reproduction", self.debug_mode)
        self.verbose_engagement = config.get("verbose_engagement", self.debug_mode)

        self.max_steps = config.get("max_steps", 10000)
        self.base_seed = config.get("seed", 42)
        self.deterministic_reset_sequence = bool(config.get("deterministic_reset_sequence", False))
        self._reset_counter = 0
        self.rng = np.random.default_rng(self.base_seed)
        # Malthusian scaffold: episode-end fitness (phi) and allocation (mu) update.
        self.enable_malthusian_update = bool(config.get("enable_malthusian_update", True))
        self.malthusian_eta = float(config.get("malthusian_eta", 0.2))
        self.malthusian_mu_learning_rate = float(config.get("malthusian_mu_learning_rate", self.malthusian_eta))
        self.malthusian_mu_entropy_coeff = float(config.get("malthusian_mu_entropy_coeff", 0.0))
        self.malthusian_mu_floor = float(config.get("malthusian_mu_floor", 0.0))
        self.malthusian_replication_mode = str(
            config.get("malthusian_replication_mode", "generalized")
        ).lower()
        if self.malthusian_replication_mode not in ("generalized", "strict"):
            raise ValueError(
                "malthusian_replication_mode must be 'generalized' or 'strict'"
            )

        default_mu_update = "multiplicative" if self.malthusian_replication_mode == "strict" else "zscore_logit"
        self.malthusian_mu_update = str(
            config.get("malthusian_mu_update", default_mu_update)
        ).lower()
        if self.malthusian_mu_update not in ("zscore_logit", "multiplicative"):
            raise ValueError(
                "malthusian_mu_update must be 'zscore_logit' or 'multiplicative'"
            )

        default_reproduction = self.malthusian_replication_mode != "strict"
        self.enable_within_episode_reproduction = bool(
            config.get("enable_within_episode_reproduction", default_reproduction)
        )
        default_phi_weights = {
            "offspring": 2.0,
            "survival": 1.0,
            "foraging": 0.5,
            "energy": 0.25,
            "death": -1.0,
            "reward": 0.0,
        }
        cfg_phi_weights = config.get("malthusian_phi_weights", {})
        self.malthusian_phi_weights = default_phi_weights.copy()
        if isinstance(cfg_phi_weights, dict):
            for key in default_phi_weights:
                if key in cfg_phi_weights:
                    self.malthusian_phi_weights[key] = float(cfg_phi_weights[key])
        self.malthusian_phi_clip = config.get("malthusian_phi_clip", None)
        if self.malthusian_phi_clip is not None:
            self.malthusian_phi_clip = float(self.malthusian_phi_clip)

        # Rewards dictionaries
        self.reward_predator_catch_prey_config = config.get("reward_predator_catch_prey", 0.0)
        self.reward_prey_eat_grass_config = config.get("reward_prey_eat_grass", 0.0)

        self.reward_predator_step_config = config.get("reward_predator_step", 0.0)
        self.reward_prey_step_config = config.get("reward_prey_step", 0.0)
        self.penalty_prey_caught_config = config.get("penalty_prey_caught", 0.0)
        self.reproduction_reward_predator_config = config.get("reproduction_reward_predator", 10.0)
        self.reproduction_reward_prey_config = config.get("reproduction_reward_prey", 10.0)

        # Energy settings
        self.energy_loss_per_step_predator = config.get("energy_loss_per_step_predator", 0.15)
        self.energy_loss_per_step_prey = config.get("energy_loss_per_step_prey", 0.05)
        self.predator_creation_energy_threshold = config.get("predator_creation_energy_threshold", 12.0)
        self.prey_creation_energy_threshold = config.get("prey_creation_energy_threshold", 8.0)

        # Learning agents
        self.n_possible_type_1_predators = config.get("n_possible_type_1_predators", 25)
        self.n_possible_type_2_predators = config.get("n_possible_type_2_predators", 25)
        self.n_possible_type_1_prey = config.get("n_possible_type_1_prey", 25)
        self.n_possible_type_2_prey = config.get("n_possible_type_2_prey", 25)

        self.n_initial_active_type_1_predator = config.get("n_initial_active_type_1_predator", 6)
        self.n_initial_active_type_2_predator = config.get("n_initial_active_type_2_predator", 0)
        self.n_initial_active_type_1_prey = config.get("n_initial_active_type_1_prey", 8)
        self.n_initial_active_type_2_prey = config.get("n_initial_active_type_2_prey", 0)
        # If enabled, each configured species (n_possible > 0) gets at least this many
        # active individuals at reset to avoid accidental permanent extinction.
        self.enforce_min_initial_mass_per_species = config.get("enforce_min_initial_mass_per_species", True)
        self.min_initial_mass_per_species = max(0, int(config.get("min_initial_mass_per_species", 1)))

        self.initial_energy_predator = config.get("initial_energy_predator", 5.0)
        self.initial_energy_prey = config.get("initial_energy_prey", 3.0)

        # Grid and Observation Settings
        self.grid_size = config.get("grid_size", 10)
        self.num_obs_channels = config.get("num_obs_channels", 4)
        self.predator_obs_range = config.get("predator_obs_range", 7)
        self.prey_obs_range = config.get("prey_obs_range", 5)
        # Optional extra observation channel (appended as last channel) showing
        # line-of-sight visibility (1 = visible, 0 = occluded by at least one wall).
        # When disabled, observation tensors retain their original channel count.
        self.include_visibility_channel = config.get("include_visibility_channel", False)
        # Movement restriction: if True, agents may only move to target cells with unobstructed LOS (no wall between current and target).
        self.respect_los_for_movement = config.get("respect_los_for_movement", False)
        # If True, dynamic observation channels (predators/prey/grass) are masked so that
        # entities behind walls (no line-of-sight) appear as 0 even if within square range.
        # Works independently of include_visibility_channel; if that is False we still mask
        # but do not append the visibility channel itself.
        self.mask_observation_with_visibility = config.get("mask_observation_with_visibility", False)

        # Grass settings
        self.initial_num_grass = config.get("initial_num_grass", 25)
        self.initial_energy_grass = config.get("initial_energy_grass", 2.0)
        self.energy_gain_per_step_grass = config.get("energy_gain_per_step_grass", 0.2)
        # Walls (static obstacles)
        self.num_walls = config.get("num_walls", 20)
        # New: wall placement mode: 'random' (default) or 'manual'.
        # When 'manual', positions come from manual_wall_positions (list of (x,y)).
        self.wall_placement_mode = config.get("wall_placement_mode", "random")
        self.manual_wall_positions = config.get("manual_wall_positions", None)
        self.wall_positions = set()

        # Mutation
        self.mutation_rate_predator = config.get("mutation_rate_predator", 0.1)
        self.mutation_rate_prey = config.get("mutation_rate_prey", 0.1)

        # Action range and movement mapping
        self.type_1_act_range = config.get("type_1_action_range", 3)
        self.type_2_act_range = config.get("type_2_action_range", 5)
        # Persistent across episodes (same env instance).
        self.mu_by_species = {}
        self.last_phi_by_species = {}
        self.last_phi_components = {}
        self.last_episode_summary = {}

    def _get_effective_initial_active_count(self, agent_type: str, type_id: int) -> int:
        """
        Return effective initial active count for (agent_type, type_id), optionally
        enforcing a minimum mass for species with non-zero configured capacity.
        """
        key = f"n_initial_active_type_{type_id}_{agent_type}"
        configured_count = max(0, int(self.config.get(key, 0)))

        if agent_type == "predator":
            n_possible = self.n_possible_type_1_predators if type_id == 1 else self.n_possible_type_2_predators
        else:
            n_possible = self.n_possible_type_1_prey if type_id == 1 else self.n_possible_type_2_prey

        effective = configured_count
        if self.enforce_min_initial_mass_per_species and n_possible > 0:
            effective = max(effective, self.min_initial_mass_per_species)

        return min(effective, max(0, int(n_possible)))

    def _tracked_species(self):
        species = []
        if self.n_possible_type_1_predators > 0:
            species.append("type_1_predator")
        if self.n_possible_type_2_predators > 0:
            species.append("type_2_predator")
        if self.n_possible_type_1_prey > 0:
            species.append("type_1_prey")
        if self.n_possible_type_2_prey > 0:
            species.append("type_2_prey")
        return species

    def _normalize_mu_vector(self, vec):
        arr = np.asarray(vec, dtype=np.float64)
        arr = np.maximum(arr, 0.0)
        total = float(arr.sum())
        if total <= 0.0:
            arr = np.ones_like(arr, dtype=np.float64)
            total = float(arr.sum())
        arr /= total

        n = arr.size
        if n == 0:
            return arr
        floor = max(0.0, min(self.malthusian_mu_floor, 1.0 / n))
        if floor > 0.0:
            arr = np.maximum(arr, floor)
            arr /= float(arr.sum())
        return arr

    @staticmethod
    def _softmax(vec):
        arr = np.asarray(vec, dtype=np.float64)
        if arr.size == 0:
            return arr
        arr = arr - float(arr.max())
        exp = np.exp(arr)
        total = float(exp.sum())
        if total <= 0.0:
            return np.ones_like(exp, dtype=np.float64) / float(exp.size)
        return exp / total

    def _initialize_or_align_malthusian_state(self):
        island_ids = sorted(self.island_id_to_cells.keys())
        self.island_ids = island_ids
        if not island_ids:
            self.mu_by_species = {}
            self.last_phi_by_species = {}
            self.last_phi_components = {}
            return

        uniform = 1.0 / len(island_ids)
        for species in self._tracked_species():
            cur = self.mu_by_species.get(species, {})
            cur_logits = self.mu_logits_by_species.get(species, {})
            if set(cur.keys()) != set(island_ids) or set(cur_logits.keys()) != set(island_ids):
                self.mu_logits_by_species[species] = {iid: 0.0 for iid in island_ids}
                self.mu_by_species[species] = {iid: uniform for iid in island_ids}
            else:
                logits = np.asarray([float(cur_logits[iid]) for iid in island_ids], dtype=np.float64)
                mu = self._softmax(logits)
                mu = self._normalize_mu_vector(mu)
                self.mu_logits_by_species[species] = {iid: float(logits[k]) for k, iid in enumerate(island_ids)}
                self.mu_by_species[species] = {iid: float(mu[k]) for k, iid in enumerate(island_ids)}

    @staticmethod
    def _species_from_agent_id(agent_id: str) -> str:
        return "_".join(agent_id.split("_")[:3])

    def _initial_energy_for_species(self, species: str) -> float:
        return float(self.initial_energy_predator) if "predator" in species else float(self.initial_energy_prey)

    def _record_lifetime_steps(self, record: dict) -> int:
        birth_step = int(record.get("birth_step", 0))
        death_step = record.get("death_step")
        end_step = int(death_step) if death_step is not None else int(self.current_step)
        return max(0, end_step - birth_step + 1)

    def _sample_species_counts_over_islands(self, species: str, n_agents: int, available_cells_by_island):
        """
        Sample how many agents of a species start on each island using mu, while
        respecting per-island remaining capacity.
        """
        island_ids = self.island_ids
        if n_agents <= 0 or not island_ids:
            return {iid: 0 for iid in island_ids}

        mu_sp = self.mu_by_species.get(species, {iid: 1.0 / len(island_ids) for iid in island_ids})
        probs = self._normalize_mu_vector([float(mu_sp.get(iid, 0.0)) for iid in island_ids])
        sampled = self.rng.multinomial(n_agents, probs)
        capacities = np.asarray([len(available_cells_by_island[iid]) for iid in island_ids], dtype=np.int64)
        assigned = np.minimum(sampled, capacities)

        deficit = int(n_agents - assigned.sum())
        while deficit > 0:
            remaining = capacities - assigned
            candidates = np.where(remaining > 0)[0]
            if candidates.size == 0:
                raise ValueError(
                    f"Not enough free cells to place species '{species}' "
                    f"({n_agents} agents requested)."
                )
            cand_probs = probs[candidates]
            total = float(cand_probs.sum())
            if total <= 0.0:
                cand_probs = np.ones_like(cand_probs, dtype=np.float64) / float(candidates.size)
            else:
                cand_probs = cand_probs / total
            chosen_idx = int(self.rng.choice(candidates, p=cand_probs))
            assigned[chosen_idx] += 1
            deficit -= 1

        return {iid: int(assigned[k]) for k, iid in enumerate(island_ids)}

    def _compute_phi_from_episode(self):
        island_ids = sorted(self.island_id_to_cells.keys())
        species_list = self._tracked_species()
        phi = {sp: {iid: 0.0 for iid in island_ids} for sp in species_list}
        counts = {sp: {iid: 0 for iid in island_ids} for sp in species_list}
        component_keys = ("offspring", "survival", "foraging", "energy", "death", "reward")
        component_sums = {
            sp: {iid: {k: 0.0 for k in component_keys} for iid in island_ids}
            for sp in species_list
        }

        # Completed agents + currently alive agents (avoid double counting dead records).
        records: list[tuple[str | None, dict]] = [(None, rec) for rec in self.death_agents_stats.values()]
        for agent_id in self.agents:
            uid = self.unique_agents.get(agent_id)
            if uid is None:
                continue
            rec = self.unique_agent_stats.get(uid)
            if rec is not None:
                records.append((agent_id, rec))

        for agent_id, rec in records:
            sp = rec.get("policy_group")
            iid = rec.get("spawn_island")
            if not isinstance(sp, str) or sp not in phi or iid not in phi[sp]:
                continue

            if self.malthusian_replication_mode == "strict":
                reward = float(rec.get("cumulative_reward", 0.0))
                phi[sp][iid] += reward
                counts[sp][iid] += 1
                component_sums[sp][iid]["reward"] += reward
                continue

            lifetime_steps = self._record_lifetime_steps(rec)
            survival = min(1.0, float(lifetime_steps) / max(1, int(self.max_steps)))
            death = 1.0 if rec.get("death_step") is not None else 0.0
            offspring = float(rec.get("offspring_count", 0.0))
            foraging = float(rec.get("times_ate", 0.0))
            reward = float(rec.get("cumulative_reward", 0.0))

            initial_energy = self._initial_energy_for_species(sp)
            final_energy = rec.get("final_energy")
            if final_energy is None and agent_id is not None and agent_id in self.agent_energies:
                final_energy = float(self.agent_energies[agent_id])
            elif final_energy is None:
                final_energy = (
                    initial_energy
                    + float(rec.get("energy_gained", 0.0))
                    - float(rec.get("energy_spent", 0.0))
                )
            energy = (float(final_energy) - initial_energy) / max(1e-8, abs(initial_energy))

            components = {
                "offspring": offspring,
                "survival": survival,
                "foraging": foraging,
                "energy": energy,
                "death": death,
                "reward": reward,
            }

            score = 0.0
            for key in component_keys:
                score += self.malthusian_phi_weights.get(key, 0.0) * components[key]
                component_sums[sp][iid][key] += components[key]

            if self.malthusian_phi_clip is not None:
                score = float(np.clip(score, -self.malthusian_phi_clip, self.malthusian_phi_clip))

            phi[sp][iid] += score
            counts[sp][iid] += 1

        component_means = {
            sp: {iid: {k: 0.0 for k in component_keys} for iid in island_ids}
            for sp in species_list
        }
        for sp in species_list:
            for iid in island_ids:
                c = counts[sp][iid]
                phi[sp][iid] = float(phi[sp][iid] / c) if c > 0 else 0.0
                if c > 0:
                    for key in component_keys:
                        component_means[sp][iid][key] = float(component_sums[sp][iid][key] / c)

        return phi, counts, component_means

    def _update_mu_from_phi(self, phi_by_species):
        island_ids = sorted(self.island_id_to_cells.keys())
        if not island_ids:
            return

        for sp in self._tracked_species():
            phi_vec = np.asarray([float(phi_by_species.get(sp, {}).get(iid, 0.0)) for iid in island_ids], dtype=np.float64)
            prev_logits = self.mu_logits_by_species.get(sp, {iid: 0.0 for iid in island_ids})
            prev_logit_vec = np.asarray([float(prev_logits.get(iid, 0.0)) for iid in island_ids], dtype=np.float64)
            prev_mu_vec = self._softmax(prev_logit_vec)

            if self.malthusian_mu_update == "multiplicative":
                updated_logits = prev_logit_vec + self.malthusian_mu_learning_rate * phi_vec
            else:
                mean = float(phi_vec.mean()) if phi_vec.size else 0.0
                std = float(phi_vec.std()) if phi_vec.size else 0.0
                z = (phi_vec - mean) / (std + 1e-8)
                updated_logits = prev_logit_vec + self.malthusian_mu_learning_rate * z

            if self.malthusian_mu_entropy_coeff != 0.0:
                updated_logits -= self.malthusian_mu_entropy_coeff * np.log(np.maximum(prev_mu_vec, 1e-12))

            updated_mu = self._softmax(updated_logits)
            updated_mu = self._normalize_mu_vector(updated_mu)
            self.mu_logits_by_species[sp] = {iid: float(updated_logits[k]) for k, iid in enumerate(island_ids)}
            self.mu_by_species[sp] = {iid: float(updated_mu[k]) for k, iid in enumerate(island_ids)}

    def _finalize_malthusian_episode(self):
        if not self.enable_malthusian_update:
            return
        if self._malthusian_finalized_at_step == self.current_step:
            return
        self._malthusian_finalized_at_step = self.current_step

        phi, counts, component_means = self._compute_phi_from_episode()
        self.last_phi_by_species = phi
        self.last_phi_components = component_means
        self._update_mu_from_phi(phi)
        self.last_episode_summary = {
            "episode_step": int(self.current_step),
            "malthusian_replication_mode": self.malthusian_replication_mode,
            "malthusian_mu_update": self.malthusian_mu_update,
            "malthusian_mu_learning_rate": self.malthusian_mu_learning_rate,
            "malthusian_mu_entropy_coeff": self.malthusian_mu_entropy_coeff,
            "phi_by_species": phi,
            "phi_components_by_species": component_means,
            "phi_weights": dict(self.malthusian_phi_weights),
            "mu_by_species": {sp: dict(self.mu_by_species.get(sp, {})) for sp in self._tracked_species()},
            "mu_logits_by_species": {sp: dict(self.mu_logits_by_species.get(sp, {})) for sp in self._tracked_species()},
            "counts_by_species": counts,
        }

    def _init_reset_variables(self, seed):
        # Agent tracking
        self.current_step = 0
        if seed is None and self.deterministic_reset_sequence:
            seed = int(self.base_seed) + self._reset_counter
        self._reset_counter += 1
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
        self._malthusian_finalized_at_step = None
        self.mu_logits_by_species = {}
        # RLlib new API stack requirement: a given agent_id may not reappear
        # after it has terminated within the same episode.
        self._used_agent_ids_this_episode = set()

        # aggregates per step
        self.active_num_predators = 0
        self.active_num_prey = 0

        self.agents: list[str] = []
        # create active agents list based on config
        for agent_type in ["predator", "prey"]:
            for type in [1, 2]:
                count = self._get_effective_initial_active_count(agent_type, type)
                for i in range(count):
                    agent_id = f"type_{type}_{agent_type}_{i}"
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

        self.action_to_move_tuple_type_1_agents = _generate_action_map(self.type_1_act_range)
        self.action_to_move_tuple_type_2_agents = _generate_action_map(self.type_2_act_range)
        # Island/component caches over non-wall cells (recomputed every reset).
        self.cell_to_island_id = {}
        self.island_id_to_cells = {}

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

        # Precompute connected non-wall components ("islands") for strict local spawning.
        self._compute_island_components()
        self._initialize_or_align_malthusian_state()

        total_entities = len(self.agents) + len(self.grass_agents)
        # Allow any config where number of free non-wall cells >= agents + grass.
        free_cells = max_cells - len(self.wall_positions)
        if total_entities > free_cells:
            raise ValueError(f"Too many agents+grass ({total_entities}) for free cells ({free_cells}) given {self.num_walls} walls on {self.grid_size}x{self.grid_size} grid")

        # Paint walls into channel 0
        for (wx, wy) in self.wall_positions:
            self.grid_world_state[0, wx, wy] = 1.0

        # Allocate agents over islands according to mu (per species), with no overlap.
        available_cells_by_island = {
            iid: set(cells) for iid, cells in self.island_id_to_cells.items()
        }
        agents_by_species = {
            species: [a for a in self.agents if self._species_from_agent_id(a) == species]
            for species in self._tracked_species()
        }

        for species in self._tracked_species():
            species_agents = agents_by_species.get(species, [])
            counts_by_island = self._sample_species_counts_over_islands(
                species=species,
                n_agents=len(species_agents),
                available_cells_by_island=available_cells_by_island,
            )

            allocated_positions = []
            for iid in self.island_ids:
                n_take = counts_by_island.get(iid, 0)
                if n_take <= 0:
                    continue
                island_cells = sorted(available_cells_by_island[iid])
                chosen_idx = self.rng.choice(len(island_cells), size=n_take, replace=False)
                chosen_positions = [island_cells[int(k)] for k in np.atleast_1d(chosen_idx)]
                for pos in chosen_positions:
                    available_cells_by_island[iid].remove(pos)
                allocated_positions.extend(chosen_positions)

            if len(allocated_positions) != len(species_agents):
                raise ValueError(
                    f"Internal allocation mismatch for {species}: "
                    f"{len(allocated_positions)} positions for {len(species_agents)} agents."
                )

            for i, agent in enumerate(species_agents):
                pos = allocated_positions[i]
                self.agent_positions[agent] = pos
                self.cumulative_rewards[agent] = 0
                self.unique_agent_stats[self.unique_agents[agent]]["spawn_island"] = self.cell_to_island_id.get(pos)
                if "predator" in agent:
                    self.predator_positions[agent] = pos
                    self.agent_energies[agent] = self.initial_energy_predator
                    self.grid_world_state[1, *pos] = self.initial_energy_predator
                else:
                    self.prey_positions[agent] = pos
                    self.agent_energies[agent] = self.initial_energy_prey
                    self.grid_world_state[2, *pos] = self.initial_energy_prey

        # Grass: fill uniformly from remaining free cells.
        remaining_cells = []
        for iid in self.island_ids:
            remaining_cells.extend(sorted(available_cells_by_island[iid]))
        if len(self.grass_agents) > len(remaining_cells):
            raise ValueError(
                f"Not enough remaining free cells for grass: need {len(self.grass_agents)}, "
                f"have {len(remaining_cells)}."
            )
        if self.grass_agents:
            chosen_idx = self.rng.choice(len(remaining_cells), size=len(self.grass_agents), replace=False)
            grass_positions = [remaining_cells[int(k)] for k in np.atleast_1d(chosen_idx)]
        else:
            grass_positions = []

        for i, grass in enumerate(self.grass_agents):
            pos = grass_positions[i]
            self.grass_positions[grass] = pos
            self.grass_energies[grass] = self.initial_energy_grass
            self.grid_world_state[3, *pos] = self.initial_energy_grass

        self.active_num_predators = len(self.predator_positions)
        self.active_num_prey = len(self.prey_positions)
        self.current_num_grass = len(self.grass_positions)

        self.current_grass_energy = sum(self.grass_energies.values())

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        return observations, {}

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        # For stepwise display eating in grid
        self.agents_just_ate.clear()
        # Reset per-step infos
        self._pending_infos = {}

        # step 0: Check for truncation
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
                self._handle_energy_decay(agent, observations, rewards, terminations, truncations)
            elif "predator" in agent:
                self._handle_predator_engagement(agent, observations, rewards, terminations, truncations)
            elif "prey" in agent:
                self._handle_prey_engagement(agent, observations, rewards, terminations, truncations)

        # Step 6: Handle agent removals
        for agent in self.agents[:]:
            if terminations[agent]:
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
        if self.enable_within_episode_reproduction:
            for agent in self.agents[:]:
                if "predator" in agent:
                    self._handle_predator_reproduction(agent, rewards, observations, terminations, truncations)
                elif "prey" in agent:
                    self._handle_prey_reproduction(agent, rewards, observations, terminations, truncations)

        # Step 8: Generate observations for all agents AFTER all engagements in the step
        for agent in self.agents:
            if agent in self.agent_positions:
                observations[agent] = self._get_observation(agent)

        # --- Build outputs with protocol guarantees ---
        # Agents alive for next step (observations only for these)
        live_next_step = set(self.agents)
        # Agents that died/terminated during this step
        dead_this_step = {a for a, t in terminations.items() if t}
        # Agents that acted this step (RLlib provided actions for them)
        acted_this_step = {
            a for a in action_dict.keys()
            if a in live_next_step or a in dead_this_step
        }

        # Observations: only for agents alive into next step
        observations = {a: observations[a] for a in live_next_step if a in observations}

        # Construct the set of agents that must appear in scalar dicts at least once
        output_agents = (
            live_next_step
            | dead_this_step
            | acted_this_step
            | set(rewards.keys())
            | set(terminations.keys())
            | set(truncations.keys())
        )

        # Ensure defaults present for all output agents
        rewards = {a: rewards.get(a, 0.0) for a in output_agents}
        terminations = {a: terminations.get(a, False) for a in output_agents}
        truncations = {a: truncations.get(a, False) for a in output_agents}

        # Global termination and truncation: fixed-horizon episodes only.
        truncations["__all__"] = False  # max_steps handled at the beginning of step
        terminations["__all__"] = False

        # Provide infos accumulated during the step for output agents
        infos = {a: self._pending_infos.get(a, {}) for a in output_agents if a in self._pending_infos}

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

        return observations, rewards, terminations, truncations, infos

    def _get_movement_energy_cost(self, agent, current_position, new_position):
        """
        Calculate energy cost for movement based on distance and a configurable factor.
        """
        distance_factor = self.config.get("move_energy_cost_factor", 0.01)
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
        """
        Generate an observation for the agent.
        """
        observation_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp, observation_range)
        channels = self.num_obs_channels + (1 if self.include_visibility_channel else 0)
        observation = np.zeros((channels, observation_range, observation_range), dtype=np.float32)
        # Channel 0: walls (binary). Already zero; stamp walls that fall inside window.
        for (wx, wy) in self.wall_positions:
            if xlo <= wx < xhi and ylo <= wy < yhi:
                lx = wx - xlo + xolo
                ly = wy - ylo + yolo
                observation[0, lx, ly] = 1.0
        # Copy dynamic channels (predators, prey, grass) into fixed locations 1..num_obs_channels-1 first
        observation[1:self.num_obs_channels, xolo:xohi, yolo:yohi] = self.grid_world_state[1:, xlo:xhi, ylo:yhi]

        need_visibility_mask = self.include_visibility_channel or self.mask_observation_with_visibility
        visibility_mask = None
        if need_visibility_mask:
            visibility_mask = np.zeros((observation_range, observation_range), dtype=np.float32)

            def bresenham(x0, y0, x1, y1):
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                x, y = x0, y0
                sx = 1 if x1 > x0 else -1
                sy = 1 if y1 > y0 else -1
                if dx >= dy:
                    err = dx / 2.0
                    while x != x1:
                        yield x, y
                        err -= dy
                        if err < 0:
                            y += sy
                            err += dx
                        x += sx
                    yield x1, y1
                else:
                    err = dy / 2.0
                    while y != y1:
                        yield x, y
                        err -= dx
                        if err < 0:
                            x += sx
                            err += dy
                        y += sy
                    yield x1, y1

            for lx in range(observation_range):
                for ly in range(observation_range):
                    gx = xlo + (lx - xolo)
                    gy = ylo + (ly - yolo)
                    if not (0 <= gx < self.grid_size and 0 <= gy < self.grid_size):
                        continue
                    blocked = False
                    for cx, cy in bresenham(xp, yp, gx, gy):
                        if (cx, cy) == (xp, yp) or (cx, cy) == (gx, gy):
                            continue
                        if (cx, cy) in self.wall_positions:
                            blocked = True
                            break
                    visibility_mask[lx, ly] = 0.0 if blocked else 1.0

            if self.mask_observation_with_visibility:
                # Multiply dynamic channels (exclude channel 0 walls, and exclude visibility channel if it'll be appended later)
                for c in range(1, self.num_obs_channels):
                    observation[c] *= visibility_mask

            if self.include_visibility_channel:
                vis_idx = channels - 1
                observation[vis_idx] = visibility_mask

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

        # Filter for unoccupied, non-wall positions
        wall_positions = self.wall_positions
        valid_positions = [
            pos for pos in potential_positions
            if pos not in occupied_positions and pos not in wall_positions
        ]

        if valid_positions:
            return valid_positions[0]  # Prefer adjacent position if available

        # Fallback: sample only inside the parent's connected non-wall component.
        parent_island_id = self.cell_to_island_id.get(reference_position)
        if parent_island_id is None:
            return None
        island_cells = self.island_id_to_cells.get(parent_island_id, set())
        free_positions = list(island_cells - occupied_positions)

        if free_positions:
            return free_positions[self.rng.integers(len(free_positions))]

        return None  # No available position found

    def _compute_island_components(self):
        """
        Build connected components of non-wall cells using 4-neighborhood adjacency.
        These components define hard islands for fallback offspring placement.
        """
        wall_positions = self.wall_positions
        grid_size = self.grid_size
        unvisited = {
            (x, y)
            for x in range(grid_size)
            for y in range(grid_size)
            if (x, y) not in wall_positions
        }

        self.cell_to_island_id = {}
        self.island_id_to_cells = {}
        island_id = 0

        while unvisited:
            start = unvisited.pop()
            stack = [start]
            component = {start}
            self.cell_to_island_id[start] = island_id

            while stack:
                cx, cy = stack.pop()
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                        continue
                    npos = (nx, ny)
                    if npos in wall_positions or npos not in unvisited:
                        continue
                    unvisited.remove(npos)
                    component.add(npos)
                    self.cell_to_island_id[npos] = island_id
                    stack.append(npos)

            self.island_id_to_cells[island_id] = component
            island_id += 1

    def _log(self, verbose: bool, message: str, color: str | None = None):
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

        prefix = colors.get(color, "") if color else ""
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
            self._finalize_malthusian_episode()
            # IMPORTANT: Only report currently active agents here.
            # Emitting already-terminated agents again violates RLlib's
            # SingleAgentEpisode contract on the new API stack.
            for agent in list(self.agents):
                observations[agent] = self._get_observation(agent)
                rewards[agent] = 0.0
                truncations[agent] = True
                terminations[agent] = False
                infos[agent] = dict(self.last_episode_summary)

            truncations["__all__"] = True
            terminations["__all__"] = False
            infos["__all__"] = dict(self.last_episode_summary)
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
        Increment the age of each active agent by one step.
        """
        for agent in action_dict:
            self.agent_ages[agent] += 1

    def _regenerate_grass_energy(self):
        """
        Increase energy of all grass patches, capped at initial energy value.
        """

        # Cap energy to maximum allowed for grass
        max_energy_grass = self.config.get("max_energy_grass", float("inf"))
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
        stat["final_energy"] = self.agent_energies[agent]
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

            rewards[agent] = self._get_type_specific("reward_predator_catch_prey", agent)
            self.cumulative_rewards.setdefault(agent, 0)
            self.cumulative_rewards[agent] += rewards[agent]

            raw_gain = min(self.agent_energies[caught_prey], self.config.get("max_energy_gain_per_prey", float("inf")))
            efficiency = self.config.get("energy_transfer_efficiency", 1.0)
            gain = raw_gain * efficiency
            self.agent_energies[agent] += gain
            self._per_agent_step_deltas[agent]["eat"] = gain

            # Cap the energy gain to max allowed for predator
            max_energy = self.config.get("max_energy_predator", float("inf"))
            self.agent_energies[agent] = min(self.agent_energies[agent], max_energy)

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
            uid = self.unique_agents[caught_prey]
            stat = self.unique_agent_stats[uid]
            stat["death_step"] = self.current_step
            stat["death_cause"] = "eaten"
            stat["final_energy"] = self.agent_energies[caught_prey]
            steps = max(stat["avg_energy_steps"], 1)
            stat["avg_energy"] = stat["avg_energy_sum"] / steps
            stat["cumulative_reward"] = self.cumulative_rewards.get(caught_prey, 0.0)

            self.death_agents_stats[uid] = stat

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
        caught_grass = next(
            (g for g, pos in self.grass_positions.items() if "grass" in g and np.array_equal(prey_position, pos)), None
        )

        if caught_grass:
            self._log(self.verbose_engagement, f"[ENGAGE] {agent} caught grass at {tuple(map(int, prey_position))}", "white")
            self.agents_just_ate.add(agent)  # Show green ring for next 1 step
            # Reward prey for eating grass
            rewards[agent] = self._get_type_specific("reward_prey_eat_grass", agent)
            # print(f"Rewards for {agent}: {rewards[agent]}")
            self.cumulative_rewards.setdefault(agent, 0)
            self.cumulative_rewards[agent] += rewards[agent]

            raw_gain = min(self.grass_energies[caught_grass], self.config.get("max_energy_gain_per_grass", float("inf")))
            efficiency = self.config.get("energy_transfer_efficiency", 1.0)
            gain = raw_gain * efficiency
            self.agent_energies[agent] += gain
            self._per_agent_step_deltas[agent]["eat"] = gain

            # Cap the energy gain to max allowed for prey
            max_energy = self.config.get("max_energy_prey", float("inf"))
            self.agent_energies[agent] = min(self.agent_energies[agent], max_energy)

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
                for i in range(self.config.get(f"n_possible_type_{new_type}_predators", 25))
                if f"type_{new_type}_predator_{i}" not in self._used_agent_ids_this_episode
            ]
            if not potential_new_ids:
                self._log(
                    self.verbose_reproduction,
                    f"[REPRODUCTION] No available predator slots at type {new_type} for spawning",
                    "red",
                )
                return

            new_agent = potential_new_ids[0]
            # Spawn position must exist in the same island; otherwise cancel reproduction.
            occupied_positions = set(self.agent_positions.values())
            new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)
            if new_position is None:
                self._log(
                    self.verbose_reproduction,
                    f"[REPRODUCTION] No available local predator spawn cell for {agent}; reproduction canceled",
                    "red",
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

            self._register_new_agent(new_agent, parent_unique_id=self.unique_agents[agent])
            child_uid = self.unique_agents[new_agent]
            self.agent_live_offspring_ids[agent].append(child_uid)
            self.agent_offspring_counts[agent] += 1

            self.unique_agent_stats[self.unique_agents[new_agent]]["mutated"] = mutated
            self.unique_agent_stats[self.unique_agents[agent]]["offspring_count"] += 1

            self.agent_positions[new_agent] = new_position
            self.predator_positions[new_agent] = new_position
            self.unique_agent_stats[self.unique_agents[new_agent]]["spawn_island"] = self.cell_to_island_id.get(new_position)

            repro_eff = self.config.get("reproduction_energy_efficiency", 1.0)
            energy_given = self.initial_energy_predator * repro_eff
            self.agent_energies[new_agent] = energy_given
            self.agent_energies[agent] -= self.initial_energy_predator
            self._per_agent_step_deltas[agent]["repro"] = -self.initial_energy_predator

            self.grid_world_state[1, *new_position] = self.initial_energy_predator
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
        cooldown = self.config.get("reproduction_cooldown_steps", 10)
        if self.current_step - self.agent_last_reproduction.get(agent, -cooldown) < cooldown:
            return

        chance_key = "reproduction_chance_predator" if "predator" in agent else "reproduction_chance_prey"
        if self.rng.random() > self.config.get(chance_key, 1.0):
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
                for i in range(self.config.get(f"n_possible_type_{new_type}_prey", 25))
                if f"type_{new_type}_prey_{i}" not in self._used_agent_ids_this_episode
            ]
            if not potential_new_ids:
                self._log(
                    self.verbose_reproduction,
                    f"[REPRODUCTION] No available prey slots at type {new_type} for spawning",
                    "red",
                )
                return

            new_agent = potential_new_ids[0]
            # Spawn position must exist in the same island; otherwise cancel reproduction.
            occupied_positions = set(self.agent_positions.values())
            new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)
            if new_position is None:
                self._log(
                    self.verbose_reproduction,
                    f"[REPRODUCTION] No available local prey spawn cell for {agent}; reproduction canceled",
                    "red",
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

            self._register_new_agent(new_agent, parent_unique_id=self.unique_agents[agent])
            child_uid = self.unique_agents[new_agent]
            self.agent_live_offspring_ids[agent].append(child_uid)

            self.agent_offspring_counts[agent] += 1
            self.unique_agent_stats[self.unique_agents[new_agent]]["mutated"] = mutated
            self.unique_agent_stats[self.unique_agents[agent]]["offspring_count"] += 1

            self.agent_positions[new_agent] = new_position
            self.prey_positions[new_agent] = new_position
            self.unique_agent_stats[self.unique_agents[new_agent]]["spawn_island"] = self.cell_to_island_id.get(new_position)

            repro_eff = self.config.get("reproduction_energy_efficiency", 1.0)
            energy_given = self.initial_energy_prey * repro_eff
            self.agent_energies[new_agent] = energy_given
            self.agent_energies[agent] -= self.initial_energy_prey
            self._per_agent_step_deltas[agent]["repro"] = -self.initial_energy_prey

            self.grid_world_state[2, *new_position] = self.initial_energy_prey
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
            "_used_agent_ids_this_episode": set(self._used_agent_ids_this_episode),
            "per_step_agent_data": self.per_step_agent_data.copy(),  # ← aligned with rest
            "cell_to_island_id": self.cell_to_island_id.copy(),
            "island_id_to_cells": {iid: set(cells) for iid, cells in self.island_id_to_cells.items()},
            "mu_by_species": {sp: dict(mu) for sp, mu in self.mu_by_species.items()},
            "last_phi_by_species": {sp: dict(phi) for sp, phi in self.last_phi_by_species.items()},
            "last_phi_components": {
                sp: {iid: dict(vals) for iid, vals in comp.items()}
                for sp, comp in self.last_phi_components.items()
            },
            "last_episode_summary": dict(self.last_episode_summary),
            "_malthusian_finalized_at_step": self._malthusian_finalized_at_step,
            "mu_logits_by_species": {sp: dict(mu) for sp, mu in self.mu_logits_by_species.items()},
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
        self._used_agent_ids_this_episode = set(
            snapshot.get("_used_agent_ids_this_episode", self._used_agent_ids_this_episode)
        )
        self.per_step_agent_data = snapshot["per_step_agent_data"].copy()
        self.cell_to_island_id = snapshot.get("cell_to_island_id", {}).copy()
        self.island_id_to_cells = {
            iid: set(cells) for iid, cells in snapshot.get("island_id_to_cells", {}).items()
        }
        self.mu_by_species = {
            sp: dict(mu) for sp, mu in snapshot.get("mu_by_species", self.mu_by_species).items()
        }
        self.last_phi_by_species = {
            sp: dict(phi) for sp, phi in snapshot.get("last_phi_by_species", self.last_phi_by_species).items()
        }
        self.last_phi_components = {
            sp: {iid: dict(vals) for iid, vals in comp.items()}
            for sp, comp in snapshot.get("last_phi_components", self.last_phi_components).items()
        }
        self.last_episode_summary = dict(snapshot.get("last_episode_summary", self.last_episode_summary))
        self._malthusian_finalized_at_step = snapshot.get(
            "_malthusian_finalized_at_step", self._malthusian_finalized_at_step
        )
        self.mu_logits_by_species = {
            sp: dict(mu) for sp, mu in snapshot.get("mu_logits_by_species", self.mu_logits_by_species).items()
        }

    def _build_possible_agent_ids(self) -> list[str]:
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

    def _build_observation_space(self, agent_id: str):
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

    def _build_action_space(self, agent_id: str):
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

    def _register_new_agent(self, agent_id: str, parent_unique_id: str | None = None):
        self._used_agent_ids_this_episode.add(agent_id)
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
            "spawn_island": None,
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
