"""
Article task reconstructions for Leibo et al. (2019).

These environments implement the mechanics that are specified in the paper text
for Clamity and Allelopathy. Several constants required for a literal
article-figure reproduction are not published in the paper; those are exposed in
config and recorded in `unpublished_reconstruction_defaults`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except Exception:  # pragma: no cover

    class MultiAgentEnv:  # type: ignore[no-redef]
        def __init__(self) -> None:
            pass

        def reset(self, *, seed=None, options=None):
            raise NotImplementedError


MOVE_DELTAS = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

# Facing: 0=N, 1=E, 2=S, 3=W. Updated by movement actions 1-4.
_ACTION_FACING = {1: 0, 2: 2, 3: 3, 4: 1}


def _world_coords(
    agent_row: int, agent_col: int, obs_r: int, obs_c: int, facing: int, radius: int
) -> tuple[int, int]:
    """Map observation-window position to world grid coordinates.

    The window is egocentric: obs_r=0 is always directly ahead of the agent.
    Facing 0=N, 1=E, 2=S, 3=W (paper Section 2.4: window follows orientation).
    """
    dr = obs_r - radius
    dc = obs_c - radius
    if facing == 0:    # N ahead
        return agent_row + dr, agent_col + dc
    elif facing == 1:  # E ahead
        return agent_row + dc, agent_col - dr
    elif facing == 2:  # S ahead
        return agent_row - dr, agent_col - dc
    else:              # W ahead
        return agent_row - dc, agent_col + dr


def _to_obs_coords(
    drow: int, dcol: int, facing: int, radius: int
) -> tuple[int, int]:
    """Map world-offset (drow, dcol) relative to agent into observation (rr, cc).

    Inverse of _world_coords: obs_r=0 is ahead, obs_c=0 is to the agent's left.
    """
    if facing == 0:    # N ahead
        return drow + radius, dcol + radius
    elif facing == 1:  # E ahead
        return -dcol + radius, drow + radius
    elif facing == 2:  # S ahead
        return -drow + radius, -dcol + radius
    else:              # W ahead
        return dcol + radius, -drow + radius


@dataclass
class AgentState:
    species: int
    island: int
    row: int
    col: int
    facing: int = 0  # 0=N,1=E,2=S,3=W — follows last movement direction
    solitary_eval: bool = False
    cumulative_reward: float = 0.0
    last_resource_type: int | None = None
    resource_streak: int = 0
    switching_costs: int = 0
    settled: bool = False
    shell_radius: int = 0


class ArticleMalthusianEnv(MultiAgentEnv):
    """Common archipelago machinery for article task reconstructions."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = dict(config or {})
        self.num_species = int(self.config.get("num_species", 4))
        self.total_individuals = int(self.config.get("total_individuals", 960))
        # Distributed-island mode: optional MuServer Ray Actor.
        # When provided, this env instance handles exactly one island; mu is
        # shared via the actor and updated once all NI islands report phi.
        # See utils/mu_server.py.  Falls back to local all-island mode when
        # all NI slots are already claimed (e.g. RLlib's local evaluator env).
        self._mu_server = self.config.get("mu_server")
        self._global_island_index: int | None = None
        self._global_num_islands: int | None = None
        _configured_num_islands = int(self.config.get("num_islands", 60))
        if self._mu_server is not None:
            import ray as _ray
            _idx = _ray.get(self._mu_server.register_worker.remote())
            if _idx is not None:
                self._global_island_index = _idx
                self._global_num_islands = _configured_num_islands
                self.num_islands = 1
            else:
                self.num_islands = _configured_num_islands
        else:
            self.num_islands = _configured_num_islands
        self.num_solitary_eval_islands_per_species = int(self.config.get("num_solitary_eval_islands_per_species", 0))
        self.episode_horizon = int(self.config.get("episode_horizon", 1000))
        self.alpha = float(self.config.get("alpha", 0.0001))
        self.eta = float(self.config.get("eta", 0.01))
        self.enable_malthusian_update = bool(self.config.get("enable_malthusian_update", True))
        self.seed_value = int(self.config.get("seed", 0))
        self.deterministic_reset_sequence = bool(self.config.get("deterministic_reset_sequence", True))
        self._reset_counter = 0
        self.rng = np.random.default_rng(self.seed_value)
        self.current_step = 0
        self.mu_logits_by_species = {
            species: np.zeros(self.num_islands, dtype=np.float64)
            for species in range(self.num_species)
        }
        if self._mu_server is not None and self._global_island_index is not None:
            _initial_p = 1.0 / float(self._global_num_islands)
            self.mu_by_species = {
                species: np.array([_initial_p], dtype=np.float64)
                for species in range(self.num_species)
            }
        else:
            self.mu_by_species = {
                species: np.ones(self.num_islands, dtype=np.float64) / self.num_islands
                for species in range(self.num_species)
            }
        self.agent_states: dict[str, AgentState] = {}
        self.agents: list[str] = []
        self.possible_agents = self._build_possible_agents()
        self.observation_spaces = {
            agent_id: self._build_observation_space()
            for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: self._build_action_space()
            for agent_id in self.possible_agents
        }
        self.last_episode_summary: dict[str, Any] = {}

    @property
    def individuals_per_species(self) -> int:
        if self.total_individuals % self.num_species != 0:
            raise ValueError("total_individuals must be divisible by num_species.")
        return self.total_individuals // self.num_species

    @property
    def total_island_count(self) -> int:
        return self.num_islands + self.num_species * self.num_solitary_eval_islands_per_species

    def _build_possible_agents(self) -> list[str]:
        archipelago_agents = [
            f"species_{species}_agent_{idx}"
            for species in range(self.num_species)
            for idx in range(self.individuals_per_species)
        ]
        solitary_agents = [
            f"species_{species}_solitary_{idx}"
            for species in range(self.num_species)
            for idx in range(self.num_solitary_eval_islands_per_species)
        ]
        return archipelago_agents + solitary_agents

    def _build_observation_space(self):
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, int(self.config.get("observation_window", 15)), int(self.config.get("observation_window", 15))),
            dtype=np.float32,
        )

    def _build_action_space(self):
        return gym.spaces.Discrete(int(self.config.get("num_actions", 7)))

    def _reset_rng(self, seed: int | None) -> None:
        if seed is None and self.deterministic_reset_sequence:
            seed = self.seed_value + self._reset_counter
        self._reset_counter += 1
        self.rng = np.random.default_rng(seed)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - float(np.max(logits))
        exp = np.exp(shifted)
        return exp / float(np.sum(exp))

    @property
    def _is_distributed_single_island(self) -> bool:
        """True when this env instance handles one island via a shared MuServer."""
        return self._mu_server is not None and self._global_island_index is not None

    def _sync_mu_from_logits(self) -> None:
        for species, logits in self.mu_logits_by_species.items():
            self.mu_by_species[species] = self._softmax(logits)

    def _allocate_agents(self) -> None:
        if self._is_distributed_single_island:
            import ray as _ray
            _global_mu = _ray.get(self._mu_server.get_mu.remote())
            for _s in range(self.num_species):
                self.mu_by_species[_s] = np.array(
                    [float(_global_mu[_s][self._global_island_index])],
                    dtype=np.float64,
                )
        self.agents = []
        self.agent_states = {}
        for species in range(self.num_species):
            counts = self._archipelago_counts_for_species(species)
            agent_idx = 0
            for island, count in enumerate(counts):
                for _ in range(int(count)):
                    agent_id = f"species_{species}_agent_{agent_idx}"
                    row, col = self._sample_agent_spawn(island)
                    self.agent_states[agent_id] = AgentState(
                        species=species,
                        island=island,
                        row=row,
                        col=col,
                    )
                    self.agents.append(agent_id)
                    agent_idx += 1
            for idx in range(self.num_solitary_eval_islands_per_species):
                island = self.num_islands + species * self.num_solitary_eval_islands_per_species + idx
                agent_id = f"species_{species}_solitary_{idx}"
                row, col = self._sample_agent_spawn(island)
                self.agent_states[agent_id] = AgentState(
                    species=species,
                    island=island,
                    row=row,
                    col=col,
                    solitary_eval=True,
                )
                self.agents.append(agent_id)
        self.agents.sort()

    def _archipelago_counts_for_species(self, species: int) -> np.ndarray:
        if bool(self.config.get("one_agent_per_island", False)):
            counts = np.ones(self.num_islands, dtype=np.int64)
            if int(np.sum(counts)) != self.individuals_per_species:
                raise ValueError("one_agent_per_island requires total_individuals == num_species * num_islands.")
            return counts

        fixed_population_per_island = self.config.get("fixed_population_per_island")
        if fixed_population_per_island is not None:
            per_island = int(fixed_population_per_island)
            if per_island % self.num_species != 0:
                raise ValueError("fixed_population_per_island must be divisible by num_species.")
            per_species_per_island = per_island // self.num_species
            counts = np.full(self.num_islands, per_species_per_island, dtype=np.int64)
            if int(np.sum(counts)) != self.individuals_per_species:
                raise ValueError("fixed_population_per_island * num_islands must equal total_individuals.")
            return counts

        if self._is_distributed_single_island:
            # mu_by_species[species][0] holds the global mu probability for our
            # island. Sample the number of individuals allocated here from
            # Binomial(M_per_species, p) — correct marginal of the multinomial.
            p = float(self.mu_by_species[species][0])
            count = int(self.rng.binomial(self.individuals_per_species, p))
            return np.array([max(1, count)], dtype=np.int64)
        return self.rng.multinomial(self.individuals_per_species, self.mu_by_species[species])

    def _sample_agent_spawn(self, island: int) -> tuple[int, int]:
        del island
        return (
            int(self.rng.integers(0, int(self.config.get("height", 25)))),
            int(self.rng.integers(0, int(self.config.get("width", 25)))),
        )

    def _finalize_malthusian_update(self) -> dict[str, Any]:
        phi_by_species: dict[str, dict[int, float]] = {}
        counts_by_species: dict[str, dict[int, int]] = {}
        switch_by_island: dict[int, int] = {island: 0 for island in range(self.num_islands)}
        solitary_returns: dict[int, list[float]] = {
            species: []
            for species in range(self.num_species)
        }
        returns: dict[int, dict[int, list[float]]] = {
            species: {island: [] for island in range(self.num_islands)}
            for species in range(self.num_species)
        }

        for state in self.agent_states.values():
            if state.solitary_eval:
                solitary_returns[state.species].append(state.cumulative_reward)
                continue
            returns[state.species][state.island].append(state.cumulative_reward)
            switch_by_island[state.island] += state.switching_costs

        for species in range(self.num_species):
            species_name = f"species_{species}"
            phi_by_species[species_name] = {}
            counts_by_species[species_name] = {}
            phi_vec = np.zeros(self.num_islands, dtype=np.float64)
            for island in range(self.num_islands):
                island_returns = returns[species][island]
                counts_by_species[species_name][island] = len(island_returns)
                mean_return = float(np.mean(island_returns)) if island_returns else 0.0
                phi_by_species[species_name][island] = mean_return
                phi_vec[island] = mean_return

            if self.enable_malthusian_update and not self._is_distributed_single_island:
                mu = self.mu_by_species[species]
                centered = phi_vec - float(np.sum(mu * phi_vec))
                entropy_grad = -(np.log(np.maximum(mu, 1e-12)) + 1.0)
                self.mu_logits_by_species[species] = (
                    self.mu_logits_by_species[species]
                    + self.alpha * (centered + self.eta * entropy_grad)
                )

        if self._is_distributed_single_island and self.enable_malthusian_update:
            # Distributed mode: report this island's phi to the MuServer.
            # The server fires the ecological update once all NI reports arrive
            # and returns the updated global mu; we cache our island's new prob.
            import ray as _ray
            _phi_for_server = {
                s: float(phi_by_species.get(f"species_{s}", {}).get(0, 0.0))
                for s in range(self.num_species)
            }
            _new_mu = _ray.get(
                self._mu_server.report_phi.remote(
                    self._global_island_index, _phi_for_server
                )
            )
            if _new_mu is not None:
                for s in range(self.num_species):
                    self.mu_by_species[s] = np.array(
                        [float(_new_mu[s][self._global_island_index])],
                        dtype=np.float64,
                    )
        elif not self._is_distributed_single_island:
            self._sync_mu_from_logits()
        summary = {
            "mu_by_species": {
                f"species_{species}": {
                    island: float(prob)
                    for island, prob in enumerate(self.mu_by_species[species])
                }
                for species in range(self.num_species)
            },
            "phi_by_species": phi_by_species,
            "counts_by_species": counts_by_species,
            "solitary_return_by_species": {
                f"species_{species}": (
                    float(np.mean(values))
                    if values
                    else 0.0
                )
                for species, values in solitary_returns.items()
            },
            "solitary_count_by_species": {
                f"species_{species}": len(values)
                for species, values in solitary_returns.items()
            },
            "switching_cost_by_island": switch_by_island,
            "malthusian_mu_learning_rate": self.alpha,
            "malthusian_mu_entropy_coeff": self.eta,
            "enable_malthusian_update": self.enable_malthusian_update,
        }
        self.last_episode_summary = summary
        return summary

    def _move_agent(self, state: AgentState, action: int, height: int, width: int) -> None:
        if action not in MOVE_DELTAS:
            return
        dr, dc = MOVE_DELTAS[action]
        state.row = int(np.clip(state.row + dr, 0, height - 1))
        state.col = int(np.clip(state.col + dc, 0, width - 1))
        if action in _ACTION_FACING:
            state.facing = _ACTION_FACING[action]

    def _empty_observation(self) -> np.ndarray:
        return np.zeros(self._build_observation_space().shape, dtype=np.float32)


class ArticleAllelopathyEnv(ArticleMalthusianEnv):
    """Text-grounded reconstruction of the paper's Allelopathy game."""

    unpublished_reconstruction_defaults = {
        "height": "Paper does not publish Allelopathy map height.",
        "width": "Paper does not publish Allelopathy map width.",
        "initial_shrub_density": "Paper says shrubs are randomly placed but does not publish density.",
        "shrub_growth_base_probability": "Paper gives inverse suppression rule but not the base probability.",
        "suppression_radius": "Paper says nearby shrubs suppress growth but does not publish radius.",
    }

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        variant = cfg.get("variant", "biased")
        if variant == "unbiased":
            cfg.setdefault("alpha", 1e-7)
            cfg.setdefault("eta", 0.3)
            cfg.setdefault("resource_spawn_probabilities", [0.5, 0.5])
            cfg.setdefault("resource_reward_caps", [250, 250])
        else:
            cfg.setdefault("alpha", 0.0001)
            cfg.setdefault("eta", 0.01)
            cfg.setdefault("resource_spawn_probabilities", [0.8, 0.2])
            cfg.setdefault("resource_reward_caps", [8, 250])
        cfg.setdefault("episode_horizon", 1000)
        # 32×32: square grid, width matches the DeepMind Melting Pot
        # allelopathic_harvest substrate (the closest related public source).
        # The 2019 paper does not publish map dimensions.
        cfg.setdefault("height", 32)
        cfg.setdefault("width", 32)
        cfg.setdefault("observation_window", 15)
        cfg.setdefault("num_actions", 7)
        super().__init__(cfg)
        self.height = int(self.config["height"])
        self.width = int(self.config["width"])
        self.num_resource_types = 2
        self.resource_spawn_probabilities = np.asarray(self.config["resource_spawn_probabilities"], dtype=np.float64)
        self.resource_spawn_probabilities = self.resource_spawn_probabilities / float(np.sum(self.resource_spawn_probabilities))
        self.resource_reward_caps = [int(v) for v in self.config["resource_reward_caps"]]
        self.initial_shrub_density = float(self.config.get("initial_shrub_density", 0.08))
        self.shrub_growth_base_probability = float(self.config.get("shrub_growth_base_probability", 0.01))
        self.suppression_radius = int(self.config.get("suppression_radius", 2))
        self.shrubs: dict[int, np.ndarray] = {}

    def reset(self, *, seed=None, options=None):
        del options
        self._reset_rng(seed)
        self.current_step = 0
        self._initialize_shrubs()
        self._allocate_agents()
        observations = {agent: self._observe_agent(self.agent_states[agent]) for agent in self.agents}
        return observations, {}

    def _initialize_shrubs(self) -> None:
        self.shrubs = {}
        for island in range(self.total_island_count):
            grid = np.full((self.height, self.width), -1, dtype=np.int8)
            mask = self.rng.random((self.height, self.width)) < self.initial_shrub_density
            types = self.rng.choice(self.num_resource_types, size=(self.height, self.width), p=self.resource_spawn_probabilities)
            grid[mask] = types[mask]
            self.shrubs[island] = grid

    def _sample_agent_spawn(self, island: int) -> tuple[int, int]:
        del island
        return (
            int(self.rng.integers(0, self.height)),
            int(self.rng.integers(0, self.width)),
        )

    def _observe_agent(self, state: AgentState) -> np.ndarray:
        obs = self._empty_observation()
        win = obs.shape[1]
        radius = win // 2
        grid = self.shrubs[state.island]
        for rr in range(win):
            for cc in range(win):
                row, col = _world_coords(state.row, state.col, rr, cc, state.facing, radius)
                if 0 <= row < self.height and 0 <= col < self.width:
                    shrub_type = int(grid[row, col])
                    if shrub_type >= 0:
                        obs[shrub_type, rr, cc] = 1.0
        # Channel 2: all agents on the same island are visible (not just self).
        # Paper Section 2.4 describes an RGB window; agents of all species appear in the
        # observation so agents can perceive competitors and coordinate.
        for other in self.agent_states.values():
            if other.island != state.island:
                continue
            rr, cc = _to_obs_coords(other.row - state.row, other.col - state.col, state.facing, radius)
            if 0 <= rr < win and 0 <= cc < win:
                obs[2, rr, cc] = 1.0
        return obs

    def _type_a_count_nearby(self, grid: np.ndarray, row: int, col: int) -> int:
        """Count type-A (index 0) shrubs within suppression radius.

        Used for one-way allelopathic suppression: type-A shrubs suppress
        type-B growth but not vice versa (the asymmetry that motivates the
        specialisation dynamics in Section 3.2).
        """
        r0 = max(0, row - self.suppression_radius)
        r1 = min(self.height, row + self.suppression_radius + 1)
        c0 = max(0, col - self.suppression_radius)
        c1 = min(self.width, col + self.suppression_radius + 1)
        patch = grid[r0:r1, c0:c1]
        return int(np.sum(patch == 0))

    def _grow_shrubs(self) -> None:
        for island, grid in self.shrubs.items():
            empty_cells = np.argwhere(grid < 0)
            if empty_cells.size == 0:
                continue
            self.rng.shuffle(empty_cells)
            for row, col in empty_cells:
                shrub_type = int(self.rng.choice(self.num_resource_types, p=self.resource_spawn_probabilities))
                if shrub_type == 1:
                    # Type-B growth is allelopathically suppressed by nearby type-A shrubs.
                    # Type A grows at base probability regardless of nearby B.
                    a_count = self._type_a_count_nearby(grid, int(row), int(col))
                    growth_prob = self.shrub_growth_base_probability / (1.0 + a_count)
                else:
                    growth_prob = self.shrub_growth_base_probability
                if self.rng.random() < growth_prob:
                    grid[int(row), int(col)] = shrub_type
            self.shrubs[island] = grid

    def _harvest_if_present(self, state: AgentState) -> float:
        grid = self.shrubs[state.island]
        shrub_type = int(grid[state.row, state.col])
        if shrub_type < 0:
            return 0.0
        grid[state.row, state.col] = -1
        if state.last_resource_type is not None and state.last_resource_type != shrub_type:
            state.switching_costs += 1
            state.resource_streak = 0
        state.last_resource_type = shrub_type
        state.resource_streak += 1
        reward = float(min(state.resource_streak, self.resource_reward_caps[shrub_type]))
        state.cumulative_reward += reward
        return reward

    def step(self, action_dict):
        self._grow_shrubs()
        rewards: dict[str, float] = {}
        observations: dict[str, np.ndarray] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}

        # Occupied cells: (island, row, col) → agent_id for collision resolution.
        # Agents are processed in a random order so no agent has a systematic
        # movement advantage (Lab2D resolves simultaneous conflicts randomly).
        occupied: dict[tuple[int, int, int], str] = {
            (s.island, s.row, s.col): a for a, s in self.agent_states.items()
        }
        agent_order = list(self.agents)
        self.rng.shuffle(agent_order)

        for agent in agent_order:
            state = self.agent_states[agent]
            action = int(action_dict.get(agent, 0))
            if action == 5:   # turn left (CCW)
                state.facing = (state.facing - 1) % 4
            elif action == 6:  # turn right (CW)
                state.facing = (state.facing + 1) % 4
            elif action in MOVE_DELTAS:
                dr, dc = MOVE_DELTAS[action]
                new_row = int(np.clip(state.row + dr, 0, self.height - 1))
                new_col = int(np.clip(state.col + dc, 0, self.width - 1))
                dest = (state.island, new_row, new_col)
                src = (state.island, state.row, state.col)
                if dest == src or dest not in occupied:
                    del occupied[src]
                    state.row, state.col = new_row, new_col
                    occupied[dest] = agent
                    if action in _ACTION_FACING:
                        state.facing = _ACTION_FACING[action]
                # If blocked: agent stays but still updates facing for continuity.
                elif action in _ACTION_FACING:
                    state.facing = _ACTION_FACING[action]

        for agent in self.agents:
            state = self.agent_states[agent]
            reward = self._harvest_if_present(state)
            rewards[agent] = reward
            observations[agent] = self._observe_agent(state)
            terminations[agent] = False
            truncations[agent] = False
            infos[agent] = {
                "island": state.island,
                "species": state.species,
                "switching_costs": state.switching_costs,
            }

        self.current_step += 1
        done = self.current_step >= self.episode_horizon
        if done:
            summary = self._finalize_malthusian_update()
            infos["__all__"] = summary
            observations = {}
            truncations = {agent: True for agent in self.agents}
        terminations["__all__"] = False
        truncations["__all__"] = done
        return observations, rewards, terminations, truncations, infos


class ArticleClamityEnv(ArticleMalthusianEnv):
    """Text-grounded reconstruction of the paper's Clamity game."""

    unpublished_reconstruction_defaults = {
        "shell_max_radius": "Paper says shell grows to a maximum size but does not publish the value.",
        "nutrient_patch_layout": "Paper refers to maps in Figure 2 but does not publish coordinates.",
        "base_filter_reward_rate": "Paper says reward is proportional to shell size but does not publish the rate.",
    }

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        cfg.setdefault("num_species", 1)
        cfg.setdefault("total_individuals", 960)
        cfg.setdefault("num_islands", 1)
        cfg.setdefault("alpha", 0.0001)
        cfg.setdefault("eta", 1.5)
        cfg.setdefault("episode_horizon", 250)
        cfg.setdefault("height", 36)
        cfg.setdefault("width", 60)
        cfg.setdefault("observation_window", 15)
        cfg.setdefault("num_actions", 7)
        super().__init__(cfg)
        self.height = int(self.config["height"])
        self.width = int(self.config["width"])
        self.shell_max_radius = int(self.config.get("shell_max_radius", 4))
        # 0.01: derived from Figure 2(E) reward scale. The "no-curiosity" agent
        # stuck at the local optimum (settle at step 0, no nutrient patch) reaches
        # ~200 total reward. With shell growing 1/step to max radius 4:
        # base × (9+25+49+81×247) ≈ base × 20,090 = 200 → base ≈ 0.01.
        self.base_filter_reward_rate = float(self.config.get("base_filter_reward_rate", 0.01))
        self.nutrient_patches = [
            tuple(p)
            for p in self.config.get(
                "nutrient_patches",
                [(6, 10), (6, 49), (29, 10), (29, 49)],
            )
        ]

    def reset(self, *, seed=None, options=None):
        del options
        self._reset_rng(seed)
        self.current_step = 0
        self._allocate_agents()
        observations = {agent: self._observe_agent(self.agent_states[agent]) for agent in self.agents}
        return observations, {}

    def _sample_agent_spawn(self, island: int) -> tuple[int, int]:
        del island
        center_row = self.height // 2
        center_col = self.width // 2
        return (
            int(np.clip(center_row + self.rng.integers(-2, 3), 0, self.height - 1)),
            int(np.clip(center_col + self.rng.integers(-2, 3), 0, self.width - 1)),
        )

    def _observe_agent(self, state: AgentState) -> np.ndarray:
        obs = self._empty_observation()
        win = obs.shape[1]
        radius = win // 2
        for other in self.agent_states.values():
            if other.island != state.island:
                continue
            rr, cc = _to_obs_coords(other.row - state.row, other.col - state.col, state.facing, radius)
            if 0 <= rr < win and 0 <= cc < win:
                obs[0, rr, cc] = 1.0
                if other.settled:
                    obs[1, rr, cc] = min(1.0, other.shell_radius / max(1, self.shell_max_radius))
        for row, col in self.nutrient_patches:
            rr, cc = _to_obs_coords(int(row) - state.row, int(col) - state.col, state.facing, radius)
            if 0 <= rr < win and 0 <= cc < win:
                obs[2, rr, cc] = 1.0
        return obs

    def _settled_neighbors(self, state: AgentState) -> int:
        count = 0
        for other in self.agent_states.values():
            if other is state or other.island != state.island or not other.settled:
                continue
            dist = abs(other.row - state.row) + abs(other.col - state.col)
            if dist <= max(1, other.shell_radius + state.shell_radius):
                count += 1
        return count

    def _shell_reward(self, state: AgentState) -> float:
        if not state.settled:
            return 0.0
        if self._settled_neighbors(state) > 0:
            return 0.0
        shell_area = (2 * state.shell_radius + 1) ** 2
        nutrient_bonus = 0
        for row, col in self.nutrient_patches:
            if abs(int(row) - state.row) <= state.shell_radius and abs(int(col) - state.col) <= state.shell_radius:
                nutrient_bonus += shell_area
        return self.base_filter_reward_rate * float(shell_area + nutrient_bonus)

    def step(self, action_dict):
        rewards: dict[str, float] = {}
        observations: dict[str, np.ndarray] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}

        # Occupied cells per island for collision resolution.
        # Settled agents are immovable anchors; unsettled agents resolve in
        # random order so no agent gets a systematic positional advantage.
        occupied: dict[tuple[int, int, int], str] = {
            (s.island, s.row, s.col): a for a, s in self.agent_states.items()
        }
        agent_order = list(self.agents)
        self.rng.shuffle(agent_order)

        for agent in agent_order:
            state = self.agent_states[agent]
            action = int(action_dict.get(agent, 0))
            if not state.settled:
                if action == 6:
                    state.settled = True
                    state.shell_radius = 1
                elif action == 5:  # turn left (CCW)
                    state.facing = (state.facing - 1) % 4
                elif action in MOVE_DELTAS:
                    dr, dc = MOVE_DELTAS[action]
                    new_row = int(np.clip(state.row + dr, 0, self.height - 1))
                    new_col = int(np.clip(state.col + dc, 0, self.width - 1))
                    dest = (state.island, new_row, new_col)
                    src = (state.island, state.row, state.col)
                    if dest == src or dest not in occupied:
                        del occupied[src]
                        state.row, state.col = new_row, new_col
                        occupied[dest] = agent
                        if action in _ACTION_FACING:
                            state.facing = _ACTION_FACING[action]
                    elif action in _ACTION_FACING:
                        state.facing = _ACTION_FACING[action]
            elif (
                state.shell_radius < self.shell_max_radius
                and self._settled_neighbors(state) == 0
            ):
                # Section 3.1: "shell growth is also restricted by the presence
                # of adjacent shells" — growth stops, not just reward.
                state.shell_radius += 1

        for agent in self.agents:
            state = self.agent_states[agent]
            reward = self._shell_reward(state)
            state.cumulative_reward += reward
            rewards[agent] = reward
            observations[agent] = self._observe_agent(state)
            terminations[agent] = False
            truncations[agent] = False
            infos[agent] = {
                "island": state.island,
                "species": state.species,
                "settled": state.settled,
                "shell_radius": state.shell_radius,
            }

        self.current_step += 1
        done = self.current_step >= self.episode_horizon
        if done:
            summary = self._finalize_malthusian_update()
            infos["__all__"] = summary
            observations = {}
            truncations = {agent: True for agent in self.agents}
        terminations["__all__"] = False
        truncations["__all__"] = done
        return observations, rewards, terminations, truncations, infos
