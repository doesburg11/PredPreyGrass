"""Continuous Predator-Prey-Grass Multi-Agent Environment (RLlib)

This is a first-pass continuous adaptation of the discrete grid-based
`PredPreyGrass` environment. Positions are now floats in a bounded square
world [0, world_size] x [0, world_size]. Actions are continuous 2D movement
vectors (dx, dy) in [-1, 1], scaled by a per-type maximum step size.

Key Simplifications vs. Discrete Version:
- Observations are vector-based, not image grids.
- Each agent observes: [self_x, self_y, self_energy, num_prey, num_predators,
  (nearest_k_other relative positions & energies), (nearest_k_grass relative positions & energies)].
- Predator catches prey if within `catch_radius`.
- Prey eats a grass patch if within `eat_radius`; grass energy transfers (capped) and grass resets to 0 then regrows.
- Reproduction retains same threshold logic but new agents spawn with slight random offset.
- Energy decay, movement cost, rewards, reproduction rewards roughly mirrored.

Config Additions (defaults shown):
    world_size: 20.0
    max_speed_predator: 0.9
    max_speed_prey: 1.0
    catch_radius: 0.6
    eat_radius: 0.5
    grass_regrow_rate: 0.05
    grass_max_energy: 2.0
    n_grass_patches: 40
    nearest_k_agents: 4
    nearest_k_grass: 3
    continuous_move_cost_factor: 0.01  # fraction of current energy * distance
    vision_radius: None  # if set (float), only agents/grass within this distance are considered before nearest-k truncation

Parity Items Not Yet Implemented:
    - Mutation between types.
    - Two predator / prey types (treated uniformly for now) -> Hooks left for extension.
    - Detailed per-step logging & lifecycle stats (lightweight versions only).

This file is intentionally separate to avoid breaking the discrete environment.
"""
from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import numpy as np
import gymnasium as gym
try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv  # type: ignore
    from ray.rllib.utils.typing import AgentID  # type: ignore
except Exception:  # minimal fallback to allow basic import without full ray[tune]
    class MultiAgentEnv:  # type: ignore
        pass
    AgentID = str  # type: ignore

Array = np.ndarray


class PredPreyGrassContinuous(MultiAgentEnv):
    def __init__(self, config: Dict | None = None):
        try:
            super().__init__()
        except Exception:
            # Allow import without full RLlib
            pass
        if config is None:
            raise ValueError("Config required.")
        self.cfg = config
        self.rng = np.random.default_rng(config.get("seed", 42))
        self._init_config_defaults()
        self._build_agent_id_sets()
        self._build_spaces()
        # Runtime flags
        self.reset_pending = True
        # Agents marked done in the last step; cleaned up at the start of the next step
        self._pending_removals = set()
        # Rendering state (lazy init)
        self._fig = None
        self._ax = None
        self._render_initialized = False
        self.metadata = {"render_modes": ["human", "rgb_array"], "name": "PredPreyGrassContinuous"}

    # ---------------------------------------------------------------
    # Config
    # ---------------------------------------------------------------
    def _init_config_defaults(self):
        c = self.cfg
        # world / dynamics
        self.world_size = c.get("world_size", 20.0)
        self.catch_radius = c.get("catch_radius", 0.6)
        self.eat_radius = c.get("eat_radius", 0.5)
        self.max_speed_predator = c.get("max_speed_predator", 0.9)
        self.max_speed_prey = c.get("max_speed_prey", 1.0)
        self.cont_move_cost_factor = c.get("continuous_move_cost_factor", 0.01)
        # populations
        self.n_initial_predators = c.get("n_initial_predators", 6)
        self.n_initial_prey = c.get("n_initial_prey", 10)
        self.n_grass = c.get("n_grass_patches", 40)
        # energy
        self.initial_energy_predator = c.get("initial_energy_predator", 5.0)
        self.initial_energy_prey = c.get("initial_energy_prey", 3.0)
        self.grass_max_energy = c.get("grass_max_energy", 2.0)
        self.grass_regrow_rate = c.get("grass_regrow_rate", 0.05)
        # per-step decay & thresholds
        self.energy_loss_per_step_predator = c.get("energy_loss_per_step_predator", 0.15)
        self.energy_loss_per_step_prey = c.get("energy_loss_per_step_prey", 0.05)
        self.predator_creation_energy_threshold = c.get("predator_creation_energy_threshold", 12.0)
        self.prey_creation_energy_threshold = c.get("prey_creation_energy_threshold", 8.0)
        # reproduction
        self.reproduction_cooldown_steps = c.get("reproduction_cooldown_steps", 10)
        self.reproduction_chance_predator = c.get("reproduction_chance_predator", 1.0)
        self.reproduction_chance_prey = c.get("reproduction_chance_prey", 1.0)
        self.reproduction_reward_predator = c.get("reproduction_reward_predator", 10.0)
        self.reproduction_reward_prey = c.get("reproduction_reward_prey", 10.0)
        self.reproduction_energy_efficiency = c.get("reproduction_energy_efficiency", 1.0)
        # capacity for dynamic agents (pre-register spaces for new API stack)
        self.max_agents_per_type = c.get("max_agents_per_type", 256)
        # rewards
        self.reward_predator_catch_prey = c.get("reward_predator_catch_prey", 0.0)
        self.reward_prey_eat_grass = c.get("reward_prey_eat_grass", 0.0)
        self.reward_predator_step = c.get("reward_predator_step", 0.0)
        self.reward_prey_step = c.get("reward_prey_step", 0.0)
        self.penalty_prey_caught = c.get("penalty_prey_caught", 0.0)
        # episode control
        self.max_steps = c.get("max_steps", 5000)
        # observation detail
        self.nearest_k_agents = c.get("nearest_k_agents", 4)
        self.nearest_k_grass = c.get("nearest_k_grass", 3)
        # observation encoding mode: 'vector' (existing) or 'grid'
        self.obs_mode = c.get("obs_mode", "vector")  # 'vector' or 'grid'
        self.obs_grid_size = c.get("obs_grid_size", 15)  # grid resolution (cells per side) if grid mode
        self.obs_grid_use_energy = c.get("obs_grid_use_energy", False)  # accumulate energy instead of presence
        self.obs_grid_wall_mode = c.get("obs_grid_wall_mode", "binary")  # 'binary' or 'distance'
        self.obs_grid_circular_mask = c.get("obs_grid_circular_mask", False)  # if True, zero cells outside radius circle
        # spawn jitter
        self.spawn_jitter = c.get("spawn_jitter", 0.3)
        # seeds
        self.seed = c.get("seed", 42)
        # rendering
        self.render_mode = c.get("render_mode")
        self.render_dpi = c.get("render_dpi", 100)
        self.render_figsize = c.get("render_figsize", (5, 5))
        self.render_show_grass_ids = c.get("render_show_grass_ids", False)
        self.render_draw_catch_radius = c.get("render_draw_catch_radius", True)
        self.render_draw_eat_radius = c.get("render_draw_eat_radius", False)
        self.render_background_color = c.get("render_background_color", "#101418")
        self.render_predator_color = c.get("render_predator_color", "crimson")
        self.render_prey_color = c.get("render_prey_color", "royalblue")
        self.render_grass_cmap = c.get("render_grass_cmap", "Greens")
        self.render_energy_alpha = c.get("render_energy_alpha", True)
        self.render_grass_size = c.get("render_grass_size", 35.0)
        self.render_agent_size = c.get("render_agent_size", 70.0)
        # perception radius (None = unlimited)
        self.vision_radius = c.get("vision_radius", None)
        self.vision_radius_predator = c.get("vision_radius_predator", self.vision_radius)
        self.vision_radius_prey = c.get("vision_radius_prey", self.vision_radius)
        # vision radius rendering (outer observation ring)
        self.render_draw_vision_radius = c.get("render_draw_vision_radius", True)
        self.render_vision_radius_color = c.get("render_vision_radius_color", "#ffaa00")
        self.render_vision_radius_lw = c.get("render_vision_radius_lw", 0.6)
        self.render_vision_radius_alpha = c.get("render_vision_radius_alpha", 0.35)
        # distinct predator / prey ring customization
        self.render_vision_radius_color_predator = c.get(
            "render_vision_radius_color_predator", self.render_vision_radius_color
        )
        self.render_vision_radius_color_prey = c.get(
            "render_vision_radius_color_prey", "#00d4ff"
        )
        self.render_vision_radius_style_predator = c.get(
            "render_vision_radius_style_predator", "--"
        )
        self.render_vision_radius_style_prey = c.get(
            "render_vision_radius_style_prey", ":"
        )
        self.render_vision_radius_lw_predator = c.get(
            "render_vision_radius_lw_predator", self.render_vision_radius_lw
        )
        self.render_vision_radius_lw_prey = c.get(
            "render_vision_radius_lw_prey", self.render_vision_radius_lw
        )

    def _build_agent_id_sets(self):
        self.predator_ids = [f"predator_{i}" for i in range(self.n_initial_predators)]
        self.prey_ids = [f"prey_{i}" for i in range(self.n_initial_prey)]
        self.possible_agents = self.predator_ids + self.prey_ids
        # Monotonic per-type ID counters (avoid reusing agent IDs within an episode)
        self.next_predator_idx = len(self.predator_ids)
        self.next_prey_idx = len(self.prey_ids)

    # ---------------------------------------------------------------
    # Spaces
    # ---------------------------------------------------------------
    def _calc_obs_size(self) -> int:
        if self.obs_mode == "grid":
            # 4 channels (walls, predators, prey, grass) * grid_size^2
            return 4 * self.obs_grid_size * self.obs_grid_size
        # vector mode
        core = 5  # self_x, self_y, energy, num_prey, num_predators
        agent_block = 4  # (dx, dy, energy, type_flag)
        grass_block = 3  # (dx, dy, energy)
        return core + self.nearest_k_agents * agent_block + self.nearest_k_grass * grass_block

    def _build_spaces(self):
        if self.obs_mode == "grid":
            shape = (4, self.obs_grid_size, self.obs_grid_size)
            self._obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)
        else:
            obs_size = self._calc_obs_size()
            self._obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self._act_space_pred = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._act_space_prey = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # RLlib new API prefers plural mapping attributes. Pre-register a wide range
        # of potential agent IDs so connectors know the action/obs spaces for newborns.
        self.observation_spaces = {}
        self.action_spaces = {}
        for i in range(self.max_agents_per_type):
            pid = f"predator_{i}"
            aid = f"prey_{i}"
            self.observation_spaces[pid] = self._obs_space
            self.observation_spaces[aid] = self._obs_space
            self.action_spaces[pid] = self._act_space_pred
            self.action_spaces[aid] = self._act_space_prey

    def observation_space(self, agent_id):  # RLlib legacy accessor
        return self._obs_space

    def action_space(self, agent_id):  # RLlib legacy accessor
        return self._act_space_pred if agent_id.startswith("predator") else self._act_space_prey

    # ---------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.current_step = 0
        # reset ID counters at episode start
        self.next_predator_idx = len(self.predator_ids)
        self.next_prey_idx = len(self.prey_ids)
        # state dicts
        self.positions = {}
        self.energies = {}
        self.last_reproduction_step = {}
        # spawn predators & prey
        for pid in self.predator_ids:
            self.positions[pid] = self.rng.random(2) * self.world_size
            self.energies[pid] = self.initial_energy_predator
            self.last_reproduction_step[pid] = -self.reproduction_cooldown_steps
        for aid in self.prey_ids:
            self.positions[aid] = self.rng.random(2) * self.world_size
            self.energies[aid] = self.initial_energy_prey
            self.last_reproduction_step[aid] = -self.reproduction_cooldown_steps
        # grass patches
        self.grass_pos = [self.rng.random(2) * self.world_size for _ in range(self.n_grass)]
        self.grass_energy = np.full(self.n_grass, self.grass_max_energy, dtype=np.float32)
        self.agents = [a for a in self.possible_agents]  # active list
        self._pending_removals.clear()
        obs = {a: self._build_observation(a) for a in self.agents}
        # If a render mode was specified at construction, allow immediate frame
        if self.render_mode == "human":
            self.render()
        return obs, {}

    # ---------------------------------------------------------------
    # Step
    # ---------------------------------------------------------------
    def step(self, action_dict):
        obs, rew, terms, truncs, infos = {}, {}, {}, {}, {}
        # First, remove any agents that were marked done in the previous step
        if self._pending_removals:
            for a in list(self._pending_removals):
                if a in self.positions:
                    del self.positions[a]
                if a in self.energies:
                    del self.energies[a]
                if a in self.agents:
                    try:
                        self.agents.remove(a)
                    except ValueError:
                        pass
            self._pending_removals.clear()
        # truncation check
        if self.current_step >= self.max_steps:
            for a in self.agents:
                obs[a] = self._build_observation(a)
                rew[a] = 0.0
                terms[a] = False
                truncs[a] = True
            terms["__all__"] = False
            truncs["__all__"] = True
            return obs, rew, terms, truncs, infos

        # energy decay base
        for a in list(self.agents):
            if a not in self.positions:  # might have died earlier in step
                continue
            if a.startswith("predator"):
                self.energies[a] -= self.energy_loss_per_step_predator
            else:
                self.energies[a] -= self.energy_loss_per_step_prey

        # apply actions
        for a, act in action_dict.items():
            if a not in self.positions:
                continue
            act = np.asarray(act, dtype=np.float32)
            act = np.clip(act, -1.0, 1.0)
            max_speed = self.max_speed_predator if a.startswith("predator") else self.max_speed_prey
            delta = act * max_speed
            old_pos = self.positions[a].copy()
            new_pos = np.clip(old_pos + delta, 0.0, self.world_size)
            dist = np.linalg.norm(new_pos - old_pos)
            # movement energy cost proportional to distance * current energy
            self.energies[a] -= dist * self.cont_move_cost_factor * max(self.energies[a], 0.0)
            self.positions[a] = new_pos

        # predator-prey interactions (catch)
        dead_this_step: set[str] = set()
        caught_prey_set: set[str] = set()
        for pred in [a for a in self.agents if a.startswith("predator") and a in self.positions]:
            ppos = self.positions[pred]
            for prey in [b for b in self.agents if b.startswith("prey") and b in self.positions and b not in dead_this_step]:
                d = np.linalg.norm(self.positions[prey] - ppos)
                if d <= self.catch_radius:
                    # transfer energy (capped by prey current energy)
                    prey_energy = self.energies[prey]
                    self.energies[pred] += prey_energy
                    rew[pred] = rew.get(pred, 0.0) + self.reward_predator_catch_prey
                    if prey not in caught_prey_set:
                        rew[prey] = rew.get(prey, 0.0) + self.penalty_prey_caught
                    caught_prey_set.add(prey)
        for prey in caught_prey_set:
            # Mark as done now; clean up next step to keep connector episode state consistent
            terms[prey] = True
            truncs[prey] = False
            dead_this_step.add(prey)
            self._pending_removals.add(prey)
        # prey-grass interactions
        for prey in [a for a in self.agents if a.startswith("prey") and a in self.positions and a not in dead_this_step]:
            ppos = self.positions[prey]
            for gi, gpos in enumerate(self.grass_pos):
                if self.grass_energy[gi] <= 0:
                    continue
                d = np.linalg.norm(ppos - gpos)
                if d <= self.eat_radius:
                    gain = self.grass_energy[gi]
                    self.energies[prey] += gain
                    rew[prey] = rew.get(prey, 0.0) + self.reward_prey_eat_grass
                    self.grass_energy[gi] = 0.0  # consumed and will regrow

        # grass regrow
        regrow = self.grass_energy < self.grass_max_energy
        self.grass_energy[regrow] = np.minimum(self.grass_energy[regrow] + self.grass_regrow_rate, self.grass_max_energy)

        # starvation deaths
        for a in list(self.agents):
            if a in self.energies and self.energies[a] <= 0 and a not in dead_this_step:
                terms[a] = True
                truncs[a] = False
                dead_this_step.add(a)
                self._pending_removals.add(a)

        # reproduction (simple, no mutation, same type new id appended index)
        new_agents: List[str] = []
        for a in list(self.agents):
            if a not in self.positions:
                continue
            if a in dead_this_step:
                continue
            if a.startswith("predator"):
                threshold = self.predator_creation_energy_threshold
                chance = self.reproduction_chance_predator
                init_energy = self.initial_energy_predator
                repro_reward = self.reproduction_reward_predator
            else:
                threshold = self.prey_creation_energy_threshold
                chance = self.reproduction_chance_prey
                init_energy = self.initial_energy_prey
                repro_reward = self.reproduction_reward_prey

            if self.energies[a] >= threshold and (self.current_step - self.last_reproduction_step[a]) >= self.reproduction_cooldown_steps:
                if self.rng.random() <= chance:
                    base_name, _idx = a.split("_")  # e.g., predator_0
                    group = base_name  # 'predator' or 'prey'
                    # Assign a globally unique (within episode) increasing index per type
                    if group == "predator":
                        child_id = f"predator_{self.next_predator_idx}"
                        self.next_predator_idx += 1
                    else:
                        child_id = f"prey_{self.next_prey_idx}"
                        self.next_prey_idx += 1
                    # spawn
                    jitter = self.rng.normal(0, self.spawn_jitter, size=2)
                    pos = np.clip(self.positions[a] + jitter, 0.0, self.world_size)
                    self.positions[child_id] = pos
                    child_energy = init_energy * self.reproduction_energy_efficiency
                    self.energies[child_id] = child_energy
                    self.energies[a] -= init_energy  # energy cost
                    self.last_reproduction_step[a] = self.current_step
                    self.last_reproduction_step[child_id] = self.current_step
                    new_agents.append(child_id)
                    rew[a] = rew.get(a, 0.0) + repro_reward
        self.agents.extend(new_agents)

        # step rewards for survivors if not already assigned (don't add step reward for agents that died this step)
        for a in self.agents:
            if a in dead_this_step:
                continue
            if a not in rew:
                if a.startswith("predator"):
                    rew[a] = self.reward_predator_step
                else:
                    rew[a] = self.reward_prey_step
            # obs + survival statuses set below

        # build observations for all agents (including those that just terminated)
        # RLlib's connectors expect an observation for agents marked done in this step.
        for a in self.agents:
            obs[a] = self._build_observation(a)
            if a not in dead_this_step:
                terms.setdefault(a, False)
                truncs.setdefault(a, False)

        # Ensure every agent that acted (i.e., appears in obs) has a reward and status keys.
        for a in list(obs.keys()):
            if a not in rew:
                rew[a] = 0.0
            terms.setdefault(a, False)
            truncs.setdefault(a, False)
        # Prune rewards for agents not in this step's obs keys.
        for a in list(rew.keys()):
            if a not in obs:
                del rew[a]

        # Global termination: only when no agents remain. Avoid ending when one side is extinct
        # to keep connector state consistent across steps.
        done_all = len(self.agents) == 0
        terms["__all__"] = done_all
        truncs["__all__"] = False

        self.current_step += 1
        return obs, rew, terms, truncs, infos

    # ---------------------------------------------------------------
    # Observation construction
    # ---------------------------------------------------------------
    def _build_observation(self, agent: str) -> Array:
        if self.obs_mode == "grid":
            return self._build_observation_grid(agent)
        pos = self.positions[agent]
        energy = self.energies[agent]
        predators = [(a, self.positions[a], self.energies[a]) for a in self.agents if a.startswith("predator")]
        prey = [(a, self.positions[a], self.energies[a]) for a in self.agents if a.startswith("prey")]
        num_pred = len(predators)
        num_prey = len(prey)
        # gather other agents excluding self
        others = []
        # Determine focal agent vision radius (per-type)
        focal_radius = self.vision_radius_predator if agent.startswith("predator") else self.vision_radius_prey
        for (a, p, e) in predators + prey:
            if a == agent:
                continue
            if focal_radius is not None and np.linalg.norm(p - pos) > focal_radius:
                continue
            others.append((a, p, e))
        # compute distances
        rel = []
        for a_id, p, e in others:
            dpos = p - pos
            rel.append((np.linalg.norm(dpos), dpos[0], dpos[1], e, 1.0 if a_id.startswith("predator") else 0.0))
        rel.sort(key=lambda x: x[0])
        kA = self.nearest_k_agents
        agent_feats: List[float] = []
        for i in range(kA):
            if i < len(rel):
                _, dx, dy, e, tflag = rel[i]
                agent_feats.extend([dx, dy, e, tflag])
            else:
                agent_feats.extend([0.0, 0.0, 0.0, 0.0])
        # nearest grass
        grass_rel = []
        for gi, gpos in enumerate(self.grass_pos):
            if focal_radius is not None and np.linalg.norm(gpos - pos) > focal_radius:
                continue
            dpos = gpos - pos
            grass_rel.append((np.linalg.norm(dpos), dpos[0], dpos[1], float(self.grass_energy[gi])))
        grass_rel.sort(key=lambda x: x[0])
        kG = self.nearest_k_grass
        grass_feats: List[float] = []
        for i in range(kG):
            if i < len(grass_rel):
                _, dx, dy, ge = grass_rel[i]
                grass_feats.extend([dx, dy, ge])
            else:
                grass_feats.extend([0.0, 0.0, 0.0])
        core = [pos[0], pos[1], energy, float(num_prey), float(num_pred)]
        obs_vec = np.array(core + agent_feats + grass_feats, dtype=np.float32)
        return obs_vec

    def _build_observation_grid(self, agent: str) -> Array:
        """Grid-based multi-channel observation.

        Channels:
          0: walls (1 where cell center falls outside world bounds, else 0)
          1: predators (presence or normalized energy)
          2: prey (presence or normalized energy)
          3: grass (presence or normalized energy)

        The grid is centered on the agent. Cells cover a square of side 2*vision_radius.
        Requires vision_radius (global or per-type) to be set.
        """
        # Determine per-type vision radius
        radius = self.vision_radius_predator if agent.startswith("predator") else self.vision_radius_prey
        if radius is None:
            raise ValueError("Grid observation mode requires a (per-type) vision_radius to be set.")
        gs = self.obs_grid_size
        half = radius
        cell_size = (2 * half) / gs
        # Initialize channels
        grid = np.zeros((4, gs, gs), dtype=np.float32)
        ax, ay = self.positions[agent]
        # Precompute entity lists
        # Walls channel: mark cells whose center lies outside bounds
        for ix in range(gs):
            cx = ax + (ix + 0.5) * cell_size - half
            for iy in range(gs):
                cy = ay + (iy + 0.5) * cell_size - half
                if cx < 0 or cy < 0 or cx > self.world_size or cy > self.world_size:
                    grid[0, iy, ix] = 1.0
                elif self.obs_grid_wall_mode == "distance":
                    # proximity to nearest world boundary (inverse distance normalized by radius)
                    min_dist = min(cx, cy, self.world_size - cx, self.world_size - cy)
                    # if outside radius from any wall, contribution is lower
                    proximity = max(0.0, (radius - min_dist) / max(radius, 1e-6))
                    if proximity > grid[0, iy, ix]:
                        grid[0, iy, ix] = proximity
        # Helper to drop out-of-radius cells if we want circular mask (optional)
        # We'll keep square; circular could zero outside circle for aesthetics.
        use_energy = self.obs_grid_use_energy
        # Normalization factors (avoid div by zero)
        pred_norm = max(self.predator_creation_energy_threshold, 1.0)
        prey_norm = max(self.prey_creation_energy_threshold, 1.0)
        grass_norm = max(self.grass_max_energy, 1.0)
        # Populate agents & grass
        for a_id, pos in self.positions.items():
            if a_id == agent:
                continue
            dx, dy = pos - np.array([ax, ay])
            dist = np.hypot(dx, dy)
            if dist > radius:
                continue
            ix = int((dx + half) / (2 * half) * gs)
            iy = int((dy + half) / (2 * half) * gs)
            if ix < 0 or ix >= gs or iy < 0 or iy >= gs:
                continue
            if a_id.startswith("predator"):
                val = self.energies[a_id] / pred_norm if use_energy else 1.0
                grid[1, iy, ix] = min(1.0, grid[1, iy, ix] + val)
            else:
                val = self.energies[a_id] / prey_norm if use_energy else 1.0
                grid[2, iy, ix] = min(1.0, grid[2, iy, ix] + val)
        for gi, gpos in enumerate(self.grass_pos):
            dx, dy = gpos - np.array([ax, ay])
            dist = np.hypot(dx, dy)
            if dist > radius:
                continue
            ix = int((dx + half) / (2 * half) * gs)
            iy = int((dy + half) / (2 * half) * gs)
            if ix < 0 or ix >= gs or iy < 0 or iy >= gs:
                continue
            val = (self.grass_energy[gi] / grass_norm) if use_energy else (1.0 if self.grass_energy[gi] > 0 else 0.0)
            grid[3, iy, ix] = min(1.0, grid[3, iy, ix] + val)
        # Optional circular mask (keeps only cells whose center lies within radius)
        if self.obs_grid_circular_mask:
            for ix in range(gs):
                cx = ax + (ix + 0.5) * cell_size - half
                for iy in range(gs):
                    cy = ay + (iy + 0.5) * cell_size - half
                    if (cx - ax) ** 2 + (cy - ay) ** 2 > radius ** 2:
                        # zero non-wall channels; keep wall encoding if already set
                        grid[1:, iy, ix] = 0.0
        return grid

    # ---------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------
    def render(self):  # Placeholder
        if self.render_mode is None:
            # allow ad-hoc usage by defaulting to human if not set
            mode = "human"
        else:
            mode = self.render_mode

        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib import cm
        except ImportError as e:
            raise RuntimeError("matplotlib is required for rendering. Install it or disable rendering.") from e

        if not self._render_initialized:
            self._fig, self._ax = plt.subplots(figsize=self.render_figsize, dpi=self.render_dpi)
            self._fig.patch.set_facecolor(self.render_background_color)
            self._ax.set_facecolor(self.render_background_color)
            self._render_initialized = True
            if mode == "human":
                plt.ion()
                plt.show(block=False)

        ax = self._ax
        ax.clear()
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_title(f"PredPreyGrassContinuous t={self.current_step}", color="w")
        ax.tick_params(colors="w", which="both")
        for spine in ax.spines.values():
            spine.set_color("w")

        # Draw grass
        if self.n_grass > 0:
            gpos = np.array(self.grass_pos)
            gE = self.grass_energy
            if self.render_energy_alpha:
                alpha_vals = (gE / (self.grass_max_energy + 1e-8)).clip(0, 1)
            else:
                alpha_vals = np.full_like(gE, 1.0)
            cmap = cm.get_cmap(self.render_grass_cmap)
            colors = [cmap(a) for a in (gE / (self.grass_max_energy + 1e-8)).clip(0, 1)]
            for (x, y), col, a in zip(gpos, colors, alpha_vals):
                ax.scatter(x, y, s=self.render_grass_size, color=col, alpha=a, edgecolors='none')

        # Draw agents
        pred_xy = []
        pred_E = []
        prey_xy = []
        prey_E = []
        for a in self.agents:
            if a not in self.positions:
                continue
            pos = self.positions[a]
            if a.startswith("predator"):
                pred_xy.append(pos)
                pred_E.append(self.energies[a])
            else:
                prey_xy.append(pos)
                prey_E.append(self.energies[a])
        def _norm(vals):
            if not vals:
                return []
            vmax = max(vals) + 1e-8
            return [v / vmax for v in vals]
        pred_norm = _norm(pred_E)
        prey_norm = _norm(prey_E)
        # Draw predators
        for (x, y), a in zip(pred_xy, pred_norm):
            rad = self.vision_radius_predator
            if rad is not None and self.render_draw_vision_radius:
                ax.add_patch(matplotlib.patches.Circle(
                    (x, y), rad, fill=False,
                    ec=self.render_vision_radius_color_predator,
                    lw=self.render_vision_radius_lw_predator,
                    alpha=self.render_vision_radius_alpha,
                    linestyle=self.render_vision_radius_style_predator,
                    zorder=1))
            ax.scatter(x, y, s=self.render_agent_size, c=self.render_predator_color,
                       alpha=0.4 + 0.6 * a, edgecolors='k', linewidths=1.0, marker='o', zorder=3)
            if self.render_draw_catch_radius:
                ax.add_patch(matplotlib.patches.Circle((x, y), self.catch_radius, fill=False, ec="#ff8080", lw=0.6, alpha=0.6, zorder=4))
        # Draw prey
        for (x, y), a in zip(prey_xy, prey_norm):
            rad = self.vision_radius_prey
            if rad is not None and self.render_draw_vision_radius:
                ax.add_patch(matplotlib.patches.Circle(
                    (x, y), rad, fill=False,
                    ec=self.render_vision_radius_color_prey,
                    lw=self.render_vision_radius_lw_prey,
                    alpha=self.render_vision_radius_alpha,
                    linestyle=self.render_vision_radius_style_prey,
                    zorder=1))
            ax.scatter(x, y, s=self.render_agent_size, c=self.render_prey_color,
                       alpha=0.4 + 0.6 * a, edgecolors='k', linewidths=1.0, marker='o', zorder=3)
            if self.render_draw_eat_radius:
                ax.add_patch(matplotlib.patches.Circle((x, y), self.eat_radius, fill=False, ec="#80c0ff", lw=0.6, alpha=0.6, zorder=4))

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X', color='w')
        ax.set_ylabel('Y', color='w')

        if mode == "human":
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            return None
        elif mode == "rgb_array":
            self._fig.canvas.draw()
            w, h = self._fig.canvas.get_width_height()
            canvas = self._fig.canvas
            # Try standard RGB first
            try:
                buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                return buf.reshape(h, w, 3)
            except Exception:
                # Fallback for backends that provide ARGB (e.g., some TkAgg variants)
                try:
                    argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
                    # Convert ARGB to RGB by dropping alpha after reordering
                    rgb = argb[:, :, 1:4]  # ignore alpha (argb[:,:,0])
                    return rgb.copy()
                except Exception as e:
                    raise RuntimeError("Failed to extract RGB frame from matplotlib canvas") from e
        else:
            raise ValueError(f"Unsupported render_mode: {mode}")

    def close(self):
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            except Exception:
                pass
            self._fig = None
            self._ax = None
            self._render_initialized = False


__all__ = ["PredPreyGrassContinuous"]
