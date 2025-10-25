import numpy as np
np.set_printoptions(precision=8, suppress=False)
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from numba import njit
import pygame


@njit(cache=True)
def _get_local_observation_numba(ix, pos, active, is_pred, energy, grass, obs_range, num_obs_channels, grid_h, grid_w):
    half = obs_range // 2
    x, y = pos[ix]
    patch = np.zeros((num_obs_channels, obs_range, obs_range), dtype=np.float32)
    xlo = max(x - half, 0)
    xhi = min(x + half + 1, grid_h)
    ylo = max(y - half, 0)
    yhi = min(y + half + 1, grid_w)
    pxlo = half - (x - xlo)
    pxhi = pxlo + (xhi - xlo)
    pylo = half - (y - ylo)
    pyhi = pylo + (yhi - ylo)
    patch[0, :, :] = 1.0
    patch[0, pxlo:pxhi, pylo:pyhi] = 0.0
    patch[3, pxlo:pxhi, pylo:pyhi] = grass[xlo:xhi, ylo:yhi]
    for i in range(pos.shape[0]):
        if not active[i]:
            continue
        ax, ay = pos[i]
        if (ax >= xlo) and (ax < xhi) and (ay >= ylo) and (ay < yhi):
            px = pxlo + (ax - xlo)
            py = pylo + (ay - ylo)
            if is_pred[i]:
                patch[1, px, py] += energy[i]
            else:
                patch[2, px, py] += energy[i]
    return patch

class PredPreyGrassEnv(MultiAgentEnv):
    """
    Vectorized multi-agent env with predators, prey, and grass.
    - Fixed universe of possible agent IDs
    - Vectorized reset
    - RLlib-style reset: returns (observations_dict, infos_dict)
    - Supports Gymnasium-style truncation via max_episode_steps

    """
    def __init__(
        self,
        grid_shape: tuple[int, int] = (25, 25),
        num_possible_predators: int = 50,
        num_possible_prey: int = 50,
        initial_num_predators: int = 10,
        initial_num_prey: int = 10,
        initial_num_grass: int = 100,                 # how many grass cells to seed
        initial_energy_grass: float = 2.0,
        initial_energy_predator: float = 5.0,
        initial_energy_prey: float = 3.0,
        seed: int | None = None,
        predator_creation_energy_threshold: float = 10.0,
        prey_creation_energy_threshold: float = 6.0,
        energy_loss_per_step_predator: float = 0.15,
        energy_loss_per_step_prey: float = 0.05,
        energy_gain_per_step_grass: float = 0.04,
        max_grass_energy: float = 2.0,
        obs_range: int = 7,
        num_obs_channels: int = 4,
        obs_build: str = "global_maps",
        max_episode_steps: int | None = None,
    ):
        self.grid_h, self.grid_w = int(grid_shape[0]), int(grid_shape[1])
        assert self.grid_h > 0 and self.grid_w > 0

        # counts
        self.num_possible_predators = int(num_possible_predators)
        self.num_possible_prey = int(num_possible_prey)
        self.initial_num_predators = int(initial_num_predators)
        self.initial_num_prey = int(initial_num_prey)
       
        self.N_agents = self.initial_num_predators + self.initial_num_prey  # num_agents is reserved for read-only properties in RLlib
        self.num_possible_agents = self.num_possible_predators + self.num_possible_prey
        self.initial_num_grass = int(initial_num_grass)

        # energies
        self.initial_energy_grass = float(initial_energy_grass)
        self.initial_energy_predator = float(initial_energy_predator)
        self.initial_energy_prey = float(initial_energy_prey)
        self.predator_creation_energy_threshold = float(predator_creation_energy_threshold)
        self.prey_creation_energy_threshold = float(prey_creation_energy_threshold)

        # Per-step maintenance costs (energy drain); adjust as needed
        self.energy_loss_per_step_predator = energy_loss_per_step_predator
        self.energy_loss_per_step_prey = energy_loss_per_step_prey

        # Grass regrowth parameters
        self.energy_gain_per_step_grass = energy_gain_per_step_grass
        self.max_grass_energy = max_grass_energy

        # RNG
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Episode step limit/truncation
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        self._episode_steps = 0  # counts steps since last reset


        # ---- Fixed universe of agent IDs (strings) ----
        self._all_ids = np.array(
            [f"predator_{i}" for i in range(self.num_possible_predators)] +
            [f"prey_{i}" for i in range(self.num_possible_prey)],
            dtype=object
        )
        self._id_to_ix = {aid: i for i, aid in enumerate(self._all_ids)}

        # RLlib MultiAgentEnv: possible_agents is the full set of possible agent IDs
        self.possible_agents = list(self._all_ids)

        # role masks (vectorized selections)
        self.is_pred = np.zeros(self.num_possible_agents, dtype=bool)
        self.is_pred[:self.num_possible_predators] = True
        self.is_prey = ~self.is_pred

        # ---- Vectorized state arrays (allocated once) ----
        self.active = np.zeros(self.num_possible_agents, dtype=bool)       # who is alive
        self.pos = np.full((self.num_possible_agents, 2), -1, dtype=np.int32)
        self.energy = np.zeros(self.num_possible_agents, dtype=np.float32)
        self.age = np.zeros(self.num_possible_agents, dtype=np.int32)
        # Track whether an agent slot has been used at any point during the current episode.
        # A slot that has been used once (active at reset or spawned later) cannot be reused until reset.
        self._used_this_episode = np.zeros(self.num_possible_agents, dtype=bool)

        # Grass field (single-channel energy map)
        self.grass = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        # Mask of original seeded grass locations (bool)
        self._original_grass_mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)

        # RLlib/PettingZoo compatibility helpers
        self._active_dirty = True
        # Immediately refresh agents cache so self.agents is up-to-date after reset
        self._refresh_agents_cache()

        # ---- Observation/action spaces ----
        # Default local window size (can be overridden in reset)
        self.obs_range = obs_range
        self.num_obs_channels = num_obs_channels
        # Observation build strategy: 'global_maps' (fast) or 'per_agent' (fallback)
        self.obs_build = str(obs_build)
        obs_shape = (self.num_obs_channels, self.obs_range, self.obs_range)

        obs_low  = np.zeros(obs_shape, dtype=np.float32)
        obs_high = np.full(obs_shape, np.finfo(np.float32).max, dtype=np.float32)

        obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        act_space = spaces.Discrete(5)

        self.observation_space = obs_space
        self.action_space = act_space
        self.observation_spaces = {aid: obs_space for aid in self.possible_agents}
        self.action_spaces = {aid: act_space for aid in self.possible_agents}

        self.reproduction_reward_predator = 10.0
        self.reproduction_reward_prey = 10.0

    # ---------- Public API ----------
    def reset(self, *, seed: int | None = None, options: dict | None = None, obs_range: int | None = None):
        """
        RLlib-style reset -> (observations_dict, infos_dict)
        Vectorized initialization:
          - Activate all predators and prey
          - Sample positions (uniform on grid)
          - Assign initial energies
          - Seed grass cells (no overlap requirement between roles; grass can share a cell with agents)
          - Resets episode step counter; can override max_episode_steps via options
        """
        if seed is not None:
            # allow RLlib to reseed us between episodes
            self._seed = seed
            self.rng = np.random.default_rng(seed)

        # Allow overriding obs_range at episode start
        if obs_range is not None and int(obs_range) != self.obs_range:
            self.obs_range = int(obs_range)
            # Rebuild observation spaces to match the new shape
            obs_shape = (self.num_obs_channels, self.obs_range, self.obs_range)
            obs_low  = np.zeros(obs_shape, dtype=np.float32)
            obs_high = np.full(obs_shape, np.finfo(np.float32).max, dtype=np.float32)
            obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
            self.observation_space = obs_space
            self.observation_spaces = {aid: obs_space for aid in self.possible_agents}

        # ChatGPT: DOES THIS NEED TO BE HERE STILL?
        # Allow overriding max steps via options if provided
        if options is not None and 'max_episode_steps' in options:
            opt_steps = options.get('max_episode_steps')
            self.max_episode_steps = None if opt_steps is None else int(opt_steps)

        # Clear all agents
        self.active[:] = False
        self.pos[:] = -1
        self.energy[:] = 0.0
        self.age[:] = 0
        self._used_this_episode[:] = False
        self._episode_steps = 0
        self._active_dirty = True

        # Activate only the first initial_num_predators and the correct prey slots
        # Predators: first num_possible_predators slots, activate only initial_num_predators
        self.active[:self.initial_num_predators] = True
        # Prey: after all possible predators, activate only initial_num_prey
        prey_start = self.num_possible_predators
        self.active[prey_start:prey_start + self.initial_num_prey] = True
        # Mark initially activated slots as used for this episode
        if self.initial_num_predators > 0:
            self._used_this_episode[:self.initial_num_predators] = True
        if self.initial_num_prey > 0:
            self._used_this_episode[prey_start:prey_start + self.initial_num_prey] = True
        self._refresh_agents_cache()  # Update the active agent list for RLlib

        # ---- Sample positions (vectorized) ----
        # We allow predators and prey to overlap each other (simple); if you want uniqueness per role, sample without replacement.
        n_cells = self.grid_h * self.grid_w

        # Predators
        if self.initial_num_predators > 0:
            pred_lin = self.rng.integers(0, n_cells, size=self.initial_num_predators, endpoint=False)
            pred_x = (pred_lin // self.grid_w).astype(np.int32)
            pred_y = (pred_lin % self.grid_w).astype(np.int32)
            self.pos[0:self.initial_num_predators, 0] = pred_x
            self.pos[0:self.initial_num_predators, 1] = pred_y

        # Prey
        if self.initial_num_prey > 0:
            prey_lin = self.rng.integers(0, n_cells, size=self.initial_num_prey, endpoint=False)
            prey_x = (prey_lin // self.grid_w).astype(np.int32)
            prey_y = (prey_lin % self.grid_w).astype(np.int32)
            prey_start = self.num_possible_predators
            self.pos[prey_start:prey_start + self.initial_num_prey, 0] = prey_x
            self.pos[prey_start:prey_start + self.initial_num_prey, 1] = prey_y

        # ---- Energies (vectorized) ----
        self.energy[:] = 0.0
        self.energy[self.active & self.is_pred] = self.initial_energy_predator
        self.energy[self.active & self.is_prey]  = self.initial_energy_prey
        self.age[:] = 0

        # ---- Grass seeding (vectorized) ----
        self.grass.fill(0.0)
        self._original_grass_mask.fill(False)
        gcount = min(self.initial_num_grass, n_cells)
        if gcount > 0:
            # Sample unique cells for grass (without replacement)
            grass_lin = self.rng.choice(n_cells, size=gcount, replace=False)
            gx = (grass_lin // self.grid_w).astype(np.int32)
            gy = (grass_lin % self.grid_w).astype(np.int32)
            self.grass[gx, gy] = self.initial_energy_grass
            self._original_grass_mask[gx, gy] = True
        # Build observations for each agent (batch, possibly via global maps)
        agent_ids = self.agents
        observations = self._batch_observations(agent_ids, self.obs_range)

        idx = np.fromiter((self._id_to_ix[aid] for aid in agent_ids), dtype=np.int32, count=len(agent_ids))
        xs = self.pos[idx, 0].astype(int)
        ys = self.pos[idx, 1].astype(int)
        es = self.energy[idx].astype(float)
        ages = self.age[idx].astype(int)

        # Add info dict with coordinates for each agent
        infos = {aid: {"x": int(x), "y": int(y), "energy": float(e), "age": int(a), "step": self._episode_steps}
                for aid, x, y, e, a in zip(agent_ids, xs, ys, es, ages)}
        return observations, infos

    def step(self, action_dict):
        """
        RLlib MultiAgentEnv step function with fully vectorized agent movement.
        Args:
            action_dict: dict mapping agent_id to action (0: noop, 1: up, 2: right, 3: down, 4: left)
        Returns:
            observations: dict of new observations
            rewards: dict of rewards
            terminations: dict of done flags (per agent)
            truncations: dict of truncation flags (per agent). When max_episode_steps is reached, all non-terminated agents are truncated and '__all__' is True.
            infos: dict of info dicts (per agent)
        """
        # Step 0: Time maintenance - costs energy drain, (re)growing grass and aging
        agents_before_time_maintenance = set(self.agents)  # Snapshot of agents before time maintenance
        rewards = {aid: 0.0 for aid in agents_before_time_maintenance}  # intialize rewards at 0.0
        self._apply_basal_metabolism_and_aging()
        # After starvation, refresh agent cache and set terminations for starved agents
        terminations = self._refresh_and_set_terminations(agents_before_time_maintenance)
        # Step 1: Apply agent movements
        acted_agents = list(action_dict.keys())
        self._apply_agent_movement(action_dict)
        agents_before_engagement = set(self.agents)  # Snapshot of agents before engagement removals
        # Step 2: Resolve engagements (predator-prey and prey-grass)
        self._resolve_prey_grass_engagement()
        # Step 3: Resolve predator-prey engagements 
        self._resolve_predator_prey_engagement()
        # After engagements, refresh agent cache and set terminations for eaten/removed agents
        terminations = self._refresh_and_set_terminations(agents_before_engagement)
        # Step 4: Apply reproduction logic
        self._apply_reproduction(agents_before_engagement, terminations, rewards)

        # Build outputs ensuring that any truncated agents receive a final observation
        # Preserve agent order to keep deterministic key ordering in outputs.
        alive_agents = list(self.agents)

        # Increment episode step counter BEFORE computing truncation status to ensure consistency
        self._episode_steps += 1

        # Start with default flags for alive agents
        truncations = {aid: False for aid in alive_agents}
        # Apply truncation for alive, non-terminated agents if we hit the episode step limit
        did_truncate = False
        if (self.max_episode_steps is not None) and (self._episode_steps >= self.max_episode_steps):
            for aid in alive_agents:
                if not terminations.get(aid, False):
                    truncations[aid] = True
            did_truncate = len(alive_agents) > 0

        # Determine the complete set of agents we must report for this step:
        # - All currently alive agents
        # - Plus any agents that were truncated this step (must receive last obs for bootstrap)
        # - Plus agents that acted this step (to guarantee last obs delivery even if they got truncated)
        truncated_agents = [aid for aid, flag in truncations.items() if flag]
        base_list = alive_agents + truncated_agents + acted_agents
        report_agents = list(dict.fromkeys(base_list))  # preserve order, dedup

        observations = self._batch_observations(report_agents, self.obs_range)
        rewards = {aid: rewards.get(aid, 0.0) for aid in report_agents}
        terminations = {aid: terminations.get(aid, False) for aid in report_agents}
        truncations = {aid: truncations.get(aid, False) for aid in report_agents}
        infos = {}
        for aid in report_agents:
            ix = self._id_to_ix[aid]
            x, y = int(self.pos[ix, 0]), int(self.pos[ix, 1])
            energy = float(self.energy[ix])
            age = int(self.age[ix])
            infos[aid] = {"x": x, "y": y, "energy": energy, "age": age, "step": self._episode_steps}
        # Custom termination: end if no predators or no prey remain
        n_pred = int(np.sum(self.active & self.is_pred))
        n_prey = int(np.sum(self.active & self.is_prey))
        hard_done = (n_pred == 0 or n_prey == 0)

        terminations['__all__'] = bool(hard_done)
        truncations['__all__']  = False if hard_done else bool(did_truncate)

        return observations, rewards, terminations, truncations, infos

    def render(self, mode=None):
        """
        Render a single frame of the environment using Pygame.
        This function does not run a simulation loop or handle events.
        It simply draws the current state of the environment to a window.
        Grass patch color intensity is proportional to energy (0.0 to max_grass_energy).
        """
        cell_size = 20
        width, height = self.grid_w * cell_size, self.grid_h * cell_size
        if not hasattr(self, "_pygame_screen"):
            pygame.init()
            self._pygame_screen = pygame.display.set_mode((width, height))
            self._pygame_clock = pygame.time.Clock()
        screen = self._pygame_screen
        screen.fill((255, 255, 255))

        # Draw grass (size proportional to energy)
        for x in range(self.grid_h):
            for y in range(self.grid_w):
                g = self.grass[x, y]
                if g > 0:
                    # Patch size scales with energy/max_grass_energy
                    frac = min(g / self.max_grass_energy, 1.0)
                    patch_size = max(2, int(cell_size * frac))
                    offset = (cell_size - patch_size) // 2
                    pygame.draw.rect(
                        screen,
                        (0, 200, 0),
                        (y*cell_size + offset, x*cell_size + offset, patch_size, patch_size)
                    )

        # Draw agents
        for i, active in enumerate(self.active):
            if not active:
                continue
            x, y = self.pos[i]
            color = (200, 0, 0) if self.is_pred[i] else (0, 0, 200)
            pygame.draw.circle(screen, color, (int(y*cell_size+cell_size//2), int(x*cell_size+cell_size//2)), cell_size//2-2)

        pygame.display.flip()
        self._pygame_clock.tick(30)

    # ---------- Internal vectorized logic methods ----------
    def _apply_basal_metabolism_and_aging(self):
        """
        Vectorized per-step updates applied before movement:
        - Decrement predator and prey energy by their respective tick costs for active agents
        - Increment age of active agents by 1
        - Deactivate (kill) agents that reached non-positive energy and clamp their energy to 0
        - Regrow grass energy only in originally seeded patches
        """
        # Grass regrowth (applies only to originally seeded patches)
        self.grass[self._original_grass_mask] += self.energy_gain_per_step_grass
        np.clip(self.grass, 0.0, self.max_grass_energy, out=self.grass)

        active = self.active
        if not np.any(active):
            return
        # Masks
        pred_mask = active & self.is_pred
        prey_mask = active & self.is_prey
        # (debug print removed)
        # Energy drain
        if self.energy_loss_per_step_predator:
            self.energy[pred_mask] -= self.energy_loss_per_step_predator
        if self.energy_loss_per_step_prey:
            self.energy[prey_mask] -= self.energy_loss_per_step_prey
        # (debug print removed)
        # Aging
        self.age[active] += 1
        # Starvation: deactivate and clamp
        dead = active & (self.energy <= 0.0)
        if np.any(dead):
            self.active[dead] = False
            self.energy[dead] = 0.0
            self._active_dirty = True

    def _apply_agent_movement(self, action_dict):
        """
        Vectorized agent movement step. Applies actions to all active agents.
        """
        # Movement deltas: 0=noop, 1=up, 2=right, 3=down, 4=left
        deltas = np.array([
            [0, 0],    # noop
            [-1, 0],   # up
            [0, 1],    # right
            [1, 0],    # down
            [0, -1],   # left
        ], dtype=np.int32)

        # --- Vectorized action gather (no Python loop) ---
        agent_ids = self.agents  # cached view of current active IDs
        n = len(agent_ids)
        if n == 0:
            return

        # Map ids -> indices as a single vector op
        agent_indices = np.fromiter(
            (self._id_to_ix[aid] for aid in agent_ids),
            dtype=np.int32, count=n
        )

        # Gather actions in the same order; default invalid/missing to 0 (noop)
        actions = np.fromiter(
            (action_dict.get(aid, 0) for aid in agent_ids),
            dtype=np.int32, count=n
        )
        actions = np.where((actions >= 0) & (actions < deltas.shape[0]), actions, 0)

        # Vectorized movement
        pos = self.pos[agent_indices]
        move = deltas[actions]
        new_pos = pos + move

        # Clamp to grid
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, self.grid_h - 1)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, self.grid_w - 1)
        self.pos[agent_indices] = new_pos

    def _resolve_prey_grass_engagement(self):
        """
        Vectorized grazing: For each cell containing one or more active prey and grass > 0,
        select one prey uniformly at random to consume all grass at that cell.
        """
        # Active prey indices
        prey_mask = self.active & self.is_prey
        if not np.any(prey_mask):
            return
        prey_ix = np.flatnonzero(prey_mask)
        pos = self.pos[prey_ix]
        x = pos[:, 0].astype(np.int64)
        y = pos[:, 1].astype(np.int64)
        lin = x * self.grid_w + y

        # Group prey by linearized cell index
        order = np.argsort(lin)
        lin_sorted = lin[order]
        prey_ix_sorted = prey_ix[order]
        unique_lin, starts, counts = np.unique(lin_sorted, return_index=True, return_counts=True)

        # Grass energy at unique cells
        ux = (unique_lin // self.grid_w).astype(np.int64)
        uy = (unique_lin % self.grid_w).astype(np.int64)
        gvals = self.grass[ux, uy]

        eligible = gvals > 0.0
        if not np.any(eligible):
            return

        # Randomly select one prey per eligible cell group
        sel_counts = counts[eligible]
        sel_starts = starts[eligible]
        rand_offsets = self.rng.integers(0, sel_counts, size=sel_counts.shape[0])
        sel_sorted_idx = sel_starts + rand_offsets
        chosen_prey_ix = prey_ix_sorted[sel_sorted_idx]

        # Apply gains and zero the grass in those cells
        gains = gvals[eligible].astype(np.float32)
        self.energy[chosen_prey_ix] += gains
        self.grass[ux[eligible], uy[eligible]] = 0.0

    def _resolve_predator_prey_engagement(self):
        """
        Vectorized per-cell matching between predators and prey:
        - For each cell, shuffle predators and prey independently and match up to k=min(n_pred,n_prey).
        - Predators gain the prey's energy; prey are deactivated.
        """
        if not np.any(self.active):
            return

        # Active indices by role
        pred_mask = self.active & self.is_pred
        prey_mask = self.active & self.is_prey
        if not np.any(pred_mask) or not np.any(prey_mask):
            return

        pred_ix_all = np.flatnonzero(pred_mask)
        prey_ix_all = np.flatnonzero(prey_mask)

        # Linearized cell ids
        ppos = self.pos[pred_ix_all]
        plin = (ppos[:, 0].astype(np.int64) * self.grid_w + ppos[:, 1].astype(np.int64))
        qpos = self.pos[prey_ix_all]
        qlin = (qpos[:, 0].astype(np.int64) * self.grid_w + qpos[:, 1].astype(np.int64))

        # Randomize order within each cell by sorting on (lin, random key)
        prand = self.rng.random(size=pred_ix_all.shape[0])
        qrand = self.rng.random(size=prey_ix_all.shape[0])
        p_order = np.lexsort((prand, plin))
        q_order = np.lexsort((qrand, qlin))

        plin_sorted = plin[p_order]
        qlin_sorted = qlin[q_order]
        pred_ix_sorted = pred_ix_all[p_order]
        prey_ix_sorted = prey_ix_all[q_order]

        # Group boundaries per species
        u_plin, p_starts, p_counts = np.unique(plin_sorted, return_index=True, return_counts=True)
        u_qlin, q_starts, q_counts = np.unique(qlin_sorted, return_index=True, return_counts=True)

        # Cells where both species occur
        common_lin, idx_p_groups, idx_q_groups = np.intersect1d(u_plin, u_qlin, assume_unique=False, return_indices=True)
        if common_lin.size == 0:
            return

        # For each common cell, pair first k elements from randomized order
        pred_pairs = []
        prey_pairs = []
        for gi_p, gi_q in zip(idx_p_groups, idx_q_groups):
            start_p = p_starts[gi_p]
            cnt_p = p_counts[gi_p]
            start_q = q_starts[gi_q]
            cnt_q = q_counts[gi_q]
            k = cnt_p if cnt_p < cnt_q else cnt_q
            if k <= 0:
                continue
            pred_pairs.append(pred_ix_sorted[start_p:start_p + k])
            prey_pairs.append(prey_ix_sorted[start_q:start_q + k])

        if not pred_pairs:
            return

        pred_ix = np.concatenate(pred_pairs).astype(np.int32, copy=False)
        prey_ix = np.concatenate(prey_pairs).astype(np.int32, copy=False)

        # Energy transfer and prey deactivation
        gains = self.energy[prey_ix].astype(np.float32)
        self.energy[pred_ix] += gains

        if hasattr(self, "max_predator_energy") and self.max_predator_energy is not None:
            np.minimum(self.energy[pred_ix], self.max_predator_energy, out=self.energy[pred_ix])

        self.energy[prey_ix] = 0.0
        self.active[prey_ix] = False
        self._active_dirty = True

    def _apply_reproduction(self, agents_before_engagement, terminations, rewards):
        """
        Vectorized reproduction logic for predators and prey.
        Spawns new agents if parents are above energy threshold and slots are available.
        Excludes agents deactivated this step from being respawned immediately.
        """
        # Find just deactivated agents (to exclude from spawn pool)
        just_deactivated = (agents_before_engagement | set(terminations.keys()) - {'__all__'}) - set(self.agents)
        # Find eligible parents (active, above threshold)
        active_ix = np.flatnonzero(self.active)
        pred_ix = active_ix[self.is_pred[active_ix] & (self.energy[active_ix] >= self.predator_creation_energy_threshold)]
        prey_ix = active_ix[self.is_prey[active_ix] & (self.energy[active_ix] >= self.prey_creation_energy_threshold)]
        # No need to track parents; assign reward directly
        # Find available slots for new agents (not active, not used previously in this episode)
        all_ix = np.arange(self.num_possible_agents)
        available_ix = all_ix[~self.active & ~self._used_this_episode]
        # Exclude just deactivated (extra safety for within-step events)
        if just_deactivated:
            just_deactivated_ix = np.array([self._id_to_ix[aid] for aid in just_deactivated], dtype=np.int32)
            available_ix = np.setdiff1d(available_ix, just_deactivated_ix, assume_unique=True)
        # Spawn new predators
        n_pred_spawn = min(len(pred_ix), np.sum(self.is_pred[available_ix]))
        if n_pred_spawn > 0:
            spawn_pred_ix = available_ix[self.is_pred[available_ix]][:n_pred_spawn]
            parent_pred_ix = pred_ix[:n_pred_spawn]
            self.active[spawn_pred_ix] = True
            self.energy[spawn_pred_ix] = self.initial_energy_predator
            self.age[spawn_pred_ix] = 0
            self.pos[spawn_pred_ix] = self.pos[parent_pred_ix]  # spawn at parent location
            self.energy[parent_pred_ix] -= self.initial_energy_predator
            # Mark spawned slots as used for this episode
            self._used_this_episode[spawn_pred_ix] = True
            # Ensure new agents have reward 0.0
            for ix in spawn_pred_ix:
                rewards[self._all_ids[ix]] = 0.0
            # Assign reproduction reward directly
            for ix in parent_pred_ix:
                aid = self._all_ids[ix]
                if aid in rewards:
                    rewards[aid] += self.reproduction_reward_predator
        # Spawn new prey
        n_prey_spawn = min(len(prey_ix), np.sum(self.is_prey[available_ix]))
        if n_prey_spawn > 0:
            spawn_prey_ix = available_ix[self.is_prey[available_ix]][:n_prey_spawn]
            parent_prey_ix = prey_ix[:n_prey_spawn]
            self.active[spawn_prey_ix] = True
            self.energy[spawn_prey_ix] = self.initial_energy_prey
            self.age[spawn_prey_ix] = 0
            self.pos[spawn_prey_ix] = self.pos[parent_prey_ix]  # spawn at parent location
            self.energy[parent_prey_ix] -= self.initial_energy_prey
            # Mark spawned slots as used for this episode
            self._used_this_episode[spawn_prey_ix] = True
            # Ensure new agents have reward 0.0
            for ix in spawn_prey_ix:
                rewards[self._all_ids[ix]] = 0.0
            # Assign reproduction reward directly
            for ix in parent_prey_ix:
                aid = self._all_ids[ix]
                if aid in rewards:
                    rewards[aid] += self.reproduction_reward_prey

        # Refresh agent cache after possible spawns
        self._refresh_agents_cache()

    def _get_local_observation(self, agent_id: str, obs_range: int = 7) -> np.ndarray:
        """
        Returns a local grid patch centered on the agent, with channels:
        0: walls (1=out-of-bounds, 0=in-bounds),
        1: predator energy, 2: prey energy, 3: grass energy.
        Patch shape: (4, obs_range, obs_range)
        Fully vectorized for all agent channels.
        """
        return _get_local_observation_numba(
            self._id_to_ix[agent_id],
            self.pos,
            self.active,
            self.is_pred,
            self.energy,
            self.grass,
            obs_range,
            self.num_obs_channels,
            self.grid_h,
            self.grid_w
        )

    def _batch_observations(self, agent_ids: list[str], obs_range: int) -> dict[str, np.ndarray]:
        """
        Build observations for a list of agent_ids.
        Uses a fast global-map slicing path when self.obs_build == 'global_maps';
        otherwise falls back to per-agent numba kernel.
        """
        if not agent_ids:
            return {}
        if getattr(self, "obs_build", "global_maps") == "global_maps":
            return self._batch_observations_global_maps(agent_ids, obs_range)
        # Fallback: per-agent construction
        return {aid: self._get_local_observation(aid, obs_range) for aid in agent_ids}

    def _batch_observations_global_maps(self, agent_ids: list[str], obs_range: int) -> dict[str, np.ndarray]:
        """
        Fast observation builder using global energy maps for predators, prey, and grass,
        then slicing a padded array around each agent.
        Channels:
          0: walls mask (1 outside grid, 0 inside)
          1: predator energy
          2: prey energy
          3: grass energy
        If num_obs_channels > 4, remaining channels are zero.
        """
        # Precompute energy maps
        pred_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        prey_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        if np.any(self.active):
            active_ix = np.flatnonzero(self.active)
            if active_ix.size:
                pred_ix = active_ix[self.is_pred[active_ix]]
                prey_ix = active_ix[self.is_prey[active_ix]]
                if pred_ix.size:
                    px = self.pos[pred_ix, 0].astype(np.intp)
                    py = self.pos[pred_ix, 1].astype(np.intp)
                    np.add.at(pred_map, (px, py), self.energy[pred_ix].astype(np.float32))
                if prey_ix.size:
                    qx = self.pos[prey_ix, 0].astype(np.intp)
                    qy = self.pos[prey_ix, 1].astype(np.intp)
                    np.add.at(prey_map, (qx, qy), self.energy[prey_ix].astype(np.float32))

        # Padded maps for easy slicing
        half = obs_range // 2
        pad = half
        if pad > 0:
            pred_pad = np.pad(pred_map, ((pad, pad), (pad, pad)), mode='constant', constant_values=0.0)
            prey_pad = np.pad(prey_map, ((pad, pad), (pad, pad)), mode='constant', constant_values=0.0)
            grass_pad = np.pad(self.grass, ((pad, pad), (pad, pad)), mode='constant', constant_values=0.0)
        else:
            pred_pad = pred_map
            prey_pad = prey_map
            grass_pad = self.grass

        # Prepare outputs
        out: dict[str, np.ndarray] = {}
        ch = int(self.num_obs_channels)
        for aid in agent_ids:
            ix = self._id_to_ix[aid]
            x = int(self.pos[ix, 0])
            y = int(self.pos[ix, 1])

            # Slices on padded maps
            sx = x + pad - half
            sy = y + pad - half
            pred_patch = pred_pad[sx:sx+obs_range, sy:sy+obs_range]
            prey_patch = prey_pad[sx:sx+obs_range, sy:sy+obs_range]
            grass_patch = grass_pad[sx:sx+obs_range, sy:sy+obs_range]

            # Walls channel: ones outside real grid, zeros inside
            walls = np.ones((obs_range, obs_range), dtype=np.float32)
            xlo = max(x - half, 0)
            xhi = min(x + half + 1, self.grid_h)
            ylo = max(y - half, 0)
            yhi = min(y + half + 1, self.grid_w)
            pxlo = half - (x - xlo)
            pxhi = pxlo + (xhi - xlo)
            pylo = half - (y - ylo)
            pyhi = pylo + (yhi - ylo)
            walls[pxlo:pxhi, pylo:pyhi] = 0.0

            patch = np.zeros((ch, obs_range, obs_range), dtype=np.float32)
            # Assign up to 4 base channels
            if ch >= 1:
                patch[0] = walls
            if ch >= 2:
                patch[1] = pred_patch
            if ch >= 3:
                patch[2] = prey_patch
            if ch >= 4:
                patch[3] = grass_patch

            out[aid] = patch

        return out

    # ---------- Override MultiAgentEnv methods ----------
    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

    @property
    def agents(self) -> list[str]:
        if self._active_dirty:
            self._refresh_agents_cache()
        return self._agents_cache

    # ----- Helpers for managing active agents cache -----
    def _refresh_agents_cache(self):
        idx = np.flatnonzero(self.active)
        # Convert indices to strings once
        self._agents_cache = [self._all_ids[i] for i in idx.tolist()]
        self._active_dirty = False

    def _refresh_and_set_terminations(self, prev_active_set):
        """
        Refresh agent cache and return a terminations dict.

        NOTE: prev_active_set must be captured BEFORE any call to _refresh_agents_cache().
        This ensures it represents the set of agents before any removals in this phase.
        self.agents only changes after _refresh_agents_cache() is called.

        - terminations[aid] = False for all agents in prev_active_set
        - terminations[aid] = True for agents deactivated in this update
        - terminations['__all__'] = True if all agents are dead
        Returns: terminations dict
        """
        self._refresh_agents_cache()
        current_active_set = set(self.agents)
        terminations = {aid: False for aid in prev_active_set}
        just_deactivated = prev_active_set - current_active_set
        for aid in just_deactivated:
            terminations[aid] = True
        terminations['__all__'] = (len(current_active_set) == 0)
        return terminations

