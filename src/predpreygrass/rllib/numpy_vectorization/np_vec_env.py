import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from numba import njit
import pygame


@njit(cache=True)
def _get_local_observation_numba(ix, pos, active, is_pred, energy, grass, obs_range, grid_h, grid_w):
    half = obs_range // 2
    x, y = pos[ix]
    patch = np.zeros((4, obs_range, obs_range), dtype=np.float32)
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
        self.energy_loss_per_step_predator = 0.15
        self.energy_loss_per_step_prey = 0.05
        # Grass regrowth parameters
        self.energy_gain_per_step_grass = 0.04
        self.max_grass_energy = 2.0

        # RNG
        self._seed = seed
        self.rng = np.random.default_rng(seed)


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

        # Grass field (single-channel energy map)
        self.grass = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        # Mask of original seeded grass locations (bool)
        self._original_grass_mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)

        # RLlib/PettingZoo compatibility helpers
        self._active_dirty = True
        # Immediately refresh agents cache so self.agents is up-to-date after reset
        self._refresh_agents_cache()
        self._agents_cache: list[str] = []

        # Declare per-agent observation/action spaces for heterogeneity
        high_xy = np.array([self.grid_h - 1, self.grid_w - 1, np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)
        low_xy  = np.array([0, 0, 0.0, 0.0], dtype=np.float32)
        obs_space = spaces.Box(low=low_xy, high=high_xy, dtype=np.float32)
        act_space = spaces.Discrete(9)
        self.observation_space = obs_space
        self.action_space = act_space
        # Per-agent dicts (homogeneous by default, but allows for heterogeneity)
        self.observation_spaces = {aid: obs_space for aid in self.possible_agents}
        self.action_spaces = {aid: act_space for aid in self.possible_agents}

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

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

    @property
    def agents(self) -> list[str]:
        if self._active_dirty:
            self._refresh_agents_cache()
        return self._agents_cache

    # ---------- Public API ----------
    def reset(self, *, seed: int | None = None, options: dict | None = None, obs_range: int = 5):
        """
        RLlib-style reset -> (observations_dict, infos_dict)
        Vectorized initialization:
          - Activate all predators and prey
          - Sample positions (uniform on grid)
          - Assign initial energies
          - Seed grass cells (no overlap requirement between roles; grass can share a cell with agents)
        """
        if seed is not None:
            # allow RLlib to reseed us between episodes
            self._seed = seed
            self.rng = np.random.default_rng(seed)

        # Clear all agents
        self.active[:] = False
        self.pos[:] = -1
        self.energy[:] = 0.0
        self.age[:] = 0
        self._active_dirty = True

        # Activate only the first initial_num_predators and the correct prey slots
        self.active[:] = False
        # Predators: first num_possible_predators slots, activate only initial_num_predators
        self.active[:self.initial_num_predators] = True
        # Prey: after all possible predators, activate only initial_num_prey
        prey_start = self.num_possible_predators
        self.active[prey_start:prey_start + self.initial_num_prey] = True
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
        self.energy[self.is_pred] = self.initial_energy_predator
        self.energy[self.is_prey] = self.initial_energy_prey
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

        # Build local grid observations for each agent
        observations = {aid: self._get_local_observation(aid, obs_range) for aid in self.agents}
        # Add info dict with coordinates for each agent
        infos = {}
        for aid in self.agents:
            ix = self._id_to_ix[aid]
            x, y = int(self.pos[ix, 0]), int(self.pos[ix, 1])
            energy = float(self.energy[ix])
            age = int(self.age[ix])
            infos[aid] = {"x": x, "y": y, "energy": energy, "age": age}
        return observations, infos

    def step(self, action_dict):
        """
        RLlib MultiAgentEnv step function with fully vectorized agent movement.
        Args:
            action_dict: dict mapping agent_id to action (0: noop, 1: up, 2: right, 3: down, 4: left)
        Returns:
            obs: dict of new observations
            rewards: dict of rewards
            terminations: dict of done flags (per agent)
            truncations: dict of truncation flags (per agent)
            infos: dict of info dicts (per agent)
        """
        # Step 0: Time maintenance - costs energy drain, (re)growing grass and aging
        agents_before_time_maintenance = set(self.agents)  # Snapshot of agents before time maintenance
        self._apply_basal_metabolism_and_aging()
        # After starvation, refresh agent cache and set terminations for starved agents
        terminations = self._refresh_and_set_terminations(agents_before_time_maintenance)
        # Print agent deletions after starvation
        just_deactivated_starvation = agents_before_time_maintenance - set(self.agents)
        if just_deactivated_starvation:
            print(f"[Starvation] Deactivated agents: {sorted(just_deactivated_starvation)}")
        # Step 1: Apply agent movements
        # Only allow currently active agents to move and participate in engagement
        self._apply_agent_movement(action_dict)
        agents_before_engagement = set(self.agents)  # Snapshot of agents before engagement removals
        # Step 2: Resolve engagements (predator-prey and prey-grass)
        self._resolve_prey_grass_engagement()
        # Step 3: Resolve predator-prey engagements 
        self._resolve_predator_prey_engagement()
        # After engagements, refresh agent cache and set terminations for eaten/removed agents
        terminations = self._refresh_and_set_terminations(agents_before_engagement | set(terminations.keys()) - {'__all__'})
        # Step 4: Apply reproduction logic
        self._apply_reproduction(agents_before_engagement, terminations)

        # Build agent-specific outputs for all agents present at start or terminated/spawned during this step
        agent_keys = (agents_before_time_maintenance | set(terminations.keys()) | set(self.agents)) - {'__all__'}
        obs = {aid: self._get_local_observation(aid) for aid in self.agents}  # Observations only for active agents
        rewards = {aid: 0.0 for aid in agent_keys}
        truncations = {aid: False for aid in agent_keys}
        infos = {}
        for aid in agent_keys:
            ix = self._id_to_ix[aid]
            x, y = int(self.pos[ix, 0]), int(self.pos[ix, 1])
            energy = float(self.energy[ix])
            age = int(self.age[ix])
            infos[aid] = {"x": x, "y": y, "energy": energy, "age": age}
        truncations['__all__'] = False
        return obs, rewards, terminations, truncations, infos

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
        # Energy drain
        if self.energy_loss_per_step_predator:
            self.energy[pred_mask] -= self.energy_loss_per_step_predator
        if self.energy_loss_per_step_prey:
            self.energy[prey_mask] -= self.energy_loss_per_step_prey
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
        # Build action array for all active agents (default to 0)
        agent_indices = np.array([self._id_to_ix[aid] for aid in self.agents], dtype=np.int32)
        actions = np.zeros(len(agent_indices), dtype=np.int32)
        for i, aid in enumerate(self.agents):
            act = action_dict.get(aid, 0)
            if 0 <= act < len(deltas):
                actions[i] = act
        # Vectorized movement
        pos = self.pos[agent_indices]
        move = deltas[actions]
        new_pos = pos + move
        # Clamp to grid
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, self.grid_h - 1)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, self.grid_w - 1)
        self.pos[agent_indices] = new_pos

        # Only movement logic here; RLlib outputs are handled in step()

    def _resolve_prey_grass_engagement(self):
        """
        Single-consumer grazing: In each cell with one or more active prey and grass > 0,
        randomly select ONE prey to consume ALL grass in that cell. That prey gains the
        grass energy; the cell's grass is set to 0. No agents are (de)activated here.
        """
        import collections
        cell_prey = collections.defaultdict(list)
        # Group active prey by their cell
        for i, active in enumerate(self.active):
            if not active or not self.is_prey[i]:
                continue
            x, y = int(self.pos[i, 0]), int(self.pos[i, 1])
            cell_prey[(x, y)].append(i)

        if not cell_prey:
            return

        prey_indices = []
        energy_gains = []
        cells_x = []
        cells_y = []

        for (x, y), prey_list in cell_prey.items():
            g = float(self.grass[x, y])
            if g <= 0.0 or len(prey_list) == 0:
                continue
            # Choose one prey uniformly at random to consume all grass at this cell
            chosen = prey_list[self.rng.integers(0, len(prey_list))]
            prey_indices.append(chosen)
            energy_gains.append(g)
            cells_x.append(x)
            cells_y.append(y)

        # Batch apply updates
        if prey_indices:
            prey_idx_arr = np.array(prey_indices, dtype=np.int64)
            gain_arr = np.array(energy_gains, dtype=np.float32)
            self.energy[prey_idx_arr] += gain_arr
            self.grass[(np.array(cells_x, dtype=np.int64), np.array(cells_y, dtype=np.int64))] = 0.0

    def _resolve_predator_prey_engagement(self):
        import collections
        # Step 1: Build cell occupancy maps
        cell_predators = collections.defaultdict(list)
        cell_prey = collections.defaultdict(list)
        for i, active in enumerate(self.active):
            if not active:
                continue
            x, y = self.pos[i]
            key = (x, y)
            if self.is_pred[i]:
                cell_predators[key].append(i)
            else:
                cell_prey[key].append(i)

        # Step 2: Resolve engagements in each cell
        prey_to_deactivate = []
        predators_to_reward = []
        for cell, preds in cell_predators.items():
            prey = cell_prey.get(cell, [])
            if not prey:
                continue  # No prey to eat
            n_pred = len(preds)
            n_prey = len(prey)
            n_pairs = min(n_pred, n_prey)
            # Randomly shuffle for fair pairing
            self.rng.shuffle(preds)
            self.rng.shuffle(prey)
            eating_preds = preds[:n_pairs]
            eaten_prey = prey[:n_pairs]
            prey_to_deactivate.extend(eaten_prey)
            predators_to_reward.extend(eating_preds)
            # Unpaired predators get nothing; unpaired prey survive

        # Step 3: Batch updates
        if prey_to_deactivate:
            self.active[prey_to_deactivate] = False
        if predators_to_reward:
            self.energy[predators_to_reward] += 5.0  # Example: reward/energy for eating

    def _apply_reproduction(self, agents_before_engagement, terminations):
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
        # Find available slots for new agents (not active, not just deactivated)
        all_ix = np.arange(self.num_possible_agents)
        available_ix = all_ix[~self.active]
        # Exclude just deactivated
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
            spawned = [self._all_ids[i] for i in spawn_pred_ix]
            parents = [self._all_ids[i] for i in parent_pred_ix]
            print(f"[Reproduction] Spawned predators: {spawned} from parents {parents}")
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
            spawned = [self._all_ids[i] for i in spawn_prey_ix]
            parents = [self._all_ids[i] for i in parent_prey_ix]
            print(f"[Reproduction] Spawned prey: {spawned} from parents {parents}")
        # Refresh agent cache after possible spawns
        self._refresh_agents_cache()

    def _get_local_observation(self, agent_id: str, obs_range: int = 5) -> np.ndarray:
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
            self.grid_h,
            self.grid_w
        )

