"""
A copy of base_environment_step_energy that replaces asexual predator
reproduction with sexual (two-parent) reproduction, and splits predators into
two sexed policies with different foraging roles:

- predator_male: high-success/low-risk hunter (also gathers fruit) -- see
  prey_vs_predator_male_* in config_env.py.
- predator_female: low-success/high-risk hunter (also gathers fruit) -- see
  prey_vs_predator_female_* in config_env.py. Both sexes can attempt to hunt
  prey and both can die doing so; see _resolve_hunting_attempt. Any resulting
  specialization is meant to emerge from RL training under this risk
  asymmetry, not from a hardcoded incapability.
- prey: unchanged -- only eats grass, reproduces asexually (solo, energy-
  threshold triggered), exactly as in base_environment_step_energy.

Grass is prey-exclusive food; fruit is predator-exclusive food (both sexes).
Sexual reproduction requires a predator_male and a predator_female to each
independently clear predator_creation_energy_threshold AND be within
mate_search_radius of each other (Chebyshev distance) -- an exact-cell match
is impossible since both sexes share one grid layer and movement collision
already forbids two predators occupying the same cell. The offspring spawns
near the female and birth cost splits asymmetrically (see config_env.py for
the parental-investment rationale).

Because the female pays the larger share of birth cost but has only a weak,
shared, depleting income source (fruit) to recover with, a predator_male also
donates a fraction of each successful hunt's energy gain to HIS RECORDED MATE
ONLY (_apply_male_gift, unidirectional, mechanically executed like
eco_evolutionary_nuptial_gift's male_donation_rate, but exclusive/pair-bonded
rather than broadcast to any nearby female -- see self.agent_mate) -- meant
to offset her post-birth energy deficit.

Both parents also share a fraction of ANY successful forage (hunt or fruit)
with their own nearby living offspring (_share_energy_with_offspring, see
self.agent_parents) -- direct parental care by both sexes, not just the
mother, matching the cooperative-breeding literature's account of human
parenting as distinctive among mammals. Split evenly across however many of
a parent's own children are currently nearby (unlike the exclusive,
single-recipient mate gift). Care stops once the offspring has reproduced
itself (self.has_reproduced) -- a reproduction-based, not age-based,
independence cutoff (this module tracks no per-agent age at all). See
config_env.py for the mate-search/cost-split/gift/parental-care rationale
and RESULTS.md (once populated) for empirical findings.
"""
from predpreygrass.non_evolutionary.predator_sexual_reproduction.config_env import config_env

# external libraries
import numpy as np

from numpy.typing import NDArray
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, Dict, List, Tuple


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config = config or config_env  # Use provided config or default config_env

        self.verbose_engagement = config.get("verbose_engagement", False)
        self.verbose_movement = config.get("verbose_movement", False)
        self.verbose_spawning = config.get("verbose_spawning", False)

        self.max_steps = config.get("max_steps", 10000)

        # Rewards
        self.reward_predator_catch_prey = config.get("reward_predator_catch_prey", 0.0)
        self.reward_predator_gather_fruit = config.get("reward_predator_gather_fruit", 0.0)
        # Energy-proportional forage reward (fruit and prey alike); see config_env.py.
        self.reward_predator_per_energy = config.get("reward_predator_per_energy", 0.0)
        if not (0.0 <= self.reward_predator_per_energy < float("inf")):
            raise ValueError(
                f"reward_predator_per_energy must be finite and >= 0 (got {self.reward_predator_per_energy})"
            )
        self.reward_prey_eat_grass = config.get("reward_prey_eat_grass", 0.0)
        self.reward_predator_step = config.get("reward_predator_step", 0.0)
        self.reward_prey_step = config.get("reward_prey_step", 0.0)
        self.penalty_prey_caught = config.get("penalty_prey_caught", 0.0)
        self.reproduction_reward_predator = config.get("reproduction_reward_predator", 10.0)
        self.reproduction_reward_prey = config.get("reproduction_reward_prey", 10.0)

        # Energy settings: homeostatic (always charged) and move (charged on
        # top, only when the agent's action isn't noop) are independent,
        # additive costs -- see base_environment_step_energy for the
        # rationale. Applies uniformly to predator_male/predator_female.
        self.homeostatic_energy_cost_per_step_predator = config.get("homeostatic_energy_cost_per_step_predator", 0.10)
        self.homeostatic_energy_cost_per_step_prey = config.get("homeostatic_energy_cost_per_step_prey", 0.035)
        self.move_energy_cost_per_step_predator = config.get("move_energy_cost_per_step_predator", 0.08)
        self.move_energy_cost_per_step_prey = config.get("move_energy_cost_per_step_prey", 0.035)
        self.predator_creation_energy_threshold = config.get("predator_creation_energy_threshold", 12.0)
        self.prey_creation_energy_threshold = config.get("prey_creation_energy_threshold", 8.0)

        # Sexual reproduction: Chebyshev-distance radius for mate-finding
        # (see config_env.py for why this can't be an exact-cell match).
        self.mate_search_radius = int(config.get("mate_search_radius", 3))

        # Male provisioning: unidirectional energy transfer from
        # predator_male to nearby predator_female neighbors on a successful
        # hunt -- offsets her post-birth energy deficit (see config_env.py
        # for the full rationale).
        self.male_gift_donation_rate = config.get("male_gift_donation_rate", 0.3)
        if not (0.0 <= self.male_gift_donation_rate <= 1.0):
            raise ValueError(f"male_gift_donation_rate must be in [0, 1] (got {self.male_gift_donation_rate})")
        self.predator_gift_range = int(config.get("predator_gift_range", 3))
        if self.predator_gift_range < 0:
            raise ValueError(f"predator_gift_range must be non-negative (got {self.predator_gift_range})")

        # Parental care: both parents share a fraction of any successful
        # forage (hunt or fruit) with their own nearby living offspring
        # (self.agent_parents), reusing predator_gift_range for proximity --
        # see config_env.py for the full rationale.
        self.parent_offspring_share_rate = config.get("parent_offspring_share_rate", 0.2)
        if not (0.0 <= self.parent_offspring_share_rate <= 1.0):
            raise ValueError(
                f"parent_offspring_share_rate must be in [0, 1] (got {self.parent_offspring_share_rate})"
            )
        # A male's successful hunt applies both donations to the SAME gross
        # gain (not sequentially off a shrinking remainder), so their sum
        # must not exceed 1.0 -- otherwise a hunt could deduct more energy
        # than it gained, pushing the male negative regardless of how much
        # energy he had banked beforehand.
        if self.male_gift_donation_rate + self.parent_offspring_share_rate > 1.0:
            raise ValueError(
                "male_gift_donation_rate + parent_offspring_share_rate must be <= 1.0 "
                f"(got {self.male_gift_donation_rate} + {self.parent_offspring_share_rate} "
                f"= {self.male_gift_donation_rate + self.parent_offspring_share_rate})"
            )

        # Birth cost split: parental-investment-theory rationale (Trivers,
        # 1972) -- the female also bears the larger share of the shared
        # reproduction cost, consistent with her being the structurally
        # riskier forager (see prey_vs_predator_female_* below).
        self.predator_birth_cost_share_female = config.get("predator_birth_cost_share_female", 0.9)
        self.predator_birth_cost_share_male = config.get("predator_birth_cost_share_male", 0.1)
        if (
            not (0.0 <= self.predator_birth_cost_share_female <= 1.0)
            or not (0.0 <= self.predator_birth_cost_share_male <= 1.0)
            or abs(self.predator_birth_cost_share_female + self.predator_birth_cost_share_male - 1.0) > 1e-9
        ):
            raise ValueError(
                "predator_birth_cost_share_female/_male must be non-negative and sum to exactly 1.0 "
                f"(got female={self.predator_birth_cost_share_female}, male={self.predator_birth_cost_share_male})"
            )

        self.penalty_predator_death_in_combat = config.get("penalty_predator_death_in_combat", 0.0)

        # Hunting is a 3-outcome stochastic contest for BOTH sexes (not a
        # deterministic, male-only catch): "success" (kill, as a deterministic
        # catch would have), "predator_dies" (the prey is left completely
        # unharmed; the predator itself is removed), or "failure" (nothing
        # happens). Failure probability is implied (1 - success - death), not
        # stored as its own key, to avoid a redundant value that could
        # silently drift out of sync with the other two. See config_env.py
        # for the full rationale.
        self.prey_vs_predator_male_success_prob = config.get("prey_vs_predator_male_success_prob", 0.90)
        self.prey_vs_predator_male_death_prob = config.get("prey_vs_predator_male_death_prob", 0.05)
        self.prey_vs_predator_female_success_prob = config.get("prey_vs_predator_female_success_prob", 0.20)
        self.prey_vs_predator_female_death_prob = config.get("prey_vs_predator_female_death_prob", 0.10)
        for _sex, _success, _death in (
            ("predator_male", self.prey_vs_predator_male_success_prob, self.prey_vs_predator_male_death_prob),
            ("predator_female", self.prey_vs_predator_female_success_prob, self.prey_vs_predator_female_death_prob),
        ):
            if not (0.0 <= _success <= 1.0) or not (0.0 <= _death <= 1.0) or _success + _death > 1.0:
                raise ValueError(
                    f"{_sex} hunting probabilities must be non-negative and sum to <= 1.0 "
                    f"(got success={_success}, death={_death})"
                )

        # Learning agents
        self.n_possible_predator_male = config.get("n_possible_predator_male", 1000)
        self.n_possible_predator_female = config.get("n_possible_predator_female", 1000)
        self.n_possible_prey = config.get("n_possible_prey", 2000)
        self.n_initial_active_predator_male = config.get("n_initial_active_predator_male", 3)
        self.n_initial_active_predator_female = config.get("n_initial_active_predator_female", 3)
        self.n_initial_active_prey = config.get("n_initial_active_prey", 8)

        self.initial_energy_predator_male = config.get("initial_energy_predator_male", 5.0)
        self.initial_energy_predator_female = config.get("initial_energy_predator_female", 5.0)
        self.initial_energy_prey = config.get("initial_energy_prey", 3.0)

        # Grid and Observation Settings
        self.grid_size = config.get("grid_size", 25)
        self.num_obs_channels = config.get("num_obs_channels", 5)
        self.predator_obs_range = config.get("predator_obs_range", 7)
        self.prey_obs_range = config.get("prey_obs_range", 9)

        # Grass settings (prey food only)
        self.initial_num_grass = config.get("initial_num_grass", 100)
        self.initial_energy_grass = config.get("initial_energy_grass", 2.0)
        self.energy_gain_per_step_grass = config.get("energy_gain_per_step_grass", 0.04)

        # Fruit settings (predator food only, both sexes)
        self.initial_num_fruit = config.get("initial_num_fruit", 100)
        self.initial_energy_fruit = config.get("initial_energy_fruit", 2.0)
        self.energy_gain_per_step_fruit = config.get("energy_gain_per_step_fruit", 0.04)

        self.cumulative_rewards = {}  # Track total rewards per agent

        # Mate tracking for exclusive (pair-bonded) male provisioning -- see
        # _apply_male_gift. Bidirectional: agent_mate[male] = female and
        # agent_mate[female] = male, set on a successful reproduction event
        # and overwritten on any later one (serial monogamy: the most recent
        # partner, not lifetime exclusivity). No entry until an agent has
        # reproduced at least once.
        self.agent_mate: Dict[AgentID, AgentID] = {}

        # Lineage tracking for parental care -- see
        # _share_energy_with_offspring. child -> (father, mother), set once
        # at birth, never updated or cleaned up on death: a dead child
        # simply never appears in predator_positions again, so a stale
        # entry here is inert rather than a dangling reference (unlike
        # agent_mate, which is looked up FROM the parent and therefore
        # needed the remating fix).
        self.agent_parents: Dict[AgentID, Tuple[AgentID, AgentID]] = {}

        # Reproduction-based independence cutoff for parental care: once an
        # agent has reproduced itself, it's a breeding adult, not a
        # dependent offspring, so _share_energy_with_offspring stops paying
        # it out regardless of proximity to its own parents. Set once,
        # alongside agent_mate/agent_parents, and never removed -- unlike
        # agent_mate (which the remating fix can clear from an abandoned
        # partner), "has this individual ever reproduced" is permanent, so
        # it needs its own flag rather than reusing agent_mate membership.
        self.has_reproduced: set[AgentID] = set()

        self._pending_removal: List[AgentID] = []
        self._next_predator_male_idx = self.n_initial_active_predator_male
        self._next_predator_female_idx = self.n_initial_active_predator_female
        self._next_prey_idx = self.n_initial_active_prey

        self.possible_agents: List[AgentID] = (
            [f"predator_male_{i}" for i in range(self.n_possible_predator_male)]
            + [f"predator_female_{i}" for i in range(self.n_possible_predator_female)]
            + [f"prey_{j}" for j in range(self.n_possible_prey)]
        )
        self.agents: List[AgentID] = (
            [f"predator_male_{i}" for i in range(self.n_initial_active_predator_male)]
            + [f"predator_female_{i}" for i in range(self.n_initial_active_predator_female)]
            + [f"prey_{j}" for j in range(self.n_initial_active_prey)]
        )

        # Non-learning agents (grass, fruit); not included in 'possible_agents' or 'agents'
        self.grass_agents: List[AgentID] = [f"grass_{k}" for k in range(self.initial_num_grass)]
        self.fruit_agents: List[AgentID] = [f"fruit_{k}" for k in range(self.initial_num_fruit)]

        # Gymnasium Spaces
        predator_obs_shape = (self.num_obs_channels, self.predator_obs_range, self.predator_obs_range)
        prey_obs_shape = (self.num_obs_channels, self.prey_obs_range, self.prey_obs_range)

        predator_obs_space = gymnasium.spaces.Box(low=0.0, high=100.0, shape=predator_obs_shape, dtype=np.float64)
        prey_obs_space = gymnasium.spaces.Box(low=0.0, high=100.0, shape=prey_obs_shape, dtype=np.float64)

        # Assign spaces based on agent type (both predator sexes share the
        # same observation space; only the species -- predator vs prey --
        # determines the space, same convention as base_environment_step_energy).
        self.observation_spaces = {
            agent: predator_obs_space if "predator" in agent else prey_obs_space for agent in self.possible_agents
        }

        self.action_to_move_tuple: Dict[int, Tuple[int, int]] = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 0),
            5: (0, 1),
            6: (1, -1),
            7: (1, 0),
            8: (1, 1),
        }
        self.noop_action_id = next(a for a, move in self.action_to_move_tuple.items() if move == (0, 0))
        action_space = gymnasium.spaces.Discrete(len(self.action_to_move_tuple))
        self.action_spaces = {agent: action_space for agent in self.possible_agents}

        # Initialize grid_world_state and agent positions
        self.agent_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.predator_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.prey_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.grass_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.fruit_positions: Dict[AgentID, Tuple[int, int]] = {}

        self.agent_energies: Dict[AgentID, float] = {}
        self.grass_energies: Dict[AgentID, float] = {}
        self.fruit_energies: Dict[AgentID, float] = {}
        self.grid_world_state_shape: Tuple[int, int, int] = (
            self.num_obs_channels,
            self.grid_size,
            self.grid_size,
        )
        self.initial_grid_world_state: NDArray[np.float64] = np.zeros(self.grid_world_state_shape, dtype=np.float64)
        self.grid_world_state: NDArray[np.float64] = self.initial_grid_world_state.copy()
        self.num_actions = len(self.action_to_move_tuple)
        self.agents_just_ate = set()  # agent_id → shows green ring this step

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self.current_step = 0
        # Per the Gymnasium reset(seed=...) contract: only reseed when an
        # explicit seed is given -- see base_environment_step_energy for why.
        if seed is not None or not hasattr(self, "rng"):
            self.rng = np.random.default_rng(seed)

        self.grid_world_state = self.initial_grid_world_state.copy()

        self.possible_agents: List[AgentID] = (
            [f"predator_male_{i}" for i in range(self.n_possible_predator_male)]
            + [f"predator_female_{i}" for i in range(self.n_possible_predator_female)]
            + [f"prey_{j}" for j in range(self.n_possible_prey)]
        )
        male_ids = [f"predator_male_{i}" for i in range(self.n_initial_active_predator_male)]
        female_ids = [f"predator_female_{i}" for i in range(self.n_initial_active_predator_female)]
        prey_ids = [f"prey_{j}" for j in range(self.n_initial_active_prey)]
        self.agents = male_ids + female_ids + prey_ids

        self.agent_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.agent_energies: Dict[AgentID, float] = {}
        self.predator_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.prey_positions: Dict[AgentID, Tuple[int, int]] = {}

        self.cumulative_rewards: Dict[AgentID, float] = {agent_id: 0 for agent_id in self.agents}

        self.agent_mate: Dict[AgentID, AgentID] = {}
        self.agent_parents: Dict[AgentID, Tuple[AgentID, AgentID]] = {}
        self.has_reproduced: set[AgentID] = set()

        self._pending_removal = []
        self._next_predator_male_idx = self.n_initial_active_predator_male
        self._next_predator_female_idx = self.n_initial_active_predator_female
        self._next_prey_idx = self.n_initial_active_prey

        self.episode_births = {"predator_male": 0, "predator_female": 0, "prey": 0}
        self.episode_deaths = {"predator_male": 0, "predator_female": 0, "prey": 0}

        # Training-time observability for the hunting/provisioning
        # mechanics -- surfaced via _build_episode_training_metrics, since
        # episode_births/episode_deaths alone can't distinguish "no
        # reproduction because no one hunts" from "no reproduction despite
        # hunting succeeding" from "hunting succeeds but mates never meet."
        self.hunting_attempts = {"predator_male": 0, "predator_female": 0}
        self.hunting_successes = {"predator_male": 0, "predator_female": 0}
        # Deaths specifically caused by a failed hunting attempt (a subset
        # of episode_deaths, which doesn't distinguish cause -- the
        # remainder, episode_deaths[sex] - hunting_deaths[sex], is
        # starvation).
        self.hunting_deaths = {"predator_male": 0, "predator_female": 0}
        self.mate_gift_events = 0
        self.mate_gift_energy_total = 0.0
        self.parental_care_events = 0
        self.parental_care_energy_total = 0.0

        def generate_random_positions(grid_size: int, num_positions: int):
            if num_positions > grid_size * grid_size:
                raise ValueError("Cannot place more unique positions than grid cells.")
            positions = set()
            while len(positions) < num_positions:
                pos = tuple(self.rng.integers(0, grid_size, size=2))
                positions.add(pos)
            return list(positions)

        total_entities = (
            self.n_initial_active_predator_male
            + self.n_initial_active_predator_female
            + self.n_initial_active_prey
            + self.initial_num_grass
            + self.initial_num_fruit
        )
        all_positions = generate_random_positions(self.grid_size, total_entities)

        idx = 0
        predator_male_positions = all_positions[idx : idx + self.n_initial_active_predator_male]
        idx += self.n_initial_active_predator_male
        predator_female_positions = all_positions[idx : idx + self.n_initial_active_predator_female]
        idx += self.n_initial_active_predator_female
        prey_positions = all_positions[idx : idx + self.n_initial_active_prey]
        idx += self.n_initial_active_prey
        grass_positions = all_positions[idx : idx + self.initial_num_grass]
        idx += self.initial_num_grass
        fruit_positions = all_positions[idx : idx + self.initial_num_fruit]
        idx += self.initial_num_fruit

        for i, agent in enumerate(male_ids):
            self.agent_positions[agent] = predator_male_positions[i]
            self.predator_positions[agent] = predator_male_positions[i]
            self.agent_energies[agent] = self.initial_energy_predator_male
            self.grid_world_state[1, *predator_male_positions[i]] = self.initial_energy_predator_male
        for i, agent in enumerate(female_ids):
            self.agent_positions[agent] = predator_female_positions[i]
            self.predator_positions[agent] = predator_female_positions[i]
            self.agent_energies[agent] = self.initial_energy_predator_female
            self.grid_world_state[1, *predator_female_positions[i]] = self.initial_energy_predator_female
        for i, agent in enumerate(prey_ids):
            self.agent_positions[agent] = prey_positions[i]
            self.prey_positions[agent] = prey_positions[i]
            self.agent_energies[agent] = self.initial_energy_prey
            self.grid_world_state[2, *prey_positions[i]] = self.initial_energy_prey

        self.grass_positions = {}
        self.grass_energies = {}
        for i, grass in enumerate(self.grass_agents):
            self.grass_positions[grass] = grass_positions[i]
            self.grass_energies[grass] = self.initial_energy_grass
            self.grid_world_state[3, *grass_positions[i]] = self.initial_energy_grass

        self.fruit_positions = {}
        self.fruit_energies = {}
        for i, fruit in enumerate(self.fruit_agents):
            self.fruit_positions[fruit] = fruit_positions[i]
            self.fruit_energies[fruit] = self.initial_energy_fruit
            self.grid_world_state[4, *fruit_positions[i]] = self.initial_energy_fruit

        self.current_num_prey = self.n_initial_active_prey
        self.current_num_predator_male = self.n_initial_active_predator_male
        self.current_num_predator_female = self.n_initial_active_predator_female
        self.current_num_grass = self.initial_num_grass
        self.current_num_fruit = self.initial_num_fruit

        observations = {agent: self._get_observation(agent) for agent in self.agents}

        return observations, {}

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        for agent in self._pending_removal:
            if agent in self.agents:
                self.agents.remove(agent)
        self._pending_removal = []

        # step 0: check for truncation
        if self.current_step >= self.max_steps:
            for agent in self.agents:
                observations[agent] = self._get_observation(agent)
                rewards[agent] = 0.0
                truncations[agent] = True
                terminations[agent] = False

            truncations["__all__"] = True
            terminations["__all__"] = False
            return observations, rewards, terminations, truncations, infos

        self.agents_just_ate.clear()

        # Step 1: homeostatic energy depletion (always charged, every step).
        # Guard membership (mirrors Step 2's guard below): a stale action for
        # an agent already removed in a prior step would otherwise KeyError.
        for agent, action in action_dict.items():
            if agent not in self.agent_positions:
                continue
            if "predator" in agent:
                self.agent_energies[agent] -= self.homeostatic_energy_cost_per_step_predator
                self.grid_world_state[1, *self.agent_positions[agent]] = self.agent_energies[agent]
            elif "prey" in agent:
                self.agent_energies[agent] -= self.homeostatic_energy_cost_per_step_prey
                self.grid_world_state[2, *self.agent_positions[agent]] = self.agent_energies[agent]

        for grass, grass_position in self.grass_positions.items():
            self.grass_energies[grass] = min(
                self.grass_energies[grass] + self.energy_gain_per_step_grass, self.initial_energy_grass
            )
            self.grid_world_state[3, *grass_position] = self.grass_energies[grass]

        for fruit, fruit_position in self.fruit_positions.items():
            self.fruit_energies[fruit] = min(
                self.fruit_energies[fruit] + self.energy_gain_per_step_fruit, self.initial_energy_fruit
            )
            self.grid_world_state[4, *fruit_position] = self.fruit_energies[fruit]

        # Step 2: movement
        for agent, action in action_dict.items():
            if agent in self.agent_positions:
                old_position = self.agent_positions[agent]
                new_position = self._get_move(agent, action)
                self.agent_positions[agent] = new_position
                move_cost = self._get_movement_energy_cost(agent, action)
                self.agent_energies[agent] -= move_cost
                if "predator" in agent:
                    self.predator_positions[agent] = new_position
                    self.grid_world_state[1, *old_position] = 0
                    self.grid_world_state[1, *new_position] = self.agent_energies[agent]
                elif "prey" in agent:
                    self.prey_positions[agent] = new_position
                    self.grid_world_state[2, *old_position] = 0
                    self.grid_world_state[2, *new_position] = self.agent_energies[agent]

                if self.verbose_movement:
                    print(f"[MOVE] Agent {agent} moved: {old_position} -> {new_position}.")

        # Step 3: removals (starvation) and engagements (hunting/foraging/grazing)
        for agent in self.agents:
            if agent not in self.agent_positions:
                continue
            if self.agent_energies[agent] <= 0:
                if self.verbose_movement:
                    print(f"[MOVE] {agent} at {self.agent_positions[agent]} ran out of energy and is removed.")
                observations[agent] = self._get_observation(agent)
                rewards[agent] = 0
                terminations[agent] = True
                truncations[agent] = False
                if "predator_male" in agent:
                    self.current_num_predator_male -= 1
                    self.episode_deaths["predator_male"] += 1
                    self.grid_world_state[1, *self.agent_positions[agent]] = 0
                    del self.predator_positions[agent]
                elif "predator_female" in agent:
                    self.current_num_predator_female -= 1
                    self.episode_deaths["predator_female"] += 1
                    self.grid_world_state[1, *self.agent_positions[agent]] = 0
                    del self.predator_positions[agent]
                elif "prey" in agent:
                    self.current_num_prey -= 1
                    self.episode_deaths["prey"] += 1
                    self.grid_world_state[2, *self.agent_positions[agent]] = 0
                    del self.prey_positions[agent]
                del self.agent_positions[agent]
                del self.agent_energies[agent]
                continue
            elif "predator_male" in agent:
                predator_position = self.agent_positions[agent]
                reward = 0.0
                ate_something = False

                caught_prey = next(
                    (
                        prey
                        for prey, prey_position in self.agent_positions.items()
                        if "prey" in prey and np.array_equal(predator_position, prey_position)
                    ),
                    None,
                )
                if caught_prey:
                    outcome, reward_delta, energy_gained = self._resolve_hunting_attempt(
                        agent,
                        predator_position,
                        caught_prey,
                        self.prey_vs_predator_male_success_prob,
                        self.prey_vs_predator_male_death_prob,
                        observations,
                        rewards,
                        terminations,
                        truncations,
                    )
                    if outcome == "predator_dies":
                        continue
                    if outcome == "success":
                        ate_something = True
                        reward += reward_delta
                        self._apply_male_gift(agent, energy_gained)
                        self._share_energy_with_offspring(agent, energy_gained)

                caught_fruit = next(
                    (
                        fruit
                        for fruit, fruit_position in self.fruit_positions.items()
                        if np.array_equal(predator_position, fruit_position)
                    ),
                    None,
                )
                if caught_fruit:
                    ate_something = True
                    self.agents_just_ate.add(agent)
                    fruit_gain = self.fruit_energies[caught_fruit]
                    reward += self.reward_predator_gather_fruit + self.reward_predator_per_energy * fruit_gain
                    self.agent_energies[agent] += fruit_gain
                    self.grid_world_state[1, *predator_position] = self.agent_energies[agent]
                    self.grid_world_state[4, *self.fruit_positions[caught_fruit]] = 0
                    self.fruit_energies[caught_fruit] = 0
                    self._share_energy_with_offspring(agent, fruit_gain)

                if not ate_something:
                    reward = self.reward_predator_step

                rewards[agent] = reward
                observations[agent] = self._get_observation(agent)
                self.cumulative_rewards[agent] += rewards[agent]
                terminations[agent] = False
                truncations[agent] = False
            elif "predator_female" in agent:
                predator_position = self.agent_positions[agent]
                reward = 0.0
                ate_something = False

                caught_prey = next(
                    (
                        prey
                        for prey, prey_position in self.agent_positions.items()
                        if "prey" in prey and np.array_equal(predator_position, prey_position)
                    ),
                    None,
                )
                if caught_prey:
                    outcome, reward_delta, energy_gained = self._resolve_hunting_attempt(
                        agent,
                        predator_position,
                        caught_prey,
                        self.prey_vs_predator_female_success_prob,
                        self.prey_vs_predator_female_death_prob,
                        observations,
                        rewards,
                        terminations,
                        truncations,
                    )
                    if outcome == "predator_dies":
                        continue
                    if outcome == "success":
                        ate_something = True
                        reward += reward_delta
                        self._share_energy_with_offspring(agent, energy_gained)

                caught_fruit = next(
                    (
                        fruit
                        for fruit, fruit_position in self.fruit_positions.items()
                        if np.array_equal(predator_position, fruit_position)
                    ),
                    None,
                )
                if caught_fruit:
                    if self.verbose_engagement:
                        print(f"[ENGAGE] {agent} gathered {caught_fruit} at {predator_position}!")
                    self.agents_just_ate.add(agent)
                    ate_something = True
                    fruit_gain = self.fruit_energies[caught_fruit]
                    reward += self.reward_predator_gather_fruit + self.reward_predator_per_energy * fruit_gain
                    self.agent_energies[agent] += fruit_gain
                    self.grid_world_state[1, *predator_position] = self.agent_energies[agent]
                    self.grid_world_state[4, *self.fruit_positions[caught_fruit]] = 0
                    self.fruit_energies[caught_fruit] = 0
                    self._share_energy_with_offspring(agent, fruit_gain)

                if not ate_something:
                    reward = self.reward_predator_step

                rewards[agent] = reward
                observations[agent] = self._get_observation(agent)
                self.cumulative_rewards[agent] += rewards[agent]
                terminations[agent] = False
                truncations[agent] = False
            elif "prey" in agent:
                if terminations.get(agent) is None or not terminations[agent]:
                    prey_position = self.agent_positions[agent]
                    caught_grass = next(
                        (
                            grass
                            for grass, grass_position in self.grass_positions.items()
                            if "grass" in grass and np.array_equal(prey_position, grass_position)
                        ),
                        None,
                    )
                    if caught_grass:
                        if self.verbose_engagement:
                            print(f"[ENGAGE] {agent} caught grass at {prey_position}!")
                        self.agents_just_ate.add(agent)
                        rewards[agent] = self.reward_prey_eat_grass
                        self.agent_energies[agent] += self.grass_energies[caught_grass]
                        self.grid_world_state[2, *prey_position] = self.agent_energies[agent]

                        self.grid_world_state[3, *self.grass_positions[caught_grass]] = 0
                        self.grass_energies[caught_grass] = 0
                    else:
                        rewards[agent] = self.reward_prey_step

                    observations[agent] = self._get_observation(agent)
                    self.cumulative_rewards[agent] += rewards[agent]
                    terminations[agent] = False
                    truncations[agent] = False

        # Step 4: schedule agent removals for the next step
        self._pending_removal = [agent for agent in self.agents if terminations.get(agent)]
        if self.verbose_engagement:
            for agent in self._pending_removal:
                print(f"[ENGAGE] Agent {agent} terminated!")

        # Step 5a: prey reproduction (asexual, unchanged from base_environment_step_energy)
        for agent in self.agents[:]:
            if agent in self._pending_removal:
                continue
            if "prey" in agent:
                if self.agent_energies[agent] >= self.prey_creation_energy_threshold:
                    if self._next_prey_idx < self.n_possible_prey:
                        occupied_positions = set(self.agent_positions.values())
                        new_position = self._find_available_spawn_position(self.agent_positions[agent], occupied_positions)
                        if new_position is None:
                            if self.verbose_spawning:
                                print(f"No free spawn position available for prey offspring of {agent}")
                        else:
                            new_agent = f"prey_{self._next_prey_idx}"
                            self._next_prey_idx += 1
                            self.agents.append(new_agent)
                            self.agent_positions[new_agent] = new_position
                            self.prey_positions[new_agent] = new_position
                            self.agent_energies[new_agent] = self.initial_energy_prey
                            self.agent_energies[agent] -= self.initial_energy_prey
                            self.grid_world_state[2, *self.agent_positions[new_agent]] = self.initial_energy_prey
                            self.grid_world_state[2, *self.agent_positions[agent]] = self.agent_energies[agent]
                            self.current_num_prey += 1
                            self.episode_births["prey"] += 1
                            rewards[new_agent] = 0
                            rewards[agent] = rewards.get(agent, 0.0) + self.reproduction_reward_prey
                            self.cumulative_rewards[agent] += self.reproduction_reward_prey
                            self.cumulative_rewards[new_agent] = 0
                            observations[new_agent] = self._get_observation(new_agent)
                            terminations[new_agent] = False
                            truncations[new_agent] = False
                            if self.verbose_spawning:
                                print(f"New prey {new_agent} spawned at {self.agent_positions[new_agent]}")
                    else:
                        if self.verbose_spawning:
                            print("No new prey agent IDs left in the pool this episode")

        # Step 5b: predator sexual reproduction. A predator_male pairs with a
        # nearby, independently-eligible predator_female (see config_env.py's
        # mate_search_radius comment for why an exact-cell match is
        # impossible). Each male can only be matched to one mate per step and
        # vice versa (paired_this_step), and one offspring is produced per pair.
        paired_this_step: set = set()
        radius = self.mate_search_radius
        eligible_males = [
            a
            for a in self.agents
            if "predator_male" in a
            and a not in self._pending_removal
            and self.agent_energies[a] >= self.predator_creation_energy_threshold
        ]
        # Snapshot female candidates and their positions before pairing starts:
        # a same-step newborn female (if her starting energy happens to clear
        # predator_creation_energy_threshold) must not be scanned as a mate
        # candidate for a later male in this same pass.
        female_snapshot = {
            female: pos
            for female, pos in self.predator_positions.items()
            if "predator_female" in female
            and female not in self._pending_removal
            and self.agent_energies[female] >= self.predator_creation_energy_threshold
        }
        for male in eligible_males:
            if male in paired_this_step:
                continue
            male_position = self.agent_positions[male]
            mate = next(
                (
                    female
                    for female, pos in female_snapshot.items()
                    if female not in paired_this_step
                    and max(abs(pos[0] - male_position[0]), abs(pos[1] - male_position[1])) <= radius
                ),
                None,
            )
            if mate is None:
                continue
            mate_position = self.agent_positions[mate]

            occupied_positions = set(self.agent_positions.values())
            new_position = self._find_available_spawn_position(mate_position, occupied_positions)
            if new_position is None:
                if self.verbose_spawning:
                    print(f"No free spawn position available for offspring of {male} and {mate}")
                continue

            child_sex = "predator_male" if self.rng.random() < 0.5 else "predator_female"
            if child_sex == "predator_male":
                if self._next_predator_male_idx >= self.n_possible_predator_male:
                    if self.verbose_spawning:
                        print("No new predator_male agent IDs left in the pool this episode")
                    continue
                new_agent = f"predator_male_{self._next_predator_male_idx}"
                self._next_predator_male_idx += 1
                offspring_energy = self.initial_energy_predator_male
            else:
                if self._next_predator_female_idx >= self.n_possible_predator_female:
                    if self.verbose_spawning:
                        print("No new predator_female agent IDs left in the pool this episode")
                    continue
                new_agent = f"predator_female_{self._next_predator_female_idx}"
                self._next_predator_female_idx += 1
                offspring_energy = self.initial_energy_predator_female

            paired_this_step.add(male)
            paired_this_step.add(mate)

            # Record (or update, serial-monogamy-style) the pair bond for
            # exclusive male provisioning -- see _apply_male_gift. Sever any
            # PREVIOUS partner's reverse pointer first: e.g. after M1<->F,
            # if F later re-mates with M2, simply overwriting
            # agent_mate[M2]/agent_mate[F] would leave agent_mate[M1] == F
            # dangling, still pointing at F even though F's own record now
            # says M2 -- M1 would keep donating to an ex indefinitely (until
            # M1 himself next reproduces). Remating with a different partner
            # is the ordinary case here (mate selection has no memory of
            # prior pairing), not a rare edge case.
            old_male_mate = self.agent_mate.get(male)
            if old_male_mate is not None and old_male_mate != mate:
                self.agent_mate.pop(old_male_mate, None)
            old_mate_mate = self.agent_mate.get(mate)
            if old_mate_mate is not None and old_mate_mate != male:
                self.agent_mate.pop(old_mate_mate, None)
            self.agent_mate[male] = mate
            self.agent_mate[mate] = male

            self.agents.append(new_agent)
            self.agent_positions[new_agent] = new_position
            self.predator_positions[new_agent] = new_position
            self.agent_energies[new_agent] = offspring_energy
            self.agent_parents[new_agent] = (male, mate)
            # Both parents are now breeding adults -- see has_reproduced's
            # docstring for why this ends their own eligibility for
            # parental care from THEIR parents (_share_energy_with_offspring).
            self.has_reproduced.add(male)
            self.has_reproduced.add(mate)

            # mate is always the predator_female here (drawn from
            # female_snapshot, already filtered to "predator_female" in female).
            cost_female = offspring_energy * self.predator_birth_cost_share_female
            cost_male = offspring_energy * self.predator_birth_cost_share_male
            self.agent_energies[mate] -= cost_female
            self.agent_energies[male] -= cost_male
            self.grid_world_state[1, *new_position] = offspring_energy
            self.grid_world_state[1, *male_position] = self.agent_energies[male]
            self.grid_world_state[1, *mate_position] = self.agent_energies[mate]

            if child_sex == "predator_male":
                self.current_num_predator_male += 1
            else:
                self.current_num_predator_female += 1
            self.episode_births[child_sex] += 1

            rewards[new_agent] = 0
            rewards[male] = rewards.get(male, 0.0) + self.reproduction_reward_predator
            rewards[mate] = rewards.get(mate, 0.0) + self.reproduction_reward_predator
            self.cumulative_rewards[new_agent] = 0
            self.cumulative_rewards[male] += self.reproduction_reward_predator
            self.cumulative_rewards[mate] += self.reproduction_reward_predator

            observations[new_agent] = self._get_observation(new_agent)
            terminations[new_agent] = False
            truncations[new_agent] = False

            if self.verbose_spawning:
                print(f"New {child_sex} {new_agent} spawned at {new_position} (parents: {male}, {mate})")

        # Step 6: generate observations for all agents AFTER all engagements/spawning
        for agent in self.agents:
            if agent in self.agent_positions:
                observations[agent] = self._get_observation(agent)

        terminations["__all__"] = (
            self.current_num_prey <= 0 or self.current_num_predator_male <= 0 or self.current_num_predator_female <= 0
        )

        observations = {agent: observations[agent] for agent in self.agents if agent in observations}
        rewards = {agent: rewards[agent] for agent in self.agents if agent in rewards}
        terminations = {agent: terminations[agent] for agent in self.agents if agent in terminations}
        truncations = {agent: truncations[agent] for agent in self.agents if agent in truncations}
        truncations["__all__"] = False

        terminations["__all__"] = (
            self.current_num_prey <= 0 or self.current_num_predator_male <= 0 or self.current_num_predator_female <= 0
        )

        self.agents.sort()

        self.current_step += 1

        return observations, rewards, terminations, truncations, infos

    def _build_episode_training_metrics(self) -> Dict[str, float]:
        """
        Ecology metrics for TensorBoard, surfaced via the EpisodeReturn callback.
        """
        return {
            "episode_length": float(self.current_step),
            "births_predator_male": float(self.episode_births["predator_male"]),
            "births_predator_female": float(self.episode_births["predator_female"]),
            "births_prey": float(self.episode_births["prey"]),
            "deaths_predator_male": float(self.episode_deaths["predator_male"]),
            "deaths_predator_female": float(self.episode_deaths["predator_female"]),
            "deaths_prey": float(self.episode_deaths["prey"]),
            "final_num_predator_male": float(self.current_num_predator_male),
            "final_num_predator_female": float(self.current_num_predator_female),
            "final_num_prey": float(self.current_num_prey),
            "extinct_predator_male": float(self.current_num_predator_male <= 0),
            "extinct_predator_female": float(self.current_num_predator_female <= 0),
            "extinct_prey": float(self.current_num_prey <= 0),
            # Hunting behavior/effectiveness by sex -- lets you distinguish
            # "never attempts" (avoidance) from "attempts but fails/dies"
            # from "never gets the chance," which births/deaths alone
            # can't. hunting_deaths is combat-caused only; the remainder,
            # deaths_predator_*[sex] - hunting_deaths_predator_*[sex], is
            # starvation.
            "hunting_attempts_predator_male": float(self.hunting_attempts["predator_male"]),
            "hunting_attempts_predator_female": float(self.hunting_attempts["predator_female"]),
            "hunting_successes_predator_male": float(self.hunting_successes["predator_male"]),
            "hunting_successes_predator_female": float(self.hunting_successes["predator_female"]),
            "hunting_success_rate_predator_male": (
                self.hunting_successes["predator_male"] / self.hunting_attempts["predator_male"]
                if self.hunting_attempts["predator_male"] > 0
                else 0.0
            ),
            "hunting_success_rate_predator_female": (
                self.hunting_successes["predator_female"] / self.hunting_attempts["predator_female"]
                if self.hunting_attempts["predator_female"] > 0
                else 0.0
            ),
            "hunting_deaths_predator_male": float(self.hunting_deaths["predator_male"]),
            "hunting_deaths_predator_female": float(self.hunting_deaths["predator_female"]),
            # Provisioning mechanics: how much energy is actually flowing
            # through the mate-gift and parental-care mechanisms this
            # episode (both are 0 if no one ever reproduces).
            "mate_gift_events": float(self.mate_gift_events),
            "mate_gift_energy_total": float(self.mate_gift_energy_total),
            "parental_care_events": float(self.parental_care_events),
            "parental_care_energy_total": float(self.parental_care_energy_total),
        }

    def _resolve_hunting_attempt(
        self,
        agent,
        predator_position,
        caught_prey,
        success_prob,
        death_prob,
        observations,
        rewards,
        terminations,
        truncations,
    ):
        """Resolve a predator (either sex) attempting to hunt a co-located prey.

        Returns (outcome, reward_delta, energy_gained), outcome in
        {"success", "predator_dies", "failure"}.

        "success": prey is fully removed (same bookkeeping as a deterministic
          catch); the caller still finalizes the predator's own turn (it may
          also gather fruit this same step). energy_gained is the raw energy
          amount the predator's own agent_energies was just credited by --
          returned so the caller can apply male provisioning
          (_apply_male_gift) on it, since the prey's own energy is gone
          (deleted) by the time this returns.
        "predator_dies": the prey is left completely untouched -- it proceeds
          to its own turn normally later in the Step 3 loop. The predator is
          removed using the same bookkeeping as the starvation-removal branch
          above. The caller MUST `continue` after this outcome (skip fruit-
          gathering -- the agent no longer exists). energy_gained is 0.0.
        "failure": no side effects; caller proceeds to fruit-gathering as
          normal. energy_gained is 0.0.
        """
        sex = "predator_male" if "predator_male" in agent else "predator_female"
        self.hunting_attempts[sex] += 1
        roll = self.rng.random()
        if roll < success_prob:
            if self.verbose_engagement:
                print(f"[ENGAGE] {agent} caught {caught_prey} at {predator_position}!")
            self.agents_just_ate.add(agent)
            # Clamp at 0: a prey that starved this same step (energy <= 0 after Step 1) can still be
            # caught if the predator is processed before it in Step 3's loop; it must not take energy
            # from the predator nor pay a negative energy-proportional reward.
            energy_gained = max(self.agent_energies[caught_prey], 0.0)
            self.agent_energies[agent] += energy_gained
            self.grid_world_state[1, *predator_position] = self.agent_energies[agent]

            observations[caught_prey] = self._get_observation(caught_prey)
            rewards[caught_prey] = self.penalty_prey_caught
            self.cumulative_rewards[caught_prey] += rewards[caught_prey]
            terminations[caught_prey] = True
            truncations[caught_prey] = False
            self.current_num_prey -= 1
            self.episode_deaths["prey"] += 1
            self.grid_world_state[2, *self.agent_positions[caught_prey]] = 0
            del self.agent_positions[caught_prey]
            del self.prey_positions[caught_prey]
            del self.agent_energies[caught_prey]
            self.hunting_successes[sex] += 1
            return (
                "success",
                self.reward_predator_catch_prey + self.reward_predator_per_energy * energy_gained,
                energy_gained,
            )

        if roll < success_prob + death_prob:
            if self.verbose_engagement:
                print(f"[ENGAGE] {agent} attacked {caught_prey} at {predator_position} and was killed!")
            observations[agent] = self._get_observation(agent)
            rewards[agent] = self.penalty_predator_death_in_combat
            self.cumulative_rewards[agent] += rewards[agent]
            terminations[agent] = True
            truncations[agent] = False
            if "predator_male" in agent:
                self.current_num_predator_male -= 1
                self.episode_deaths["predator_male"] += 1
            else:
                self.current_num_predator_female -= 1
                self.episode_deaths["predator_female"] += 1
            self.hunting_deaths[sex] += 1
            self.grid_world_state[1, *predator_position] = 0
            del self.predator_positions[agent]
            del self.agent_positions[agent]
            del self.agent_energies[agent]
            return "predator_dies", 0.0, 0.0

        return "failure", 0.0, 0.0

    def _apply_male_gift(self, agent, energy_gained):
        """Exclusive (pair-bonded), unidirectional male -> female provisioning:
        a predator_male donates male_gift_donation_rate of a successful
        hunt's energy gain to HIS RECORDED MATE ONLY (self.agent_mate),
        never to any other nearby female -- more biologically apt for the
        human pair-bonding this module studies than broadcasting to whoever
        happens to be nearby (contrast eco_evolutionary_nuptial_gift, which
        deliberately broadcasts to any nearby female, modeling a
        non-pair-bonded species). Mechanically executed (not a learned
        action) -- same credit-assignment rationale as
        eco_evolutionary_nuptial_gift's male_donation_rate (a donor's own
        reward stream never reflects a recipient's downstream fitness, so a
        learned "donate" action would face a real credit-assignment gap).
        Meant to offset predator_female's post-birth energy deficit: she
        pays the larger share of birth cost (predator_birth_cost_share_female)
        but her only reliable income (fruit) is weak, shared, and depleting,
        unlike the male's much higher hunting success rate.

        No-ops if he has no recorded mate yet (never successfully
        reproduced), if she's no longer alive, if she's already at <= 0
        energy this step (about to starve -- see the historical note below),
        or if she's outside predator_gift_range (Chebyshev distance) -- the
        gift is still a physical exchange, so proximity still matters even
        though the recipient is now a specific individual rather than
        anyone nearby.

        Historical note on the <= 0 energy guard: Step 3 processes agents in
        self.agents order, and on the very first step of an episode (before
        self.agents has ever been sorted) predator_male_* entries precede
        predator_female_* -- so a mate who already took a fatal Step-1
        homeostatic hit this step, but whose own starvation check hasn't run
        yet, would otherwise still be reachable here and receive a gift that
        pushes her energy back above zero, making her survive purely as an
        accident of iteration order. On every later step (self.agents is
        sorted at the end of step(), putting predator_female_* first) a
        starving mate is already removed before any male's turn runs, so
        this guard just makes that (already-starving-is-already-decided)
        behavior consistent across steps instead of order-dependent.
        """
        if energy_gained <= 0.0 or self.male_gift_donation_rate <= 0.0:
            return
        mate = self.agent_mate.get(agent)
        if mate is None or mate not in self.agent_positions or self.agent_energies[mate] <= 0.0:
            return
        position = self.agent_positions[agent]
        mate_position = self.agent_positions[mate]
        radius = self.predator_gift_range
        if max(abs(mate_position[0] - position[0]), abs(mate_position[1] - position[1])) > radius:
            return
        donation = self.male_gift_donation_rate * energy_gained
        self.agent_energies[agent] -= donation
        self.grid_world_state[1, *position] = self.agent_energies[agent]
        self.agent_energies[mate] += donation
        self.grid_world_state[1, *mate_position] = self.agent_energies[mate]
        self.mate_gift_events += 1
        self.mate_gift_energy_total += donation

    def _share_energy_with_offspring(self, agent, energy_gained):
        """Parental care: both parents share parent_offspring_share_rate of
        any successful forage (hunt or fruit -- called from every
        successful-forage branch of both predator sexes) with their own
        nearby living offspring (self.agent_parents), within
        predator_gift_range. Mechanically executed, like _apply_male_gift,
        for the same credit-assignment reasoning. Unlike _apply_male_gift
        (exclusive to one recorded mate), this splits evenly across
        however many of the forager's own children are currently nearby,
        since a parent can have multiple living offspring at once.

        Cuts off once an offspring has reproduced itself
        (self.has_reproduced): it's then a breeding adult, not a dependent
        juvenile, regardless of how close it still stands to its own
        parents. This is a reproduction-based independence cutoff, not an
        age-based one -- this module tracks no per-agent age at all, and
        "started its own family" is a cheaper, already-available proxy for
        "grown up" than adding age/weaning-duration bookkeeping would be.
        Distance still tapers care off naturally too: a grown offspring
        that simply wanders away (without yet reproducing) falls out of
        range on its own, no extra state needed for that part.

        Note on same-step ordering: this runs during Step 3, which precedes
        Step 5b (where has_reproduced gets updated). An offspring that
        reproduces for the first time later in THIS step can therefore
        still receive care earlier in this same step -- it only becomes
        ineligible starting next step. Deterministic, not a bug.
        """
        if energy_gained <= 0.0 or self.parent_offspring_share_rate <= 0.0:
            return
        position = self.agent_positions[agent]
        radius = self.predator_gift_range
        children = [
            other
            for other, pos in self.predator_positions.items()
            if agent in self.agent_parents.get(other, ())
            and other not in self.has_reproduced
            and self.agent_energies[other] > 0.0
            and max(abs(pos[0] - position[0]), abs(pos[1] - position[1])) <= radius
        ]
        if not children:
            return
        donation_total = self.parent_offspring_share_rate * energy_gained
        share = donation_total / len(children)
        self.agent_energies[agent] -= donation_total
        self.grid_world_state[1, *position] = self.agent_energies[agent]
        for child in children:
            self.agent_energies[child] += share
            self.grid_world_state[1, *self.agent_positions[child]] = self.agent_energies[child]
        self.parental_care_events += 1
        self.parental_care_energy_total += donation_total

    def _get_movement_energy_cost(self, agent, action):
        if action == self.noop_action_id:
            return 0.0
        return self.move_energy_cost_per_step_predator if "predator" in agent else self.move_energy_cost_per_step_prey

    def _get_move(self, agent: AgentID, action: int) -> Tuple[int, int]:
        agent_type_nr = 1 if "predator" in agent else 2
        current_position = self.agent_positions[agent]
        move_vector = self.action_to_move_tuple[action]
        new_position = (current_position[0] + move_vector[0], current_position[1] + move_vector[1])
        new_position = tuple(np.clip(new_position, 0, self.grid_size - 1))
        if self.grid_world_state[agent_type_nr, *new_position] > 0:
            new_position = current_position

        return new_position

    def _get_observation(self, agent):
        observation_range = self.predator_obs_range if "predator" in agent else self.prey_obs_range
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp, observation_range)
        observation = np.zeros(
            (self.num_obs_channels, observation_range, observation_range),
            dtype=np.float64,
        )
        observation[0].fill(1)
        observation[0, xolo:xohi, yolo:yohi] = 0
        observation[1:, xolo:xohi, yolo:yohi] = self.grid_world_state[1:, xlo:xhi, ylo:yhi]

        return observation

    def _obs_clip(self, x, y, observation_range):
        observation_offset = (observation_range - 1) // 2
        xld, xhd = x - observation_offset, x + observation_offset
        yld, yhd = y - observation_offset, y + observation_offset
        xlo, xhi = np.clip(xld, 0, self.grid_size - 1), np.clip(xhd, 0, self.grid_size - 1)
        ylo, yhi = np.clip(yld, 0, self.grid_size - 1), np.clip(yhd, 0, self.grid_size - 1)
        xolo, yolo = abs(np.clip(xld, -observation_offset, 0)), abs(np.clip(yld, -observation_offset, 0))
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def _get_agent_by_position(self) -> dict:
        return {position: agent for agent, position in self.agent_positions.items()}

    def _remove_agent(self, agent: AgentID):
        """Removes an agent from all tracking dictionaries."""
        del self.agent_positions[agent]
        del self.agent_energies[agent]

        if "predator_male" in agent:
            del self.predator_positions[agent]
            self.current_num_predator_male -= 1
        elif "predator_female" in agent:
            del self.predator_positions[agent]
            self.current_num_predator_female -= 1
        elif "prey" in agent:
            del self.prey_positions[agent]
            self.current_num_prey -= 1

    def _find_available_spawn_position(self, reference_position, occupied_positions):
        """
        Finds an available position for spawning a new agent.
        Tries to spawn near the reference position first before selecting a random free position.
        """
        x, y = reference_position
        potential_positions = [
            (x + dx, y + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size
        ]

        valid_positions = [pos for pos in potential_positions if pos not in occupied_positions]

        if valid_positions:
            return valid_positions[0]

        all_positions = {(i, j) for i in range(self.grid_size) for j in range(self.grid_size)}
        free_positions = list(all_positions - occupied_positions)

        if free_positions:
            return free_positions[self.rng.integers(len(free_positions))]

        return None

    def get_state_snapshot(self):
        return {
            "current_step": self.current_step,
            "agent_positions": self.agent_positions.copy(),
            "agent_energies": self.agent_energies.copy(),
            "predator_positions": self.predator_positions.copy(),
            "prey_positions": self.prey_positions.copy(),
            "grass_positions": self.grass_positions.copy(),
            "grass_energies": self.grass_energies.copy(),
            "fruit_positions": self.fruit_positions.copy(),
            "fruit_energies": self.fruit_energies.copy(),
            "grid_world_state": self.grid_world_state.copy(),
            "agents": self.agents.copy(),
            "cumulative_rewards": self.cumulative_rewards.copy(),
            "agent_mate": self.agent_mate.copy(),
            "agent_parents": self.agent_parents.copy(),
            "has_reproduced": self.has_reproduced.copy(),
            "current_num_predator_male": self.current_num_predator_male,
            "current_num_predator_female": self.current_num_predator_female,
            "current_num_prey": self.current_num_prey,
            "agents_just_ate": self.agents_just_ate.copy(),
            "pending_removal": self._pending_removal.copy(),
            "next_predator_male_idx": self._next_predator_male_idx,
            "next_predator_female_idx": self._next_predator_female_idx,
            "next_prey_idx": self._next_prey_idx,
        }

    def restore_state_snapshot(self, snapshot):
        self.current_step = snapshot["current_step"]
        self.agent_positions = snapshot["agent_positions"].copy()
        self.agent_energies = snapshot["agent_energies"].copy()
        self.predator_positions = snapshot["predator_positions"].copy()
        self.prey_positions = snapshot["prey_positions"].copy()
        self.grass_positions = snapshot["grass_positions"].copy()
        self.grass_energies = snapshot["grass_energies"].copy()
        self.fruit_positions = snapshot["fruit_positions"].copy()
        self.fruit_energies = snapshot["fruit_energies"].copy()
        self.grid_world_state = snapshot["grid_world_state"].copy()
        self.agents = snapshot["agents"].copy()
        self.cumulative_rewards = snapshot["cumulative_rewards"].copy()
        self.agent_mate = snapshot["agent_mate"].copy()
        self.agent_parents = snapshot["agent_parents"].copy()
        self.has_reproduced = snapshot["has_reproduced"].copy()
        self.current_num_predator_male = snapshot["current_num_predator_male"]
        self.current_num_predator_female = snapshot["current_num_predator_female"]
        self.current_num_prey = snapshot["current_num_prey"]
        self.agents_just_ate = snapshot["agents_just_ate"].copy()
        self._pending_removal = snapshot["pending_removal"].copy()
        self._next_predator_male_idx = snapshot["next_predator_male_idx"]
        self._next_predator_female_idx = snapshot["next_predator_female_idx"]
        self._next_prey_idx = snapshot["next_prey_idx"]
