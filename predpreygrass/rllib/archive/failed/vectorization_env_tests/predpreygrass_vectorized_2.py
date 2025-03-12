import numpy as np
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, Dict, List, Tuple
from config_env_15 import config_env


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config = config or config_env

        self.max_steps = config["max_steps"]
        self.grid_size = config["grid_size"]
        self.num_obs_channels = config["num_obs_channels"]
        self.predator_obs_range = config["predator_obs_range"]
        self.prey_obs_range = config["prey_obs_range"]

        self.n_possible_predators = config["n_possible_predators"]
        self.n_possible_prey = config["n_possible_prey"]
        self.n_initial_active_predator = config["n_initial_active_predator"]
        self.n_initial_active_prey = config["n_initial_active_prey"]
        self.initial_num_grass = config["initial_num_grass"]

        self.initial_energy_predator = config["initial_energy_predator"]
        self.initial_energy_prey = config["initial_energy_prey"]
        self.initial_energy_grass = config["initial_energy_grass"]

        self.energy_loss_per_step_predator = config["energy_loss_per_step_predator"]
        self.energy_loss_per_step_prey = config["energy_loss_per_step_prey"]
        self.energy_gain_per_step_grass = config["energy_gain_per_step_grass"]

        self.reward_predator_catch_prey = config["reward_predator_catch_prey"]
        self.reward_prey_eat_grass = config["reward_prey_eat_grass"]
        self.penalty_prey_caught = config["penalty_prey_caught"]
        self.reproduction_reward_predator = config["reproduction_reward_predator"]
        self.reproduction_reward_prey = config["reproduction_reward_prey"]

        # Action mapping
        self.action_to_move_tuple = np.array([
            (0, 0),  # Stay
            (-1, 0), # Up
            (1, 0),  # Down
            (0, -1), # Left
            (0, 1),  # Right
        ])

        # Define agent arrays
        self.agent_positions = np.full((self.n_possible_predators + self.n_possible_prey, 2), -1, dtype=np.int32)
        self.agent_energies = np.zeros(self.n_possible_predators + self.n_possible_prey, dtype=np.float64)

        self.grid_world_state = np.zeros((self.num_obs_channels, self.grid_size, self.grid_size), dtype=np.float64)

        # List of active agents
        self.agents = [f"predator_{i}" for i in range(self.n_initial_active_predator)] + \
                      [f"prey_{i}" for i in range(self.n_initial_active_prey)]
        self.possible_agents = [
            f"predator_{i}" for i in range(self.n_possible_predators)
        ] + [
            f"prey_{j}" for j in range(self.n_possible_prey)
        ]


        self.rng = np.random.default_rng()

       # Spaces
       # Compute observation shapes
        predator_obs_shape = (self.num_obs_channels, self.predator_obs_range, self.predator_obs_range)
        prey_obs_shape = (self.num_obs_channels, self.prey_obs_range, self.prey_obs_range)

        # Define observation spaces
        predator_obs_space = gymnasium.spaces.Box(
            low=0.0, high=100.0, shape=predator_obs_shape, dtype=np.float64
        )
        prey_obs_space = gymnasium.spaces.Box(
            low=0.0, high=100.0, shape=prey_obs_shape, dtype=np.float64
        )

        # Assign spaces based on agent type
        self.observation_spaces = {
            agent: predator_obs_space if "predator" in agent else prey_obs_space
            for agent in self.possible_agents
        }


        action_space = gymnasium.spaces.Discrete(
            5
        )  # 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
        self.action_spaces = {agent: action_space for agent in self.possible_agents}


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.rng = np.random.default_rng(seed)

        # Reset grid state and agent arrays
        self.grid_world_state.fill(0)
        self.agent_positions.fill(-1)
        self.agent_energies.fill(0)

        # Generate random positions
        total_entities = self.n_initial_active_predator + self.n_initial_active_prey + self.initial_num_grass
        all_positions = self.rng.choice(self.grid_size * self.grid_size, total_entities, replace=False)
        all_positions = np.column_stack((all_positions // self.grid_size, all_positions % self.grid_size))

        # Assign positions and energy
        self.agent_positions[:self.n_initial_active_predator] = all_positions[:self.n_initial_active_predator]
        self.agent_energies[:self.n_initial_active_predator] = self.initial_energy_predator
        self.grid_world_state[1, all_positions[:self.n_initial_active_predator, 0], all_positions[:self.n_initial_active_predator, 1]] = self.initial_energy_predator

        self.agent_positions[self.n_initial_active_predator:self.n_initial_active_predator + self.n_initial_active_prey] = \
            all_positions[self.n_initial_active_predator:self.n_initial_active_predator + self.n_initial_active_prey]
        self.agent_energies[self.n_initial_active_predator:self.n_initial_active_predator + self.n_initial_active_prey] = self.initial_energy_prey
        self.grid_world_state[2, all_positions[self.n_initial_active_predator:self.n_initial_active_predator + self.n_initial_active_prey, 0], 
                                 all_positions[self.n_initial_active_predator:self.n_initial_active_predator + self.n_initial_active_prey, 1]] = self.initial_energy_prey

        self.grid_world_state[3, all_positions[self.n_initial_active_predator + self.n_initial_active_prey:, 0], 
                                 all_positions[self.n_initial_active_predator + self.n_initial_active_prey:, 1]] = self.initial_energy_grass

        # Generate initial observations
        observations = {agent: self._get_observation(idx) for idx, agent in enumerate(self.agents)}
        return observations, {}

    def step(self, action_dict):
        # Convert action_dict to NumPy array
        action_array = np.array([action_dict[agent] for agent in self.agents])

        # Apply movement
        move_vectors = self.action_to_move_tuple[action_array]
        new_positions = self.agent_positions[:len(self.agents)] + move_vectors
        np.clip(new_positions, 0, self.grid_size - 1, out=new_positions)

        # Handle collisions
        _, unique_indices = np.unique(new_positions, axis=0, return_index=True)
        mask = np.zeros(len(new_positions), dtype=bool)
        mask[unique_indices] = True
        self.agent_positions[:len(self.agents)][mask] = new_positions[mask]

        # Update energy (predators and prey lose energy)
        predator_mask = np.arange(len(self.agents)) < self.n_initial_active_predator
        self.agent_energies[predator_mask] -= self.energy_loss_per_step_predator
        self.agent_energies[~predator_mask] -= self.energy_loss_per_step_prey

        # Update grid state
        self.grid_world_state[1:3, :, :] = 0
        predator_positions = self.agent_positions[:self.n_initial_active_predator]
        prey_positions = self.agent_positions[self.n_initial_active_predator:len(self.agents)]
        self.grid_world_state[1, predator_positions[:, 0], predator_positions[:, 1]] = self.agent_energies[predator_mask]
        self.grid_world_state[2, prey_positions[:, 0], prey_positions[:, 1]] = self.agent_energies[~predator_mask]

        # Compute observations
        observations = {agent: self._get_observation(i) for i, agent in enumerate(self.agents)}

        # Return updated state
        return observations, {}, {}, {}, {}

    def _get_observation(self, agent_idx):
        observation_range = self.predator_obs_range if agent_idx < self.n_initial_active_predator else self.prey_obs_range
        obs_offset = (observation_range - 1) // 2
        x, y = self.agent_positions[agent_idx]
        xlo, xhi = max(0, x - obs_offset), min(self.grid_size, x + obs_offset + 1)
        ylo, yhi = max(0, y - obs_offset), min(self.grid_size, y + obs_offset + 1)

        observation = np.zeros((self.num_obs_channels, observation_range, observation_range), dtype=np.float64)
        observation[0].fill(1)
        xolo, yolo = max(0, obs_offset - x), max(0, obs_offset - y)
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        observation[0, xolo:xohi, yolo:yohi] = 0
        observation[1:, xolo:xohi, yolo:yohi] = self.grid_world_state[1:, xlo:xhi, ylo:yhi]

        return observation
