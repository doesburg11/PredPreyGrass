import numpy as np
from numpy.typing import NDArray
import gymnasium 
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, Dict, List, Tuple

class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.max_steps: int = config.get("max_steps", 200) if config else 200
        self.reward_predator_catch: float = config.get("reward_predator_catch", 15.0)
        self.reward_prey_survive : float= config.get("reward_prey_survive", 0.5)
        self.penalty_predator_miss : float = config.get("penalty_predator_miss", -0.2)
        self.penalty_prey_caught: float = config.get("penalty_prey_caught", -20.0)

        self.current_step: int = 0

        # Learning agents
        self.max_num_predators: int = 4
        self.max_num_prey: int = 4
        self.num_predators: int = 4
        self.num_prey: int = 4
        # self.num_agents: int = self.num_predators + self.num_prey  # read only property inherited from MultiAgentEnv
        self.possible_agents: List[AgentID] = [   # max_num of learning agents, placeholder inherited from MultiAgentEnv
            f"predator_{i}" for i in range(self.max_num_predators)
        ] + [f"prey_{j}" for j in range(self.max_num_prey)]
        self.agents: List[AgentID] = [f"predator_{i}" for i in range(self.num_predators)] + [
            f"prey_{j}" for j in range(self.num_prey)
        ]

        # Non-learning agents (grass); not included in 'possible_agents' or 'agents'
        self.max_num_grass: int = 4
        self.num_grass: int = 4
        self.grass_agents: List[AgentID] = [f"grass_{k}" for k in range(self.max_num_grass)]

        # Grid and observation settings
        self.grid_size: int = 5
        self.num_obs_channels: int = 4  # Border, Predator, Prey, Grass
        self.max_obs_range: int = 7
        self.max_obs_offset: int = (self.max_obs_range - 1) // 2

        # Spaces
        obs_space_shape = (
            self.num_obs_channels,
            self.max_obs_range,
            self.max_obs_range,
        )
        observation_space = gymnasium.spaces.Box(
            low=-1.0, high=100.0, shape=obs_space_shape, dtype=np.float64
        )
        self.observation_spaces = {
            agent: observation_space for agent in self.possible_agents
        }
        action_space = gymnasium.spaces.Discrete(5)  # 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
        self.action_spaces = {agent: action_space for agent in self.possible_agents}

        # Initialize grid and agent positions
        self.positions: Dict[str, NDArray[np.int_]] = {} 
        # self.agent_positions: Dict[str, tuple] = {}
        self.energies: Dict[str, float] = {}
        self.grid: NDArray[np.float64] = np.zeros(
            (self.num_obs_channels, self.grid_size, self.grid_size), dtype=np.float64
        )
        # Mapping actions to movements: (dx, dy)
        self.action_to_move_tmp: Dict[int,NDArray[int_]] = {
            0: (0, 0),  # Stay
            1: (-1, 0),  # Up
            2: (1, 0),  # Down
            3: (0, -1),  # Left
            4: (0, 1),  # Right
        }
        # Mapping actions to movements: np.array experimental
        self.action_to_move: Dict[int, NDArray[np.int_]] = {
            0: np.array([0, 0], dtype=np.int_),
            1: np.array([-1, 0], dtype=np.int_),
            2: np.array([1, 0], dtype=np.int_),
            3: np.array([0, -1], dtype=np.int_),
            4: np.array([0, 1], dtype=np.int_),
        }

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.rng = np.random.default_rng(seed)

        # Initialize grid
        self.grid = np.zeros(
            (self.num_obs_channels, self.grid_size, self.grid_size),
            dtype=np.float64,
        )

        # Place entities
        def place_entities(entity_list, grid_channel, energy_value=None) -> Tuple[Dict[str, NDArray[np.int_]], Dict[str, float]]:
            positions, energies = {}, {}
            for entity in entity_list:
                while True:
                    position = self.rng.integers(self.grid_size, size=2) # <class 'numpy.ndarray'> with 2 random integers
                    if self.grid[grid_channel, *position] == 0:  # * is unpacking operator needed for position (numpy.ndarray)
                        self.grid[grid_channel, *position] = energy_value or 1
                        positions[entity] = position  # dict with entity as key and position as np array 
                        if energy_value is not None:
                            energies[entity] = energy_value
                        break
            return positions, energies

        # Place agents and grass
        predator_positions, predator_energies = place_entities(
            [agent for agent in self.agents if "predator" in agent],
            grid_channel=1,
            energy_value=5,
        )
        prey_positions, prey_energies = place_entities(
            [agent for agent in self.agents if "prey" in agent],
            grid_channel=2,
            energy_value=3,
        )
        grass_positions, _ = place_entities(
            self.grass_agents,
            grid_channel=3,
        )

        # Store state
        self.agent_positions = {**predator_positions, **prey_positions}
        self.agent_energies = {**predator_energies, **prey_energies}
        self.grass_positions = grass_positions

        # Generate observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        return observations, {}

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        # Process moves for each agent
        for agent, action in action_dict.items():
            # Process actions only for agents still in the environment
            print(f"{agent} : {self.agent_positions[agent]}", end=" -> ")
            self._apply_move(agent, action)
            print(f"{self.agent_positions[agent]}")
         
        # Check termination and truncation conditions
        for agent in list(self.agents):
            if agent not in self.agent_positions:  # Agent already removed
                continue
            agent_type_nr = 1 if "predator" in agent else 2
            if self.agent_energies[agent] <= 0:
                # Agent has no energy left
                print(f"Agent {agent} dies of lack of energy")
                observations[agent] = self._get_observation(agent)
                x, y = self.agent_positions[agent]
                self.grid[agent_type_nr, x, y] = 0  # Clear from grid
                del self.agent_positions[agent]
                del self.agent_energies[agent]
                terminations[agent] = True
                truncations[agent] = False
            elif "prey" in agent:
                # Check if any predator is on the same position as this prey
                prey_position = self.agent_positions[agent]
                for predator, predator_position in self.agent_positions.items():
                    if "predator" in predator and np.array_equal(predator_position, prey_position):
                        # Prey is caught
                        if agent in self.agent_positions:
                            print(f"Predator {predator} caught prey {agent}")
                            self.num_prey -= 1
                            observations[agent] = self._get_observation(agent)
                            x, y = prey_position
                            self.grid[agent_type_nr, x, y] = 0  # Clear the prey from the grid (channel 2)
                            del self.agent_positions[agent]
                            del self.agent_energies[agent]
                            del self.agents[self.agents.index(agent)]
                            terminations[agent] = True
                            truncations[agent] = False
                        break
                else:
                    terminations[agent] = False
                    truncations[agent] = False
            elif self.current_step >= self.max_steps:
                truncations[agent] = True
                terminations[agent] = False
                observations[agent] = self._get_observation(agent)  # Ensure last observation

            else:
                terminations[agent] = False
                truncations[agent] = False

        # Increment step counter
        self.current_step += 1

        # Global termination and truncation
        terminations["__all__"] = self.num_prey <= 0
        if terminations["__all__"]:
            print("All prey are gone. Environment is terminating.")

        truncations["__all__"] = self.current_step >= self.max_steps

        return observations, rewards, terminations, truncations, infos

    def _get_movement_energy_cost(self, agent, current_position, new_position, distance_factor=0.1):
        """
        Calculate the energy cost for moving an agent.
        
        Args:
            current_position (np.array): Current position of the agent [x, y].
            new_position (np.array): New position of the agent [x, y].
            current_energy (float): Current energy level of the agent.
            max_energy (float): Maximum possible energy level for the agent.
            base_cost (float): Fixed cost for any move.
            distance_factor (float): Scaling factor for the cost based on distance.

        Returns:
            float: Energy cost of the move.
        """
        current_position = self.agent_positions[agent]
        current_energy = self.agent_energies[agent]
        distance = np.linalg.norm(new_position - current_position)  # Euclidean distance
        
        # Calculate the energy cost
        energy_cost = distance * distance_factor * current_energy 
        return energy_cost

    def _get_time_step_energy_cost(self, agent, step_factor=0.1):
        """
        Calculate the energy cost for a time step of the agent.
        """
        return step_factor * self.agent_energies[agent]

    def _get_move(self, agent, action) -> NDArray[np.int_]:
        """
        Get the new position of the agent based on the action.
        """
        # Current position as an np.array
        current_position = self.agent_positions[agent]  # NDArray[np.int_]
        # Movement vector from action
        move_vector = self.action_to_move[action]  # NDArray[np.int_]
        # Calculate new position as an np.array
        new_position = current_position + move_vector  # Element-wise addition
        return new_position.astype(np.int_)  # Ensure dtype is np.int_

    def _apply_move(self, agent, action):
        """
        Apply the agent's action, ensuring moves are within bounds and do not collide with occupied cells.
        """
        # Current position
        x_old, y_old = self.agent_positions[agent]

        # Compute new position based on action
        dx, dy = self.action_to_move[action]
        x_new = np.clip(x_old + dx, 0, self.grid_size - 1)
        y_new = np.clip(y_old + dy, 0, self.grid_size - 1)

        # Determine agent type
        agent_type_nr = 1 if "predator" in agent else 2

        # Check for collisions and finalize position
        if action != 0 and self.grid[agent_type_nr, x_new, y_new] == 0:
            self.agent_positions[agent] = np.array([x_new, y_new], dtype=np.int_)
            self.grid[agent_type_nr, x_old, y_old] = 0  # Clear old position
            self.grid[agent_type_nr, x_new, y_new] = self.agent_energies[agent]
           
    def _get_observation(self, agent):
        """
        Generate an observation for the agent.
        """
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp)
        observation = np.zeros((self.num_obs_channels, self.max_obs_range, self.max_obs_range), dtype=np.float64)
        observation[0].fill(1)
        observation[0, xolo:xohi, yolo:yohi] = 0
        observation[1:, xolo:xohi, yolo:yohi] = self.grid[1:, xlo:xhi, ylo:yhi]
        return observation

    def _get_reward(self, agent):
        reward = 0.0
        if "predator" in agent:
            # Positive reward for predator when catching prey
            prey_positions = [
                self.agent_positions[prey] 
                for prey in self.agents 
                if "prey" in prey and prey in self.agent_positions
            ]
            if any(np.array_equal(self.agent_positions[agent], pos) for pos in prey_positions):
                reward = self.reward_predator_catch  # Reward for catching prey
            else:
                reward = self.penalty_predator_miss  # Penalty for not catching prey
        elif "prey" in agent:
            # Reward prey for survival
            if agent in self.agent_positions:
                reward = self.reward_prey_survive  # Small reward for survival
            else:
                reward = self.penalty_prey_caught  # Penalty for being caught
        return reward
        
    def _obs_clip(self, x, y):
        """
        Clip the observation window to the boundaries of the grid.
        """
        xld, xhd = x - self.max_obs_offset, x + self.max_obs_offset
        yld, yhd = y - self.max_obs_offset, y + self.max_obs_offset
        xlo, xhi = np.clip(xld, 0, self.grid_size - 1), np.clip(xhd, 0, self.grid_size - 1)
        ylo, yhi = np.clip(yld, 0, self.grid_size - 1), np.clip(yhd, 0, self.grid_size - 1)
        xolo, yolo = abs(np.clip(xld, -self.max_obs_offset, 0)), abs(np.clip(yld, -self.max_obs_offset, 0))
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1
