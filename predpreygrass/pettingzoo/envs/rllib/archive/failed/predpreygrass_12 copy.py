import numpy as np
from numpy.typing import NDArray
import gymnasium 
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, Dict, List, Tuple

class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()

        self.max_steps: int = 200
        self.reward_predator_catch: float = 15.0
        self.reward_prey_survive : float= 0.5
        self.penalty_predator_miss : float = -0.2
        self.penalty_prey_caught: float = -20.0

        self.current_step: int = 0

        # Learning agents
        self.max_num_predators: int = 4
        self.max_num_prey: int = 4
        self.num_predators: int = 4
        self.num_prey: int = 4
        self.initial_energy_predator: float = 5.0
        self.initial_energy_prey: float = 3.0
        self.energy_depletion_rate: float = 0.01

        self.episode_rewards = {}  # Track total rewards per agent

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
        self.grid_size: int = 10
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

        self.agents = [f"predator_{i}" for i in range(4)] + [
            f"prey_{j}" for j in range(4)
        ]

        # ✅ Reset agent positions and energies
        self.agent_positions = {}
        self.agent_energies = {}

        # ✅ Reset Rewards
        self.episode_rewards = {agent_id: 0 for agent_id in self.agents}


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
            energy_value=self.initial_energy_predator,
        )
        prey_positions, prey_energies = place_entities(
            [agent for agent in self.agents if "prey" in agent],
            grid_channel=2,
            energy_value=self.initial_energy_prey,
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
        """
        Executes one step in the environment.

        - Processes agent movements.
        - Computes rewards before removing agents.
        - Checks termination and truncation conditions.
        """
        #print("\n=== DEBUG: Actions Received ===")
        for agent, action in action_dict.items():
            #print(f"Agent {agent}: Action {action}")
            if "predator" in agent:
                #print(f"Energies: {self.agent_energies}")
                self.agent_energies[agent] -= self._get_time_step_energy_cost(agent,self.energy_depletion_rate)
            elif "prey" in agent:
                self.agent_energies[agent] -= self._get_time_step_energy_cost(agent,self.energy_depletion_rate) 


        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        # Step 1: Process all movements before removing any agents
        for agent, action in action_dict.items():
            if agent in self.agent_positions:
                new_position = self._get_move(agent, action)
                self.agent_positions[agent] = new_position

        # Step 2: Ensure every agent gets an observation
        for agent in self.agents:
            if agent in self.agent_positions:
                observations[agent] = self._get_observation(agent)
            else:
                observations[agent] = np.zeros((self.num_obs_channels, self.max_obs_range, self.max_obs_range))  # Placeholder observation

        # Step 3: Assign rewards **before** removing agents
        for agent in self.agents:
            if agent in self.agent_positions:
                step_reward = self._get_reward(agent)  
                rewards[agent] = step_reward
                self.episode_rewards[agent] += step_reward  # ✅ Accumulate total episode reward
            else:
                rewards[agent] = 0.0  # Prevent missing rewards

        # Step 4: Handle agent removals (Prey caught, Energy depleted)
        for agent in list(self.agents):
            if agent not in self.agent_positions:
                continue

            # Check if energy is depleted
            if self.agent_energies[agent] <= 0:
                rewards[agent] = self._get_reward(agent)  # Assign reward before deletion
                terminations[agent] = True
                truncations[agent] = False

                # Remove from the grid
                x, y = self.agent_positions[agent]
                self.grid[1 if "predator" in agent else 2, x, y] = 0  

                # Delete agent from state
                del self.agent_positions[agent]
                del self.agent_energies[agent]
                self.agents.remove(agent)  # ✅ Ensures agent is removed safely
                #print(f"[DEBUG] {agent} ran out of energy and was removed.")
                if "predator" in agent:
                    self.num_predators -= 1
                elif "prey" in agent:
                    self.num_prey -= 1

            # Check if prey is caught (process rewards before removal)
            elif "prey" in agent:
                prey_position = self.agent_positions[agent]
                for predator, predator_position in self.agent_positions.items():
                    if "predator" in predator and np.array_equal(predator_position, prey_position):
                        # Reward predator before removing prey
                        rewards[predator] = self.reward_predator_catch
                        #print(f"[DEBUG] Predator {predator} caught prey {agent}! Predator Reward: {self.reward_predator_catch}")

                        # Assign penalty before removal
                        if agent in self.agent_positions:
                            rewards[agent] = self._get_reward(agent)  # Assign penalty
                            #print(f"[DEBUG] Prey {agent} caught. Penalty: {self.penalty_prey_caught}")

                            # Remove prey
                            terminations[agent] = True
                            truncations[agent] = False
                            self.num_prey -= 1

                            # Remove from grid and lists
                            x, y = prey_position
                            self.grid[2, x, y] = 0  # Remove prey from the grid
                            del self.agent_energies[agent]
                            del self.agent_positions[agent]
                            self.agents.remove(agent)  # ✅ Ensures agent is removed safely
                        break  # Stop checking after first predator catches prey

        # Step 5: Global termination conditions
        self.current_step += 1
        terminations["__all__"] = self.num_prey <= 0 or self.num_predators <= 0
        truncations["__all__"] = self.current_step >= self.max_steps


        if terminations["__all__"] or truncations["__all__"]:
            print(f"End of Episode: Total rewards: {self.episode_rewards}")

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

    def _get_time_step_energy_cost(self, agent, step_factor=0.005):
        """
        Calculate the energy cost for a time step of the agent.
        """
        if "predator" in agent:
            return step_factor * self.initial_energy_predator
        elif "prey" in agent:
            return step_factor * self.initial_energy_prey

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
        # ✅ Clip new position to stay within grid bounds
        new_position = np.clip(new_position, 0, self.grid_size - 1)

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
        """
        Compute the reward for the given agent.

        - Predators are rewarded for catching prey.
        - Predators are penalized for missing prey.
        - Prey receive small survival rewards.
        - Prey are penalized if caught.
        """
        reward = 0.0

        if "predator" in agent:
            prey_positions = [
                self.agent_positions[prey] 
                for prey in self.agents 
                if "prey" in prey and prey in self.agent_positions
            ]

            if any(np.array_equal(self.agent_positions[agent], pos) for pos in prey_positions):
                reward = self.reward_predator_catch
            else:
                reward = self.penalty_predator_miss

        elif "prey" in agent:
            if agent in self.agent_positions:
                reward = self.reward_prey_survive
            else:
                reward = self.penalty_prey_caught

        return reward  # This now only returns the immediate reward


        
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
