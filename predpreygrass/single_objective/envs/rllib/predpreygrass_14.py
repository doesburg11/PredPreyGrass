# external libraries
import numpy as np
from numpy.typing import NDArray
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, Dict, List, Tuple


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()

        self.max_steps: int = 100
        self.reward_predator_catch: float = 15.0
        self.reward_prey_eat_grass: float = 10.0
        self.reward_prey_step: float = 0.5
        self.reward_predator_step: float = 0.3
        self.penalty_prey_caught: float = -20.0

        self.current_step: int = 0

        # Learning agents
        self.max_num_predators: int = 6
        self.max_num_prey: int = 6
        self.initial_num_predators: int = 4
        self.initial_num_prey: int = 4
        self.num_predators: int = 4
        self.num_prey: int = 4

        self.initial_energy_predator: float = 5.0
        self.initial_energy_prey: float = 3.0
        self.initial_energy_grass: float = 2.0
        self.energy_depletion_rate: float = 0.01

        self.cumulative_rewards = {}  # Track total rewards per agent

        # self.num_agents: int = self.num_predators + self.num_prey  # read only property inherited from MultiAgentEnv
        self.possible_agents: List[AgentID] = (
            [  # max_num of learning agents, placeholder inherited from MultiAgentEnv
                f"predator_{i}" for i in range(self.max_num_predators)
            ]
            + [f"prey_{j}" for j in range(self.max_num_prey)]
        )
        self.agents: List[AgentID] = [
            # placeholder for learning agents, inherited from MultiAgentEnv
            f"predator_{i}" for i in range(self.initial_num_predators)
            ] + [f"prey_{j}" for j in range(self.initial_num_prey)
        ]

        # Non-learning agents (grass); not included in 'possible_agents' or 'agents'
        self.max_num_grass: int = 4
        self.initial_num_grass: int = 4
        self.num_grass: int = 4
        self.initial_energy_grass: float = 2.0
        self.energyaccumulation_rate: float = 0.1
        self.grass_agents: List[AgentID] = [
            f"grass_{k}" for k in range(self.initial_num_grass)
        ]

        # grid_world_state and observation settings
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
        action_space = gymnasium.spaces.Discrete(
            5
        )  # 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
        self.action_spaces = {agent: action_space for agent in self.possible_agents}

        # Initialize grid_world_state and agent positions
        self.agent_positions: Dict[AgentID, NDArray[np.int_]] = {}
        self.agent_energies: Dict[AgentID, float] = {}
        self.grass_energies: Dict[AgentID, float] = {}
        self.grid_world_state_shape: Tuple = (
            self.num_obs_channels,
            self.grid_size,
            self.grid_size,
        )
        self.grid_world_state: NDArray[np.float64] = np.zeros(
            self.grid_world_state_shape, dtype=np.float64
        )
        # Mapping actions to movements
        self.action_to_move: Dict[int, NDArray[np.int_]] = {
            0: np.array([0, 0], dtype=np.int_),
            1: np.array([-1, 0], dtype=np.int_),
            2: np.array([1, 0], dtype=np.int_),
            3: np.array([0, -1], dtype=np.int_),
            4: np.array([0, 1], dtype=np.int_),
        }
        self.num_actions = len(self.action_to_move)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.rng = np.random.default_rng(seed)

        # Initialize grid_world_state
        self.grid_world_state = np.zeros(
            self.grid_world_state_shape,
            dtype=np.float64,
        )
        self.agents = [f"predator_{i}" for i in range(self.initial_num_predators)] + [
            f"prey_{j}" for j in range(self.initial_num_prey)
        ]
        # Reset agent positions and energies
        self.agent_positions = {}
        self.agent_energies = {}

        # Reset cumulative rewards to zero
        self.cumulative_rewards = {agent_id: 0 for agent_id in self.agents}

        # Place entities
        def place_entities(
            entity_list: List[AgentID], grid_channel: int, energy_value: float
        ) -> Tuple[Dict[AgentID, NDArray[np.int_]], Dict[AgentID, float]]:
            agent_positions, agent_energies = {}, {}
            for entity in entity_list:
                while True:
                    position = self.rng.integers(
                        self.grid_size, size=2
                    ) 
                    if (
                        self.grid_world_state[grid_channel, *position] == 0
                    ): 
                        self.grid_world_state[grid_channel, *position] = (
                            energy_value
                        )
                        agent_positions[entity] = (
                            position  # dict with entity as key and position as np array
                        )
                        if energy_value is not None:
                            agent_energies[entity] = energy_value
                        break
            return agent_positions, agent_energies

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
        grass_positions, grass_energies = place_entities(
            self.grass_agents,
            grid_channel=3,
            energy_value=self.initial_energy_grass,
        )

        # Store state
        self.agent_positions = {**predator_positions, **prey_positions}
        self.agent_energies = {**predator_energies, **prey_energies}
        self.grass_positions = grass_positions
        self.grass_energies = grass_energies

        self.num_prey = self.initial_num_prey
        self.num_predators = self.initial_num_predators
        self.num_grass = self.initial_num_grass

        # Generate observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        return observations, {}

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        # Step 1: Process energy depletion due to time steps
        for agent, action in action_dict.items():
            # energy_before = self.agent_energies[agent]
            self.agent_energies[agent] += self._get_time_step_energy_cost(
                agent, self.energy_depletion_rate
            )
        for grass, grass_position in self.grass_positions.items():
            self.grass_energies[grass] += 0.2 # TODO hardcoded
            if (
                self.grass_energies[grass] >= self.initial_energy_grass 
                and grass not in self.grass_agents
            ):
                self.grass_agents.append(grass)
                self.num_grass += 1
                self.grid_world_state[3, *grass_position] = self.grass_energies[grass]
            elif (
                self.grass_energies[grass] >= self.initial_energy_grass
                and grass in self.grass_agents
            ):
                self.grid_world_state[3, *grass_position] = self.grass_energies[grass]


        #print(f"---Agent energies after time step: {self.agent_energies}")
        #print()

        # Step 2: Process movements
        for agent, action in action_dict.items():
            if agent in self.agent_positions:
                agent_type_nr = 1 if "predator" in agent else 2
                old_position = self.agent_positions[agent].copy()
                self.grid_world_state[agent_type_nr,  *old_position] = 0 
                new_position = self._get_move(agent, action)
                self.agent_positions[agent] = new_position
                #print(f"Agent {agent} moved: {old_position} -> {new_position}")
                 
                move_cost = self._get_movement_energy_cost(agent,old_position,new_position)
                #print(f"movement cost {agent}: {move_cost}")
                self.agent_energies[agent] -= move_cost
                self.grid_world_state[agent_type_nr, *new_position] = self.agent_energies[agent]
                #print(f"Agent {agent} moved: {old_position} -> {new_position}")

        #print(f"---Agent energies after move step: {self.agent_energies}")
        #print()

        # Step 3: Prepare agent removals (Prey caught, Energy depleted)
        for agent in self.agents:
            # Agent not active
            if agent not in self.agent_positions:  
                continue
            # Agent has no energy left
            if self.agent_energies[agent] <= 0:
                print(f"[DEBUG] {agent} ran out of energy and is removed.")
                observations[agent] = self._get_observation(agent) # Ensure last observation
                rewards[agent] = 0  # TODO remove hardcoded
                terminations[agent] = True
                truncations[agent] = False
                if "predator" in agent:
                    self.num_predators -= 1
                    self.grid_world_state[1,*self.agent_positions[agent]] = 0
                elif "prey" in agent:
                    self.num_prey -= 1
                    self.grid_world_state[2,*self.agent_positions[agent]] = 0
                del self.agent_positions[agent]
                del self.agent_energies[agent]
                continue
            elif "predator" in agent:
                predator_position = self.agent_positions[agent]
                # Find the first prey at the same position
                caught_prey = next(
                    (prey for prey, prey_position in self.agent_positions.items()
                    if "prey" in prey and np.array_equal(predator_position, prey_position)), None
                )
                if caught_prey:
                    print(f"[DEBUG] {agent} caught {caught_prey}! Predator Reward: {self.reward_predator_catch}")
                    
                    # Assign rewards predator and penaly prey
                    rewards[agent] = self.reward_predator_catch
                    self.cumulative_rewards[agent] += rewards[agent]

                    observations[caught_prey] = self._get_observation(caught_prey)
                    rewards[caught_prey] = self.penalty_prey_caught
                    self.cumulative_rewards[caught_prey] += rewards[caught_prey]   

                    # Remove prey
                    terminations[caught_prey] = True
                    truncations[caught_prey] = False
                    self.num_prey -= 1
                    self.grid_world_state[2, *self.agent_positions[caught_prey]] = 0
                    del self.agent_positions[caught_prey]
                    del self.agent_energies[caught_prey]
                else:
                    # Predator did not catch prey
                    rewards[agent] = self.reward_predator_step

                self.cumulative_rewards[agent] += rewards[agent]
                terminations[agent] = False
                truncations[agent] = False

            elif "prey" in agent:
                if terminations.get(agent) is None or not terminations[agent]:
                    prey_position = self.agent_positions[agent]
                    # Check if prey is on the same cell as grass
                    caught_grass = next(
                        (grass for grass, grass_position in self.grass_positions.items()
                        if "grass" in grass and np.array_equal(prey_position, grass_position)), None
                    )
                    if caught_grass:
                        print(f"[DEBUG] {agent} caught grass! Prey Reward: {self.reward_prey_eat_grass}")
                        
                        # Reward prey for eating grass
                        rewards[agent] = self.reward_prey_eat_grass
                        self.cumulative_rewards[agent] += rewards[agent]
                        
                        # Remove grass from the cell
                        self.grid_world_state[3, self.grass_positions[caught_grass]] = 0
                        del self.grass_positions[caught_grass]
                        self.grass_energies[caught_grass] = 0
                        self.grass_agents.remove(caught_grass)
                        self.num_grass -= 1
                    else:
                        rewards[agent] = self.reward_prey_step
                    
                    observations[agent] = self._get_observation(agent)
                    self.cumulative_rewards[agent] += rewards[agent]
                    terminations[agent] = False
                    truncations[agent] = False
 
            # max_steps reached
            elif self.current_step >= self.max_steps:
                # Ensure last observation is recorded for bootstrapping
                if agent in self.agent_positions:
                    observations[agent] = self._get_observation(agent)  
                else:
                    # If the agent was already removed, provide a placeholder observation
                    observations[agent] = np.zeros((self.num_obs_channels, self.max_obs_range, self.max_obs_range))

                rewards[agent] = 0.0
                truncations[agent] = True
                terminations[agent] = False

        # Step 4: Handle agent removals 
        for agent in self.agents[:]:
            if terminations[agent]:
                print(f"[DEBUG] Agent {agent} terminated")
                self.agents.remove(agent)

        # Increment step counter
        self.current_step += 1

        # Ensure all truncated agents receive their final observations
        for agent in self.agents:
            if truncations.get(agent, False):
                if agent in self.agent_positions:
                    observations[agent] = self._get_observation(agent)  
                else:
                    # If the agent was already removed, provide a placeholder observation
                    observations[agent] = np.zeros((self.num_obs_channels, self.max_obs_range, self.max_obs_range))

        # Global termination and truncation
        terminations["__all__"] = self.num_prey <= 0 or self.num_predators <= 0
        truncations["__all__"] = self.current_step >= self.max_steps

        # Ensure all agents receive their final observations when "__all__" is truncated
        if truncations["__all__"]:
            for agent in self.agents:
                if agent not in observations:
                    observations[agent] = self._get_observation(agent)

        # Debugging print statements to verify truncation behavior
        #print(f"[DEBUG] Terminations: {terminations}")
        #print(f"[DEBUG] Truncations: {truncations}")
        #print(f"[DEBUG] Observations at truncation: {observations.keys()}")


        return observations, rewards, terminations, truncations, infos
  
    def _get_movement_energy_cost(
        self, agent, current_position, new_position, distance_factor=0.1
    ):
        """
        Calculate the energy cost for moving an agent.

        Args:
            current_position (np.array): Current position of the agent [x, y].
            new_position (np.array): New position of the agent [x, y].
            current_energy (float): Current energy level of the agent.
            distance_factor (float): Scaling factor for the movement energy cost based on distance.

        Returns:
            float: Energy cost of the move.
        """
        current_energy = self.agent_energies[agent]
        distance = np.linalg.norm(new_position - current_position)  # Euclidean distance
        #print(f"Distance: {distance}")
        #print(f"Distance factor: {distance_factor}")
        #print(f"Current energy: {current_energy}")

        # Calculate the energy cost
        energy_cost = distance * distance_factor * current_energy
        return energy_cost

    def _get_time_step_energy_cost(self, agent, energy_depletion_rate=0.01):
        """
        Calculate the energy cost for a time step of the agent.

        """
        if "predator" in agent:
            return energy_depletion_rate * self.initial_energy_predator
        elif "prey" in agent:
            return energy_depletion_rate * self.initial_energy_prey
        
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

    def _get_observation(self, agent):
        """
        Generate an observation for the agent.
        """
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp)
        observation = np.zeros(
            (self.num_obs_channels, self.max_obs_range, self.max_obs_range),
            dtype=np.float64,
        )
        observation[0].fill(1)
        observation[0, xolo:xohi, yolo:yohi] = 0
        observation[1:, xolo:xohi, yolo:yohi] = self.grid_world_state[
            1:, xlo:xhi, ylo:yhi
        ]

        return observation

    def _obs_clip(self, x, y):
        """
        Clip the observation window to the boundaries of the grid_world_state.
        """
        xld, xhd = x - self.max_obs_offset, x + self.max_obs_offset
        yld, yhd = y - self.max_obs_offset, y + self.max_obs_offset
        xlo, xhi = np.clip(xld, 0, self.grid_size - 1), np.clip(
            xhd, 0, self.grid_size - 1
        )
        ylo, yhi = np.clip(yld, 0, self.grid_size - 1), np.clip(
            yhd, 0, self.grid_size - 1
        )
        xolo, yolo = abs(np.clip(xld, -self.max_obs_offset, 0)), abs(
            np.clip(yld, -self.max_obs_offset, 0)
        )
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1
