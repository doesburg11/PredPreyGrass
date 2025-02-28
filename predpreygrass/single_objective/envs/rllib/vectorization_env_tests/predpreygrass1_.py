# external libraries
import numpy as np
import math
from numpy.typing import NDArray
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, Dict, List, Tuple


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()

        self.verbose_engagement: bool = False
        self.verbose_movement: bool = False

        self.max_steps: int = 10000
        self.reward_predator_catch: float = 15.0
        self.reward_prey_eat_grass: float = 5.0
        self.reward_prey_step: float = 0.0
        self.reward_predator_step: float = 0.0
        self.penalty_prey_caught: float = 0.0

        self.current_step: int = 0

        self.energy_loss_per_step_predator= 0.15  #-0.15,  # -0.15 # default
        self.energy_loss_per_step_prey= 0.05 #-0.05,  # -0.05 # default

        # Learning agents
        self.max_num_predators: int = 20
        self.max_num_prey: int = 20
        self.initial_num_predators: int = 20
        self.initial_num_prey: int = 20
        self.current_num_predators: int = 20
        self.current_num_prey: int = 20

        self.initial_energy_predator: float = 5.0
        self.initial_energy_prey: float = 3.0

        self.cumulative_rewards = {}  # Track total rewards per agent

        # self.num_agents: int = self.current_num_predators + self.current_num_prey  # read only property inherited from MultiAgentEnv
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
        self.max_num_grass: int = 30
        self.initial_num_grass: int = 30
        self.current_num_grass: int = 30
        self.initial_energy_grass: float = 2.0
        self.grass_agents: List[AgentID] = [
            f"grass_{k}" for k in range(self.initial_num_grass)
        ]
        self.energy_gain_per_step_grass: float = 0.2

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
        self.agent_positions: Dict[AgentID, Tuple[int,int]] = {}
        self.predator_positions: Dict[AgentID, Tuple[int, int]]  = {}
        self.prey_positions: Dict[AgentID, Tuple[int, int]]  = {}
        self.grass_positions: Dict[AgentID, Tuple[int, int]]  = {}

        # TODO still to implement for reversed lookup
        self.reversed_predator_positions: Dict[Tuple[int, int], AgentID]  = {}
        self.reversed_prey_positions: Dict[Tuple[int, int], AgentID]  = {}
        self.reversed_grass_positions: Dict[Tuple[int, int], AgentID]  = {}

        self.agent_energies: Dict[AgentID, float] = {}
        self.grass_energies: Dict[AgentID, float] = {}
        self.grid_world_state_shape: Tuple[int,int,int] = (
            self.num_obs_channels,
            self.grid_size,
            self.grid_size,
        )

        self.grid_world_state: NDArray[np.float64] = np.zeros(
            self.grid_world_state_shape, dtype=np.float64
        )

        # Mapping actions to movements
        self.action_to_move_tuple: Dict[int, Tuple[int, int]] = {
            0: (0,0),
            1: (-1,0),
            2: (1,0),
            3: (0,-1),
            4: (0,1),
        }
        self.num_actions = len(self.action_to_move_tuple)

        self.observations = {}
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)

        # Step 1: Initialize grid world state
        self.grid_world_state: np.ndarray = np.zeros(
            self.grid_world_state_shape, dtype=np.float64
        )

        # Step 2: Initialize environment variables
        self.current_step = 0
        self.rng = np.random.default_rng(seed)
        # Step 3: initialize the agents
        self.possible_agents: List[AgentID] = (
            [  # max_num of learning agents, placeholder inherited from MultiAgentEnv
                f"predator_{i}" for i in range(self.max_num_predators)
            ]
            + [f"prey_{j}" for j in range(self.max_num_prey)]
        )
        self.agents = [f"predator_{i}" for i in range(self.initial_num_predators)] + [
            f"prey_{j}" for j in range(self.initial_num_prey)
        ]

        # Step 4: Initialize agent positions and energies
        self.agent_positions: Dict[AgentID, Tuple[int, int]] = {}
        self.predator_positions: Dict[AgentID, Tuple[int, int]]  = {}
        self.prey_positions: Dict[AgentID, Tuple[int, int]]  = {}
        self.grass_positions: Dict[AgentID, Tuple[int, int]]  = {}

        self.agent_energies: Dict[AgentID, float] = {}
        self.predator_energies: Dict[AgentID, float] = {}   
        self.prey_energies: Dict[AgentID, float] = {}
        self.grass_energies: Dict[AgentID, float] = {}

        self.reversed_agent_positions: Dict[Tuple[int, int], AgentID] = {}
        self.reversed_predator_positions: Dict[Tuple[int, int], AgentID]  = {}
        self.reversed_prey_positions: Dict[Tuple[int, int], AgentID]  = {}
        self.reversed_grass_positions: Dict[Tuple[int, int], AgentID]  = {}
        
        # Step 5: initialize the grid world state
        self.obsservations: Dict[AgentID, NDArray[np.float64]] = {}
        self.rewards: Dict[AgentID, float] = {agent_id: 0 for agent_id in self.agents}
        self.cumulative_rewards: Dict[AgentID, float] = {agent_id: 0 for agent_id in self.agents}
        self.terminations: Dict[AgentID, bool] = {agent_id: False for agent_id in self.agents}
        self.truncations: Dict[AgentID, bool] = {agent_id: False for agent_id in self.agents}
        self.infos: Dict[AgentID, dict] = {agent_id: {} for agent_id in self.agents}

       # Initialize grid_world_state
        self.grid_world_state: NDArray[np.float64] = np.zeros(
            self.grid_world_state_shape, dtype=np.float64
        )

        # Step 2: Generate unique random positions for predators, prey and grass and assign energy levels
        # Generate all possible positions on the grid
        eligible_predator_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        # Randomly shuffle the positions to ensure randomness
        self.rng.shuffle(eligible_predator_positions)

        # Assign positions to predators
        for i in range(self.initial_num_predators):
            predator_position = eligible_predator_positions[i]  # Get the next shuffled position
            self.predator_positions[f"predator_{i}"] = predator_position
            self.predator_energies[f"predator_{i}"] = self.initial_energy_predator
            self.reversed_predator_positions[predator_position] = f"predator_{i}"  # Update reversed dictionary
            self.grid_world_state[1, *predator_position] = self.initial_energy_predator  # Update grid state


        eligible_prey_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        self.rng.shuffle(eligible_prey_positions)

        for j in range(self.initial_num_prey):
            prey_position = eligible_prey_positions[j]
            self.prey_positions[f"prey_{j}"] = prey_position
            self.prey_energies[f"prey_{j}"] = self.initial_energy_prey 
            self.reversed_prey_positions[prey_position] = f"prey_{j}" 
            self.grid_world_state[2, *prey_position] = self.initial_energy_prey
        
        eligible_grass_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        self.rng.shuffle(eligible_grass_positions)
        for k in range(self.initial_num_grass):
            grass_position = eligible_grass_positions[k]
            self.grass_positions[f"grass_{k}"] = grass_position
            self.grass_energies[f"grass_{k}"] = self.initial_energy_grass
            self.reversed_grass_positions[grass_position] = f"grass_{k}"
            self.grid_world_state[3, *grass_position] = self.initial_energy_grass

        self.agent_positions = {**self.predator_positions, **self.prey_positions}
        self.agent_energies = {**self.predator_energies, **self.prey_energies}
        self.reversed_agent_positions = {**self.reversed_predator_positions, **self.reversed_prey_positions}


        # Step 3: Reset counters
        self.current_num_agents = self.current_num_predators + self.current_num_prey
        self.current_num_predators = self.initial_num_predators
        self.current_num_prey = self.initial_num_prey
        self.current_num_grass = self.initial_num_grass

        # Generate observations
        self.observations = {agent: self._get_observation(agent) for agent in self.agents}

        #print (f"Predator positions: {self.predator_positions}")

        return self.observations, {}

    def step(self, action_dict):
        (
            self.observations, 
            self.rewards, 
            self.terminations, 
            self.truncations, 
            self.infos
         ) = {}, {}, {}, {}, {}

        if len(action_dict) != len(self.agents):
            raise ValueError("Number of actions must match the number of active agents.")

        # Step 1.a: Energy depletion for predators and prey
        for agent in self.agents:
            if "predator" in agent:
                self.agent_energies[agent] -= self.energy_loss_per_step_predator

            else:
                self.agent_energies[agent] -= self.energy_loss_per_step_prey

        # Step 1.b: Energy gain for grass
        for grass in self.grass_agents:
            self.grass_energies[grass] = np.clip(
                self.grass_energies[grass] + self.energy_gain_per_step_grass,
                None,  # No lower bound
                self.initial_energy_grass  # Upper bound
            )

        # Step 2: Compute theoretical movements

        print(f"Agents positions: \n {self.agent_positions}")
        print()
        print(f"Predator positions: \n {self.predator_positions}")
        print()
        print(f"Prey positions: \n {self.prey_positions}")

        

        # Convert positions to NumPy array
        #agent_positions_array = np.array(list(self.agent_positions.values()))  # Shape (N, 2)
        predator_positions_array = np.array(list(self.predator_positions.values()))  # Shape (N, 2)
        prey_positions_array = np.array(list(self.prey_positions.values()))  # Shape (N, 2)



        #print(f"Positions array: \n {positions_array}")

        # Convert actions to move tuples
        predator_actions_array = np.array([
            self.action_to_move_tuple[action_dict[agent]] for agent in self.predator_positions.keys()
        ])
        prey_actions_array = np.array([
            self.action_to_move_tuple[action_dict[agent]] for agent in self.prey_positions.keys()
        ])

        # Define grid boundaries (example: 0 <= x, y < grid_size)
        grid_min = np.array([0, 0])
        grid_max = np.array([self.grid_size-1, self.grid_size-1])   

        # Compute new positions and clip to boundaries
        predator_new_positions = np.clip(predator_positions_array + predator_actions_array, grid_min, grid_max)
        prey_new_positions = np.clip(prey_positions_array + prey_actions_array, grid_min, grid_max)

        # Convert dictionary keys to NumPy array
        predator_agent_ids = np.array(list(self.predator_positions.keys()))
        prey_agent_ids = np.array(list(self.prey_positions.keys()))

        # Detect collisions using NumPy
        unique_predator_positions, predator_indices, predator_counts = np.unique(predator_new_positions, axis=0, return_inverse=True, return_counts=True)
        unique_prey_positions, prey_indices, prey_counts = np.unique(prey_new_positions, axis=0, return_inverse=True, return_counts=True)

        # Find positions where multiple agents exist
        predator_collision_indices = np.where(predator_counts > 1)[0]
        prey_collision_indices = np.where(prey_counts > 1)[0]

        # Build a dictionary of colliding agents
        colliding_predator_agents = {
            tuple(unique_predator_positions[idx]): predator_agent_ids[predator_indices == idx].tolist()
            for idx in predator_collision_indices
        }
        colliding_prey_agents = {
            tuple(unique_prey_positions[idx]): prey_agent_ids[prey_indices == idx].tolist()
            for idx in prey_collision_indices
        }

        print()
        print("{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15}".format(
            "Agent", "Tuple Position", "Array Position", "Action", "Action Array", "New Position", "Energy"))
        print("-" * 100)

        for agent, position in self.predator_positions.items():
            array_position = np.array(position)  # Convert tuple to NumPy array
            action_number = action_dict[agent]  # Get action number from action_dict
            action_array = np.array(self.action_to_move_tuple[action_number])  # Convert action to array
            new_position = np.clip(array_position + action_array, 0, self.grid_size - 1)  # Apply movement
            energy = self.agent_energies[agent]  # Get agent energy

            print("{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15}".format(
                agent, str(position), str(array_position), action_number, str(action_array), str(new_position), str(energy)))
        print("-" * 100)
        print()
        # Print results
        print("Colliding Predators:", colliding_predator_agents)

        print()
        print("{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15}".format(
            "Agent", "Tuple Position", "Array Position", "Action", "Action Array", "New Position", "Energy"))
        print("-" * 100)

        for agent, position in self.prey_positions.items():
            array_position = np.array(position)  # Convert tuple to NumPy array
            action_number = action_dict[agent]  # Get action number from action_dict
            action_array = np.array(self.action_to_move_tuple[action_number])  # Convert action to array
            new_position = np.clip(array_position + action_array, 0, self.grid_size - 1)  # Apply movement
            energy = self.agent_energies[agent]  # Get agent energy

            print("{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15}".format(
                agent, str(position), str(array_position), action_number, str(action_array), str(new_position), str(energy)))
        print("-" * 100)
        print()
        # Print results
        print("Colliding Prey:", colliding_prey_agents)

        # Handle predator collisions using the new array
        self._resolve_agents_of_same_type_collisions(colliding_predator_agents, predator_new_positions)

        # Handle prey collisions using the new array
        self._resolve_agents_of_same_type_collisions(colliding_prey_agents, prey_new_positions)

        # print(f"Agent new Positions: /n {self.agent_positions}")


        # Step 4: Update agent positions **AFTER** resolving collisions
        updated_positions = {
            **{agent: tuple(predator_new_positions[i]) for i, agent in enumerate(self.predator_positions.keys())},
            **{agent: tuple(prey_new_positions[i]) for i, agent in enumerate(self.prey_positions.keys())}
        }

        # Apply the final updated positions
        self.agent_positions.update(updated_positions)


        # Print final agent positions
        print(f"Agent new Positions:\n {self.agent_positions}")

        """
        for agent, position in updated_positions.items():
            self.predator_positions[agent] = position
            self.grid_world_state[1, *position] = self.agent_energies[agent]
            print(f"{agent}: {position} with energy {self.agent_energies[agent]}")
        """




        # Step 5: Update `grid_world_state`
        for agent, position in self.agent_positions.items():
            if "predator" in agent:
                self.predator_positions[agent] = position
                self.grid_world_state[1, *position] = self.agent_energies[agent]
            elif "prey" in agent:
                self.prey_positions[agent] = position
                self.grid_world_state[2, *position] = self.agent_energies[agent]

        #print(f"Agent positions: {self.agent_positions}")

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos
    
    def _resolve_agents_of_same_type_collisions(self, colliding_agents, new_positions_array):
        """
        Resolve collisions among agents of the same type.

        Args:
            colliding_agents: A dictionary mapping positions to lists of agents colliding at that position.
            new_positions_array: NumPy array containing the new positions of all agents of the same type.
        """
        agents_to_remove = []
        
        for position, agents in colliding_agents.items():
            # Sort agents by energy (highest stays, others must move)
            sorted_agents = sorted(agents, key=lambda a: self.agent_energies[a], reverse=True)

            # The highest-energy agent stays at the position, others must move
            for agent in sorted_agents[1:]:
                # Find a new available position using the updated function
                new_pos = self._find_closest_eligible_adjacent_cell(position, new_positions_array)

                if new_pos:
                    self.agent_positions[agent] = new_pos  # Assign new position
                else:
                    # If no position is available, remove the agent
                    agents_to_remove.append(agent)

        # Remove agents that couldn't be relocated
        for agent in agents_to_remove:
            del self.agent_positions[agent]


    def _find_closest_eligible_adjacent_cell(self, position, new_positions_array):
        """
        Find the closest eligible adjacent spot that is not occupied by the same agent type in the new positions.

        Args:
            position: The current position (x, y).
            new_positions_array: NumPy array of new positions for the same agent type.

        Returns:
            Tuple[int, int]: The closest eligible adjacent spot, or None if no spot is available.
        """
        x, y = position
        adjacent_spots = [
            (x + 1, y),  # Right
            (x - 1, y),  # Left
            (x, y + 1),  # Down
            (x, y - 1),  # Up
        ]

        # Ensure spots are within grid boundaries
        adjacent_spots = [
            spot for spot in adjacent_spots 
            if 0 <= spot[0] < self.grid_size and 0 <= spot[1] < self.grid_size
        ]

        # Convert new positions to a set for faster lookups
        occupied_positions = {tuple(pos) for pos in new_positions_array}

        # Find the closest available spot
        available_spots = [spot for spot in adjacent_spots if spot not in occupied_positions]

        # If there is an available spot, return the closest one
        if available_spots:
            available_spots.sort(key=lambda spot: np.linalg.norm(np.array(position) - np.array(spot)))
            return available_spots[0]

        # If no spots are available, return None
        return None

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
    
    def _get_agent_by_position(self) -> dict:
        """
        Reverse the agent_positions dictionary to map positions to agents.

        Returns:
            dict: A dictionary where keys are positions (tuples) and values are agent IDs.
        """
        return {position: agent for agent, position in self.agent_positions.items()}
    
    def _remove_agent(self, agent: AgentID):
        """
        Removes an agent from all tracking dictionaries.
        """
        position = self.agent_positions[agent]
        del self.agent_positions[agent]
        del self.agent_energies[agent]

        if "predator" in agent:
            del self.predator_positions[position]
            self.current_num_predators -= 1
        elif "prey" in agent:
            del self.prey_positions[position]
            self.current_num_prey -= 1


