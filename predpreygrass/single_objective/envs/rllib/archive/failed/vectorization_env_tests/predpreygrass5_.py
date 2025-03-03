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
        self.verbose_resulution_same_type_collisions: bool = False
        self.verbose_movement_table: bool = False

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
        self.max_num_predators: int = 10
        self.max_num_prey: int = 10
        self.initial_num_predators: int = 10
        self.initial_num_prey: int = 10
        self.current_num_predators: int = 10
        self.current_num_prey: int = 10

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
        self.max_num_grass: int = 10
        self.initial_num_grass: int = 10
        self.current_num_grass: int = 10
        self.initial_energy_grass: float = 2.0
        self.grass_agents: List[AgentID] = [
            f"grass_{k}" for k in range(self.initial_num_grass)
        ]
        self.energy_gain_per_step_grass: float = 0.2

         # grid_world_state and observation settings
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

        return self.observations, {}

    def step(self, action_dict):
        (
            self.observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos
        ) = {}, {}, {}, {}, {}

        # Initialize termination & truncation dictionaries
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}


        # Apply energy depletion logic
        agent_keys = np.array(list(self.agent_energies.keys()))  # Convert to NumPy array
        # Create an energy array with both predator and prey energy values
        agent_energies = np.array(list(self.agent_energies.values()))
        # Create a mask to differentiate predators and prey
        is_predator = np.array([agent.startswith("predator") for agent in self.agent_energies.keys()])
        is_prey = ~is_predator  # Everything else is prey
        # Apply energy depletion using vectorized operations
        agent_energies[is_predator] -= self.energy_loss_per_step_predator
        agent_energies[is_prey] -= self.energy_loss_per_step_prey
        # Identify agents with zero or negative energy
        dead_agents_mask = agent_energies <= 0
        dead_agents = agent_keys[dead_agents_mask]  # Get agent names

        # Remove dead agents efficiently
        for agent in dead_agents:
            self.terminations[agent] = True  # Mark as terminated
            print(f"Agent {agent} removed due to energy depletion.")
            # Remove from tracking dictionaries
            self.agent_positions.pop(agent, None)
            self.agent_energies.pop(agent, None)
            if "predator" in agent:
                self.predator_positions.pop(agent, None)
                self.current_num_predators -= 1
            elif "prey" in agent:
                self.prey_positions.pop(agent, None)
                self.current_num_prey -= 1       

        # Remove dead agents from active agents list
        self.agents = [agent for agent in self.agents if agent not in dead_agents]


        """
        # Check for simulation termination
        if self.current_num_predators == 0:
            print("All predators are gone. Ending simulation.")
            self.terminations = {agent: True for agent in self.agents}  # Terminate all agents
            self.agents = []  # Clear all agents to stop the simulation
            return self.observations, self.rewards, self.terminations, self.truncations, self.infos
        """

        # Step 3: Vectorized Energy gain for grass (clipped to max)
        grass_energies_array = np.array(list(self.grass_energies.values()))
        grass_energies_array = np.clip(
            grass_energies_array + self.energy_gain_per_step_grass,
            None,  # No lower bound
            self.initial_energy_grass  # Upper bound (max energy)
        )

        # Step 4: Convert NumPy arrays back to dictionaries
        # is this necessary?
        self.agent_energies = dict(zip(self.agent_energies.keys(), agent_energies))
        self.grass_energies = dict(zip(self.grass_energies.keys(), grass_energies_array))
        
        # Step 5: Compute poisition after actions
        predator_positions_array = np.array(list(self.predator_positions.values()))
        prey_positions_array = np.array(list(self.prey_positions.values()))

        predator_actions_array = np.array([
            self.action_to_move_tuple[action_dict[agent]] for agent in self.predator_positions.keys()
        ])
        prey_actions_array = np.array([
            self.action_to_move_tuple[action_dict[agent]] for agent in self.prey_positions.keys()
        ])

        # Define grid boundaries
        grid_min = np.array([0, 0])
        grid_max = np.array([self.grid_size - 1, self.grid_size - 1])

        # Compute initial new positions after actions and clip within boundaries
        predator_new_position = np.clip(predator_positions_array + predator_actions_array, grid_min, grid_max)
        prey_new_unresolved_positions = np.clip(prey_positions_array + prey_actions_array, grid_min, grid_max)

        predator_position_after_action = predator_new_position.copy()
        prey_position_after_action = prey_new_unresolved_positions.copy()

        # Convert dictionary keys to NumPy arrays
        predator_agent_ids = np.array(list(self.predator_positions.keys()))
        prey_agent_ids = np.array(list(self.prey_positions.keys()))

        # Detect collisions using NumPy
        unique_predator_positions, predator_indices, predator_counts = np.unique(predator_new_position, axis=0, return_inverse=True, return_counts=True)
        unique_prey_positions, prey_indices, prey_counts = np.unique(prey_new_unresolved_positions, axis=0, return_inverse=True, return_counts=True)

        predator_collision_indices = np.where(predator_counts > 1)[0]
        prey_collision_indices = np.where(prey_counts > 1)[0]

        # Build dictionaries of colliding agents
        colliding_predator_agents = {
            tuple(unique_predator_positions[idx]): predator_agent_ids[predator_indices == idx].tolist()
            for idx in predator_collision_indices
        }
        colliding_prey_agents = {
            tuple(unique_prey_positions[idx]): prey_agent_ids[prey_indices == idx].tolist()
            for idx in prey_collision_indices
        }

        # Handle collisions
        self._resolve_agents_of_same_type_collisions(colliding_predator_agents, predator_new_position)
        self._resolve_agents_of_same_type_collisions(colliding_prey_agents, prey_new_unresolved_positions)

        # Step 4: Update agent positions **AFTER** resolving collisions
        resolved_positions = {
            **{agent: tuple(predator_new_position[i]) for i, agent in enumerate(self.predator_positions.keys())},
            **{agent: tuple(prey_new_unresolved_positions[i]) for i, agent in enumerate(self.prey_positions.keys())}
        }

        # Step 5: Call the **new movement table function**
        if self.verbose_movement_table:
            self._print_movement_table(action_dict, predator_position_after_action, prey_position_after_action, resolved_positions, colliding_predator_agents, colliding_prey_agents)



        # Apply the final updated positions
        self.agent_positions.update(resolved_positions)

        # Fix: Ensure predator_positions & prey_positions are correctly updated
        self.predator_positions.clear()
        self.prey_positions.clear()

        for agent, pos in self.agent_positions.items():
            if "predator" in agent:
                self.predator_positions[agent] = pos
            elif "prey" in agent:
                self.prey_positions[agent] = pos


        # Step 6: Update `grid_world_state`
        # Reset only predator (1st) and prey (2nd) channels to zero
        self.grid_world_state[1:3, :, :] = 0  

        # Convert dictionary keys & values into NumPy arrays for fast processing
        positions_array = np.array(list(self.agent_positions.values()))
        energies_array = np.array(list(self.agent_energies.values()))

        # Convert agent names & positions to arrays
        agent_keys = np.array(list(self.agent_positions.keys()))
        positions_array = np.array(list(self.agent_positions.values()))
        energies_array = np.array(list(self.agent_energies.values()))

        # Ensure only existing predator/prey are processed
        is_predator = np.array(["predator" in agent for agent in agent_keys])
        is_prey = np.array(["prey" in agent for agent in agent_keys])

        # Update grid state using NumPy
        self.grid_world_state[1, positions_array[is_predator, 0], positions_array[is_predator, 1]] = energies_array[is_predator]
        self.grid_world_state[2, positions_array[is_prey, 0], positions_array[is_prey, 1]] = energies_array[is_prey]

        # ✅ Ensure `__all__` exists
        self.terminations["__all__"] = len(self.agents) == 0
        self.truncations["__all__"] = len(self.agents) == 0


        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def _resolve_agents_of_same_type_collisions(self, colliding_agents, new_positions_array):
        """
        Resolve collisions among agents of the same type dynamically.

        Args:
            colliding_agents: A dictionary mapping positions to lists of agents colliding at that position.
            new_positions_array: NumPy array containing the new positions of all agents of the same type.
        """

        if not colliding_agents:
            return  # No collisions to resolve

        # Determine agent type (predator or prey)
        agent_type = "predator" if "predator" in list(colliding_agents.values())[0][0] else "prey"

        # Separate handling for prey
        if agent_type == "predator":
            agent_list = list(self.predator_positions.keys())
        else:
            agent_list = list(self.prey_positions.keys())  # ✅ Track prey separately

        agent_index_map = {agent: i for i, agent in enumerate(agent_list)}
        occupied_positions = set(map(tuple, new_positions_array))  # Track occupied positions dynamically

        for disputed_position, agents in colliding_agents.items():
            x, y = disputed_position
            adjacent_spots = [
                (x + 1, y),  # Right
                (x - 1, y),  # Left
                (x, y + 1),  # Down
                (x, y - 1),  # Up
            ]
            
            # Prevent Out-of-Bounds Movement
            adjacent_spots = [
                (nx, ny) for nx, ny in adjacent_spots 
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size
            ]

            # Sort agents by energy (highest stays, others move)
            sorted_agents = sorted(agents, key=lambda a: self.agent_energies[a], reverse=True)

            # The strongest agent stays in place
            winner_agent = sorted_agents[0]
            self.agent_positions[winner_agent] = disputed_position
            if self.verbose_resulution_same_type_collisions:
                print(f"Winner at {disputed_position}: {winner_agent}")
            occupied_positions.add(disputed_position)

            # ✅ Safely update `new_positions_array` using fixed index map
            if winner_agent in agent_index_map:
                winner_idx = agent_index_map[winner_agent]
                if winner_idx < len(new_positions_array):  # Prevent IndexError
                    new_positions_array[winner_idx] = np.array(disputed_position)


            # losing agents need to move to an adjacent spot or, if that is not possible, die
            loser_agents = sorted_agents[1:]
            if self.verbose_resulution_same_type_collisions:
                print(f"Losers at {disputed_position}: {loser_agents}")

            for loser_agent in loser_agents:
                found_spot = False
                if self.verbose_resulution_same_type_collisions:
                    print(f"Agent {loser_agent} is looking for a new spot.")

                for spot in adjacent_spots:
                    if spot in occupied_positions:
                        if self.verbose_resulution_same_type_collisions:
                            print(f"Spot {spot} is occupied. Skipping.")
                        continue

                    # Assign agent to the newbvailable spot
                    if self.verbose_resulution_same_type_collisions:
                        print(f"Spot {spot} is available for {loser_agent}. Moving there.")
                    self.agent_positions[loser_agent] = spot
                    occupied_positions.add(spot)

                    # Safely update `new_positions_array` using fixed index map
                    if loser_agent in agent_index_map:
                        loser_idx = agent_index_map[loser_agent]
                        if loser_idx < len(new_positions_array):  # 🔥 Prevent IndexError
                            new_positions_array[loser_idx] = np.array(spot)

                    found_spot = True
                    break

                if not found_spot:
                    # If no adjacent spot is available, remove the agent
                    if self.verbose_resulution_same_type_collisions:    
                        print(f"Agent {loser_agent} died due to lack of available spots.")
                    self._remove_agent(loser_agent)
        print()

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
        if agent not in self.agent_positions:
            print(f"Warning: Trying to remove {agent}, but it is not in agent_positions.")
            return  # Avoid KeyError if agent is already removed

        position = self.agent_positions[agent]  # Get the correct position

        # ✅ Safely remove from agent_positions
        del self.agent_positions[agent]
        del self.agent_energies[agent]

        if "predator" in agent:
            if agent in self.predator_positions:
                del self.predator_positions[agent]  # ✅ Remove using agent key, not position
            self.current_num_predators -= 1
        elif "prey" in agent:
            if agent in self.prey_positions:
                del self.prey_positions[agent]  # ✅ Remove using agent key, not position
            self.current_num_prey -= 1

    def _print_grid_state(self):
        print("\nCurrent Grid State (IDs):\n")

        predator_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        prey_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grass_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Populate Predator Grid
        for agent, pos in self.predator_positions.items():
            x, y = pos
            agent_num = int(agent.split('_')[1])
            predator_grid[y][x] = f"P{agent_num:02d}".center(5)

        # Populate Prey Grid
        for agent, pos in self.prey_positions.items():
            x, y = pos
            agent_num = int(agent.split('_')[1])
            prey_grid[y][x] = f"p{agent_num:02d}".center(5)

        # Populate Grass Grid
        for agent, pos in self.grass_positions.items():
            x, y = pos
            agent_num = int(agent.split('_')[1])
            grass_grid[y][x] = f"G{agent_num:02d}".center(5)

        # Print Headers
        print(f"{'Predators'.center(self.grid_size * 6)}   {'Prey'.center(self.grid_size * 6)}   {'Grass'.center(self.grid_size * 6)}")
        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

        # Print Grids
        for y in range(self.grid_size):
            predator_row = " ".join(predator_grid[y])
            prey_row = " ".join(prey_grid[y])
            grass_row = " ".join(grass_grid[y])
            print(f"{predator_row}     {prey_row}     {grass_row}")

        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

    def _print_grid_from_state(self):
        print("\nCurrent Grid State (Energy Levels):\n")

        predator_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        prey_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grass_grid = [["  .  " for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Iterate over the grid state to fill energy values
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                predator_energy = self.grid_world_state[1, x, y]  # 1st channel: Predators
                prey_energy = self.grid_world_state[2, x, y]      # 2nd channel: Prey
                grass_energy = self.grid_world_state[3, x, y]     # 3rd channel: Grass

                if predator_energy > 0:
                    predator_grid[y][x] = f"{predator_energy:4.2f}".center(5)
                if prey_energy > 0:
                    prey_grid[y][x] = f"{prey_energy:4.2f}".center(5)
                if grass_energy > 0:
                    grass_grid[y][x] = f"{grass_energy:4.2f}".center(5)

        # Print Headers
        print(f"{'Predators'.center(self.grid_size * 6)}   {'Prey'.center(self.grid_size * 6)}   {'Grass'.center(self.grid_size * 6)}")
        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

        # Print Grids
        for y in range(self.grid_size):
            predator_row = " ".join(predator_grid[y])
            prey_row = " ".join(prey_grid[y])
            grass_row = " ".join(grass_grid[y])
            print(f"{predator_row}     {prey_row}     {grass_row}")

        print("=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6, "  ", "=" * self.grid_size * 6)

    def _print_movement_table(self, action_dict, predator_position_after_action, prey_new_unresolved_positions, resolved_positions, colliding_predator_agents, colliding_prey_agents):
        """
        Prints the movement table for predators and prey, including actions, positions, and energy levels.
        """

        print("\nPredator Position Table:")
        print("{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<20}".format(
            "Agent", "Tuple", "Energy",  "Array", "Action", "Action Array", "New", "Resolved"
        ))
        print("-" * 120)

        for i, (agent, position) in enumerate(self.predator_positions.items()):
            array_position = np.array(position)
            action_number = action_dict[agent]
            action_array = np.array(self.action_to_move_tuple[action_number])
            new_position = predator_position_after_action[i]
            resolved_position = resolved_positions[agent]  # Position after collision resolution
            energy = self.agent_energies[agent]

            print("{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<20}".format(
                agent, str(position), f"{energy:.2f}", str(array_position), action_number, str(action_array),
                str(new_position), str(resolved_position)
            ))

        print("-" * 120)
        print()
        print("Colliding Predators:", colliding_predator_agents)
        print()

        print("\nPrey Position Table:")
        print("{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<20}".format(
            "Agent", "Tuple", "Energy", "Array", "Action", "Action Array", "New", "Resolved"
        ))
        print("-" * 120)

        for i, (agent, position) in enumerate(self.prey_positions.items()):
            array_position = np.array(position)
            action_number = action_dict[agent]
            action_array = np.array(self.action_to_move_tuple[action_number])
            new_position = prey_new_unresolved_positions[i]
            resolved_position = resolved_positions[agent]  # Position after collision resolution
            energy = self.agent_energies[agent]

            print("{:<12} {:<15} {:<15} {:<10} {:<15} {:<15} {:<15} {:<20}".format(
                agent, str(position), f"{energy:.2f}", str(array_position), action_number, str(action_array),
                str(new_position), str(resolved_position)
            ))

        print("-" * 120)
        print()
        print("Colliding Prey:", colliding_prey_agents)
        print()
