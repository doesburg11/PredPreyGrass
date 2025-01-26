import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        # learning agents
        self.max_num_predators = 5
        self.max_num_prey = 5
        self.num_predators = 2
        self.num_prey = 3
        self.possible_agents = [
            f"predator_{i}" for i in range(self.max_num_predators)
        ] + [f"prey_{j}" for j in range(self.max_num_prey)]
        self.agents = [f"predator_{i}" for i in range(self.num_predators)] + [
            f"prey_{j}" for j in range(self.num_prey)
        ]

        # grass agents
        self.max_num_grass = 10
        self.num_grass = 10
        self.grass_agents = [f"grass_{k}" for k in range(self.max_num_grass)]

        # configuration grid
        self.x_grid_size = 8
        self.y_grid_size = 8
        # configuration agents
        self.num_obs_channels = 4
        self.max_obs_range = 7
        self.max_obs_offset = (self.max_obs_range - 1) // 2

        # Observation and action spaces
        obs_space_shape = (
            self.max_obs_range,
            self.max_obs_range,
            self.num_obs_channels,
        )
        observation_space = gym.spaces.Box(
            low=-1.0, high=100.0, shape=obs_space_shape, dtype=np.float64
        )
        self.observation_spaces = {
            agent: observation_space for agent in self.possible_agents
        }
        action_space = gym.spaces.Discrete(5)  # 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
        self.action_spaces = {agent: action_space for agent in self.possible_agents}


    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Args:
            seed (int, optional): Seed for random number generator.
            options (dict, optional): Additional reset options.

        Returns:
            dict: Initial observations for all agents.
            dict: Empty info dictionary.
        """
        super().reset(seed=seed)

        # Seed random number generator
        if seed is not None:
            np.random.seed(seed)

        # Initialize the grid
        self.grid = np.zeros((self.num_obs_channels, self.x_grid_size, self.y_grid_size), dtype=np.float64)

        # Helper function for random placement on the grid of learning agents (predators and prey) and 
        # non-learning agents (grass)
        def place_entity(entity_list, grid_channel, energy_value=None):
            positions = {}
            energies = {}
            for entity in entity_list:
                while True:
                    x, y = np.random.randint(self.x_grid_size), np.random.randint(self.y_grid_size)
                    if self.grid[grid_channel, x, y] == 0:  # Ensure no overlap
                        self.grid[grid_channel, x, y] = energy_value or 1  # Default to 1 if energy not provided
                        positions[entity] = [x, y]
                        if energy_value is not None:
                            energies[entity] = energy_value
                        break
            return positions, energies

        # Place predators and prey
        predator_positions, predator_energies = place_entity(
            [agent for agent in self.agents if "predator" in agent],
            grid_channel=1,
            energy_value=5
        )
        prey_positions, prey_energies = place_entity(
            [agent for agent in self.agents if "prey" in agent],
            grid_channel=2,
            energy_value=3
        )

        # Place grass agents
        grass_positions, grass_energies = place_entity(
            self.grass_agents,
            grid_channel=3,
            energy_value=2
        )

        # Combine agent positions and energies
        self.agent_positions = {**predator_positions, **prey_positions}
        self.agent_energies = {**predator_energies, **prey_energies}
        self.grass_positions = grass_positions
        self.grass_energies = grass_energies

        # Debugging: Print the grid if needed
        if options and options.get("debug", False):
            for i, channel in enumerate(self.grid):
                print(f"Grid channel {i}:\n{channel}\n")

        # Generate initial observations
        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }

        return observations, {}

    def step(self, action_dict):
        rewards = {}
        dones = {agent: False for agent in self.agents}
        observations = {}

        # Process actions
        for agent, action in action_dict.items():
            self._apply_action(agent, action)
            observations[agent] = self._get_observation(agent)
            rewards[agent] = self._get_reward(agent)

        # Example termination condition
        dones["__all__"] = all(dones.values())

        return observations, rewards, dones, {}

    def _obs_clip(self, x, y):
        """
        Clip the observation window to the boundaries of the grid.

        Args:
            x (int): X-coordinate of the agent's position.
            y (int): Y-coordinate of the agent's position.

        Returns:
            tuple: Clipped coordinates for grid and observation matrix:
                - xlo, xhi: Low and high bounds for the X-axis in the grid.
                - ylo, yhi: Low and high bounds for the Y-axis in the grid.
                - xolo, xohi: Low and high bounds for the X-axis in the observation matrix.
                - yolo, yohi: Low and high bounds for the Y-axis in the observation matrix.
        """
        # Calculate raw bounds for the observation window based on max offset
        xld = x - self.max_obs_offset  # Low boundary for X (raw, may go negative)
        xhd = x + self.max_obs_offset  # High boundary for X (raw, may exceed grid size)
        yld = y - self.max_obs_offset  # Low boundary for Y (raw, may go negative)
        yhd = y + self.max_obs_offset  # High boundary for Y (raw, may exceed grid size)

        # Clip the raw bounds to stay within the grid size
        # xlo, xhi: Clipped low and high X bounds for the grid
        # ylo, yhi: Clipped low and high Y bounds for the grid
        xlo, xhi, ylo, yhi = (
            np.clip(xld, 0, self.x_grid_size - 1),  # Ensure X low bound is within the grid
            np.clip(xhd, 0, self.x_grid_size - 1),  # Ensure X high bound is within the grid
            np.clip(yld, 0, self.y_grid_size - 1),  # Ensure Y low bound is within the grid
            np.clip(yhd, 0, self.y_grid_size - 1),  # Ensure Y high bound is within the grid
        )

        # Compute offsets for the observation matrix to align with the clipped grid range
        # xolo, yolo: Offset in the observation matrix for clipped low bounds
        xolo, yolo = (
            abs(np.clip(xld, -self.max_obs_offset, 0)),  # Adjust for negative raw X low bound
            abs(np.clip(yld, -self.max_obs_offset, 0)),  # Adjust for negative raw Y low bound
        )

        # xohi, yohi: Offset in the observation matrix for clipped high bounds
        xohi, yohi = (
            xolo + (xhi - xlo),  # Adjust for grid width included in observation
            yolo + (yhi - ylo),  # Adjust for grid height included in observation
        )

        # Return grid and observation matrix bounds
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def _apply_action(self, agent, action):
        x, y = self.agent_positions[agent]

        # Move the agent based on the action
        if action == 1 and x > 0:  # Move up
            x -= 1
        elif action == 2 and x < self.x_grid_size - 1:  # Move down
            x += 1
        elif action == 3 and y > 0:  # Move left
            y -= 1
        elif action == 4 and y < self.y_grid_size - 1:  # Move right
            y += 1

        # Update the grid and agent position
        self.grid[
            self.agent_positions[agent][0], self.agent_positions[agent][1]
        ].remove(agent)
        self.agent_positions[agent] = [x, y]
        self.grid[x, y].append(agent)

    def _get_observation(self, agent):
        """
        Generate a local observation for the given agent.

        Channel 0: Wall indicator (1 for wall/outside grid, 0 for inside grid).
        Channels 1-3: Grid data (e.g., predators, prey, grass).

        Args:
            agent (str): The name of the agent (e.g., "predator_0", "prey_1").

        Returns:
            np.ndarray: A 3D observation matrix of shape
                        (num_obs_channels, max_obs_range, max_obs_range).
        """
        # Determine agent type and position
        agent_type_nr = 1 if "predator" in agent else 2
        xp, yp = self.agent_positions[agent]

        # Clip the observation range to grid boundaries
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp)

        # Initialize the observation matrix
        observation = np.zeros(
            (self.num_obs_channels, self.max_obs_range, self.max_obs_range),
            dtype=np.float64,
        )

        # Channel 0: Wall indicator (default to walls)
        observation[0].fill(1)  # Start with walls everywhere

        # Set grid cells inside the boundaries to 0
        observation[0, xolo:xohi, yolo:yohi] = 0

        # Fill other channels (1-3) with the relevant grid data
        observation[1:, xolo:xohi, yolo:yohi] = self.grid[1:, xlo:xhi, ylo:yhi]

        return observation

    def _get_reward(self, agent):
        # Example reward logic
        return 0.0


if __name__ == "__main__":
    predpregrass = PredPreyGrass()
    observations, _ = predpregrass.reset()
    print("Initial Observations:")
    for agent, obs in observations.items():
        print(f"{agent}: {obs}")
