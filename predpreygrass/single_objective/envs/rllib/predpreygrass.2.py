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

        print(f"Possible agents: {self.possible_agents}")
        print(f"Agents: {self.agents}")
        print(f"Number of possible agents: {self.max_num_agents}")
        print(f"Number of agents: {self.num_agents}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize grid
        self.grid = np.zeros(
            (self.num_obs_channels, self.x_grid_size, self.y_grid_size),
            dtype=np.float64,
        )
        self.grid[0].fill(1.0)  # Initialize the first channel with 1.0

        # Randomly place predator and prey agents
        self.agent_positions = {}
        self.agent_energies = {}
        for agent in self.agents:
            agent_type_nr = 1 if "predator" in agent else 2
            agent_energy = 5 if "predator" in agent else 3
            while True:
                x, y = np.random.randint(self.x_grid_size), np.random.randint(
                    self.y_grid_size
                )
                if self.grid[agent_type_nr, x, y] == 0:
                    self.grid[agent_type_nr, x, y] = (
                        agent_energy  # initial energy of predator
                    )
                    self.agent_energies[agent] = agent_energy
                    self.agent_positions[agent] = [x, y]
                    break
        # Randomly place grass agents
        self.grass_positions = {}
        self.grass_energies = {}
        for grass in self.grass_agents:
            while True:
                x, y = np.random.randint(self.x_grid_size), np.random.randint(
                    self.y_grid_size
                )
                if self.grid[3, x, y] == 0:
                    self.grid[3, x, y] = 1  # initial energy of grass
                    self.grass_positions[grass] = [x, y]
                    self.grass_energies[grass] = 2
                    break

        # Print the grid
        """
        print(self.grid[0])
        print()
        print(self.grid[1])
        print()
        print(self.grid[2])
        print()
        print(self.grid[3])
        """

        # Generate initial observations

        observations = {agent: self._get_observation(agent) for agent in self.agents}
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
        xld = x - self.max_obs_offset
        xhd = x + self.max_obs_offset
        yld = y - self.max_obs_offset
        yhd = y + self.max_obs_offset
        xlo, xhi, ylo, yhi = (
            np.clip(xld, 0, self.x_grid_size - 1),
            np.clip(xhd, 0, self.x_grid_size - 1),
            np.clip(yld, 0, self.y_grid_size - 1),
            np.clip(yhd, 0, self.y_grid_size - 1),
        )
        xolo, yolo = abs(np.clip(xld, -self.max_obs_offset, 0)), abs(
            np.clip(yld, -self.max_obs_offset, 0)
        )
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
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
