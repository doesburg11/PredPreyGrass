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
        ] + [
            f"prey_{j}" for j in range(self.max_num_prey)
        ]
        self.agents = [
            f"predator_{i}" for i in range(self.num_predators)
        ] + [
            f"prey_{j}" for j in range(self.num_prey)
        ]
        
        # configuration grid
        self.x_grid_size = 16
        self.y_grid_size = 16

        # configuration agents
        self.num_obs_channels = 4
        self.max_obs_range = 7

        # Observation and action spaces
        obs_space_shape = (self.max_obs_range, self.max_obs_range, self.num_obs_channels)
        observation_space = gym.spaces.Box(
            low=-1.0, high=100.0, shape=obs_space_shape, dtype=np.float64
        )
        self.observation_spaces = {
            agent: observation_space for agent in self.possible_agents
        }
        action_space = gym.spaces.Discrete(5)  # 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
        self.action_spaces = {
            agent: action_space for agent in self.possible_agents
        }

        print(f"Possible agents: {self.possible_agents}")
        print(f"Agents: {self.agents}")
        print(f"Number of possible agents: {self.max_num_agents}")
        print(f"Number of agents: {self.num_agents}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize grid
        self.grid = np.zeros((self.num_obs_channels, self.x_grid_size, self.y_grid_size), dtype=np.float64)

        # Randomly place agents
        self.agent_positions = {}
        for agent in self.agents:
            while True:
                x, y = np.random.randint(self.x_grid_size), np.random.randint(self.y_grid_size)
                if len(self.grid[x, y]) == 0:  # Ensure no overlap
                    self.grid[x, y].append(agent)
                    self.agent_positions[agent] = [x, y]
                    break

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
        self.grid[self.agent_positions[agent][0], self.agent_positions[agent][1]].remove(agent)
        self.agent_positions[agent] = [x, y]
        self.grid[x, y].append(agent)

    def _get_observation(self, agent):
        max_obs_range = self.max_obs_range
        max_obs_offset = (max_obs_range - 1) // 2
        x_grid_size, y_grid_size = self.x_grid_size, self.y_grid_size
        nr_channels = self.num_obs_channels
        
        x, y = self.agent_positions[agent]
        obs = np.zeros((max_obs_range, max_obs_range, self.num_obs_channels), dtype=np.float64)
        obs[0].fill(1.0)

        # Extract a 7x7 window around the agent
        for dx in range(-max_obs_offset, max_obs_offset + 1):
            for dy in range(-max_obs_offset, max_obs_offset + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.x_grid_size and 0 <= ny < self.y_grid_size:
                    obs[dx + 3, dy + 3, 0] = len(self.grid[nx, ny])  # Example: Number of agents in the cell

        return obs

    def _get_reward(self, agent):
        # Example reward logic
        return 0.0


if __name__ == "__main__":
    predpregrass = PredPreyGrass()
    observations, _ = predpregrass.reset()
    print("Initial Observations:")
    for agent, obs in observations.items():
        print(f"{agent}: {obs}")
