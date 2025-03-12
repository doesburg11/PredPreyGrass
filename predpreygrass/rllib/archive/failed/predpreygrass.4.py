import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        # Learning agents
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

        # Non-learning agents (grass)
        self.max_num_grass = 10
        self.num_grass = 10
        self.grass_agents = [f"grass_{k}" for k in range(self.max_num_grass)]

        # Configuration: grid and observation
        self.x_grid_size = 8
        self.y_grid_size = 8
        self.num_obs_channels = 4
        self.max_obs_range = 7
        self.max_obs_offset = (self.max_obs_range - 1) // 2

        # Observation and action spaces
        obs_space_shape = (
            self.num_obs_channels,
            self.max_obs_range,
            self.max_obs_range,
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
        if seed is not None:
            np.random.seed(seed)

        # Initialize the grid
        self.grid = np.zeros(
            (self.num_obs_channels, self.x_grid_size, self.y_grid_size),
            dtype=np.float64,
        )

        # Helper function for placing entities on the grid
        def place_entities(entity_list, grid_channel, energy_value=None):
            positions, energies = {}, {}
            for entity in entity_list:
                while True:
                    x, y = np.random.randint(self.x_grid_size), np.random.randint(self.y_grid_size)
                    if self.grid[grid_channel, x, y] == 0:  # Ensure no overlap
                        self.grid[grid_channel, x, y] = energy_value or 1
                        positions[entity] = [x, y]
                        if energy_value is not None:
                            energies[entity] = energy_value
                        break
            return positions, energies

        # Place predators and prey
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

        # Place grass
        grass_positions, grass_energies = place_entities(
            self.grass_agents,
            grid_channel=3,
            energy_value=2,
        )

        # Store positions and energies
        self.agent_positions = {**predator_positions, **prey_positions}
        self.agent_energies = {**predator_energies, **prey_energies}
        self.grass_positions = grass_positions
        self.grass_energies = grass_energies

        # Generate initial observations
        observations = {
            agent: self._get_observation(agent) for agent in self.agents
        }

        return observations, {}

    def step(self, action_dict):
        """
        Process a step in the environment.

        Args:
            action_dict (dict): A dictionary mapping agents to their actions.

        Returns:
            tuple: Observations, rewards, terminations, truncations, and info dictionaries.
        """
        observations, rewards = {}, {}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # Process agent actions
        for agent, action in action_dict.items():
            self._apply_action(agent, action)
            observations[agent] = self._get_observation(agent)
            rewards[agent] = self._get_reward(agent)

        # Example termination condition (placeholder)
        terminations["__all__"] = all(terminations.values())
        truncations["__all__"] = False  # Adjust for max step limit if needed

        return observations, rewards, terminations, truncations, {}

    def _obs_clip(self, x, y):
        """
        Clip the observation window to the boundaries of the grid.
        """
        xld, xhd = x - self.max_obs_offset, x + self.max_obs_offset
        yld, yhd = y - self.max_obs_offset, y + self.max_obs_offset
        xlo, xhi = np.clip(xld, 0, self.x_grid_size - 1), np.clip(xhd, 0, self.x_grid_size - 1)
        ylo, yhi = np.clip(yld, 0, self.y_grid_size - 1), np.clip(yhd, 0, self.y_grid_size - 1)
        xolo, yolo = abs(np.clip(xld, -self.max_obs_offset, 0)), abs(np.clip(yld, -self.max_obs_offset, 0))
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1

    def _apply_action(self, agent, action):
        """
        Apply the agent's action to update its position.
        Block the move if the target cell is already occupied by another agent of the same type.
        """
        # Determine agent type (1 for predator, 2 for prey)
        agent_type_nr = 1 if "predator" in agent else 2

        # Get the agent's current position
        x, y = x_old, y_old = self.agent_positions[agent]

        # Calculate the new position based on the action
        if action == 1 and x > 0:  # Move up
            x -= 1
        elif action == 2 and x < self.x_grid_size - 1:  # Move down
            x += 1
        elif action == 3 and y > 0:  # Move left
            y -= 1
        elif action == 4 and y < self.y_grid_size - 1:  # Move right
            y += 1

        # Check if the target cell is occupied by an agent of the same type
        if self.grid[agent_type_nr, x, y] > 0:  # Cell is occupied by the same type
            # Block the move and keep the agent at its original position
            x, y = x_old, y_old

        # Update the agent's position
        self.agent_positions[agent] = [x, y]

        # Update the grid to reflect the new position
        self.grid[agent_type_nr, x_old, y_old] = 0  # Clear the old position
        self.grid[agent_type_nr, x, y] = self.agent_energies[agent]  # Update new position



    def _get_observation(self, agent):
        """
        Generate an observation for the given agent.
        """
        xp, yp = self.agent_positions[agent]
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self._obs_clip(xp, yp)
        observation = np.zeros((self.num_obs_channels, self.max_obs_range, self.max_obs_range), dtype=np.float64)
        observation[0].fill(1)  # Wall channel
        observation[0, xolo:xohi, yolo:yohi] = 0  # Inside the grid
        observation[1:, xolo:xohi, yolo:yohi] = self.grid[1:, xlo:xhi, ylo:yhi]
        return observation

    def _get_reward(self, agent):
        """
        Calculate the reward for the given agent.
        """
        return 0.0


if __name__ == "__main__":
    env = PredPreyGrass()
    observations, _ = env.reset()
    print("Initial Observations:")
    for agent, obs in observations.items():
        print(f"{agent}: \n {obs}")
