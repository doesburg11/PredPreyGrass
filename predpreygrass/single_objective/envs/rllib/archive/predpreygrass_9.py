import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from renderer import GridVisualizer


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.max_steps = config.get("max_steps", 100) if config else 100
        self.current_step = 0

        # Learning agents
        self.max_num_predators = 10
        self.max_num_prey = 5
        self.num_predators = 10
        self.num_prey = 5
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

        # Grid and observation settings
        self.x_grid_size = 8
        self.y_grid_size = 8
        self.num_obs_channels = 4
        self.max_obs_range = 7
        self.max_obs_offset = (self.max_obs_range - 1) // 2

        # Spaces
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
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.rng = np.random.default_rng(seed)

        # Initialize grid
        self.grid = np.zeros(
            (self.num_obs_channels, self.x_grid_size, self.y_grid_size),
            dtype=np.float64,
        )

        # Place entities
        def place_entities(entity_list, grid_channel, energy_value=None):
            positions, energies = {}, {}
            for entity in entity_list:
                while True:
                    x, y = self.rng.integers(self.x_grid_size), self.rng.integers(self.y_grid_size)
                    if self.grid[grid_channel, x, y] == 0:
                        self.grid[grid_channel, x, y] = energy_value or 1
                        positions[entity] = [x, y]
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
        """
        Perform one step in the environment.
        """
        observations, rewards = {}, {}
        terminations = {}
        truncations = {}
        info = {}

        # Process actions
        for agent, action in action_dict.items():
            if agent in self.agent_positions:  # Only act on existing agents
                self._apply_action(agent, action)
                observations[agent] = self._get_observation(agent)
                rewards[agent] = self._get_reward(agent)
                info[agent] = {
                    "energy": self.agent_energies.get(agent, 0),
                    "position": self.agent_positions.get(agent),
                }

        # Check termination and truncation conditions
        for agent in list(self.agents):
            if agent not in self.agent_positions:  # Agent already removed
                continue
            agent_type_nr = 1 if "predator" in agent else 2
            if self.agent_energies[agent] <= 0:
                # Agent has no energy left
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
                    if "predator" in predator and predator_position == prey_position:
                        print(f"Predator {predator} caught prey {agent}!")
                        # Prey is caught
                        self.num_prey -= 1
                        print(f"Number of prey left: {self.num_prey}")
                        x, y = prey_position
                        self.grid[agent_type_nr, x, y] = 0  # Clear the prey from the grid (channel 2)
                        del self.agent_positions[agent]
                        del self.agent_energies[agent]
                        terminations[agent] = True
                        truncations[agent] = False
                        break
            else:
                terminations[agent] = False
                truncations[agent] = False

        # Increment step counter
        self.current_step += 1

        # Global termination and truncation
        terminations["__all__"] = self.num_prey == 0 or self.current_step >= self.max_steps
        truncations["__all__"] = self.current_step >= self.max_steps

        return observations, rewards, terminations, truncations, info

    def _apply_action(self, agent, action):
        """
        Apply the agent's action and block moves into occupied cells.
        """
        agent_type_nr = 1 if "predator" in agent else 2
        x, y = x_old, y_old = self.agent_positions[agent]

        if action == 1 and x > 0:
            x -= 1
        elif action == 2 and x < self.x_grid_size - 1:
            x += 1
        elif action == 3 and y > 0:
            y -= 1
        elif action == 4 and y < self.y_grid_size - 1:
            y += 1

        if self.grid[agent_type_nr, x, y] > 0:
            x, y = x_old, y_old

        self.agent_positions[agent] = [x, y]
        self.grid[agent_type_nr, x_old, y_old] = 0
        self.grid[agent_type_nr, x, y] = self.agent_energies[agent]

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
        if "predator" in agent:
            # Reward predators for catching prey
            return 10 if any(
                predator_pos == self.agent_positions[agent]
                for predator_pos in self.agent_positions.values()
            ) else -0.1  # Small penalty for not catching prey
        elif "prey" in agent:
            # Negative reward for prey when caught
            return -10 if agent not in self.agent_positions else 0.1  # Small reward for survival
        return 0.0

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

from time import sleep

if __name__ == "__main__":
    # Grid size
    env = PredPreyGrass()

    # Reset the environment and get initial observations
    observations, _ = env.reset(seed=42)
    print("Initial Observations:")
    grid_size = (env.x_grid_size, env.y_grid_size)

    # Combine predator, prey, and grass agents
    all_agents = env.agents + env.grass_agents

    # Initialize the visualizer
    visualizer = GridVisualizer(grid_size, all_agents, trace_length=1)

    for step in range(1000):  # Arbitrary large number to test termination
        print(f"Step {step + 1}")
        # Generate random actions for all agents
        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}

        # Perform a step
        observations, rewards, terminations, truncations, info = env.step(action_dict)

        # Merge agent and grass positions
        merged_positions = {**env.agent_positions, **env.grass_positions}

        # Update visualization
        visualizer.update(merged_positions)

        # Debug information
        print(f"Number of prey left: {env.num_prey}")
        print(f"Terminations: {terminations['__all__']}, Truncations: {truncations['__all__']}")

        # Check if the environment is properly terminating
        if terminations["__all__"]:
            print("Environment terminated properly.")
            break

        sleep(0.1)  # Slow down visualization

    visualizer.close()
