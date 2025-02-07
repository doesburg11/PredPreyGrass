# external libraries
from pettingzoo.utils.env import AgentID
import numpy as np
from gymnasium import spaces
from typing import Tuple

class DiscreteAgent:
    def __init__(
        self,
        agent_type_nr: int,
        agent_id_nr: int,
        agent_name: AgentID,
        model_state_agent: np.ndarray,
        observation_range: int = 7,
        n_channels: int = 4,  # number of observation channels
        motion_range: np.ndarray = np.array([
            [-1, 0],  # move left
            [0, -1],  # move up
            [0, 0],   # stay
            [0, 1],   # move down
            [1, 0],   # move right
        ]),
        initial_energy: float = 10,
        energy_gain_per_step: float = -0.1,
        is_torus: bool = False,
        motion_energy_per_distance_unit: float = -0.0,
        random_action_prob: float = 0.1,  # NEW
    ):
        self.agent_type_nr: int = agent_type_nr
        self.agent_name: AgentID = agent_name
        self.agent_id_nr: int = agent_id_nr
        self.model_state_agent: np.ndarray = model_state_agent
        self.observation_range: int = observation_range
        self.observation_shape: Tuple[int, int, int] = (observation_range, observation_range, n_channels)
        self.motion_range: np.ndarray = motion_range
        self.n_actions_agent: int = len(self.motion_range)
        self.action_space_agent = spaces.Discrete(self.n_actions_agent)
        self.position: np.ndarray = np.zeros(2, dtype=np.int32)
        self.energy: float = initial_energy
        self.energy_gain_per_step: float = energy_gain_per_step
        self.motion_energy_per_distance_unit = motion_energy_per_distance_unit
        self.is_torus: bool = is_torus
        self.is_active: bool = False
        self.age: int = 0
        self.x_grid_dim: int = self.model_state_agent.shape[0]
        self.y_grid_dim: int = self.model_state_agent.shape[1]
        self.random_action_prob = random_action_prob  # NEW

    def random_action(self) -> int:
        """Selects a random valid action."""
        return np.random.randint(self.n_actions_agent)

    def step(self, action: int) -> np.ndarray:
        if np.random.rand() < self.random_action_prob and self.agent_id_nr == 2:
            action = self.random_action()  # Override action with random

        self.age += 1
        self.energy += self.energy_gain_per_step
        next_position = self.position + np.array(self.motion_range[action])

        if self.is_torus:
            next_position %= [self.x_grid_dim, self.y_grid_dim]
        else:
            next_position = np.clip(next_position, [0, 0], [self.x_grid_dim - 1, self.y_grid_dim - 1])

        if self.model_state_agent[tuple(next_position)] > 0:
            return self.position

        distance_traveled = np.linalg.norm(self.position - next_position)
        self.energy += distance_traveled * self.energy * self.motion_energy_per_distance_unit
        self.position = next_position
        return self.position