# external libraries
from pettingzoo.utils.env import AgentID
import numpy as np
from gymnasium import spaces
from typing import List, Tuple
import random

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
            [0, 0],  # stay
            [0, 1],  # move down
            [1, 0],  # move right
        ]),
        initial_energy: float = 10,
        energy_gain_per_step: float = -0.1,
        torus: bool = True,
        random_action_prob: float = 0.1,  # probability of taking a random action
    ):
        # identification agent
        self.agent_type_nr: int = agent_type_nr  # also channel number of agent
        self.agent_name: AgentID = agent_name  # string like "prey_1"
        self.agent_id_nr: int = agent_id_nr  # unique integer per agent

        self.model_state_agent: np.ndarray = model_state_agent
        self.observation_range: int = observation_range
        self.observation_shape: Tuple[int, int, int] = (observation_range, observation_range, n_channels)
        self.motion_range: np.ndarray = motion_range
        self.n_actions_agent: int = len(self.motion_range)
        self.action_space_agent = spaces.Discrete(self.n_actions_agent)
        self.position: np.ndarray = np.zeros(2, dtype=np.int32)  # x and y position
        self.energy: float = initial_energy  # still to implement
        self.energy_gain_per_step: float = energy_gain_per_step
        self.torus: bool = torus
        self.random_action_prob: float = random_action_prob

        self.is_active: bool = False
        self.age: int = 0

        self.x_grid_dim: int = self.model_state_agent.shape[0]
        self.y_grid_dim: int = self.model_state_agent.shape[1]

    def step(self, action: int) -> np.ndarray:
        # Introduce randomness in action selection
        if random.random() < self.random_action_prob:
            action = self.action_space_agent.sample()

        # returns new position of agent "self" given action "action"
        next_position = self.position + np.array(self.motion_range[action])

        if self.torus:
            # Apply torus transformation to handle out-of-bounds movement
            next_position %= [self.x_grid_dim, self.y_grid_dim]
        else:
            # Clip next position to stay within bounds
            next_position = np.clip(next_position, [0, 0], [self.x_grid_dim - 1, self.y_grid_dim - 1])
            

        # Check if the next position is occupied by the same agent type
        if self.model_state_agent[tuple(next_position)] > 0:
            return self.position  # if intended to move to occupied cell of same agent type: don't move

        # Update position
        self.position = next_position
        return self.position