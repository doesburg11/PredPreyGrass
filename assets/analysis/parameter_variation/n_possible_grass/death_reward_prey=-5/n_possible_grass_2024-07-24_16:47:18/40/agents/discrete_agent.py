import numpy as np
from gymnasium import spaces
from typing import List, Tuple

from pettingzoo.utils.env import AgentID

class DiscreteAgent:
    def __init__(
        self,
        agent_type_nr: int,
        agent_id_nr: int,
        agent_name: AgentID,
        model_state_agent: np.ndarray,
        observation_range: int = 7,
        n_channels: int = 4,  # number of observation channels
        motion_range: List[List[int]] = [
            [-1, 0],  # move left
            [0, -1],  # move up
            [0, 0],  # stay
            [0, 1],  # move down
            [1, 0],  # move right
        ],
        initial_energy: float = 10,
        energy_gain_per_step: float = -0.1,
    ):
        # identification agent
        self.agent_type_nr: int = agent_type_nr  # also channel number of agent
        self.agent_name: AgentID = agent_name  # string like "prey_1"
        self.agent_id_nr: int = agent_id_nr  # unique integer per agent

        self.model_state_agent: np.ndarray = model_state_agent
        self.observation_range: int = observation_range
        self.observation_shape: Tuple[int, int, int] = (observation_range, observation_range, n_channels)
        self.motion_range: List[List[int]] = motion_range
        self.n_actions_agent: int = len(self.motion_range)
        self.action_space_agent = spaces.Discrete(self.n_actions_agent)
        self.position: np.ndarray = np.zeros(2, dtype=np.int32)  # x and y position
        self.energy: float = initial_energy  # still to implement
        self.energy_gain_per_step: float = energy_gain_per_step

        self.is_alive: bool = False
        self.age: int = 0

        self.x_grid_dim: int = self.model_state_agent.shape[0]
        self.y_grid_dim: int = self.model_state_agent.shape[1]
        # TODO: yet to implement for generalization
        # in general, an agent type can have more pursuers or recource agents; 
        # therefore a list
        self.pursuer_agent_type_nr: List[int] = None
        self.resource_agent_type_nr: List[int] = None

    def step(self, action: int) -> np.ndarray:
        # returns new position of agent "self" given action "action"

        self.age += 1

        next_position: np.ndarray = np.zeros(2, dtype=np.int32)
        next_position[0], next_position[1] = self.position[0], self.position[1]

        next_position += self.motion_range[action]
        if not (
            0 <= next_position[0] < self.x_grid_dim
            and 0 <= next_position[1] < self.y_grid_dim
        ):
            return self.position  # if moved out of borders: dont move
        elif self.model_state_agent[next_position[0], next_position[1]] > 0:
            return (
                self.position
            )  # if intended to moved to occupied cell of same agent type: dont move
        else:
            self.position = next_position
            return self.position