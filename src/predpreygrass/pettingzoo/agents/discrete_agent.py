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
        motion_range: np.ndarray = np.array(
            [
                [-1, 0],  # move left
                [0, -1],  # move up
                [0, 0],  # stay
                [0, 1],  # move down
                [1, 0],  # move right
            ]
        ),
        initial_energy: float = 10,
        energy_gain_per_step: float = -0.1,
        is_torus: bool = False,
        motion_energy_per_distance_unit: float = -0.0,
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
        self.motion_energy_per_distance_unit = motion_energy_per_distance_unit
        self.is_torus: bool = is_torus

        self.is_active: bool = False
        self.age: int = 0

        self.x_grid_dim: int = self.model_state_agent.shape[0]
        self.y_grid_dim: int = self.model_state_agent.shape[1]

    def step(self, action: int) -> np.ndarray:
        self.age += 1
        # update step energy
        # print("self.motion_energy_per_distance_unit", self.motion_energy_per_distance_unit)
        # print(self.agent_name, "=> energy", round(self.energy,2),"steps => energy ",end="")
        self.energy += self.energy_gain_per_step
        # print(round(self.energy,2),end="")
        # returns new position of agent "self" given action "action"
        next_position = self.position + np.array(self.motion_range[action])

        if self.is_torus:
            # Calculate distance to next position in is_torus space
            distance_traveled = np.linalg.norm(self.position - next_position)
            # Apply is_torus transformation to handle out-of-bounds movement
            next_position %= [self.x_grid_dim, self.y_grid_dim]
        else:
            # Clip next position to stay within bounds
            next_position = np.clip(next_position, [0, 0], [self.x_grid_dim - 1, self.y_grid_dim - 1])
            distance_traveled = np.linalg.norm(self.position - next_position)

        # Check if the next position is occupied by the same agent type
        if self.model_state_agent[tuple(next_position)] > 0:
            distance_traveled = 0
            # print(" moves distance", round(distance_traveled,2)," => energy",round(self.energy,2))
            return self.position  # if intended to move to occupied cell of same agent type: don't move
        # update move energy
        energy_gain_per_move = distance_traveled * self.energy * self.motion_energy_per_distance_unit
        self.energy += energy_gain_per_move
        # print(" moves distance", round(distance_traveled,2)," => energy",round(self.energy,2), "energy_gain_per_move", round(energy_gain_per_move,2))
        # Update position
        self.position = next_position
        return self.position
