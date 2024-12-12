# disrectionary libraries
from predpreygrass.single_objective.agents.discrete_super_agent import DiscreteSuperAgent

# external libraries
from pettingzoo.utils.env import AgentID
import numpy as np

class DiscreteGrassAgent(DiscreteSuperAgent):
    def __init__(
        self,
        agent_type_nr: int,
        agent_id_nr: int,
        agent_name: AgentID,
        initial_energy: float = 4,
        energy_gain_per_step: float = 0.1,
        max_energy_grass: float = 4,
    ):
        super().__init__(agent_type_nr, agent_id_nr, agent_name, initial_energy, energy_gain_per_step)
        # identification agent
        self.max_energy_grass: float = max_energy_grass


