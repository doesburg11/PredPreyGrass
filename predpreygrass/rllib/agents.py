import numpy as np

class Agent:
    """
    Base class for all agents in the environment.
    """
    def __init__(self, agent_id, position, energy, energy_delta_per_step):
        self.agent_id = agent_id
        self.position = np.array(position)
        self.energy = energy
        self.energy_delta_per_step = energy_delta_per_step




class Predator(Agent):
    """
    Predator agent that hunts prey.
    """
    def __init__(self, agent_id, position, energy, grid_size, energy_loss_per_step, catch_reward):
        super().__init__(agent_id, position, energy, grid_size, energy_loss_per_step)
        self.catch_reward = catch_reward

    def eat(self, prey):
        """
        Consume prey and gain its energy.
        """
        self.energy += prey.energy
        prey.energy = 0  # Mark prey as dead


class Prey(Agent):
    """
    Prey agent that eats grass and avoids predators.
    """
    def __init__(self, agent_id, position, energy, grid_size, energy_loss_per_step, eat_reward):
        super().__init__(agent_id, position, energy, grid_size, energy_loss_per_step)
        self.eat_reward = eat_reward

    def eat(self, grass):
        """
        Consume grass and gain its energy.
        """
        self.energy += grass.energy
        grass.energy = 0  # Mark grass as consumed


class Grass:
    """
    Grass agent that regrows over time.
    """
    def __init__(self, agent_id, position, energy, regrowth_rate):
        self.agent_id = agent_id
        self.position = np.array(position)
        self.energy = energy
        self.regrowth_rate = regrowth_rate

    def grow(self):
        """
        Regrow grass at a fixed rate.
        """
        self.energy = min(self.energy + self.regrowth_rate, 2.0)  # Cap energy at a maximum value
