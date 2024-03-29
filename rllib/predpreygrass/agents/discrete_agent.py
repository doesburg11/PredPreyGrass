import numpy as np

class DiscreteAgent():
    def __init__(
        self,
        agent_type_nr, 
        agent_id_nr,
        agent_name,
        model_state_agent: np.ndarray,
        observation_range=7,
        motion_range = [
                [-1, 0], # move left
                [0, -1], # move up
                [0, 0], # stay
                [0, 1], # move down
                [1, 0], # move right
                ],
        initial_energy=10,
        catch_grass_reward=5.0,
        catch_prey_reward=5.0,
        energy_loss_per_step=-0.1

    ):
        #identification agent
        self.agent_type_nr = agent_type_nr   # also channel number of agent 
        self.agent_name = agent_name   # string like "prey_1"
        self.agent_id_nr = agent_id_nr       # unique integer per agent

        self.model_state_agent = model_state_agent
        self.observation_range = observation_range
        self.motion_range = motion_range
        self.position = np.zeros(2, dtype=np.int32)  # x and y position
        self.energy = initial_energy  
        self.energy_loss_per_step = energy_loss_per_step
        self.catch_grass_reward = catch_grass_reward
        self.catch_prey_reward = catch_prey_reward

        self.is_alive = False

        self.x_grid_dim = self.model_state_agent.shape[0]
        self.y_grid_dim = self.model_state_agent.shape[1]

    def move(self, action : int):
        # returns new position of agent "self" given action "action"

        next_position = np.zeros(2, dtype=np.int32) 
        next_position[0], next_position[1] = self.position[0], self.position[1]

        next_position += self.motion_range[action]
        if not (0 <= next_position[0] < self.x_grid_dim and 0 <= next_position[1] < self.y_grid_dim):
            return self.position   # if moved out of borders: dont move
        elif self.model_state_agent[next_position[0], next_position[1]] > 0:
            return self.position   # if moved to occupied cell of same agent type: dont move
        else:
            self.position = next_position
            return self.position

