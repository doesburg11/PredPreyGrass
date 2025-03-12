import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class PredPreyGrass(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.max_num_predators = 5
        self.max_num_prey = 5
        self.num_predators = 2
        self.num_prey = 3
        self.possible_agents = ["predator_"+str(i) for  i in range(self.max_num_predators)] + ["prey_"+str(j) for  j in range(self.max_num_prey)] 
        self.agents = ["predator_"+str(i) for  i in range(self.num_predators)] + ["prey_"+str(j) for  j in range(self.num_prey)]    
        # self.max_num_agents = self.max_num_predators + self.max_num_prey # already set as read only property in MultiAgentEnv
        # self.num_agents = self.num_predators + self.num_prey # already set in MultiAgentEnv

        print(f"Possible agents: {self.possible_agents}")
        print(f"Agents: {self.agents}")
        print(f"Number of possible agents: {self.max_num_agents}")
        print(f"Number of agents: {self.num_agents}")


        self.x_grid_size = 16
        self.y_grid_size = 16
        self.nr_observation_channels = 3

        # Define the agents in the game.

        # observations space
        # TODO: handle the low/high values correctly; maybe clip energy levels to -1 and-100
        observation_space = gym.spaces.Box(low=-1.0, high=100, shape=(7,7,3), dtype=np.float64)
        self.observation_spaces = {
            agent: observation_space for agent in self.possible_agents
        }
        # end observations

        # action space
        action_space = gym.spaces.Discrete(9)
        self.action_spaces = {
            agent: action_space for agent in self.possible_agents
        }
        # end actions


    def reset(self, *, seed=None, options=None):
        pass

    def step(self, action_dict):
        pass    


if __name__ == "__main__":
    predpregrass = PredPreyGrass()